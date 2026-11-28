//! Build a search index from a raw dataset dump and time single-threaded, one-at-a-time
//! queries on one pinned core, printing one JSON line per (build, query parameters).
//!
//! The dump directory (see the VIBE repository's dev/rinnd/dump_raw.py) holds `meta.txt`
//! ("n dim n_test k n_learn learn_width distance"), `train.f32`, `test.f32`, `neighbors.i32`
//! and optionally `learn.f32` and `learn_neighbors.i32`, little-endian. Distances are
//! "euclidean", "cosine"/"normalized" (searched as 1 - dot on unit vectors) or "ip" (maximum
//! inner product over points augmented to a common norm, as the Python `metric="mips"`).
//! Recall is computed as VIBE does: a returned point counts if its distance (euclidean,
//! 1 - cosine, or minus the inner product) is within 1e-3 of the true k-th neighbour's.
//!
//! Usage: search_bench <dump dir> key=value...
//!   nn=30 pdm=1.5 prob=1.0 alpha=1.0 order=tree|rcm quant=none|sq8|sq4 half=0|1 learn=0|1
//!   rev=0|1 (diversify reverse edges too)
//!   eps=0,0.01,0.02 cw=100,150,200 (quantized only) k=100 nq=1000 runs=3 cpu=<query core>
//!   threads=<build threads> seed=42 label=<free text>
//!   variants=<overrides>;<overrides>... (e.g. rev=0:alpha=1.2;rev=1): build each, interleave timing
//! Build with RAYON_NUM_THREADS or threads=; the query loop runs on `cpu` alone.

use rinnd_core::distance::affine::{
    AffineCodes, AffineKind, AffineQuery, HalfQuery, HalfRows, DEFAULT_SQ4_CLIP_QUANTILE,
};
use rinnd_core::distance::{DirectNormalizedCosine, Distance, Metric, SquaredEuclidean};
use rinnd_core::index::{NNDescentBuilder, NNDescentIndex, VertexOrder};
use rinnd_core::query_sample::{QuerySample, QuerySampleOptions};
use rinnd_core::search::SearchWorkspace;
use std::collections::HashMap;
use std::time::Instant;

struct Dump {
    n: usize,
    dim: usize,
    n_test: usize,
    k_true: usize,
    train: Vec<f32>,
    test: Vec<f32>,
    neighbors: Vec<i32>,
    learn: Option<(Vec<f32>, Vec<i32>, usize, usize)>,
    distance: String,
}

impl Dump {
    fn row(&self, index: i32) -> &[f32] {
        &self.train[index as usize * self.dim..(index as usize + 1) * self.dim]
    }
}

/// The distance VIBE's ground truth uses.
fn vibe_distance(distance: &str, a: &[f32], b: &[f32]) -> f32 {
    let dot = |x: &[f32], y: &[f32]| {
        x.iter()
            .zip(y)
            .map(|(p, q)| (*p as f64) * (*q as f64))
            .sum::<f64>()
    };
    match distance {
        "euclidean" => a
            .iter()
            .zip(b)
            .map(|(p, q)| ((p - q) as f64).powi(2))
            .sum::<f64>()
            .sqrt() as f32,
        "cosine" | "normalized" => (1.0 - dot(a, b) / (dot(a, a).sqrt() * dot(b, b).sqrt())) as f32,
        "ip" => -dot(a, b) as f32,
        other => panic!("distance {other}"),
    }
}

fn read_f32(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_i32(path: &str) -> Vec<i32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn load(dir: &str, with_learn: bool) -> Dump {
    let meta = std::fs::read_to_string(format!("{dir}/meta.txt")).unwrap();
    let f: Vec<&str> = meta.split_whitespace().collect();
    let p = |i: usize| f[i].parse::<usize>().unwrap();
    let (n, dim, n_test, k_true, n_learn, learn_width) = (p(0), p(1), p(2), p(3), p(4), p(5));
    let learn = (with_learn && n_learn > 0).then(|| {
        (
            read_f32(&format!("{dir}/learn.f32")),
            read_i32(&format!("{dir}/learn_neighbors.i32")),
            n_learn,
            learn_width,
        )
    });
    Dump {
        n,
        dim,
        n_test,
        k_true,
        train: read_f32(&format!("{dir}/train.f32")),
        test: read_f32(&format!("{dir}/test.f32")),
        neighbors: read_i32(&format!("{dir}/neighbors.i32")),
        learn,
        distance: f[6].to_string(),
    }
}

fn normalize(rows: &mut [f32], dim: usize) {
    for row in rows.chunks_exact_mut(dim) {
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            row.iter_mut().for_each(|x| *x /= norm);
        }
    }
}

fn pad(rows: &[f32], dim: usize) -> Vec<f32> {
    rows.chunks_exact(dim)
        .flat_map(|row| row.iter().copied().chain([0.0]))
        .collect()
}

#[cfg(target_os = "linux")]
fn pin(cpu: usize) {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu, &mut set);
        assert_eq!(
            libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set),
            0,
            "pin to {cpu}"
        );
    }
}

/// Pinning is Linux-only; elsewhere the query loop runs wherever it is scheduled.
#[cfg(not(target_os = "linux"))]
fn pin(_cpu: usize) {}

#[derive(Clone)]
struct Args(HashMap<String, String>);

impl Args {
    fn get<'a>(&'a self, key: &str, default: &'a str) -> &'a str {
        self.0.get(key).map(String::as_str).unwrap_or(default)
    }
    fn num<T: std::str::FromStr>(&self, key: &str, default: &str) -> T
    where
        T::Err: std::fmt::Debug,
    {
        self.get(key, default).parse().unwrap()
    }
    fn list(&self, key: &str, default: &str) -> Vec<f32> {
        self.get(key, default)
            .split(',')
            .map(|v| v.parse().unwrap())
            .collect()
    }
}

/// One build of the index (the base arguments with a variant's overrides).
struct Variant<'a, D: Distance<f32>> {
    args: Args,
    label: &'a str,
    index: NNDescentIndex<D>,
    graph_build: f64,
}

#[allow(clippy::too_many_arguments)]
fn build<D: Distance<f32>>(
    args: &Args,
    dump: &Dump,
    distance: D,
    correction: Option<fn(f32) -> f32>,
    metric: Metric,
    data: Vec<f32>,
    index_dim: usize,
    learn: &Option<(Vec<f32>, &[i32], usize, usize)>,
) -> (NNDescentIndex<D>, f64) {
    let alpha: f32 = args.num("alpha", "1.0");
    let order = match args.get("order", "tree") {
        "tree" => VertexOrder::Tree,
        "rcm" => VertexOrder::Rcm,
        other => panic!("order {other}"),
    };
    let n_trees = ((2.0 * (dump.n as f64).log10()).round() as usize).clamp(3, 12);
    let started = Instant::now();
    let mut index: NNDescentIndex<D> = NNDescentBuilder::from_vec(data, dump.n, index_dim)
        .metric(metric)
        .n_neighbors(args.num("nn", "30"))
        .n_trees(n_trees)
        .random_seed(args.num("seed", "42"))
        .diversify_prob(args.num("prob", "1.0"))
        // euclidean, 1 - dot and the augmented (mips) distance behave like squared distances
        .diversify_relax(alpha * alpha)
        .reverse_diversify(args.get("rev", "0") == "1")
        .pruning_degree_multiplier(args.num("pdm", "1.5"))
        .vertex_order(order)
        .build_unprepared_with_distance(distance, correction);
    match learn {
        Some((queries, neighbors, n_queries, width)) => index.prepare_with_query_sample(
            &QuerySample {
                queries,
                neighbors,
                n_queries: *n_queries,
                width: *width,
            },
            &QuerySampleOptions::default(),
        ),
        None => index.prepare(),
    }
    (index, started.elapsed().as_secs_f64())
}

#[allow(clippy::too_many_arguments)]
fn run<D: Distance<f32> + Clone>(
    args: &Args,
    dump: &Dump,
    distance: D,
    correction: Option<fn(f32) -> f32>,
    metric: Metric,
    kind: AffineKind,
    data: Vec<f32>,
    index_dim: usize,
    queries: &[f32],
    learn: Option<(Vec<f32>, &[i32], usize, usize)>,
) {
    // variants=rev=0:alpha=1.2;rev=1 builds one index per variant (the base arguments with
    // its overrides) and interleaves their timing runs, so slow drifts in machine load
    // affect them alike.
    let specs: Vec<&str> = args
        .0
        .get("variants")
        .map_or(vec![""], |v| v.split(';').collect());
    let mut variants = Vec::new();
    for spec in &specs {
        let mut merged = args.0.clone();
        for pair in spec.split(':').filter(|p| !p.is_empty()) {
            let (key, value) = pair
                .split_once('=')
                .expect("variant overrides are key=value");
            merged.insert(key.to_string(), value.to_string());
        }
        let variant_args = Args(merged);
        let (index, graph_build) = build(
            &variant_args,
            dump,
            distance.clone(),
            correction,
            metric,
            data.clone(),
            index_dim,
            &learn,
        );
        variants.push(Variant {
            args: variant_args,
            label: spec,
            index,
            graph_build,
        });
    }
    drop(data);

    let k: usize = args.num("k", "100");
    let nq = args.num::<usize>("nq", "1000").min(dump.n_test);
    let runs: usize = args.num("runs", "3");
    let features = dump.dim;
    let mut workspace = SearchWorkspace::new(dump.n, k);
    let mut pinned = false;
    for quant in args.get("quant", "none").split(',') {
        let mut encoded = Vec::new();
        for variant in &variants {
            let encode_start = Instant::now();
            let index = &variant.index;
            let (codes, half) = match quant {
                "none" => (None, None),
                "sq8" | "sq4" => {
                    let bits = if quant == "sq8" { 8 } else { 4 };
                    let clip = (bits == 4).then_some(DEFAULT_SQ4_CLIP_QUANTILE);
                    let codes = AffineCodes::encode_columns(
                        &index.data,
                        index_dim,
                        features,
                        kind,
                        bits,
                        clip,
                    );
                    let half = (args.get("half", "1") == "1")
                        .then(|| HalfRows::encode_columns(&index.data, index_dim, features, kind));
                    (Some(codes), half)
                }
                other => panic!("quant {other}"),
            };
            encoded.push((
                codes,
                half,
                variant.graph_build + encode_start.elapsed().as_secs_f64(),
            ));
        }
        if let (Some(cpu), false) = (args.0.get("cpu"), pinned) {
            pin(cpu.parse().unwrap());
            pinned = true;
        }
        let mut prepared = AffineQuery::default();
        let mut half_query = HalfQuery::default();
        let quantized = quant != "none";
        let widths = if quantized {
            args.list("cw", "100,150,200")
        } else {
            vec![0.0]
        };
        for &epsilon in &args.list("eps", "0.0,0.01,0.02") {
            for &width in &widths {
                let width = (width as usize).max(k);
                let mut best = vec![f64::INFINITY; variants.len()];
                // The first run's results give the recall.
                let mut found = vec![Vec::with_capacity(nq * k); variants.len()];
                for run in 0..runs {
                    for (v, (variant, (codes, half, _))) in
                        variants.iter().zip(&encoded).enumerate()
                    {
                        let index = &variant.index;
                        let t0 = Instant::now();
                        for query in queries.chunks_exact(index_dim).take(nq) {
                            let result = match codes {
                                None => index.query_one_indices_with_workspace(
                                    query,
                                    k,
                                    epsilon,
                                    &mut workspace,
                                ),
                                Some(codes) => {
                                    codes.prepare_query(&query[..features], &mut prepared);
                                    let half = half.as_ref().map(|rows| {
                                        rows.prepare_query(&query[..features], &mut half_query);
                                        (rows, &half_query)
                                    });
                                    index
                                        .query_one_codes(
                                            query,
                                            &prepared,
                                            codes,
                                            half,
                                            k,
                                            width,
                                            width,
                                            epsilon,
                                            &mut workspace,
                                        )
                                        .0
                                }
                            };
                            if run == 0 {
                                found[v].extend(result);
                            }
                        }
                        best[v] = best[v].min(t0.elapsed().as_secs_f64());
                    }
                }
                for (v, variant) in variants.iter().enumerate() {
                    let mut hits = 0usize;
                    for (q, row) in found[v].chunks_exact(k).enumerate() {
                        let query = &dump.test[q * dump.dim..(q + 1) * dump.dim];
                        let kth = dump.neighbors[q * dump.k_true + k - 1];
                        let threshold = vibe_distance(&dump.distance, query, dump.row(kth)) + 1e-3;
                        hits += row
                            .iter()
                            .filter(|&&i| {
                                i >= 0
                                    && vibe_distance(&dump.distance, query, dump.row(i))
                                        <= threshold
                            })
                            .count();
                    }
                    let a = &variant.args;
                    let degree = variant.index.search_graph.indices.len() as f64 / dump.n as f64;
                    println!(
                        "{{\"label\":\"{}\",\"variant\":\"{}\",\"distance\":\"{}\",\"nn\":{},\"pdm\":{},\"alpha\":{},\"order\":\"{}\",\
\"quant\":\"{}\",\"rev\":{},\"half\":{},\"learn\":{},\"eps\":{},\"cw\":{},\"recall\":{:.5},\"qps\":{:.1},\"build\":{:.2},\"degree\":{:.2}}}",
                        args.get("label", ""),
                        variant.label,
                        dump.distance,
                        a.get("nn", "30"),
                        a.get("pdm", "1.5"),
                        a.num::<f32>("alpha", "1.0"),
                        a.get("order", "tree"),
                        quant,
                        a.get("rev", "0"),
                        encoded[v].1.is_some(),
                        learn.is_some(),
                        epsilon,
                        if quantized { width } else { 0 },
                        hits as f64 / (nq * k) as f64,
                        nq as f64 / best[v],
                        encoded[v].2,
                        degree,
                    );
                }
            }
        }
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    if argv.len() < 2 {
        eprintln!("usage: search_bench <dump dir> key=value...");
        std::process::exit(2);
    }
    let args = Args(
        argv[2..]
            .iter()
            .map(|a| {
                let (key, value) = a.split_once('=').expect("arguments are key=value");
                (key.to_string(), value.to_string())
            })
            .collect(),
    );
    if let Some(threads) = args.0.get("threads") {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads.parse().unwrap())
            .build_global()
            .unwrap();
    }
    let dump = load(&argv[1], args.get("learn", "0") == "1");
    let dim = dump.dim;
    let learn_rows = dump.learn.as_ref().map(|(rows, _, _, _)| rows.clone());
    let learn_meta = dump
        .learn
        .as_ref()
        .map(|(_, neighbors, n, w)| (neighbors.as_slice(), *n, *w));
    let with_learn = |rows: Option<Vec<f32>>| {
        rows.zip(learn_meta)
            .map(|(rows, (neighbors, n, width))| (rows, neighbors, n, width))
    };
    match dump.distance.as_str() {
        "euclidean" => run(
            &args,
            &dump,
            SquaredEuclidean,
            Some(|d: f32| d.sqrt()),
            Metric::Euclidean,
            AffineKind::SquaredEuclidean,
            dump.train.clone(),
            dim,
            &dump.test,
            with_learn(learn_rows),
        ),
        "cosine" | "normalized" => {
            let mut data = dump.train.clone();
            normalize(&mut data, dim);
            let mut queries = dump.test.clone();
            normalize(&mut queries, dim);
            let learn = learn_rows.map(|mut rows| {
                normalize(&mut rows, dim);
                rows
            });
            run(
                &args,
                &dump,
                DirectNormalizedCosine,
                None,
                Metric::Cosine,
                AffineKind::Dot,
                data,
                dim,
                &queries,
                with_learn(learn),
            )
        }
        "ip" => {
            let norms: Vec<f32> = dump
                .train
                .chunks_exact(dim)
                .map(|r| r.iter().map(|x| x * x).sum())
                .collect();
            let norm2 = norms.iter().fold(0.0f32, |m, &v| m.max(v));
            let mut data = Vec::with_capacity(dump.n * (dim + 1));
            for (row, &norm) in dump.train.chunks_exact(dim).zip(&norms) {
                data.extend_from_slice(row);
                data.push((norm2 - norm).max(0.0).sqrt());
            }
            let queries = pad(&dump.test, dim);
            let learn = learn_rows.map(|rows| pad(&rows, dim));
            run(
                &args,
                &dump,
                SquaredEuclidean,
                None,
                Metric::SquaredEuclidean,
                AffineKind::Mips { norm2 },
                data,
                dim + 1,
                &queries,
                with_learn(learn),
            )
        }
        other => panic!("distance {other}"),
    }
}
