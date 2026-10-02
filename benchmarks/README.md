# Recall-gated performance work

This implementation establishes references and measurement tools for graph
construction and query optimizations.
Graph delivery and single-query throughput are separate workloads. The current
priority is full-corpus graph delivery, especially for large datasets; query timing
is not an adoption gate for lossless construction changes with identical graphs.

## Current default decision (2026-10-02)

Default Rust and Python builds now enable six optimization features:
`batched-euclidean`, `batched-angular`, `compact-candidates`, `batched-leaves`,
`candidate-membership`, and `candidate-workspace`. The latter includes worker-side
update-bucket allocation from the Fashion-MNIST regression fix. Python inherits
the core defaults. The first four were approved on 2026-09-30; the user approved
the two candidate features on 2026-10-02 after the lossless construction screens
and regression repair. Candidate-reference packing remains opt-in. Direct packing,
buffer tuning and the rejected FP16 features were removed on 2026-10-02.

The final fix validation checked 54 measured pairs and ten warm-up pairs across
four full corpora at degrees 15/30, including a direct old/new Fashion-MNIST
comparison. Graph indices, distance bits and iteration counts were identical.
That establishes unchanged graph quality for those builds; there is no lossy
recall tradeoff. All measured large-data pairs improved against the previous four
defaults, but individual image timings vary and peak RSS is mixed (for example,
NYTimes k30 was about 16 MB higher). This is an approved engineering default, not
a universal performance/memory guarantee or a claim that ten-seed statistical
confirmation passed. Query performance was not measured in these screens, and
the existing artifacts retain their original exploratory status.

Promotion validation: 158 release core tests pass with the new defaults and 158
with all six optimizations disabled. Cargo feature resolution confirmed packing,
buffer tuning and FP16 were off. The normal CPython 3.10 wheel was rebuilt
without explicit optimization flags into `target/wheels/default-candidates-v1/`
and installed into `ann-benchmarks/.venv` without changing dependencies. Its imported
native binary matches the wheel, and all 61 Python API/adapter/harness tests pass
without a `PYTHONPATH` override. Earlier isolated wheels remain unchanged.

Cleanup validation: 158 release core tests pass with defaults and with all six
optimizations disabled; 159 pass with all remaining features. All core targets,
including benchmarks, compile with all features. The replacement default CPython
3.10 wheel in `target/wheels/default-no-direct-packing-v1/` is installed into
`ann-benchmarks/.venv` without dependency changes. Its imported native binary matches
the wheel, and all 59 Python API/adapter/harness tests pass without a `PYTHONPATH`
override. The six approved defaults are unchanged.

Removed support includes the direct-record codec and destination layouts, configurable
source buffers, direct-only replay benchmark, buffer-sweep CLI/provenance, and FP16
kernels, runner, benchmarks and experimental build-stat fields. General construction
timings, candidate profiling, exact-output checks and candidate-reference replay
remain. The profiler no longer exposes FP16 encoding/recomputation categories;
archived timings for those phases fall into `other` if re-summarized with current
code. Saved results and isolated wheels are unchanged. No new performance screen
was run for this cleanup.

Use `cargo build -p rinnd-core --no-default-features --features std,rayon` for the
all-optimizations-off core configuration. For Python, add `--no-default-features`
to `maturin build --manifest-path crates/rinnd/Cargo.toml`; the binding supplies
`std,rayon` explicitly. Add individual `rinnd-core/<feature>` flags to build a subset.
The matrix runner already disables defaults explicitly, so its original four-feature
masks and candidate-profile baselines remain valid. To retain only the previous
four core optimizations, use:

```bash
cargo build -p rinnd-core --release --no-default-features \
  --features std,rayon,batched-euclidean,batched-angular,compact-candidates,batched-leaves
```

The sections below are a chronological experiment record. Their statements about
features being opt-in, defaults being unchanged, and promotion being blocked
describe the state at that time, superseded for the six promoted features by this
decision. Historical isolated
feature builds must use `--no-default-features` to reproduce their original masks.
FP16/direct-packing feature flags, buffer tuning, and their dedicated benchmark
options are no longer available in the current source.

## Optimization experiments

### Lossless candidate-reference updates (2026-09-30)

`packed-updates` is a new opt-in feature, separate from the historical four-feature
matrix. The baseline is the current four defaults, not mask 00. Both update formats
now share disjoint mutable producer/heap slices and the same heap insertion logic.
Eight-byte records reference immutable candidate slots within the current 16,384-row
source block, preserving all FP32 distance bits, endpoint IDs, bucket ordering,
reciprocal updates and convergence. Widths above 64 use the twelve-byte fallback.
No reduced-precision distances or vectors are implemented by this feature.

`cargo bench -p rinnd-core --bench heap --features bench-internals -- update_replay`
compares both formats in one binary. An untimed diagnostic pass reports emitted
records (including cross-owner duplicates), duplicate records, nonempty bucket
observations across source blocks, peak live/capacity bytes and bucket-vector
metadata. Timed replay excludes these counters and input-heap cloning, but includes
buffer allocation, distance generation, packing/decoding and heap application.
Capacity is not RSS. Generation/application phase times are also available through
the existing build statistics; those exclude buffer allocation.

Initial synthetic screen: Xeon Platinum 8370C, affinity 0,2,4,6,8,10,12,14,
100-dimensional FP32 data, 256/4096 points, degree 15/30, 1/8 workers, ten Criterion
samples, 0.2 s warm-up and 0.5 s requested measurement. Packed live/capacity bytes
were exactly two thirds of wide bytes, with equal record/duplicate counts and
unchanged bucket metadata. Seven of eight cases were slower (roughly 17-39%);
the 4096-point, degree-30, eight-worker case was about 7% faster (21.65 to 20.19 ms).
These short synthetic timings are exploratory, not end-to-end improvement or
default-promotion evidence. Broader dataset, RSS, query and statistical gates remain
required. Historical matrix masks and benchmark artifacts are unchanged.

The source-matched CPython 3.10 wheels in `target/packed-updates-v1/{wide,packed}`
are isolated from the normal environment. A seed-42, one-repetition check used all
four 10k graph references (degrees 15/30) and full-training indexes with 200 tuning
queries (k10, epsilon 0/0.1/0.3/0.6/1). Graph IDs/distance bits, update counts,
query fingerprints and per-row recalls matched exactly. Results and binary
provenance are in `results/packed-updates-v1/`. Graph timing ratios ranged from
0.868x to 1.348x, including a Fashion-MNIST degree-30 regression; query ratios were
0.956x to 1.069x. One observation per endpoint cannot establish performance.
Core validation passed 153 tests with defaults, 154 with packed diagnostics,
153 with optimization opt-out, and seven focused unbatched packed tests. All 48
Python benchmark/adapter/API tests passed against the isolated packed wheel.
No FP16/FP24 modes or new defaults have been introduced.

### Full-corpus packed-update profiling (2026-09-30)

The follow-up focuses exclusively on graph production, including first delivery of
both Python arrays. Identical graph outputs give no algorithmic reason for update
packing to change query work; the earlier single-run query fluctuations are not
evidence of a packing regression. Shared-host contention, cache state and binary
layout can affect timing, but contention was not established for those earlier runs.

The new `profile-updates` action uses the source-matched isolated wide/packed wheels.
It checks native source, interpreter, feature and imported binary identities, then
runs one graph degree per fresh subprocess. For each dataset/degree there is one
excluded warm-up pair followed by seeds 42/43/44 with three paired repetitions each.
Wide/packed order alternates every pair; timed processes run serially on physical
cores 0,2,4,6,8,10,12,14 with BLAS/OMP/MKL thread counts set to one. Both graph arrays
are fingerprinted in full and iteration update counts must match on every pair.
All 72 measured pairs and eight warm-up pairs matched. This is repeated exploratory
profiling, not ten-seed statistical confirmation; recall scoring is omitted because
this comparison requires exact output equality.

Complete-delivery medians in seconds; speedup is the median of paired wide/packed
ratios, not the ratio of the two displayed medians:

| Dataset | Full Rows | Degree | Wide | Packed | Paired Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| GloVe-100 | 1,183,514 | 15 | 18.548 | 19.243 | 0.966x |
| GloVe-100 | 1,183,514 | 30 | 37.846 | 39.589 | 0.954x |
| NYTimes-256 | 290,000 | 15 | 5.124 | 5.310 | 0.965x |
| NYTimes-256 | 290,000 | 30 | 10.899 | 11.379 | 0.960x |
| MNIST | 60,000 | 15 | 0.882 | 0.898 | 0.977x |
| MNIST | 60,000 | 30 | 1.698 | 1.745 | 0.973x |
| Fashion-MNIST | 60,000 | 15 | 0.933 | 0.956 | 0.963x |
| Fashion-MNIST | 60,000 | 30 | 1.673 | 1.731 | 0.964x |

The existing native timers isolate generation and application. The harness now
adds disjoint phase summaries, including `update_bookkeeping = update_seconds -
generation - application` (allocation, clearing, destruction and surrounding work),
plus forest, leaf initialization, candidates, sorting, correction, export and a
residual `other` category. Each record retains its per-iteration native timings.
Median phase seconds for the largest corpora:

| Dataset | Degree | Generation Wide / Packed | Application Wide / Packed |
| --- | ---: | ---: | ---: |
| GloVe-100 | 15 | 5.791 / 6.143 | 1.976 / 2.324 |
| GloVe-100 | 30 | 14.646 / 15.624 | 4.920 / 5.707 |
| NYTimes-256 | 15 | 1.903 / 1.990 | 0.507 / 0.603 |
| NYTimes-256 | 30 | 5.005 / 5.221 | 1.296 / 1.565 |

Both large corpora were slower in every measured pair. Forest/leaf/candidate phases
were close, while packed generation and application consistently took longer.
Process CPU time rose too: GloVe degree 30 from 271.06 to 285.29 CPU seconds, and
NYTimes degree 30 from 76.02 to 80.03. This supports an implementation cost rather
than simply whole-process descheduling, without proving its microarchitectural cause.
The reference format does extra encoding and candidate-ID loads; it is not a
direct bucket-relative endpoint format. No hardware-counter conclusion is implied.

Peak process RSS barely changed on the large corpora (GloVe degree 30: 2.831 to
2.827 GB; NYTimes degree 30: 1.181 to 1.178 GB). Record payload still falls by one
third, but buffers cover at most 16,384 source rows, so their size does not scale
linearly with corpus size. RSS includes other phases and input/graph storage;
capacity reductions are not peak-RSS guarantees. Context switches, process CPU,
page faults and before/after host load are recorded outside the delivery timer.
Host load includes the benchmark itself and cannot attribute unrelated contention.

Raw records, excluded warm-ups, phase/resource summaries and provenance are in
`results/packed-updates-full-glove-v1/` and `results/packed-updates-full-other-v1/`.
The feature remains opt-in; current evidence does not justify enabling this
candidate-reference implementation. Direct bucket-relative packing remains a
separate unimplemented experiment that could avoid candidate lookup costs.

Reproduce from `ann-benchmarks` with the existing isolated wheels and a new output
directory (the runner refuses to overwrite results):

```bash
taskset -c 0,2,4,6,8,10,12,14 \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  conda run -n uv --no-capture-output uv run python ../benchmarks/feature_matrix.py \
  profile-updates --artifacts ../target/packed-updates-v1 \
  --output ../benchmarks/results/packed-updates-full-rerun \
  --datasets glove-100-angular nytimes-256-angular \
  --seeds 42 43 44 --repeats 3 --threads 8
```

The standalone `benchmark_recall.py profile` mode also accepts `--degrees 15` or
`--degrees 30` for single-build resource measurements; its default remains both.

### Direct packing and buffer size (2026-10-01)

**Removed on 2026-10-02.** The following description and commands record the archived
experiment, not supported current build options. Its small, inconsistent delivery
gain did not justify the layout complexity and memory tradeoff. Saved results and
wheels remain; the direct replay benchmark and buffer sweep were also removed.

The opt-in `direct-packed-updates` implementation avoids candidate-array decoding.
It allocates 32 bits to the unchanged FP32 distance and splits the remaining 32
between a global neighbor ID and the local endpoint offset within its destination
bucket. Cross-bucket copies are oriented toward their local endpoint; same-bucket
pairs remain single records applied symmetrically. No global/public ID narrowing
or distance quantization is introduced. GloVe uses 21 global-ID bits and an
11-bit offset (2,048 vertices/bucket, 578 logical buckets); NYTimes uses 19+13 bits
(8,192 vertices/bucket, 36 buckets). Rayon workers and logical buckets are separate.
Layouts requiring more than 1,024 buckets or 8 MiB of producer-by-bucket vector
metadata fall back to the wide format (or candidate references if also enabled).

The independent `update-buffer-tuning` feature enables `RINND_UPDATE_BLOCK_ROWS`
in 1..=65536, read once per update iteration; unset retains 16,384. This setting
must stay fixed during a build. Normal builds ignore it. Candidate references
cannot encode larger source blocks and fall back to wide records above 16,384;
direct records have no such source-row limit. The profiler's `--buffer-rows`
option requires tuning-enabled wheels for both formats and records the value in
each subprocess's provenance. It requires exact full graph fingerprints and
iteration update counts to match between formats, repetitions and buffer sizes.

Source-matched isolated wheels/manifests are in `target/direct-updates-v1/{wide,packed}`;
the normal Python environment and feature defaults were not changed. The initial
full-corpus screen covered 4,096/16,384/65,536 rows on GloVe and NYTimes, degrees
15/30, seed 42, one measured pair and one excluded warm-up pair per configuration.
The 4,096-row setting did not show a benefit. Severe outliers occurred for both
wide and direct NYTimes builds, so those one-pair apparent speedups are not evidence.
All graph arrays and update counts matched, including across sizes.

The repeated run covered 16,384 and 65,536 rows on both corpora/degrees, seeds
42/43/44, three paired repetitions per seed and one excluded warm-up pair per
endpoint. It retained the physical-core affinity and serial, alternating fresh-
process protocol above. All 72 measured pairs and eight warm-up pairs matched
exactly, including across buffer sizes. No query timings were used as a gate.
Delivery medians are seconds; speedup is the median paired wide/direct ratio:

| Dataset | Degree | Buffer Rows | Wide | Direct | Paired Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| GloVe-100 | 15 | 16,384 | 18.574 | 18.854 | 0.995x |
| GloVe-100 | 30 | 16,384 | 37.651 | 38.446 | 0.979x |
| GloVe-100 | 15 | 65,536 | 18.380 | 18.187 | 1.013x |
| GloVe-100 | 30 | 65,536 | 37.694 | 37.176 | 1.011x |
| NYTimes-256 | 15 | 16,384 | 5.080 | 5.187 | 0.978x |
| NYTimes-256 | 30 | 16,384 | 10.821 | 10.995 | 0.987x |
| NYTimes-256 | 15 | 65,536 | 5.110 | 5.088 | 1.010x |
| NYTimes-256 | 30 | 65,536 | 10.703 | 10.775 | 0.994x |

GloVe degree 15 at 16,384 rows includes substantial timing interference indicators:
host load reached 48, unaffected phases also varied, and seed 43's paired median
was 1.478x versus 0.985x/0.988x for seeds 42/44. All observations remain in the
summary; no post-hoc exclusions were made. Host load cannot identify the cause.
The 65,536-row GloVe results were tighter: all nine pairs favored direct at each
degree, with ratios 1.002-1.020x (k15) and 1.008-1.018x (k30). NYTimes k15 had a
seed-level regression and k30 was slightly slower overall. Three seeds are still
exploratory evidence, not the predeclared ten-seed statistical confirmation.

At 65,536 rows, median phase seconds (wide / direct) show where the time goes:

| Dataset | Degree | Generation | Application | Update Bookkeeping |
| --- | ---: | ---: | ---: | ---: |
| GloVe-100 | 15 | 5.712 / 6.052 | 1.867 / 1.187 | 0.030 / 0.138 |
| GloVe-100 | 30 | 14.457 / 15.524 | 4.854 / 3.201 | 0.076 / 0.168 |
| NYTimes-256 | 15 | 1.888 / 1.957 | 0.413 / 0.408 | 0.145 / 0.018 |
| NYTimes-256 | 30 | 4.944 / 5.224 | 1.240 / 1.097 | 0.079 / 0.039 |

Larger buffers benefit direct application on GloVe, but generation still costs
more and bookkeeping can increase. Compared across the separate buffer-size runs,
direct GloVe k30 application falls from 4.497 to 3.201 s, while wide application
stays around 4.85 s; direct delivery falls from 38.446 to 37.176 s. This cross-size
comparison is descriptive, not an alternating paired estimate: buffer sizes were
swept sequentially, while record formats were paired within each size.

The memory tradeoff remains important. GloVe k30 median peak RSS at 65,536 rows
was 2.835 GB wide versus 3.047 GB direct (about 212 MB more), despite smaller
individual records. Direct uses more buckets and cross-bucket copies; reservation,
growth and phase high-water marks also matter. NYTimes k30 at that size was
1.394 GB wide versus 1.367 GB direct, both above their roughly 1.18 GB measurements
at 16,384 rows. Eight-byte records do not guarantee lower peak process memory.

The existing heap bench adds `direct_update_replay`: wide, wide with the same
logical buckets as direct, and direct, at all three buffer sizes. Its synthetic
131,073-point, dimension-100, width-6, eight-worker screen found at 65,536 rows
91.15/113.82/103.07 ms respectively. This separates bucket-count overhead from
record width locally; it is not a full-graph result or proof of the cause of the
large-corpus phase changes. Timings used ten samples, 0.2 s warm-up and 0.5 s
requested measurement time.

Results: `results/direct-updates-buffer-screen-v1/` and
`results/direct-updates-buffer-repeat-v1/`. Native validation passed 155 default,
156 combined experimental/diagnostic, and 155 unbatched-direct tests; 51 Python
benchmark/adapter/API tests passed with the isolated direct wheel. Focused tests
cover exact codec values, metadata fallback, reciprocal ownership and replay at
buffer sizes 1/7/4096/16384/65536. The current result supports retaining an opt-in
experiment, not changing defaults. Neither image-dataset performance nor newer
precision modes were measured in this direct-format follow-up.

Historical reproduction command (requires the removed profiler options), from
`ann-benchmarks` with the archived wheels and a new result directory:

```bash
taskset -c 0,2,4,6,8,10,12,14 \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  conda run -n uv --no-capture-output uv run python ../benchmarks/feature_matrix.py \
  profile-updates --artifacts ../target/direct-updates-v1 \
  --output ../benchmarks/results/direct-updates-buffer-rerun \
  --packing direct --buffer-rows 16384 65536 \
  --datasets glove-100-angular nytimes-256-angular \
  --seeds 42 43 44 --repeats 3 --threads 8
```

### FP16 construction screen (2026-10-01)

**Removed on 2026-10-02.** The following description and commands record the archived
experiment, not supported current build options. Saved results and wheels remain.

**Decision: neither FP16 option is promoted.** The graph-production screen found
no end-to-end speedup, and each option has an endpoint with mean recall loss greater
than the agreed 0.001 absolute allowance. These are three-seed exploratory results,
not statistical confirmation. No query performance claims are made.

Two independent opt-in features were implemented:

- `fp16-update-probe`: apply round-to-nearest binary16 after the existing FP32
  generation filter, then widen into the unchanged twelve-byte record. The heap
  stays FP32 with mixed exact/rounded keys. Non-finite values and magnitudes above
  65,504 remain bit-exact; no saturation. This tests numerical viability only.
- `fp16-vectors`: convert a working copy once after the existing FP32 normalization;
  use FP16 storage with FP32 arithmetic/accumulation for candidate updates only.
  Trees and leaf initialization remain FP32. Any non-finite/out-of-range coordinate
  disables the whole copy. Supported metrics are squared Euclidean, AlternativeDot
  and DirectNormalizedCosine; custom metrics default to lossless behavior.

This Xeon 8370C supports F16C but not native AVX512-FP16 arithmetic. The kernels
independently detect AVX2, FMA and F16C, widen in registers, and preserve the existing
FP32 reduction order. Tests cover single/four-candidate equivalence, short/unaligned
vectors and tails, scalar references, all finite half encodings, escapes, changed
threshold/tie decisions, unsupported metrics and overflow fallback. The dependency
is pinned to `half = 2.4.1` (declared Rust minimum 1.70, below this project's 1.75).

Original FP32 vectors are retained. Both builder paths recompute every valid
retained-edge distance before sorting/correction. This restores distances for the
selected IDs, not missing neighbors. Build stats expose `vector_encoding_seconds`,
`fp16_vectors_used` and `distance_recompute_seconds`; total graph delivery includes
conversion, allocation/destruction, recomputation, correction and both Python arrays.
The half working copy increases live vector payload from four to six bytes per
coordinate, although iterative vector loads use two bytes per coordinate.

`fp16_screen.py` builds isolated source-matched baseline/update/vector wheels and
runs one degree per fresh process. The existing lossless fingerprint gate is
unchanged; this screen has a separate paired lossy comparison. It uses the original
full-corpus, 2,048-row exact references, seeds 42/43/44, one measured run per seed,
and one excluded warm-up per variant/endpoint. Variant order rotates across rounds;
all timed builds run serially on physical cores 0,2,4,6,8,10,12,14. Each screen has
72 measured builds and 24 warm-ups. CPU/resource and full phase records are retained.
Public distances on sampled rows are checked against original-data Float64 distances
with rtol 5e-5 / atol 5e-6, preserving the existing log-mode saturation at distance
one for nonpositive dot products. Native tests additionally require bit-exact
original-kernel distances for both builder APIs.

The first implementation retained out-of-line metric dispatch. Static metric
specialization reduced warmed-cache angular kernel cost and graph generation cost,
but did not produce a delivery win. Both screens are preserved. All 72 measured
graph fingerprints remained identical between versions. Final artifacts:
`target/fp16-screen-v2/{baseline,updates,vectors}`; final measurements:
`results/fp16-full-screen-v2/summary.json`. Native source hash:
`dc60e946a9b0b9acfbb41d44a6837bbb00fa79b068631daea1fdec24154a8113`.

Final complete-delivery medians (seconds); paired speedups are medians of baseline /
candidate ratios, not ratios of the displayed medians. Recall changes are mean
paired seed differences in tie-aware recall:

| Dataset | Degree | FP32 | Update Probe | Half Vectors | Paired Speedup Update / Vector | Recall Change Update / Vector |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| MNIST | 15 | 0.877 | 0.904 | 0.879 | 0.968x / 0.989x | 0 / 0 |
| MNIST | 30 | 1.675 | 1.780 | 1.727 | 0.940x / 0.970x | 0 / 0 |
| Fashion-MNIST | 15 | 0.913 | 0.963 | 0.942 | 0.939x / 0.961x | 0 / 0 |
| Fashion-MNIST | 30 | 1.666 | 1.775 | 1.700 | 0.941x / 0.981x | 0 / 0 |
| GloVe-100 | 15 | 18.396 | 18.782 | 19.320 | 0.981x / 0.953x | +0.000543 / -0.001161 |
| GloVe-100 | 30 | 37.781 | 38.544 | 40.758 | 0.981x / 0.927x | -0.000776 / +0.000570 |
| NYTimes-256 | 15 | 5.048 | 5.215 | 5.330 | 0.970x / 0.948x | -0.002865 / -0.000835 |
| NYTimes-256 | 30 | 10.864 | 11.081 | 11.456 | 0.980x / 0.957x | +0.000168 / +0.000760 |

Strict recall gives the same failure decision: NYTimes k15 update loss is 0.002962;
GloVe k15 vector loss is 0.001161. Some individual seeds lose more than 0.001 even
where the mean does not: GloVe k30 updates -0.001546, NYTimes k15 vectors -0.001367.
The harness reports insufficient seeds for every formal gate. Promotion would still
require at least ten paired seeds, five repetitions and simultaneous lower bounds;
there is no reason to spend that confirmation budget on these slower candidates.

For GloVe k30, FP32/vector generation is 14.594/16.971 s, application 4.856/4.838 s,
encoding 0.042 s and exact recomputation 0.561 s. Peak RSS is 2.800/3.011 GB.
NYTimes k30 generation is 4.970/5.316 s, application 1.306/1.300 s, encoding 0.026 s,
recomputation 0.221 s and RSS 1.189/1.332 GB. Thus conversion/finalization costs do
not alone explain the regressions: iterative generation is slower as well.

Warmed-cache four-candidate kernels (20 Criterion samples, 0.2 s warm-up, 0.5 s
measurement) also remain slower after specialization: normalized logarithmic dot
FP32/FP16 is 44.99/63.89 ns at dimension 100 and 61.62/91.30 ns at 256; squared L2
is 54.04/71.73 ns at 256 and 134.97/197.43 ns at 784. These microbenchmarks do not
establish cache/memory causality or performance on other hardware.

Both image datasets have zero coordinates changed by FP16 conversion. Their sampled
nearest-edge squared distances exceed 65,504 in 99.9984% (MNIST) and 99.9951%
(Fashion-MNIST) of cases. All six measured graphs per image dataset are bit-identical
across the three variants. FP16 update packing would therefore need predominantly
exact escapes on these unscaled distances; silently rescaling the metric was not
attempted. No eight-byte FP16-distance record, FP24 codec, FP16 heap or half-query
traversal was added after these screens.

Validation: release core tests pass for baseline (155), updates (162), vectors
(162); combined FP16/packing/tuning/diagnostics passes 163 tests. All 29 benchmark
tests and 25 Python API/adapter tests per isolated wheel pass. Editor Python
missing-import diagnostics remain an environment issue; the explicit benchmark
interpreter imports and tests pass. Default features and the normal installed wheel
are unchanged.

Historical reproduction commands (require the removed implementation and runner):

```bash
conda run -n uv --no-capture-output uv run --directory ann-benchmarks \
  python ../benchmarks/fp16_screen.py build --artifacts ../target/fp16-new
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 0,2,4,6,8,10,12,14 \
  conda run -n uv --no-capture-output uv run --directory ann-benchmarks \
  python ../benchmarks/fp16_screen.py run --artifacts ../target/fp16-new \
  --output ../benchmarks/results/fp16-new
cargo bench -p rinnd-core --features fp16-kernels --bench distance -- fp16_
```

### Candidate profiling and membership (2026-10-01)

**Decision: retain `candidate-membership` as opt-in; defaults are unchanged.**
The first candidate optimization improves large-corpus k30 delivery without changing
graph bits or convergence. This is repeated three-seed screening, not ten-seed
statistical confirmation. Allocation/workspace reuse, copy-free distance tiles and
profile-guided compilation have not yet been implemented.

Candidate construction now reports per-iteration `candidate_phases` through Python
build statistics: `initialization`, `forward`, `reverse`, `mark`, and `release`.
Coarse timers surround the existing loops, with an untimed native path retained for
equivalence tests. The sequential implementation interleaves reverse insertion with
forward processing, so its `reverse` timer is zero. Parallel `forward` includes
local reverse insertion and allocation/growth of cross-worker reverse buckets;
`reverse` measures their merge. `release` measures priority/bucket scratch disposal,
not later destruction of the returned candidate indices. Subphases partition
`candidate_seconds`; they are not added again to build totals. Unattributed overhead
is retained in summaries. Tests compare candidate arrays, graph flags, distances and
RNG state between timed and untimed paths.

A seed-42 diagnostic profile of both complete corpora, at eight physical cores,
gave the following cumulative candidate seconds:

| Dataset | Degree | Initialization | Forward | Reverse | Mark | Release |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GloVe-100 | 15 | 0.908 | 2.200 | 1.989 | 0.155 | 0.044 |
| GloVe-100 | 30 | 1.381 | 4.228 | 4.917 | 0.324 | 0.071 |
| NYTimes-256 | 15 | 0.217 | 0.428 | 0.282 | 0.039 | 0.001 |
| NYTimes-256 | 30 | 0.309 | 0.948 | 0.815 | 0.108 | 0.009 |

All four graphs and update-count vectors match the previous FP32 baseline exactly.
These diagnostics are in `results/candidate-phases-v1/`, using the isolated wheel
in `target/candidate-phases-v1/mask-15`. The diagnostic process builds both degrees;
subsequent paired measurements use one degree per fresh process.

The selected optimization affects candidate-heap membership only. At widths >=16,
an integer equality/OR reduction replaces the scalar early-exit duplicate scan.
The measured release wheel contains SIMD `pcmpeqd`/`por` instructions in the candidate
builder; no manual ISA dispatch, new heap representation, RNG change or accumulation
change is introduced. Heap sift and flag marking are unchanged. Widths below 16
retain the scalar scan because the initial full reduction regressed the k15 replay.
Same-binary heap replays exclude initial allocation and compare exact priorities and
indices. At width30, the guarded reduction takes about 1.019 us versus scalar
1.109 us; width60 is 2.469 versus 3.819 us. These synthetic gains do not imply equal
end-to-end gains.

Final source-matched wheels are `target/candidate-membership-v1/{wide,packed}`.
The legacy runner labels mean baseline/optimized here: both use twelve-byte FP32
updates, and only the optimized wheel enables `candidate-membership`. Native hash:
`4153632914550dcab5ce5abf798ae1207bf5dc8c04397b35e89a7124d8de6b3e`.
The existing four-feature matrix masks remain unchanged. The new `build-profile`
action builds paired variants; `--candidate-feature candidate-membership` selects
this experiment instead of update packing and disallows buffer tuning in that run.

The full-corpus screen uses seeds 42/43/44, three paired repetitions per seed,
one excluded warm-up pair per dataset/degree, and alternating variant order.
Timed processes run serially on physical cores 0,2,4,6,8,10,12,14 with OMP/BLAS/MKL
threads set to one. All 72 measured pairs and eight warm-up pairs have identical
complete graph fingerprints and iteration update counts. Delivery includes both
Python arrays. Raw results and candidate subphase summaries are in
`results/candidate-membership-large-v1/` and `results/candidate-membership-images-v1/`.

| Dataset | Degree | Baseline Seconds | Optimized Seconds | Median Paired Speedup |
| --- | ---: | ---: | ---: | ---: |
| GloVe-100 | 15 | 18.387 | 18.405 | 0.998x |
| GloVe-100 | 30 | 37.697 | 36.597 | 1.030x |
| NYTimes-256 | 15 | 5.052 | 5.035 | 1.004x |
| NYTimes-256 | 30 | 10.761 | 10.608 | 1.017x |
| MNIST | 15 | 0.869 | 0.868 | 1.006x |
| MNIST | 30 | 1.685 | 1.686 | 0.998x |
| Fashion-MNIST | 15 | 0.912 | 0.904 | 0.997x |
| Fashion-MNIST | 30 | 1.669 | 1.656 | 1.004x |

Paired speedups are medians of paired ratios, not ratios of the displayed medians.
Every measured large-corpus k30 pair is faster: GloVe 1.015-1.045x, NYTimes 1.005-1.024x.
Candidate construction itself improves 1.096x and 1.072x, respectively. GloVe k30
forward time falls from 4.232 to 3.988 s and reverse merge from 4.798 to 4.104 s;
NYTimes k30 forward falls from 0.955 to 0.938 s and reverse from 0.805 to 0.706 s.
Median process CPU time also falls (GloVe 268.0 to 260.5 CPU-seconds; NYTimes 74.82
to 73.79). Degree 15 uses the scalar scan
and its near-neutral variation is not credited as an algorithmic gain.

The image endpoints show no established delivery benefit. Their candidate phase is
small, and some paired timings vary substantially (MNIST k30 minimum ratio 0.904x;
Fashion k15 range 0.873-1.126x). No observations were discarded. Large-corpus peak
RSS is near baseline, but not identically measured: GloVe k30 is 2.827/2.843 GB and
NYTimes k30 is 1.176/1.183 GB. The optimization adds no candidate payload buffers.
No hardware-counter or cross-platform performance conclusion is claimed.

Validation: 157 release core tests pass for baseline, optimized and optimized with
the four defaults disabled; 165 pass with membership plus FP16, packing, buffer
tuning and benchmark internals. 59 Python API/adapter/harness tests pass against the
isolated optimized wheel, including candidate timing partition checks across both
builder APIs and 1/4 workers. Native editor diagnostics are clean; Python editor
missing-import diagnostics remain, while the explicit benchmark interpreter passes.
The normal installed wheel is unchanged. No query-speed or formal recall gate is
claimed; exact full-graph equality is the lossless construction check here.

Reproduce from the repository root with unused output directories:

```bash
conda run -n uv --no-capture-output uv run --directory ann-benchmarks \
  python ../benchmarks/feature_matrix.py build-profile \
  --artifacts ../target/candidate-membership-new --candidate-feature candidate-membership
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 0,2,4,6,8,10,12,14 \
  conda run -n uv --no-capture-output uv run --directory ann-benchmarks \
  python ../benchmarks/feature_matrix.py profile-updates \
  --artifacts ../target/candidate-membership-new \
  --output ../benchmarks/results/candidate-membership-new \
  --candidate-feature candidate-membership --seeds 42 43 44 --repeats 3
cargo bench -p rinnd-core --features bench-internals --bench heap -- candidate_membership
```

### Candidate workspace reuse (2026-10-01)

**Decision: keep `candidate-workspace` opt-in.** Reusing candidate arrays lowers
large-corpus initialization time and gives small delivery gains, but the image
results are mixed and memory use is not uniformly lower. The comparison is against
the four current defaults, without `candidate-membership`; this screen does not
measure combined performance. See the later combined-feature screen below.
No defaults changed.

`CandidateWorkspace` retains both index arrays and both FP32 priority arrays across
NN-descent iterations. On each use, the arrays are resized as needed and fully reset
to -1 or infinity. Reverse-edge buckets are not retained. The candidate arrays are
recycled only after update application, so packed candidate references remain valid
throughout their use. No processing order, priority generation, flag policy, distance
calculation or convergence parameter changes. Repeated-build tests compare fresh
and reused outputs, graph state and RNG across changing shapes and 1/4/8 workers.

Resets are charged to candidate initialization. The final workspace destruction is
charged to the last candidate `release` phase and its containing `candidate_seconds`
entry; whole delivery also includes all resets, recycling, cleanup and both Python
arrays. The timing phases are not counted twice. Compared with fresh allocation,
the two priority arrays now stay live during update generation/application: an
additional `8 * n_points * max_candidates` bytes in that phase, about 284 MB for
GloVe k30. This is not necessarily an increase in process peak RSS, since candidate
construction already requires those arrays and other buffers can determine the peak.

Source-matched CPython 3.10 wheels are in
`target/candidate-workspace-v1/{wide,packed}`; `packed` is the legacy runner label
for the workspace variant here, not update packing. Native source hash:
`8ac4f46250dc367e4d9e5476a0c8b514d4d2e7435b6269ac0f8049d3c23a5d53`.
The full profile is in `results/candidate-workspace-full-v1/summary.json`, with raw
per-build records, binary identities, candidate subphases and process resources.

Protocol: four full corpora, degrees 15/30, seeds 42/43/44, three paired repetitions
per seed, one excluded warm-up pair per endpoint, alternating variant order, and
one graph build per process. Runs use eight physical cores 0,2,4,6,8,10,12,14 with
OMP/BLAS/MKL threads set to one. All 72 measured pairs and eight warm-up pairs have
identical full graph fingerprints and iteration update counts. Imported binary
hashes match their build manifests. This is exploratory profiling, not ten-seed
statistical confirmation; no query speed or lossy recall claim is made.

Complete-delivery medians in seconds. Paired speedups are medians of the paired
baseline/workspace ratios, not ratios of the displayed medians:

| Dataset | Degree | Baseline | Workspace | Paired Speedup | Paired Range |
| --- | ---: | ---: | ---: | ---: | --- |
| GloVe-100 | 15 | 18.410 | 17.978 | 1.022x | 1.014-1.033x |
| GloVe-100 | 30 | 37.589 | 37.047 | 1.014x | 0.997-1.028x |
| NYTimes-256 | 15 | 5.043 | 5.006 | 1.009x | 0.994-1.026x |
| NYTimes-256 | 30 | 10.782 | 10.669 | 1.015x | 0.973-1.032x |
| MNIST | 15 | 0.875 | 0.872 | 0.998x | 0.961-1.017x |
| MNIST | 30 | 1.705 | 1.680 | 1.014x | 1.006-1.136x |
| Fashion-MNIST | 15 | 0.916 | 0.934 | 0.979x | 0.952-1.232x |
| Fashion-MNIST | 30 | 1.677 | 1.673 | 1.002x | 0.997-1.025x |

The initialization reduction is consistent with the intended optimization:

| Dataset | Degree | Initialization Baseline / Workspace | All Candidate Work Baseline / Workspace |
| --- | ---: | --- | --- |
| GloVe-100 | 15 | 0.873 / 0.479 s | 5.181 / 4.774 s |
| GloVe-100 | 30 | 1.366 / 0.776 s | 10.826 / 10.262 s |
| NYTimes-256 | 15 | 0.228 / 0.126 s | 0.984 / 0.896 s |
| NYTimes-256 | 30 | 0.343 / 0.193 s | 2.237 / 2.112 s |

Large-corpus forward and reverse processing remain close to baseline. Every seed's
median delivery improves at these four endpoints, but individual pairs are slower
at three of them. Median process CPU changes are much smaller than wall-time gains
in some cases (GloVe k30: 268.16 / 267.85 CPU-seconds). These results do not establish
a general scaling or memory-bandwidth explanation. No observations were excluded.

Peak RSS medians (decimal GB) are GloVe k15 2.036 / 2.045, k30 2.826 / 2.815;
NYTimes k15 0.966 / 0.992, k30 1.180 / 1.185. The mixed image results also matter:
Fashion-MNIST k15 has a slower median for every seed, almost unchanged candidate
time (0.0524 / 0.0519 s), and peak RSS 0.363 / 0.390 GB. It is not treated as a
speedup or dismissed as host interference. MNIST k30 improves in every pair, but
has a wide timing range. The retained scratch is a workload-dependent tradeoff.

Validation: 158 release core tests pass for the source-matched baseline and workspace
wheels, and for workspace reuse with the four defaults disabled. All 166 tests pass
with workspace, membership, FP16, packing, buffer tuning and benchmark internals
enabled together. All 59 Python API/adapter/harness tests pass against the isolated
workspace wheel. Native editor diagnostics are clean; Python editor missing-import
diagnostics persist despite passing tests in the explicit benchmark interpreter.

Reproduce from the repository root, choosing unused artifact/output directories:

```bash
conda run -n uv --no-capture-output uv run --directory ann-benchmarks \
  python ../benchmarks/feature_matrix.py build-profile \
  --artifacts ../target/candidate-workspace-new --candidate-feature candidate-workspace
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 0,2,4,6,8,10,12,14 \
  conda run -n uv --no-capture-output uv run --directory ann-benchmarks \
  python ../benchmarks/feature_matrix.py profile-updates \
  --artifacts ../target/candidate-workspace-new \
  --output ../benchmarks/results/candidate-workspace-new \
  --candidate-feature candidate-workspace --seeds 42 43 44 --repeats 3
```

### Combined candidate features (2026-10-02)

**Decision: retain both candidate features as opt-in.** The combination improves
every measured large-corpus pair, especially degree 30, but consistently regresses
Fashion-MNIST degree 15. This screen compares the four approved defaults with those
same defaults plus `candidate-membership,candidate-workspace`. Neither packing,
buffer tuning nor FP16 is enabled. Native implementation and default features are
unchanged. The runner-only `--candidate-feature candidate-combined` option selects
the two existing Cargo features; it is not a new Cargo feature.

Isolated CPython 3.10 wheels: `target/candidate-combined-v1/{wide,packed}`.
The legacy `wide`/`packed` labels mean baseline/combined here, not update packing.
Both manifests match current native source hash
`8ac4f46250dc367e4d9e5476a0c8b514d4d2e7435b6269ac0f8049d3c23a5d53`.
Results: [full paired summary](results/candidate-combined-full-v1/summary.json),
with raw per-build records and logs alongside it. All endpoint summary fields were
recomputed from the raw records. Native binary and harness hashes, CPU affinity
and thread environment were verified against the profile manifest.

Protocol: the four complete training corpora, degrees 15/30, seeds 42/43/44 and
three paired repetitions per seed. One excluded warm-up pair per endpoint,
alternating variant order, one graph per process, eight physical cores
0,2,4,6,8,10,12,14, and OMP/OpenBLAS/MKL threads set to one. All **72 measured pairs
and eight warm-up pairs** have identical full graph hashes (indices and distance
bits) and iteration update counts, including across repetitions. Complete delivery
includes candidate resets, final workspace cleanup and both Python arrays. No
observations were discarded. Three seeds remain exploratory, not formal ten-seed
confirmation; query performance was not measured.

Complete-delivery medians in seconds; paired ratios are medians of baseline/combined
ratios, not ratios of the displayed medians:

| Dataset | Degree | Baseline | Combined | Paired Speedup | Paired Range | Faster Pairs |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| GloVe-100 | 15 | 18.313 | 17.971 | 1.020x | 1.011-1.025x | 9/9 |
| GloVe-100 | 30 | 37.598 | 36.062 | 1.046x | 1.005-1.066x | 9/9 |
| NYTimes-256 | 15 | 5.065 | 5.014 | 1.017x | 1.001-1.119x | 9/9 |
| NYTimes-256 | 30 | 10.913 | 10.454 | 1.037x | 1.022-1.061x | 9/9 |
| MNIST | 15 | 0.877 | 0.875 | 0.997x | 0.967-1.011x | 3/9 |
| MNIST | 30 | 1.696 | 1.672 | 1.011x | 1.001-1.022x | 9/9 |
| Fashion-MNIST | 15 | 0.913 | 0.936 | 0.972x | 0.961-0.993x | 0/9 |
| Fashion-MNIST | 30 | 1.665 | 1.653 | 1.008x | 0.991-1.022x | 8/9 |

Candidate-phase medians (baseline / combined, seconds):

| Dataset | Degree | Initialization | Forward | Reverse | Total Candidate Work |
| --- | ---: | --- | --- | --- | --- |
| GloVe-100 | 15 | 0.869 / 0.482 | 2.161 / 2.168 | 1.943 / 1.952 | 5.163 / 4.801 |
| GloVe-100 | 30 | 1.382 / 0.781 | 4.208 / 4.010 | 4.846 / 4.098 | 10.855 / 9.297 |
| NYTimes-256 | 15 | 0.225 / 0.127 | 0.445 / 0.453 | 0.279 / 0.281 | 0.988 / 0.901 |
| NYTimes-256 | 30 | 0.337 / 0.193 | 1.001 / 0.922 | 0.815 / 0.710 | 2.256 / 1.929 |

At degree 30 both intended improvements are visible: lower initialization/reset
time and lower duplicate-check processing time. Candidate time falls about 14% on
both large corpora. Degree 15 retains scalar membership, so its candidate savings
are mainly initialization. GloVe k30 process CPU medians fall from 268.80 to 260.62
CPU-seconds; NYTimes k30 from 75.97 to 74.00. These paired results establish the
combination's benefit against baseline here, not a direct ranking against the
earlier independently timed single-feature screens or a claim of additive gains.

Retained priority arrays still cost `8 * n_points * max_candidates` bytes during
updates, about 284 MB at GloVe k30. Process peak RSS medians (decimal GB,
baseline / combined): GloVe k15 2.048 / 2.043, k30 2.821 / 2.817;
NYTimes k15 0.966 / 0.991, k30 1.179 / 1.182; MNIST k15 0.388 / 0.390,
k30 0.534 / 0.540; Fashion-MNIST k15 0.363 / 0.390, k30 0.537 / 0.537.
The small lower peaks at some endpoints do not imply a memory-saving optimization.

Fashion-MNIST k15 regresses for every pair and every seed median. Candidate time is
essentially unchanged (0.0518 / 0.0520 s), while the measured update-bookkeeping
residual rises from 0.0016 to 0.0241 s. NYTimes k15 also has a larger residual
(0.0057 / 0.0611 s), partly offsetting its initialization savings. This residual is
total update time minus generation/application time; it does not identify a specific
allocator or cleanup cause. No such cause is asserted without further profiling.
The NYTimes k15 upper paired ratio of 1.119x is retained, not presented as typical.

Validation: source-matched baseline and combined wheels each passed 158 release
core tests. All 61 Python API/adapter/harness tests pass against the combined wheel,
including CLI dispatch and exact combined feature selection. Native editor
diagnostics are clean; the existing Python editor missing-import diagnostics remain
despite passing tests in the explicit benchmark interpreter. The normal installed
wheel is unchanged.

Reproduce from the repository root using unused output directories:

```bash
conda run -n uv --no-capture-output uv run --directory ann-benchmarks \
  python ../benchmarks/feature_matrix.py build-profile \
  --artifacts ../target/candidate-combined-new --candidate-feature candidate-combined
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 0,2,4,6,8,10,12,14 \
  conda run -n uv --no-capture-output uv run --directory ann-benchmarks \
  python ../benchmarks/feature_matrix.py profile-updates \
  --artifacts ../target/candidate-combined-new \
  --output ../benchmarks/results/candidate-combined-new \
  --candidate-feature candidate-combined --seeds 42 43 44 --repeats 3
```

### Fashion-MNIST allocation regression (2026-10-02)

The initial combined-feature regression is localized to recurring update-bucket
allocation, not distance calculations or final workspace cleanup. Originally,
`update_iteration_configured` reserved every producer's destination buckets serially
on the calling thread. Fashion-MNIST k15 reserves about 65 MB total bucket capacity.
Its original combined profile has roughly 6 ms of extra update residual per
iteration after the first, even in late iterations with little update work.

A temporary `RINND_PROFILE_UPDATE_STORAGE` probe separated allocation, reset and
release. In the combined wheel, later allocations cost about 5.3 ms each, release
0.14-0.56 ms, and reset less than 1 microsecond. Logging also made the baseline
allocations slow: an important observer effect, not a fair baseline for the original
regression. Disabling logging on the same wheels restored the total residual gap
(1.63 / 24.51 ms), with delivery 0.905 / 0.931 s.

On this glibc 2.39 host, a process-local `MALLOC_MMAP_THRESHOLD_=131072` diagnostic
reduced later allocations to about 0.16 ms in both logged variants, supporting
allocator-policy sensitivity. Complete delivery was worse (0.967 / 0.940 s), so
this setting is not recommended. The exact allocator-internal state transition
was not established. No system setting or persistent environment change was made.
Temporary probe code and logging were removed from the final source.

The retained fix is limited to `candidate-workspace`: allocate each producer's
bucket vector with an indexed Rayon parallel iterator. Default serial allocation
is unchanged. Indexed collection preserves producer/destination ordering; capacity,
candidate IDs, distances, update order and convergence are unchanged. Buffers still
drop each iteration, with all allocation/release charged to full update and graph
delivery time. No additional retained update workspace or metric changes.

Diagnostic artifacts are preserved separately:

- `target/candidate-storage-probe-v1`, native SHA
  `4c4af958f019e4d285b857469405e04ddc5fc68deb12ac6785a4db80e5e88646`;
  results `candidate-storage-probe-fmnist-v1`,
  `candidate-storage-probe-disabled-fmnist-v1`, and
  `candidate-storage-probe-mmap-fmnist-v1` under `benchmarks/results`.
  Each uses seed 42, one measured and one warm-up pair per degree. Per-build logs
  contain allocation details for the two logging-enabled runs.
- `target/candidate-storage-workers-v1`, native SHA
  `d1df6f74d3aefd3b595ffc57acef163e1802498c4d06bb1d26ec845708be4bb6`;
  results `candidate-storage-workers-fmnist-v1`. This intermediate worker-allocation
  variant still contained the disabled probe and is not the final implementation.
- Clean wheels: `target/candidate-storage-workers-v2/{wide,packed}`, native SHA
  `7b3f8522c30c9a0808f964145e41eef5d27443367baa6550f89e8ea3cbf5174f`.
  Here `wide` means approved defaults; `packed` means combined candidate features
  including worker allocation, not packed update records.

The direct [before/after comparison](results/candidate-storage-workers-fmnist-direct-v2/summary.json)
pairs the immutable pre-fix combined wheel in `target/candidate-combined-v1/packed`
with the clean fixed combined wheel. Both have identical Cargo feature sets; the
profile records their distinct source/binary identities. Full Fashion-MNIST,
seeds 42/43/44, three repetitions each, one excluded warm-up pair per degree,
alternating order, fresh processes, eight physical cores 0,2,4,6,8,10,12,14,
OMP/OpenBLAS/MKL threads one, no logging or allocator tuning. All 18 measured and
two warm-up pairs have identical full graph hashes and iteration counts.

| Degree | Before / Fixed Delivery | Paired Speedup | Paired Range | Update Residual Before / Fixed | Peak RSS Before / Fixed |
| ---: | --- | ---: | --- | --- | --- |
| 15 | 0.928 / 0.903 s | 1.028x | 1.018-1.038x | 23.913 / 2.276 ms | 390.5 / 372.5 MB |
| 30 | 1.642 / 1.632 s | 1.006x | 1.000-1.011x | 15.633 / 13.346 ms | 536.9 / 533.8 MB |

Every measured pair improves at both degrees, including all resets, release and
both Python arrays. CPU medians are 6.394 / 6.333 seconds at k15 and 11.909 / 11.877
at k30. RSS is decimal MB. This is three-seed evidence on this host/allocator,
not a portable guarantee or formal statistical confirmation.

A separate [default-versus-fixed screen](results/candidate-storage-workers-fmnist-v2/summary.json)
uses the same Fashion-MNIST protocol. Delivery medians are 0.922 / 0.918 s at k15
(median paired speedup 1.022x, range 0.983-1.038x), and 1.677 / 1.642 s at k30
(1.022x, range 1.004-1.028x). Every seed median improves, though individual k15
pairs still vary. All 18 measured and two warm-up pairs are exact. The baseline's
allocation residual/RSS varies between builds/runs; the direct old/new comparison
avoids crediting that unrelated baseline variation to the fix.

The [cross-dataset check](results/candidate-storage-workers-other-v2/summary.json)
uses the same cores, full corpora and three seeds, with one measured repetition
and one warm-up pair per endpoint. All 18 measured and six warm-up pairs are exact.
Paired ratios compare approved defaults to the fixed combined features:

| Dataset | Degree | Baseline / Fixed Delivery | Paired Speedup | Peak RSS Baseline / Fixed |
| --- | ---: | --- | ---: | --- |
| MNIST | 15 | 0.867 / 0.859 s | 1.012x | 387.9 / 380.8 MB |
| MNIST | 30 | 1.692 / 1.658 s | 1.021x | 537.1 / 547.4 MB |
| GloVe-100 | 15 | 18.414 / 17.968 s | 1.025x | 2036.6 / 2054.5 MB |
| GloVe-100 | 30 | 37.290 / 35.725 s | 1.043x | 2827.7 / 2826.5 MB |
| NYTimes-256 | 15 | 5.022 / 4.899 s | 1.026x | 967.9 / 972.6 MB |
| NYTimes-256 | 30 | 10.869 / 10.428 s | 1.033x | 1176.9 / 1192.9 MB |

Every large-data pair improves; MNIST k15 ranges 0.986-1.032x. Memory is mixed,
not uniformly reduced: NYTimes k30 is about 16 MB higher. This smaller check is
not a direct before/after measurement of the allocation fix on the large corpora.
All 54 measured and ten warm-up pairs across the three clean runs were rechecked
for graph hashes, iteration counts, binary provenance, affinity and every summary
field, including equality across runs. No observations were discarded.

Validation: 158 release core tests pass for both clean wheels and workspace reuse
with the four defaults disabled; 166 pass with all experimental features enabled.
All 61 Python API/adapter/harness tests pass against the fixed wheel. Native editor
diagnostics are clean. Defaults and the normal installed wheel remain unchanged.

Reproduce the clean Fashion-MNIST comparison using unused directories:

```bash
conda run -n uv --no-capture-output uv run --directory ann-benchmarks \
  python ../benchmarks/feature_matrix.py build-profile \
  --artifacts ../target/candidate-storage-new --candidate-feature candidate-combined
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 0,2,4,6,8,10,12,14 \
  conda run -n uv --no-capture-output uv run --directory ann-benchmarks \
  python ../benchmarks/feature_matrix.py profile-updates \
  --artifacts ../target/candidate-storage-new \
  --output ../benchmarks/results/candidate-storage-new \
  --candidate-feature candidate-combined --datasets fashion-mnist-784-euclidean \
  --seeds 42 43 44 --repeats 3
```

### Earlier distance batching experiments

The `rinnd-core/batched-euclidean` feature enables four-candidate AVX2/FMA distance
evaluation in NN-descent updates and greedy query traversal. Each candidate keeps
the original accumulation and reduction order. Heap insertion, candidate order,
termination bounds and convergence settings are unchanged. With this feature alone,
non-Euclidean metrics retain their original update/traversal loops; unsupported SIMD platforms retain
the scalar distance fallback. This feature is experimental and disabled by default.

At 784 dimensions, the pinned single-core Criterion benchmark measured four
existing distance calls at about 324 ns versus 133 ns for the new kernel. Complete
graph comparisons on 10,000-row subsets across all four datasets and three seeds
produced identical indices/distances and per-row recall. Early candidates were
also screened at 1, 8 and 16 physical threads. These are not release gates.

The final isolated wheel screen used eight physical cores, seed 42, three timing
repetitions, graph k=15/30, and full training corpora with 200 fixed official tuning
queries. Speedups are median paired baseline time / candidate time:

| Dataset | Graph k=15 | Graph k=30 | Query QPS range over epsilon 0/0.1/0.3/0.6/1 |
| --- | ---: | ---: | ---: |
| MNIST | 1.116x | 1.220x | 1.020-1.146x |
| Fashion-MNIST | 1.103x | 1.202x | 1.059-1.117x |
| GloVe-100 | 1.000x | 1.011x | 0.993-1.009x |
| NYTimes-256 | 1.019x | 1.258x | 0.993-1.027x |

Every graph and query-output fingerprint matched the baseline in this final screen.
NYTimes construction timings varied substantially across screens; its apparent
gain is not attributed to Euclidean SIMD, which it does not use. Angular query
changes are near measurement noise, not established improvements. Raw final data
is in `results/batched-final/`. Earlier experiments are in `leaf-buckets-compare`,
`four-build-compare`, `four-query-compare`, `four-isolated-query`, and `batched-scaling`.
The earlier combined traversal introduced an angular regression; the final version
isolates batching from original angular/quantized loops to remove that overhead.

Destination-bucket leaf initialization was tested first and rejected: output was
identical, but end-to-end gains were generally only 0-2%, with NYTimes regressions.
The production routing change was removed; its regression test remains.

Build the opt-in candidate from inside `ann-benchmarks`:

```bash
uv run --with maturin maturin build --release \
  --manifest-path ../crates/rinnd/Cargo.toml \
  --no-default-features \
  --features rinnd-core/batched-euclidean \
  --interpreter "$PWD/.venv/bin/python" --out ../target/wheels/batched-isolated-final
```

The candidate is installed only in `target/perf-python/batched-isolated-final`,
not in the ordinary benchmark environment. From `ann-benchmarks`, prefix harness
commands with `PYTHONPATH="$PWD/../target/perf-python/batched-isolated-final"` to
select it. Baseline imports are isolated in `target/perf-python/baseline`.
Binary hashes in measurement output verify which implementation ran.

Before enabling by default: finish the declared ten-seed, five-repeat paired
confirmation, full-corpus sampled graph recall, held-out query curves, memory/tail
checks and final-candidate scaling across threads/platforms. Exact outputs in these
screens are strong local evidence, not a universal recall or performance guarantee.

## Angular batching and cache experiments

`rinnd-core/batched-angular` independently enables batching for `Cosine`,
`InnerProduct`, `Dot`, `AlternativeDot`, and `DirectNormalizedCosine`. Euclidean
also exposes four-result square-root correction under `batched-euclidean`.
Other metrics keep their existing individual-distance implementation and default
four-call fallback; they are not silently opted into the batched traversal.

Raw cosine shares query loads and query-norm accumulation across four candidates.
Inner product and clamped dot use four independent accumulators. Normalized cosine
and logarithmic dot preserve the existing four accumulators *per candidate*,
including their horizontal reduction and scalar tail. Their four-result API uses
two paired-candidate kernels, avoiding sixteen simultaneously live accumulators
on AVX2. No log/clamp, normalization or search-bound policy is changed.

Pinned hot-data Criterion results (15 samples, 0.5 s warm-up, 1 s measurement):

| Kernel | 100 dimensions | 256 dimensions | 784 dimensions |
| --- | ---: | ---: | ---: |
| Raw cosine | 1.78x | 2.61x | 2.92x |
| Inner product | - | 2.02x | 2.66x |
| Logarithmic normalized dot | 1.15x | 1.21x | 1.42x |
| Direct normalized cosine | 1.06x | 1.10x | 1.35x |

These kernel ratios do not predict complete graph gains. A three-seed (42-44),
three-repeat, eight-physical-core screen on the same 10k graph subsets compared
`batched-angular-v1` against the previous Euclidean-only `batched-isolated-final`:

| Dataset | Graph delivery k=15 | Graph delivery k=30 |
| --- | ---: | ---: |
| MNIST | 1.009x | 1.001x |
| Fashion-MNIST | 1.006x | 1.007x |
| GloVe-100 | 1.008x | 1.016x |
| NYTimes-256 | 1.009x | 1.020x |

The full-training query screen used seed 42, five repetitions and 200 tuning
queries, compared at identical epsilon values. Query QPS speedups:

| Dataset | epsilon=0 | 0.1 | 0.3 | 0.6 | 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MNIST | 0.989x | 0.989x | 0.987x | 0.993x | 1.000x |
| Fashion-MNIST | 0.991x | 1.031x | 1.036x | 1.051x | 1.038x |
| GloVe-100 | 1.062x | 1.082x | 1.105x | 1.088x | 1.072x |
| NYTimes-256 | 0.983x | 1.050x | 1.069x | 1.093x | 1.102x |

Graph fingerprints (indices and distance bits), query fingerprints and per-row
recalls matched in every comparison. Image kernels are unchanged in this comparison;
their timing variation is not credited to angular batching. NYTimes at epsilon=0
is slower in this screen and must be included in confirmation, not averaged away.
Data: `results/angular-cache-graph/` and `results/angular-query/`.

Two separately gated cache experiments were implemented, measured and removed:

- Upcoming candidate-vector and destination-heap prefetching: graph speedups versus
  batching alone were 0.987/0.972x MNIST, 0.965/0.951x Fashion, 0.960/0.965x GloVe,
  1.011/1.027x NYTimes (k=15/30). Results in `angular-cache-graph` with the isolated
  `angular-prefetch-v1` wheel. Prefetching generation and application was tested
  together; these results do not identify which part caused the regressions.
- Compact heap-root threshold snapshots, refreshed before each update-generation
  block: 0.993/1.004x MNIST, 1.004/0.995x Fashion, 0.979/1.002x GloVe,
  0.979/1.005x NYTimes. Results in `results/angular-thresholds-graph/` with the
  `angular-thresholds-v1` wheel. Copy/allocation costs were included in timing.

All cache-experiment graph outputs also matched. These are subset results, not
evidence against those strategies at every corpus size. Update records remain
12 bytes, with natural 4-byte alignment; heap IDs/distances/flags remain separate
arrays totaling 9 bytes per neighbor. Padding updates to 16 bytes would add 33%
record traffic; padding to cache lines is not justified without false-sharing
evidence. Destination ownership already limits concurrent heap writes to disjoint
rows. Finer destination tiling and reusable candidate packing remain hypotheses:
measure full-scale generation/application times and cache/TLB misses separately,
and preserve per-row insertion order and block-level threshold timing.

The closest next metric kernel is Manhattan (existing AVX2 absolute-difference
reduction). Bit-packed Hamming/Jaccard need wider integer/popcount kernels;
probability metrics need metric-specific sqrt/log and zero-case handling.
Neither those kernels nor a new packed/aligned heap representation are implemented.

For the retained angular candidate, add
`--features rinnd-core/batched-euclidean,rinnd-core/batched-angular` to the build
command above and use a new output directory. The measured wheel is isolated in
`target/wheels/batched-angular-v1`, with imports in `target/perf-python/batched-angular-v1`.
The ordinary Python environment remains baseline. Tests cover exact kernel bits
at SIMD boundaries, signed/zero vectors, complete construction at multiple thread
counts and degrees, and reused query workspaces across epsilon values. Full recall
confirmation, performance confidence bounds, non-AVX2 execution and final scaling
remain outstanding. Neither batching feature is enabled by default.

## Profiling without perf

`build_stats` now includes per-iteration `update_generation_seconds` and
`update_application_seconds`. Each is wall time summed over source blocks around
the corresponding parallel phase, including scheduling and waiting. Timers are
outside worker inner loops. The existing `update_seconds` remains the enclosing
total: its remainder includes buffer setup/clear/drop and instrumentation overhead.
The new fields are subsets of that total, not extra terms in `nn_descent_seconds`.
They do not distinguish CPU arithmetic, memory stalls or individual worker imbalance.

The `profile` CLI mode measures graph construction through first array delivery
against the complete training corpus, without needing an exact graph reference:

```bash
# From ann-benchmarks, after building/installing an instrumented wheel separately:
PYTHONPATH="$PWD/../target/perf-python/phase-profile-v1" \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
taskset -c 0,2,4,6,8,10,12,14 uv run python ../benchmarks/benchmark_recall.py \
  profile data/glove-100-angular.hdf5 ../benchmarks/results/profile-glove-t8.json \
  --threads 8 --seeds 42 --repeats 2
```

It records complete graph fingerprints, per-stage timings, binary provenance and
process peak RSS. Recall dictionaries are empty and `detail.recall_status` is
`not_evaluated`; this mode cannot satisfy recall promotion gates. Output paths
must be new. `--ordinary-index` measures graph delivery for a lazy search-capable
index, not search preparation or query traversal. Query profiling remains separate.

The first full-scale screen uses the isolated `phase-profile-v1` wheel with both
batching features, k=15/30, seed 42 and two repetitions at one/eight physical cores.
An uninstrumented `batched-angular-v1` wheel is compared at eight cores with
alternating binary order by dataset. Raw artifacts: `results/phase-profile-full/`.
This is bottleneck triage, not a statistically confirmed timing comparison.
Cross-thread speedups are operational measurements: candidate RNG/ordering depends
on thread count, so the amount of algorithmic work can differ between one and eight
cores. Fingerprints are compared between binaries at the same thread count.

The one-core GloVe run was stopped after more than eleven minutes; the original
series is incomplete. Completed MNIST/Fashion and eight-core GloVe artifacts remain
available. The subsequent candidate/leaf screen below covers all four full corpora
at eight physical cores, including NYTimes.

## Candidate and leaf experiments

Two additional features remain opt-in, with default behavior unchanged:

- `rinnd-core/compact-candidates`: stores each reverse-candidate record in 12 bytes,
  encoding the new/old flag in the sign of the nonnegative candidate ID. Destination
  routing, priorities, insertion order and RNG consumption are unchanged.
- `rinnd-core/batched-leaves`: uses the existing bit-preserving distance batches
  during leaf pair generation, preserving pair order and the strict update threshold.
  Requires the corresponding distance batching feature and dimension >= 32.

The isolated `compact-leaves-v1` wheel enables these plus `batched-euclidean` and
`batched-angular`. Its comparison baseline is `phase-profile-v1`, which already
enables both distance features. The following gains are incremental, not relative
to the original unoptimized build.

Full-corpus screen: eight physical cores, seeds 42/43, one repetition per seed,
baseline-first for seed 42 and candidate-first for seed 43. Ratios are medians of
paired baseline/candidate times, including first delivery of both graph arrays:

| Dataset | Graph k15 speedup | Graph k30 speedup | Baseline / candidate peak RSS (GB) |
| --- | ---: | ---: | ---: |
| MNIST | 1.080x | 1.093x | 0.582 / 0.547 |
| Fashion-MNIST | 1.088x | 1.096x | 0.581 / 0.551 |
| GloVe-100 | 1.030x | 1.035x | 3.464 / 2.737 |
| NYTimes-256 | 1.020x | 1.064x | 1.423 / 1.251 |

GloVe candidate construction improves 1.094/1.101x at k15/k30; NYTimes improves
1.102/1.142x. Image leaf initialization improves 1.439-1.460x. GloVe peak RSS is
about 21% lower, NYTimes about 12% lower. RSS is a process-lifetime high-water mark
covering both degrees, not isolated candidate storage. NYTimes build time varies
substantially by seed; paired comparisons are essential. All complete graph
fingerprints and update counts match. Artifacts: `results/compact-leaves-full/`.

Follow-up exact 10k-subset checks across all four datasets, three seeds and three
repetitions match graph fingerprints and per-row recalls. Query checks use the full
training corpora, 200 tuning queries, seed 42, three repetitions and epsilon
0/0.1/0.3/0.6/1: all query fingerprints and recalls match. Query speed ratios range
from 0.907x to 1.059x, with the largest slowdown on MNIST. A reversed-order MNIST
repeat with five repetitions yields 0.976-1.009x. There is no established query gain
from these construction changes, and query timing nonregression remains unresolved.
Artifacts: `results/compact-leaves-validation/`.

Three implemented alternatives were removed after isolated full-corpus screens
(seed 42, one repetition, eight physical cores). All preserved graph fingerprints:

- Gathering whole candidate-row vectors into reusable contiguous scratch: total
  speedups 0.865-0.952x across MNIST/GloVe/NYTimes; update generation 0.583-0.875x.
  The packing cost outweighed reuse in this implementation.
- Cached four-way tree projections: forest speedups 0.991-1.015x across those
  datasets, insufficient evidence of a benefit despite exact partitions and RNG.
- Hole-based candidate heap sift: GloVe candidate speedups 0.992/0.966x at k15/k30;
  reducing stores did not improve this workload.

Raw isolated artifacts are under `results/packed-vectors-full/`,
`results/batched-tree-full/`, and `results/candidate-hole-full/`. Positive isolated
screens are under `results/compact-candidates-full/` and `results/batched-leaves-full/`.
Rejected wheels remain separate immutable artifacts; their features are removed.

Next hypotheses worth measuring are candidate allocation/reset versus reverse merge
cost, and small two-dimensional distance tiles that reuse vector loads without
gathering full rows or changing accumulation order. Tree hyperplane scratch reuse
is another bounded experiment. The existing breadth-first tree builder changes
random choices, so it is not an exact-output substitution. Without hardware counters,
these are hypotheses, not diagnosed cache or bandwidth bottlenecks.

Validation: 141 core tests pass with default features and with all four features
in release mode; 45 benchmark/adapter/Python API tests pass against the combined
wheel. These screens remain exploratory. Ten paired seeds, five repetitions,
held-out query curves, full-corpus sampled exact references and platform/thread
checks are still required before promotion. The normal Python installation is
unchanged; the combined wheel is in `target/wheels/compact-leaves-v1`, with an
isolated import target at `target/perf-python/compact-leaves-v1`.

## Feature matrix and default confirmation

`feature_matrix.py` builds all 16 combinations of the four optimization flags.
Bit order is Euclidean batching (1), angular batching (2), compact candidates (4),
leaf batching (8). Mask 0 is the control and mask 15 enables all four. Each build
uses `--no-default-features`, explicit core features, an isolated Python target,
source/binary manifests and release core tests. The binding crate explicitly
forwards core defaults only through its own default feature, preserving an opt-out.
The runner guards resumption with job manifests and verifies imported binary hashes.

All 16 variants passed 141 release core tests each. The initial screen used seed 42,
three repetitions, eight physical cores, all four exact 10k graph subsets, and full
training corpora with 200 tuning queries per query run. Every graph and query
fingerprint and per-row recall matched mask 0. Raw files:
`results/feature-matrix-screen/`; binaries: `target/feature-matrix-v1/mask-*/`.

Mask 15 is the fixed confirmation candidate, chosen before inspecting held-out data:

| Dataset | Subset graph speedup k15 / k30 | Tuning query QPS speedup range |
| --- | ---: | ---: |
| MNIST | 1.215 / 1.316 | 1.079-1.177 |
| Fashion-MNIST | 1.205 / 1.315 | 1.069-1.146 |
| GloVe | 1.005 / 1.015 | 1.041-1.153 |
| NYTimes | 1.044 / 1.295 | 1.019-1.074 |

These are exploratory ratios against the all-flags-off control. No defaults have
yet changed. The user chose the original full confirmation rather than a relaxed
local promotion. The frozen confirmation protocol is:

- Masks 0/15; fresh seeds 100-109; five timing repetitions; eight physical cores
  0,2,4,6,8,10,12,14; BLAS/OMP/MKL threads one. Binary order alternates by seed.
- Graph k15/30 on each full corpus, scored on 2,048 sampled training rows searched
  exhaustively against the full corpus. Reference seed 20260930; immutable references
  `results/ground_truth/*-full-s2048-k30.hdf5`. Full graph fingerprints also compared.
- Query k10 on the entire untouched 80% confirmation split, epsilon
  0/0.1/0.3/0.6/1, unchanged constructor/search parameters. Invalid NYTimes official
  truth rows remain reported; no post-selection filtering or repair.
- Recall family: 36 endpoints (four datasets times two graph degrees times two
  recall metrics, plus four datasets times five query epsilons). One-sided 95%
  Student-t/Bonferroni bounds on paired seed means; minimum difference -0.001.
- Timing family: 28 endpoints (eight graph cases and twenty query cases). Within
  each seed take the median timing over repetitions, then bound the mean log paired
  speedup with the same simultaneous one-sided confidence level. Require lower
  speedup bound >= 1.0 at every endpoint; inconclusive endpoints block promotion.
- Report process peak RSS and query p99 ratios separately. Seed-level inference is
  conditional on evaluated rows. Identical full graph fingerprints establish zero
  observed graph recall difference on every row for those seeds, not just the sample;
  if fingerprints differ, stop rather than infer full-corpus quality from seed-only
  bounds. This does not establish guarantees for unseen seeds or other machines.

The confirmation report refuses missing/unpaired repetitions and seeds. No candidate
switching or epsilon retuning is allowed based on confirmation results. This confirms
the unchanged policy's measured curve, not a newly tuned high-recall Pareto frontier.
Defaults remain unchanged if any required endpoint fails or remains inconclusive.

### Confirmation blocked by degenerate leaf

The original angular confirmation stopped at NYTimes seed 105, mask 15, before
the first graph record: leaf initialization requested 2,070,458,127,744 bytes.
A bounded diagnostic reproduced the identical abort in mask 0, so this is a
shared baseline defect, not evidence of a feature-specific regression. The eager
reserve multiplied the global maximum leaf size across every leaf in a block.
It now uses saturating arithmetic and caps the initial reserve at 1,048,576 update
records per worker; vectors can still grow to hold actual updates. Pair generation,
filtering and insertion order are unchanged. This is not a total-memory bound.

Both repaired configurations pass 142 release core tests. Their isolated artifacts
are in `target/feature-matrix-reserve-v2/` (masks 0/15). Under a 16 GiB address-space
limit both passed the failed reservation but remained in leaf initialization until
a 180-second diagnostic timeout each. Oversized leaves still require exhaustive
pair work; completion and full-output equality for seed 105 are not yet verified.
Changing degenerate tree partition policy would be a separate correctness change
and require a new source-matched confirmation, not resuming mixed binaries.

Original artifacts and the failure log remain in `results/feature-default-confirmation/`.
All image graph confirmation runs completed and passed; angular runs completed
GloVe seeds 100-105 and NYTimes seeds 100-104 before the abort. Held-out query
confirmation has not run. Default promotion is blocked; no benchmark is running.

### Degenerate split repair

The production depth-first builder now rejects non-finite planes and retries
empty-sided partitions, up to eight attempts. A regression fixture reproduces
empty-sided angular partitions on nearly collinear float32 vectors: normalizing
their tiny directional difference can produce a plane that puts every point on
the same side. Previously a single such partition terminated the entire node as
an oversized leaf.

If every attempt fails, construction shuffles the node's indices and divides them
at the midpoint, storing a zero plane and zero offset for retained trees. Query
traversal uses the existing randomized tie behavior for that plane. This is a
non-geometric fallback, not an assertion that duplicate vectors are separable.
Successful ordinary splits and successful randomized tie partitions retain their
existing behavior. An initial attempt to reject all zero planes was discarded
after it changed NYTimes low-epsilon tuning recall substantially.

The configured maximum geometric-tree depth remains enforced. At leaf
initialization, all inputs are subdivided into borrowed groups no larger than the
configured leaf size (minimum one), after removing sentinel tails. This bounds
pair work for a leaf with N valid IDs to O(N * leaf_size), rather than O(N^2).
It deliberately omits cross-group initialization pairs; ordinary NN-descent still
refines the graph afterward. Retained query-tree leaves are not truncated. This
handling safeguard also covers depth-limited and experimental breadth-first
leaves; the breadth-first split policy itself is unchanged. The earlier bounded
eager-reserve fix remains in place.

Final artifacts: `target/feature-matrix-tree-v4/`, masks 0/15, and
`results/degenerate-tree-repair-v4/`. Both configurations pass 146 release core
tests, including near-collinearity, duplicates, zero/non-finite planes, depth
limits, point coverage, retained/leaf-only agreement and bounded initialization
with sentinels. The final wheel passes 48 Python benchmark/adapter/API tests.

Full NYTimes seed 105 now completes graph k15/k30 and retained-tree query
preparation/search. Graph delivery took about 5-12 seconds in this one-repeat
diagnostic, with 1.19-1.39 GB process peak RSS, instead of an allocation abort or
the earlier 180-second leaf-initialization timeout. Both feature configurations
produce identical graph/query fingerprints and recalls on this seed.

An all-four-dataset seed-42 tuning check also produces identical outputs between
repaired masks 0/15. Compared with the original tree policy, all query fingerprints
match, as do MNIST/Fashion/GloVe subset graph fingerprints. NYTimes subset graphs
change: strict recall delta -0.0001533 at k15 and +0.0003800 at k30 (tie-aware
-0.0001600/+0.0003733). This is exploratory evidence, not a statistical recall gate.
Timings were variable; no new speedup claim is made from these diagnostic runs.

The repair changes construction semantics for formerly degenerate leaves. The
original frozen default-confirmation results cannot be combined with these new
binaries. Default features are unchanged; full source-matched confirmation must
restart before promotion. No held-out queries were used to tune this repair, and
no benchmark is currently running.

### NYTimes repeated timing check

After suspected shared-host interference, the source-verified `tree-v4` masks
0/15 were rerun on the same eight physical cores with five repetitions. Separate
artifacts are in `results/nytimes-rerun-tree-v4/`; earlier results are preserved.
The subset graph and 200-tuning-query runs use seeds 42/43/44, alternating binary
order. Summaries below take each seed's median timing, then the median paired ratio.

| Query epsilon | QPS speedup | Mean recall across three seeds, both binaries |
| --- | ---: | ---: |
| 0 | 1.019x | 52.750% |
| 0.1 | 1.034x | 65.683% |
| 0.3 | 1.039x | 85.550% |
| 0.6 | 1.071x | 97.650% |
| 1 | 1.078x | 99.733% |

The severe earlier query slowdown did not repeat. Epsilon-zero seed 42 is still
0.974x, so this is not an all-endpoint statistical nonregression result. Subset
graph median speedups are 1.054x/1.053x at k15/k30, with seed ratios spanning
0.895-1.309x and 0.942-1.102x respectively: timing variability remains substantial.

Full-corpus graph seed 105, five repetitions, gives median delivery times
5.662 -> 5.128 seconds at k15 (1.104x), and 12.309 -> 11.246 seconds at k30
(1.094x). Strict sampled full-corpus recall is 0.678158/0.790706; tie-aware recall
is 0.686198/0.795443. Process peak RSS is 1.411 -> 1.259 GB. All paired graph/query
fingerprints and per-row recalls match exactly throughout these reruns.

Host load averages were 1.13/2.96/6.12 before and 5.32/5.65/5.97 after the runs.
These snapshots do not establish which processes interfered or prove causality.
Results are consistent with transient interference, but remain exploratory;
defaults and the full confirmation requirement are unchanged.

## Protocol

- Required datasets: MNIST, Fashion-MNIST, GloVe-100-angular, NYTimes-256-angular.
- Graph output degree: 15 and 30; query result count: 10.
- Graph timing starts with a canonical in-memory float32 array and includes the
  constructor, internal normalization/copies, sorting and first delivery of both
  NumPy arrays. Disk loading, canonical input conversion and scoring are excluded.
  `--ordinary-index` measures ordinary lazy indices instead of `graph_only=True`.
- Query mode always indexes the full training corpus. Adapter fitting explicitly
  prepares search structures, so preparation is charged to query-ready build time.
  Each epsilon receives a warm-up query. Timings include the Python adapter call;
  result validation and reference-distance calculation are outside the timed region.
- Query recall uses ann-benchmarks' distance function and `get_recall_values` with
  its stored kth test-neighbor distance and default additive 0.001 tolerance.
  This distance tolerance is unrelated to the 0.001 recall non-inferiority margin.
- Test-query IDs have a fixed 20% tuning / 80% confirmation split. Query limits
  are accepted only on the tuning split. Never tune on the confirmation split.
- Results retain per-row recall, all repetitions, latency samples, binary hashes,
  source provenance, CPU affinity, thread environment and build-stage statistics.
  RSS is the process lifetime high-water mark, not an incremental index-size metric.
  Recent graph runs also record an untimed complete indices/distances fingerprint.
- Output JSON files are exclusive-create. All measurements are labelled
  `exploratory_only`; these tools do not automatically promote configurations.

## Exact training graphs

`ground_truth.py` uses SciPy exhaustive float64 distances on canonical float32
input, in bounded query/reference blocks. It stores the top 30 non-self neighbors;
the first 15 give the smaller reference. No approximate backend or dimensionality
reduction is used. Neighbor IDs are local to `corpus_ids`; that array maps back to
original dataset rows. `query_rows` addresses rows within that corpus.

Subset sampling is deterministic and nested by membership for a fixed seed.
Ties sort by distance and then original dataset row ID, including ties spanning
reference blocks. Self exclusion is by row identity; distinct duplicate vectors
remain eligible. Graph scoring reports both strict ID overlap and distance-based
tie-aware recall with a fixed 1e-12 numerical tolerance. Missing, self, invalid or
duplicate returned IDs cannot earn additional credit.

HDF5 references contain source/generator hashes, numerical-library versions,
sampling/metric conventions and completion metadata. Block completion is flushed
before progress advances. Repeating a command resumes only with an identical
manifest; after generator or configuration changes, select a new output path.
Existing references remain usable for scoring; the measurement file records their
original manifest and checksum.

### NYTimes zero vectors

The downloaded NYTimes dataset has 239 zero training vectors and 9 zero test
vectors. Its official test-distance array contains non-finite values. Training
references explicitly define angular distance involving a zero vector as 1.0,
consistent with RINND's cosine zero-vector convention; this extension is recorded
in each manifest. Zero rows are not silently discarded.

Query mode leaves official ground truth and recall semantics unchanged, reporting
invalid kth-distance rows separately, both in the evaluated split and across the
test set. Claims about meaningful accuracy on undefined ground truth require a
separately agreed dataset repair; do not treat the retained official score as such
a guarantee. The initial 200-query tuning sample contains no invalid truth rows.

## Run locally

Use the `uv` conda environment, then run these commands inside `ann-benchmarks`.
The dataset loader downloads missing official files:

```bash
uv run python -c 'from ann_benchmarks.datasets import get_dataset; data, _ = get_dataset("mnist-784-euclidean"); data.close()'
uv run python ../benchmarks/ground_truth.py \
  data/mnist-784-euclidean.hdf5 \
  ../benchmarks/results/ground_truth/mnist-n10000-k30-new.hdf5 --size 10000
```

Substitute each required dataset name. Run a small subset pilot before increasing
the size. With `--size 0 --sample-rows 2048`, every sampled row is compared against
the complete training corpus. This produces sampled full-corpus truth, not a
complete full-corpus exact graph. The corpus is held in RAM; only distance matrices
are block-bounded. The default 128-by-8192 distance block is 8 MiB, with additional
top-k merge arrays and float64 input blocks.

Build/install the source-matched wheel for the local interpreter before measuring:

```bash
uv run --with maturin maturin build --release \
  --manifest-path ../crates/rinnd/Cargo.toml \
  --no-default-features \
  --interpreter "$PWD/.venv/bin/python" --out ../target/wheels/performance-baseline
uv pip install --python "$PWD/.venv/bin/python" --force-reinstall --no-deps \
  ../target/wheels/performance-baseline/rinnd-0.2.0-cp310-cp310-manylinux_2_34_x86_64.whl
```

The wheel filename is the one produced in this checkout; use the actual build
output after changing package version or platform. The baseline wheel was built
without introducing new Rust compiler flags. Keep flags identical in comparisons.

Example runs on this host's eight physical cores in socket 0 (adapt affinity on
other hosts; do not use SMT siblings accidentally):

```bash
taskset -c 0,2,4,6,8,10,12,14 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python ../benchmarks/benchmark_recall.py graph \
  data/mnist-784-euclidean.hdf5 ../benchmarks/results/graph-new.json \
  --reference ../benchmarks/results/ground_truth/mnist-n10000-k30-new.hdf5 \
  --threads 8 --seeds 42 43 44 --repeats 2

taskset -c 0,2,4,6,8,10,12,14 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python ../benchmarks/benchmark_recall.py query \
  data/mnist-784-euclidean.hdf5 ../benchmarks/results/query-new.json \
  --threads 8 --seeds 42 --repeats 2 --query-limit 200

uv run python -m pytest ../benchmarks/test_benchmark_recall.py \
  test/rinnd_adapter_test.py ../tests/test_rinnd.py -q
```

Use `--parameters '{"n_trees": 8, "n_iters": 12}'` to test explicit build policies.
Query mode also accepts adapter search/quantization parameters; use `--epsilons`
for an epsilon sweep. Do not assume every binding option is forwarded by the
adapter. Standalone graph mode forwards its parameters directly to the binding.
Single-query measurements are not batch-throughput results. Official benchmark
runs remain available through `uv run python run.py --local --algorithm rinnd
--dataset <name> --count 10 --runs 5` with a scoped definitions set.

## Initial measurements (2026-09-30)

Complete exact graphs were generated for all four 10,000-row subsets, seed 42.
Artifacts are in `results/ground_truth/*-n10000-k30.hdf5`. Approximate graph runs
used three build seeds, two repetitions and eight threads. Times below are median
constructor-through-return times; recall is mean strict ID recall across runs.

| Dataset | k=15 seconds | k=15 recall | k=30 seconds | k=30 recall |
| --- | ---: | ---: | ---: | ---: |
| MNIST | 0.1471 | 0.991233 | 0.3192 | 0.998330 |
| Fashion-MNIST | 0.1371 | 0.994998 | 0.3127 | 0.999513 |
| GloVe-100 | 0.1021 | 0.796798 | 0.1987 | 0.937997 |
| NYTimes-256 | 0.1419 | 0.616122 | 0.2567 | 0.827338 |

Raw records are in `results/baseline-graph/`. The separate
`results/baseline-query-pilot/` contains full-corpus builds with one seed, 200 fixed
tuning queries, two repetitions and epsilon 0/0.1/0.3/0.6/1.0 on all four datasets.
These older pilot records include a small array-coercion overhead that the final
query harness excludes; rerun both sides with the same harness before comparison.
Neither suite is sufficient for release decisions, a tuned frontier or a claimed
speedup. Historical artifacts keep their original code fingerprints.

Leaf initialization consumed roughly 14-25% of measured subset construction;
export was below 0.4 ms. Candidate updates dominate, especially at k=30. Therefore
the first proposed optimization is destination-bucket routing in leaf initialization,
following the existing iteration implementation, not a new graph-export API.
Preserve per-destination insertion order and threshold snapshots, test bitwise
output equality, then measure scaling at 1/8/16 physical cores before proceeding.

## Remaining promotion work

1. Generate full-corpus sampled references and establish baseline graph/query
   frontiers, including attainable high-recall endpoints. The subset defaults above
   are not target-quality configurations for the angular datasets.
2. Add a paired baseline/candidate experiment driver with alternating run order,
   fixed targets and independent confirmation seeds. `paired_gate` is a tested
   primitive, not a complete promotion workflow: it applies a paired Student-t
   bound to seed means conditional on the evaluated rows. Sampled full-corpus
   inference also needs row-sampling uncertainty.
3. Require at least ten paired build seeds, five timing repetitions, and simultaneous
   one-sided 95% bounds across all declared datasets/degrees/targets. The lower
   recall-difference bound must be at least -0.001 everywhere. Declare the full
   multiplicity family before testing; the primitive's `family_size` must include
   the entire family, not just one file. Insufficient/inconclusive evidence blocks
   promotion. Pairing and family bookkeeping are not yet automated.
4. Profile and implement one work-preserving construction change, then validate
   both graph and query outputs. Optimize allocation, distance and heap hotpaths
   before exploring convergence or quantization tradeoffs.
5. Confirm on untouched official test queries, with full recall curves, matched-
   recall timing confidence intervals, memory and tail-latency checks. Retain FP32
   fallbacks and baseline paths whenever a workload or recall endpoint regresses.

Generated datasets and results are local ignored artifacts. The adapter changes
and its tests belong to the nested `ann-benchmarks` Git repository; the harness
and this documentation belong to the outer RINND repository.