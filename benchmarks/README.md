# Recall-gated performance work

This implementation establishes references and measurement tools for graph
construction and query optimizations.
Graph delivery and single-query throughput are separate, equally important workloads.

## Current default decision (2026-09-30)

Default builds now enable all four features: `batched-euclidean`, `batched-angular`,
`compact-candidates`, and `batched-leaves`. The Python bindings inherit these core
defaults. The user explicitly approved this change after the combination screens,
tree repair and repeated NYTimes measurements, accepting the remaining timing
variability and overriding the earlier requirement to finish full confirmation
before changing defaults. This is an engineering decision, not a claim that the
ten-seed held-out statistical gates passed. Full confirmation remains incomplete;
existing measurement artifacts retain their original exploratory status.

Use `cargo build -p rinnd-core --no-default-features --features std,rayon` for the
all-optimizations-off core configuration. For Python, add `--no-default-features`
to `maturin build --manifest-path crates/rinnd/Cargo.toml`; the binding supplies
`std,rayon` explicitly. Add individual `rinnd-core/<feature>` flags to build a subset.
The matrix runner already disables defaults explicitly, so its masks remain valid.

The sections below are a chronological experiment record. Their statements about
features being opt-in, defaults being unchanged, and promotion being blocked
describe the state at that time, superseded by this decision. Historical isolated
feature builds must use `--no-default-features` to reproduce their original masks.

## Optimization experiments

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