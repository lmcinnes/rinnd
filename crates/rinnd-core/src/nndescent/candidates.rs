//! Candidate set management for NN-Descent.
//!
//! This implements the new/old candidate separation that is key to
//! NN-Descent's efficiency - we skip comparing (old, old) pairs.

use crate::heap::NeighborHeap;
use crate::rng::FastRng;
use rayon::prelude::*;
use std::time::Instant;

#[derive(Clone, Debug, Default)]
pub struct CandidateBuildStats {
    pub initialization_seconds: f64,
    pub forward_seconds: f64,
    pub reverse_seconds: f64,
    pub mark_seconds: f64,
    pub release_seconds: f64,
    /// Vertices with at least one new candidate.
    pub active_vertices: usize,
}

#[cfg(not(feature = "compact-candidates"))]
type ReverseCandidate = (usize, i32, f32, bool);

#[cfg(feature = "compact-candidates")]
#[derive(Clone, Copy)]
#[repr(C)]
struct ReverseCandidate {
    destination: i32,
    flagged_candidate: i32,
    priority: f32,
}

#[inline]
fn reverse_candidate(destination: i32, candidate: i32, priority: f32, is_new: bool) -> ReverseCandidate {
    #[cfg(feature = "compact-candidates")]
    {
        debug_assert!(destination >= 0 && candidate >= 0);
        ReverseCandidate {
            destination,
            flagged_candidate: if is_new { !candidate } else { candidate },
            priority,
        }
    }
    #[cfg(not(feature = "compact-candidates"))]
    (destination as usize, candidate, priority, is_new)
}

#[inline]
fn unpack_reverse_candidate(record: ReverseCandidate) -> (usize, i32, f32, bool) {
    #[cfg(feature = "compact-candidates")]
    {
        let is_new = record.flagged_candidate < 0;
        let candidate = if is_new { !record.flagged_candidate } else { record.flagged_candidate };
        (record.destination as usize, candidate, record.priority, is_new)
    }
    #[cfg(not(feature = "compact-candidates"))]
    record
}

/// Candidate sets using flat arrays for cache efficiency.
///
/// This separation is crucial for NN-Descent performance:
/// - (new, new) pairs: always compare
/// - (new, old) pairs: always compare  
/// - (old, old) pairs: skip (already compared in previous iteration)
#[derive(Clone, Debug)]
pub struct CandidateSets {
    /// New candidate indices, flat array shape (n_vertices × max_candidates)
    /// -1 indicates empty slot
    pub new_indices: Vec<i32>,
    /// Old candidate indices, flat array shape (n_vertices × max_candidates)
    pub old_indices: Vec<i32>,
    /// Number of vertices
    pub n_vertices: usize,
    /// Maximum candidates per vertex
    pub max_candidates: usize,
}

#[derive(Default)]
pub(crate) struct CandidateWorkspace {
    new_indices: Vec<i32>,
    old_indices: Vec<i32>,
    new_priority: Vec<f32>,
    old_priority: Vec<f32>,
    /// Rows the last active-set build may have written; `None` means all rows.
    #[cfg(feature = "active-candidates")]
    dirty_rows: Option<Vec<bool>>,
}

impl CandidateWorkspace {
    fn take_arrays(&mut self, size: usize) -> (Vec<i32>, Vec<f32>, Vec<i32>, Vec<f32>) {
        #[cfg(feature = "active-candidates")]
        {
            self.dirty_rows = None;
        }
        fn reset<T: Clone>(stored: &mut Vec<T>, size: usize, value: T) -> Vec<T> {
            let mut values = std::mem::take(stored);
            values.truncate(size);
            values.fill(value.clone());
            values.resize(size, value);
            values
        }
        (
            reset(&mut self.new_indices, size, -1),
            reset(&mut self.new_priority, size, f32::INFINITY),
            reset(&mut self.old_indices, size, -1),
            reset(&mut self.old_priority, size, f32::INFINITY),
        )
    }

    /// Like `take_arrays`, but only resets rows written by the previous active-set build.
    #[cfg(feature = "active-candidates")]
    fn take_arrays_active(&mut self, size: usize, width: usize) -> (Vec<i32>, Vec<f32>, Vec<i32>, Vec<f32>) {
        let reusable = |dirty: &Vec<bool>| {
            width > 0
                && dirty.len() * width == size
                && self.new_indices.len() == size
                && self.new_priority.len() == size
                && self.old_indices.len() == size
                && self.old_priority.len() == size
        };
        let Some(dirty) = self.dirty_rows.take().filter(reusable) else {
            return self.take_arrays(size);
        };
        let mut new_indices = std::mem::take(&mut self.new_indices);
        let mut new_priority = std::mem::take(&mut self.new_priority);
        let mut old_indices = std::mem::take(&mut self.old_indices);
        let mut old_priority = std::mem::take(&mut self.old_priority);
        new_indices
            .par_chunks_mut(width)
            .zip(new_priority.par_chunks_mut(width))
            .zip(old_indices.par_chunks_mut(width))
            .zip(old_priority.par_chunks_mut(width))
            .zip(dirty.par_iter())
            .filter(|(_, dirty)| **dirty)
            .for_each(|((((new_index, new_rank), old_index), old_rank), _)| {
                new_index.fill(-1);
                new_rank.fill(f32::INFINITY);
                old_index.fill(-1);
                old_rank.fill(f32::INFINITY);
            });
        (new_indices, new_priority, old_indices, old_priority)
    }

    #[cfg(any(feature = "candidate-workspace", test))]
    pub(crate) fn recycle(&mut self, candidates: CandidateSets) {
        self.new_indices = candidates.new_indices;
        self.old_indices = candidates.old_indices;
    }
}

impl CandidateSets {
    #[cfg(feature = "bench-internals")]
    #[doc(hidden)]
    pub fn replay_heap_updates(priorities: &mut [f32], indices: &mut [i32], updates: &[(f32, i32)], vector: bool) {
        if vector {
            for &(priority, index) in updates {
                checked_heap_push_flat_with::<true>(priorities, indices, priority, index);
            }
        } else {
            for &(priority, index) in updates {
                checked_heap_push_flat_with::<false>(priorities, indices, priority, index);
            }
        }
    }

    /// Get new candidates for a vertex as a slice.
    #[inline]
    pub fn get_new(&self, vertex: usize) -> &[i32] {
        let start = vertex * self.max_candidates;
        &self.new_indices[start..start + self.max_candidates]
    }

    /// Get old candidates for a vertex as a slice.
    #[inline]
    pub fn get_old(&self, vertex: usize) -> &[i32] {
        let start = vertex * self.max_candidates;
        &self.old_indices[start..start + self.max_candidates]
    }

    /// Build candidate sets from the current neighbor graph (parallel version).
    ///
    /// This uses a block-based parallel approach matching PyNNDescent:
    /// - Vertices are divided into blocks
    /// - Each thread handles one block but iterates all vertices
    /// - A thread only writes to heaps in its block
    /// - This avoids write conflicts without locks
    pub fn build_from_graph(
        graph: &mut NeighborHeap,
        max_candidates: usize,
        rng: &mut FastRng,
    ) -> Self {
        Self::build_from_graph_impl::<false, false>(graph, max_candidates, rng, &mut CandidateWorkspace::default()).0
    }

    pub(crate) fn build_from_graph_profiled(
        graph: &mut NeighborHeap,
        max_candidates: usize,
        rng: &mut FastRng,
    ) -> (Self, CandidateBuildStats) {
        Self::build_from_graph_impl::<true, false>(graph, max_candidates, rng, &mut CandidateWorkspace::default())
    }

    #[cfg(any(feature = "candidate-workspace", test))]
    pub(crate) fn build_from_graph_reusing(
        graph: &mut NeighborHeap,
        max_candidates: usize,
        rng: &mut FastRng,
        workspace: &mut CandidateWorkspace,
    ) -> (Self, CandidateBuildStats) {
        Self::build_from_graph_impl::<true, true>(graph, max_candidates, rng, workspace)
    }

    fn build_from_graph_impl<const PROFILE: bool, const REUSE: bool>(
        graph: &mut NeighborHeap,
        max_candidates: usize,
        rng: &mut FastRng,
        workspace: &mut CandidateWorkspace,
    ) -> (Self, CandidateBuildStats) {
        let n_vertices = graph.n_points;
        let n_threads = rayon::current_num_threads().max(1);

        #[cfg(feature = "blocked-candidates")]
        if blocked::enabled() {
            return blocked::build::<PROFILE, REUSE>(graph, max_candidates, rng, workspace);
        }

        // Use the parallel version if we have enough vertices
        if n_vertices >= 256 && n_threads > 1 {
            #[cfg(feature = "active-candidates")]
            return Self::build_from_graph_parallel_active::<PROFILE, REUSE>(graph, max_candidates, rng, n_threads, workspace);
            #[cfg(not(feature = "active-candidates"))]
            Self::build_from_graph_parallel::<PROFILE, REUSE>(graph, max_candidates, rng, n_threads, workspace)
        } else {
            Self::build_from_graph_sequential::<PROFILE, REUSE>(graph, max_candidates, rng, workspace)
        }
    }

    /// Sequential version of candidate building.
    fn build_from_graph_sequential<const PROFILE: bool, const REUSE: bool>(
        graph: &mut NeighborHeap,
        max_candidates: usize,
        rng: &mut FastRng,
        workspace: &mut CandidateWorkspace,
    ) -> (Self, CandidateBuildStats) {
        let mut stats = CandidateBuildStats::default();
        let started = PROFILE.then(Instant::now);
        let n_vertices = graph.n_points;
        let k = graph.k;

        // Flat arrays for cache efficiency
        let size = n_vertices * max_candidates;
        let (mut new_indices, mut new_priority, mut old_indices, mut old_priority) = if REUSE {
            workspace.take_arrays(size)
        } else {
            (vec![-1; size], vec![f32::INFINITY; size], vec![-1; size], vec![f32::INFINITY; size])
        };

        stats.initialization_seconds = started.map_or(0.0, |time| time.elapsed().as_secs_f64());
        let started = PROFILE.then(Instant::now);

        // Process all edges: forward (i -> neighbor) and reverse (neighbor -> i)
        for i in 0..n_vertices {
            let row_start = i * k;

            for j in 0..k {
                let neighbor = graph.indices[row_start + j];
                if neighbor < 0 {
                    continue;
                }
                let neighbor_idx = neighbor as usize;
                let is_new = graph.flags[row_start + j] != 0;
                let priority = rng.next_float();

                // Add forward edge (i -> neighbor) - neighbor is candidate for i
                let offset_i = i * max_candidates;
                if is_new {
                    checked_heap_push_flat(
                        &mut new_priority[offset_i..offset_i + max_candidates],
                        &mut new_indices[offset_i..offset_i + max_candidates],
                        priority,
                        neighbor,
                    );
                } else {
                    checked_heap_push_flat(
                        &mut old_priority[offset_i..offset_i + max_candidates],
                        &mut old_indices[offset_i..offset_i + max_candidates],
                        priority,
                        neighbor,
                    );
                }

                // Add reverse edge (neighbor -> i) - i is candidate for neighbor
                let reverse_priority = rng.next_float();
                let offset_n = neighbor_idx * max_candidates;
                if is_new {
                    checked_heap_push_flat(
                        &mut new_priority[offset_n..offset_n + max_candidates],
                        &mut new_indices[offset_n..offset_n + max_candidates],
                        reverse_priority,
                        i as i32,
                    );
                } else {
                    checked_heap_push_flat(
                        &mut old_priority[offset_n..offset_n + max_candidates],
                        &mut old_indices[offset_n..offset_n + max_candidates],
                        reverse_priority,
                        i as i32,
                    );
                }
            }
        }

        stats.forward_seconds = started.map_or(0.0, |time| time.elapsed().as_secs_f64());
        let started = PROFILE.then(Instant::now);
        // Mark neighbors that appear in new_candidates as old (flag=0) in the graph
        Self::mark_old_flags(graph, &new_indices, max_candidates);
        stats.mark_seconds = started.map_or(0.0, |time| time.elapsed().as_secs_f64());
        let started = PROFILE.then(Instant::now);
        if REUSE {
            workspace.new_priority = new_priority;
            workspace.old_priority = old_priority;
        } else {
            drop(new_priority);
            drop(old_priority);
        }
        stats.release_seconds = started.map_or(0.0, |time| time.elapsed().as_secs_f64());

        (Self {
            new_indices,
            old_indices,
            n_vertices,
            max_candidates,
        }, stats)
    }

    /// Parallel version of candidate building using two-phase approach.
    ///
    /// Phase 1: Each thread processes its own block's rows, pushing forward edges.
    ///          Also collects reverse edges bucketed by destination block.
    /// Phase 2: Each thread applies reverse edges destined for its block.
    ///
    /// This is O(n*k) total scan work instead of O(n_threads * n*k).
    #[cfg_attr(all(feature = "active-candidates", not(test)), allow(dead_code))]
    fn build_from_graph_parallel<const PROFILE: bool, const REUSE: bool>(
        graph: &mut NeighborHeap,
        max_candidates: usize,
        rng: &mut FastRng,
        n_threads: usize,
        workspace: &mut CandidateWorkspace,
    ) -> (Self, CandidateBuildStats) {
        let mut stats = CandidateBuildStats::default();
        let started = PROFILE.then(Instant::now);
        let n_vertices = graph.n_points;
        let k = graph.k;
        let block_size = (n_vertices + n_threads - 1) / n_threads;

        // Flat arrays for all blocks
        let size = n_vertices * max_candidates;
        let (mut new_indices, mut new_priority, mut old_indices, mut old_priority) = if REUSE {
            workspace.take_arrays(size)
        } else {
            (vec![-1; size], vec![f32::INFINITY; size], vec![-1; size], vec![f32::INFINITY; size])
        };

        // Create per-thread RNG seeds
        let thread_seeds: Vec<u64> = (0..n_threads).map(|_| rng.next_u64()).collect();

        // Read-only references
        let graph_indices = &graph.indices;
        let graph_flags = &graph.flags;

        stats.initialization_seconds = started.map_or(0.0, |time| time.elapsed().as_secs_f64());
        let started = PROFILE.then(Instant::now);

        // Phase 1: Each thread processes its own rows (forward edges) and collects
        // reverse edges bucketed by destination block.
        // reverse_buckets[src_thread][dest_block] = Vec of (dest_vertex, candidate, priority, is_new)
        let reverse_buckets: Vec<Vec<Vec<ReverseCandidate>>> = (0..n_threads)
            .into_par_iter()
            .map(|thread_idx| {
                let block_start = thread_idx * block_size;
                let block_end = ((thread_idx + 1) * block_size).min(n_vertices);

                if block_start >= n_vertices {
                    return vec![Vec::new(); n_threads];
                }

                let mut local_rng = FastRng::new(thread_seeds[thread_idx]);

                // Per-destination-block reverse edge buckets
                let mut buckets: Vec<Vec<ReverseCandidate>> = vec![Vec::new(); n_threads];

                // SAFETY: Each thread writes to disjoint portions [block_start*mc .. block_end*mc)
                let new_idx_ptr = new_indices.as_ptr() as *mut i32;
                let new_pri_ptr = new_priority.as_ptr() as *mut f32;
                let old_idx_ptr = old_indices.as_ptr() as *mut i32;
                let old_pri_ptr = old_priority.as_ptr() as *mut f32;

                for i in block_start..block_end {
                    let row_start = i * k;

                    for j in 0..k {
                        let flat_idx = row_start + j;
                        let neighbor = graph_indices[flat_idx];

                        if neighbor < 0 {
                            continue;
                        }
                        let neighbor_idx = neighbor as usize;
                        let is_new = graph_flags[flat_idx] != 0;
                        let priority = local_rng.next_float();

                        // Forward edge: push (neighbor) as candidate for vertex i
                        let offset = i * max_candidates;
                        if is_new {
                            unsafe {
                                let pri = std::slice::from_raw_parts_mut(
                                    new_pri_ptr.add(offset),
                                    max_candidates,
                                );
                                let idx = std::slice::from_raw_parts_mut(
                                    new_idx_ptr.add(offset),
                                    max_candidates,
                                );
                                checked_heap_push_flat(pri, idx, priority, neighbor);
                            }
                        } else {
                            unsafe {
                                let pri = std::slice::from_raw_parts_mut(
                                    old_pri_ptr.add(offset),
                                    max_candidates,
                                );
                                let idx = std::slice::from_raw_parts_mut(
                                    old_idx_ptr.add(offset),
                                    max_candidates,
                                );
                                checked_heap_push_flat(pri, idx, priority, neighbor);
                            }
                        }

                        // Reverse edge: push (i) as candidate for vertex neighbor
                        // Bucket by which block the neighbor belongs to
                        let dest_block = neighbor_idx / block_size;
                        if dest_block < n_threads {
                            if dest_block == thread_idx {
                                // Neighbor is in our own block - handle directly
                                let n_offset = neighbor_idx * max_candidates;
                                if is_new {
                                    unsafe {
                                        let pri = std::slice::from_raw_parts_mut(
                                            new_pri_ptr.add(n_offset),
                                            max_candidates,
                                        );
                                        let idx = std::slice::from_raw_parts_mut(
                                            new_idx_ptr.add(n_offset),
                                            max_candidates,
                                        );
                                        checked_heap_push_flat(pri, idx, priority, i as i32);
                                    }
                                } else {
                                    unsafe {
                                        let pri = std::slice::from_raw_parts_mut(
                                            old_pri_ptr.add(n_offset),
                                            max_candidates,
                                        );
                                        let idx = std::slice::from_raw_parts_mut(
                                            old_idx_ptr.add(n_offset),
                                            max_candidates,
                                        );
                                        checked_heap_push_flat(pri, idx, priority, i as i32);
                                    }
                                }
                            } else {
                                buckets[dest_block].push(reverse_candidate(
                                    neighbor,
                                    i as i32,
                                    priority,
                                    is_new,
                                ));
                            }
                        }
                    }
                }

                buckets
            })
            .collect();

        stats.forward_seconds = started.map_or(0.0, |time| time.elapsed().as_secs_f64());
        let started = PROFILE.then(Instant::now);

        // Phase 2: Each thread applies reverse edges destined for its block
        (0..n_threads).into_par_iter().for_each(|thread_idx| {
            let block_start = thread_idx * block_size;
            if block_start >= n_vertices {
                return;
            }

            // SAFETY: Each thread writes to disjoint portions [block_start*mc .. block_end*mc)
            let new_idx_ptr = new_indices.as_ptr() as *mut i32;
            let new_pri_ptr = new_priority.as_ptr() as *mut f32;
            let old_idx_ptr = old_indices.as_ptr() as *mut i32;
            let old_pri_ptr = old_priority.as_ptr() as *mut f32;

            // Read reverse edges from all source threads destined for this block
            for src_thread in 0..n_threads {
                if src_thread == thread_idx {
                    continue; // Already handled in Phase 1
                }
                for &record in &reverse_buckets[src_thread][thread_idx] {
                    let (dest_vertex, candidate, priority, is_new) = unpack_reverse_candidate(record);
                    let offset = dest_vertex * max_candidates;
                    if is_new {
                        unsafe {
                            let pri = std::slice::from_raw_parts_mut(
                                new_pri_ptr.add(offset),
                                max_candidates,
                            );
                            let idx = std::slice::from_raw_parts_mut(
                                new_idx_ptr.add(offset),
                                max_candidates,
                            );
                            checked_heap_push_flat(pri, idx, priority, candidate);
                        }
                    } else {
                        unsafe {
                            let pri = std::slice::from_raw_parts_mut(
                                old_pri_ptr.add(offset),
                                max_candidates,
                            );
                            let idx = std::slice::from_raw_parts_mut(
                                old_idx_ptr.add(offset),
                                max_candidates,
                            );
                            checked_heap_push_flat(pri, idx, priority, candidate);
                        }
                    }
                }
            }
        });

        stats.reverse_seconds = started.map_or(0.0, |time| time.elapsed().as_secs_f64());
        let started = PROFILE.then(Instant::now);
        // Mark neighbors that appear in new_candidates as old (flag=0) in the graph
        Self::mark_old_flags(graph, &new_indices, max_candidates);
        stats.mark_seconds = started.map_or(0.0, |time| time.elapsed().as_secs_f64());
        let started = PROFILE.then(Instant::now);
        drop(reverse_buckets);
        if REUSE {
            workspace.new_priority = new_priority;
            workspace.old_priority = old_priority;
        } else {
            drop(new_priority);
            drop(old_priority);
        }
        stats.release_seconds = started.map_or(0.0, |time| time.elapsed().as_secs_f64());

        (Self {
            new_indices,
            old_indices,
            n_vertices,
            max_candidates,
        }, stats)
    }

    /// Parallel candidate build that fills old lists only for active vertices.
    ///
    /// The join never reads old candidates of a vertex without new candidates, so
    /// new lists are built first and each thread's RNG stream is replayed for old
    /// lists. Every active row sees the same push sequence as the default build,
    /// so outputs are identical; inactive rows' old lists are left empty.
    #[cfg(feature = "active-candidates")]
    fn build_from_graph_parallel_active<const PROFILE: bool, const REUSE: bool>(
        graph: &mut NeighborHeap,
        max_candidates: usize,
        rng: &mut FastRng,
        n_threads: usize,
        workspace: &mut CandidateWorkspace,
    ) -> (Self, CandidateBuildStats) {
        let mut stats = CandidateBuildStats::default();
        let started = PROFILE.then(Instant::now);
        let n_vertices = graph.n_points;
        let k = graph.k;
        let block_size = (n_vertices + n_threads - 1) / n_threads;

        let size = n_vertices * max_candidates;
        let (mut new_indices, mut new_priority, mut old_indices, mut old_priority) = if REUSE {
            workspace.take_arrays_active(size, max_candidates)
        } else {
            (vec![-1; size], vec![f32::INFINITY; size], vec![-1; size], vec![f32::INFINITY; size])
        };
        let thread_seeds: Vec<u64> = (0..n_threads).map(|_| rng.next_u64()).collect();
        let graph_indices = &graph.indices;
        let graph_flags = &graph.flags;
        let new_heaps = RawHeaps::new(&mut new_indices, &mut new_priority, max_candidates);
        let old_heaps = RawHeaps::new(&mut old_indices, &mut old_priority, max_candidates);

        stats.initialization_seconds = started.map_or(0.0, |time| time.elapsed().as_secs_f64());

        // Scan this thread's rows, replaying its RNG; push edges with the requested flag
        // whose destination passes `active`, bucketing cross-block reverse edges.
        let scan = |thread_idx: usize, want_new: bool, active: &(dyn Fn(usize) -> bool + Sync)| {
            let block_start = thread_idx * block_size;
            let block_end = ((thread_idx + 1) * block_size).min(n_vertices);
            let mut buckets: Vec<Vec<ReverseCandidate>> = vec![Vec::new(); n_threads];
            if block_start >= n_vertices {
                return buckets;
            }
            let heaps = if want_new { new_heaps } else { old_heaps };
            let mut local_rng = FastRng::new(thread_seeds[thread_idx]);
            for i in block_start..block_end {
                let source_active = active(i);
                let row_start = i * k;
                for j in 0..k {
                    let neighbor = graph_indices[row_start + j];
                    if neighbor < 0 {
                        continue;
                    }
                    let priority = local_rng.next_float();
                    if (graph_flags[row_start + j] != 0) != want_new {
                        continue;
                    }
                    // SAFETY: rows in [block_start, block_end) are written only by this thread.
                    if source_active {
                        unsafe { heaps.push(i, priority, neighbor) };
                    }
                    let neighbor_idx = neighbor as usize;
                    if !active(neighbor_idx) {
                        continue;
                    }
                    let dest_block = neighbor_idx / block_size;
                    if dest_block == thread_idx {
                        unsafe { heaps.push(neighbor_idx, priority, i as i32) };
                    } else {
                        buckets[dest_block].push(reverse_candidate(neighbor, i as i32, priority, want_new));
                    }
                }
            }
            buckets
        };
        let apply = |buckets: &[Vec<Vec<ReverseCandidate>>], want_new: bool| {
            let heaps = if want_new { new_heaps } else { old_heaps };
            (0..n_threads).into_par_iter().for_each(|thread_idx| {
                for (src_thread, source) in buckets.iter().enumerate() {
                    if src_thread == thread_idx {
                        continue;
                    }
                    for &record in &source[thread_idx] {
                        let (dest_vertex, candidate, priority, _) = unpack_reverse_candidate(record);
                        // SAFETY: bucket `thread_idx` only holds rows owned by this thread.
                        unsafe { heaps.push(dest_vertex, priority, candidate) };
                    }
                }
            });
        };

        let started = PROFILE.then(Instant::now);
        let new_buckets: Vec<_> = (0..n_threads)
            .into_par_iter()
            .map(|thread_idx| scan(thread_idx, true, &|_| true))
            .collect();
        let mut forward_seconds = started.map_or(0.0, |time| time.elapsed().as_secs_f64());

        let started = PROFILE.then(Instant::now);
        apply(&new_buckets, true);
        drop(new_buckets);
        let mut reverse_seconds = started.map_or(0.0, |time| time.elapsed().as_secs_f64());

        let started = PROFILE.then(Instant::now);
        let active: Vec<bool> = new_indices
            .par_chunks(max_candidates.max(1))
            .map(|row| row.iter().any(|&candidate| candidate >= 0))
            .collect();
        stats.active_vertices = active.iter().filter(|&&value| value).count();
        let old_buckets: Vec<_> = (0..n_threads)
            .into_par_iter()
            .map(|thread_idx| scan(thread_idx, false, &|vertex| active[vertex]))
            .collect();
        forward_seconds += started.map_or(0.0, |time| time.elapsed().as_secs_f64());

        let started = PROFILE.then(Instant::now);
        apply(&old_buckets, false);
        reverse_seconds += started.map_or(0.0, |time| time.elapsed().as_secs_f64());
        stats.forward_seconds = forward_seconds;
        stats.reverse_seconds = reverse_seconds;

        let started = PROFILE.then(Instant::now);
        Self::mark_old_flags(graph, &new_indices, max_candidates);
        stats.mark_seconds = started.map_or(0.0, |time| time.elapsed().as_secs_f64());
        let started = PROFILE.then(Instant::now);
        drop(old_buckets);
        if REUSE {
            workspace.new_priority = new_priority;
            workspace.old_priority = old_priority;
            // Every push lands in an active row, so only those rows need resetting next time.
            workspace.dirty_rows = Some(active);
        } else {
            drop(new_priority);
            drop(old_priority);
        }
        stats.release_seconds = started.map_or(0.0, |time| time.elapsed().as_secs_f64());

        (Self {
            new_indices,
            old_indices,
            n_vertices,
            max_candidates,
        }, stats)
    }

    /// Mark flags in graph as old for neighbors that appear in new_indices.
    fn mark_old_flags(graph: &mut NeighborHeap, new_indices: &[i32], max_candidates: usize) {
        let n_vertices = graph.n_points;
        let k = graph.k;
        let n_threads = rayon::current_num_threads().max(1);

        if n_vertices < 256 || n_threads <= 1 {
            // Sequential version
            for i in 0..n_vertices {
                let row_offset = i * k;
                let new_offset = i * max_candidates;

                for j in 0..k {
                    let neighbor = graph.indices[row_offset + j];
                    if neighbor < 0 || graph.flags[row_offset + j] == 0 {
                        continue;
                    }

                    for nc_idx in 0..max_candidates {
                        if new_indices[new_offset + nc_idx] == neighbor {
                            graph.flags[row_offset + j] = 0;
                            break;
                        }
                    }
                }
            }
        } else {
            // Parallel version - each thread handles a disjoint range of vertices
            // SAFETY: Each thread writes to a disjoint range of flags (different rows).
            // We use a usize wrapper to pass the pointer safely across threads.
            let flags_base = graph.flags.as_mut_ptr() as usize;
            let indices_ref = &graph.indices;

            (0..n_threads).into_par_iter().for_each(|t| {
                let block_size = (n_vertices + n_threads - 1) / n_threads;
                let start = t * block_size;
                let end = ((t + 1) * block_size).min(n_vertices);
                let flags_ptr = flags_base as *mut u8;

                for i in start..end {
                    let row_offset = i * k;
                    let new_offset = i * max_candidates;

                    for j in 0..k {
                        let neighbor = indices_ref[row_offset + j];
                        if neighbor < 0 {
                            continue;
                        }
                        if unsafe { *flags_ptr.add(row_offset + j) } == 0 {
                            continue;
                        }

                        for nc_idx in 0..max_candidates {
                            if new_indices[new_offset + nc_idx] == neighbor {
                                unsafe {
                                    *flags_ptr.add(row_offset + j) = 0;
                                }
                                break;
                            }
                        }
                    }
                }
            });
        }
    }

    /// Get total number of new candidates across all vertices.
    pub fn total_new(&self) -> usize {
        self.new_indices.iter().filter(|&&x| x >= 0).count()
    }

    /// Get total number of old candidates across all vertices.
    pub fn total_old(&self) -> usize {
        self.old_indices.iter().filter(|&&x| x >= 0).count()
    }
}

/// Shared raw view of flat candidate heaps for row-disjoint parallel writes.
#[cfg(feature = "active-candidates")]
#[derive(Clone, Copy)]
struct RawHeaps {
    indices: *mut i32,
    priorities: *mut f32,
    width: usize,
}

#[cfg(feature = "active-candidates")]
unsafe impl Send for RawHeaps {}
#[cfg(feature = "active-candidates")]
unsafe impl Sync for RawHeaps {}

#[cfg(feature = "active-candidates")]
impl RawHeaps {
    fn new(indices: &mut [i32], priorities: &mut [f32], width: usize) -> Self {
        debug_assert_eq!(indices.len(), priorities.len());
        Self { indices: indices.as_mut_ptr(), priorities: priorities.as_mut_ptr(), width }
    }

    /// SAFETY: the caller must have exclusive access to `row` and `row` must be in bounds.
    #[inline(always)]
    unsafe fn push(self, row: usize, priority: f32, index: i32) {
        let offset = row * self.width;
        checked_heap_push_flat(
            std::slice::from_raw_parts_mut(self.priorities.add(offset), self.width),
            std::slice::from_raw_parts_mut(self.indices.add(offset), self.width),
            priority,
            index,
        );
    }
}

/// Cache-blocked candidate construction (feature `blocked-candidates`).
///
/// The classic build pushes every reverse edge into the heap row of its
/// destination, a random row of n x max_candidates arrays. Here:
///
/// 1. Scatter: rows are streamed in order and each reverse edge becomes an
///    8-byte record (destination, source | new-flag) in a bucket for its
///    destination block. Blocks are sized so a block's candidate heaps fit in
///    L2 (`RINND_CAND_BLOCK_BYTES`, default 512 KiB).
/// 2. Per block: forward pushes, then the block's reverse records, then the
///    graph flags of the block's rows are marked, all while the block's heaps
///    are cache-resident. Priority scratch is per block, not n x max_candidates.
///
/// Priorities are a hash of (seed, source, destination) with one seed drawn
/// from the RNG per build, so records carry no priority and the result does
/// not depend on the thread count: records reach each block in source-row
/// order however rows are split across threads.
///
/// `RINND_BLOCKED_CANDIDATES=0` selects the original algorithm at runtime.
#[cfg(feature = "blocked-candidates")]
pub(crate) mod blocked {
    use super::{checked_heap_push_flat, CandidateBuildStats, CandidateSets, CandidateWorkspace};
    use crate::heap::NeighborHeap;
    use crate::rng::FastRng;
    use rayon::prelude::*;
    use std::time::Instant;

    const NEW_BIT: u32 = 0x8000_0000;

    pub(crate) fn enabled() -> bool {
        !matches!(std::env::var("RINND_BLOCKED_CANDIDATES").as_deref(), Ok("0"))
    }

    /// Rows per destination block: a power of two whose candidate heaps
    /// (new + old indices and priorities) fit the byte budget.
    pub(crate) fn block_rows(max_candidates: usize) -> usize {
        let budget = std::env::var("RINND_CAND_BLOCK_BYTES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(512 * 1024);
        let per_row = 16 * max_candidates.max(1);
        let rows = (budget / per_row).max(1);
        (1usize << (usize::BITS - 1 - rows.leading_zeros())).clamp(64, 1 << 16)
    }

    #[inline(always)]
    fn priority(seed: u64, source: u32, destination: u32) -> f32 {
        let mut z = seed ^ (((source as u64) << 32) | destination as u64);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
    }

    pub(crate) fn build<const PROFILE: bool, const REUSE: bool>(
        graph: &mut NeighborHeap,
        max_candidates: usize,
        rng: &mut FastRng,
        workspace: &mut CandidateWorkspace,
    ) -> (CandidateSets, CandidateBuildStats) {
        let rows_per_block = block_rows(max_candidates);
        build_with_block_rows::<PROFILE, REUSE>(graph, max_candidates, rng, workspace, rows_per_block)
    }

    pub(crate) fn build_with_block_rows<const PROFILE: bool, const REUSE: bool>(
        graph: &mut NeighborHeap,
        max_candidates: usize,
        rng: &mut FastRng,
        workspace: &mut CandidateWorkspace,
        rows_per_block: usize,
    ) -> (CandidateSets, CandidateBuildStats) {
        let mut stats = CandidateBuildStats::default();
        let started = PROFILE.then(Instant::now);
        let n = graph.n_points;
        let k = graph.k;
        let mc = max_candidates;
        let seed = rng.next_u64();
        let n_blocks = n.div_ceil(rows_per_block).max(1);
        let n_threads = rayon::current_num_threads().max(1);
        let rows_per_thread = n.div_ceil(n_threads).max(1);

        // Output arrays: reuse allocations without a full fill (each block
        // initializes its own rows).
        let size = n * mc;
        let take = |v: &mut Vec<i32>| {
            let mut v = if REUSE { std::mem::take(v) } else { Vec::new() };
            v.resize(size, -1);
            v.truncate(size);
            v
        };
        let mut new_indices = take(&mut workspace.new_indices);
        let mut old_indices = take(&mut workspace.old_indices);
        #[cfg(feature = "active-candidates")]
        {
            workspace.dirty_rows = None;
        }
        stats.initialization_seconds = started.map_or(0.0, |t| t.elapsed().as_secs_f64());

        // Phase 1: scatter reverse edges into destination-block buckets.
        let started = PROFILE.then(Instant::now);
        let indices = &graph.indices;
        let flags = &graph.flags;
        let expected = (n * k) / (n_blocks * n_threads) + 16;
        let buckets: Vec<Vec<Vec<u64>>> = (0..n_threads)
            .into_par_iter()
            .map(|t| {
                let mut out: Vec<Vec<u64>> = (0..n_blocks).map(|_| Vec::with_capacity(expected + expected / 4)).collect();
                let start = (t * rows_per_thread).min(n);
                let end = ((t + 1) * rows_per_thread).min(n);
                for i in start..end {
                    let row = &indices[i * k..(i + 1) * k];
                    let row_flags = &flags[i * k..(i + 1) * k];
                    for (&neighbor, &flag) in row.iter().zip(row_flags) {
                        if neighbor < 0 {
                            continue;
                        }
                        let source = i as u32 | if flag != 0 { NEW_BIT } else { 0 };
                        out[neighbor as usize / rows_per_block].push(((neighbor as u64) << 32) | source as u64);
                    }
                }
                out
            })
            .collect();
        stats.forward_seconds = started.map_or(0.0, |t| t.elapsed().as_secs_f64());

        // Phase 2: per destination block, build heaps and mark flags in cache.
        let started = PROFILE.then(Instant::now);
        let indices = &graph.indices;
        let flags = &mut graph.flags;
        new_indices
            .par_chunks_mut(rows_per_block * mc)
            .zip(old_indices.par_chunks_mut(rows_per_block * mc))
            .zip(flags.par_chunks_mut(rows_per_block * k))
            .enumerate()
            .for_each_init(
                || (vec![f32::INFINITY; rows_per_block * mc], vec![f32::INFINITY; rows_per_block * mc]),
                |(new_priority, old_priority), (block, ((new_rows, old_rows), block_flags))| {
                    let first_row = block * rows_per_block;
                    let rows = new_rows.len() / mc;
                    new_rows.fill(-1);
                    old_rows.fill(-1);
                    new_priority[..rows * mc].fill(f32::INFINITY);
                    old_priority[..rows * mc].fill(f32::INFINITY);
                    let mut push = |row: usize, candidate: i32, prio: f32, is_new: bool| {
                        let range = row * mc..(row + 1) * mc;
                        if is_new {
                            checked_heap_push_flat(&mut new_priority[range.clone()], &mut new_rows[range], prio, candidate);
                        } else {
                            checked_heap_push_flat(&mut old_priority[range.clone()], &mut old_rows[range], prio, candidate);
                        }
                    };
                    // Forward edges of this block's rows.
                    for r in 0..rows {
                        let i = first_row + r;
                        for j in 0..k {
                            let neighbor = indices[i * k + j];
                            if neighbor < 0 {
                                continue;
                            }
                            let p = priority(seed, i as u32, neighbor as u32);
                            push(r, neighbor, p, block_flags[r * k + j] != 0);
                        }
                    }
                    // Reverse edges, in source-row order.
                    for producer in &buckets {
                        for &record in &producer[block] {
                            let destination = (record >> 32) as u32;
                            let source = record as u32;
                            let candidate = source & !NEW_BIT;
                            let p = priority(seed, candidate, destination);
                            push(destination as usize - first_row, candidate as i32, p, source & NEW_BIT != 0);
                        }
                    }
                    // Mark sampled new neighbours as old.
                    for r in 0..rows {
                        let i = first_row + r;
                        let new_row = &new_rows[r * mc..(r + 1) * mc];
                        for j in 0..k {
                            let flag = &mut block_flags[r * k + j];
                            if *flag != 0 {
                                let neighbor = indices[i * k + j];
                                if neighbor >= 0 && new_row.contains(&neighbor) {
                                    *flag = 0;
                                }
                            }
                        }
                    }
                },
            );
        stats.reverse_seconds = started.map_or(0.0, |t| t.elapsed().as_secs_f64());
        let started = PROFILE.then(Instant::now);
        drop(buckets);
        stats.release_seconds = started.map_or(0.0, |t| t.elapsed().as_secs_f64());

        (CandidateSets { new_indices, old_indices, n_vertices: n, max_candidates: mc }, stats)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::collections::HashSet;

        fn random_graph(n: usize, k: usize, seed: u64) -> NeighborHeap {
            let mut rng = FastRng::new(seed);
            let mut graph = NeighborHeap::new(n, k);
            for i in 0..n {
                for _ in 0..k + 3 {
                    let j = rng.next_index(n);
                    if j != i {
                        graph.checked_flagged_push(i, j as i32, rng.next_float(), rng.next_bool());
                    }
                }
            }
            graph
        }

        /// Each row holds exactly min(mc, distinct) candidates drawn from the
        /// forward and reverse edges of the right flag, without duplicates, and
        /// graph flags are cleared exactly for new neighbours that were sampled.
        #[test]
        fn blocked_candidates_are_valid_samples() {
            for &(n, k, mc, block) in &[(3000, 12, 12, 64), (2500, 20, 15, 2048), (700, 8, 8, 128), (900, 6, 6, 4096)] {
                let mut graph = random_graph(n, k, n as u64);
                let before = graph.clone();
                let mut expected_new = vec![HashSet::new(); n];
                let mut expected_old = vec![HashSet::new(); n];
                for i in 0..n {
                    for j in 0..k {
                        let nb = before.indices[i * k + j];
                        if nb < 0 { continue; }
                        let a = if before.flags[i * k + j] != 0 { &mut expected_new } else { &mut expected_old };
                        a[i].insert(nb);
                        a[nb as usize].insert(i as i32);
                    }
                }
                let (c, _) = build_with_block_rows::<false, false>(&mut graph, mc, &mut FastRng::new(1), &mut CandidateWorkspace::default(), block);
                for i in 0..n {
                    for (row, expected) in [(c.get_new(i), &expected_new[i]), (c.get_old(i), &expected_old[i])] {
                        let got: Vec<i32> = row.iter().copied().filter(|&x| x >= 0).collect();
                        let set: HashSet<i32> = got.iter().copied().collect();
                        assert_eq!(set.len(), got.len(), "duplicate in row {i}");
                        assert!(set.is_subset(expected), "row {i} has a non-candidate");
                        assert_eq!(got.len(), expected.len().min(mc), "row {i} size");
                    }
                    for j in 0..k {
                        let nb = before.indices[i * k + j];
                        if nb < 0 { continue; }
                        let was_new = before.flags[i * k + j] != 0;
                        let sampled = c.get_new(i).contains(&nb);
                        assert_eq!(graph.flags[i * k + j] != 0, was_new && !sampled, "flag {i},{j}");
                    }
                }
            }
        }

        /// Output does not depend on the thread count.
        #[test]
        fn blocked_candidates_are_thread_independent() {
            let base = random_graph(4000, 10, 3);
            let run = |threads: usize| {
                let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
                pool.install(|| {
                    let mut g = base.clone();
                    let (c, _) = build_with_block_rows::<false, false>(&mut g, 10, &mut FastRng::new(5), &mut CandidateWorkspace::default(), 256);
                    (c.new_indices, c.old_indices, g.flags)
                })
            };
            let one = run(1);
            assert_eq!(one, run(3));
            assert_eq!(one, run(8));
        }
    }
}

/// Push to a bounded priority max-heap with duplicate checking (flat slice version).
/// Uses the shift-down technique matching PyNNDescent's `checked_heap_push`.
#[inline(always)]
fn checked_heap_push_flat(priorities: &mut [f32], indices: &mut [i32], priority: f32, index: i32) {
    checked_heap_push_flat_with::<{ cfg!(feature = "candidate-membership") }>(priorities, indices, priority, index);
}

#[inline(always)]
fn checked_heap_push_flat_with<const VECTOR: bool>(priorities: &mut [f32], indices: &mut [i32], priority: f32, index: i32) {
    // Early exit if priority is worse than current max
    // SAFETY: We know priorities is non-empty (max_candidates > 0)
    if priority >= unsafe { *priorities.get_unchecked(0) } {
        return;
    }

    // Check for duplicate (linear scan - OK since max_size is small ~30-60)
    let n = priorities.len();
    if VECTOR && n >= 16 {
        if indices.iter().fold(0u32, |found, &candidate| found | u32::from(candidate == index)) != 0 {
            return;
        }
    } else {
        for position in 0..n {
            if unsafe { *indices.get_unchecked(position) } == index {
                return;
            }
        }
    }

    // Insert at root and sift down using shift technique
    unsafe {
        *priorities.get_unchecked_mut(0) = priority;
        *indices.get_unchecked_mut(0) = index;
    }

    let mut pos = 0usize;
    loop {
        let left = 2 * pos + 1;
        let right = 2 * pos + 2;
        let mut largest = pos;

        unsafe {
            if left < n && *priorities.get_unchecked(left) > *priorities.get_unchecked(largest) {
                largest = left;
            }
            if right < n && *priorities.get_unchecked(right) > *priorities.get_unchecked(largest) {
                largest = right;
            }
        }

        if largest != pos {
            unsafe {
                let child_pri = *priorities.get_unchecked(largest);
                let child_idx = *indices.get_unchecked(largest);
                *priorities.get_unchecked_mut(pos) = child_pri;
                *indices.get_unchecked_mut(pos) = child_idx;
                *priorities.get_unchecked_mut(largest) = priority;
                *indices.get_unchecked_mut(largest) = index;
            }
            pos = largest;
        } else {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_workspace_reuse_preserves_outputs_and_rng() {
        let mut workspace = CandidateWorkspace::default();
        for threads in [1, 4, 8, 1] {
            rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
                for (points, width) in [(513, 30), (513, 30), (71, 15), (513, 60), (0, 1), (257, 1)] {
                    let mut initial = NeighborHeap::new(points, 30);
                    for point in 0..points {
                        for slot in 0..30 {
                            if (point + slot) % 11 != 0 {
                                initial.checked_flagged_push(point, ((point * 17 + slot * 7) % points) as i32,
                                    (slot + 1) as f32, (point + slot) % 3 == 0);
                            }
                        }
                    }
                    let mut fresh_graph = initial.clone();
                    let mut reused_graph = initial;
                    let mut fresh_rng = FastRng::new(42);
                    let mut reused_rng = FastRng::new(42);
                    for iteration in 0..4 {
                        if iteration % 2 == 1 {
                            fresh_graph.flags.fill(1);
                            reused_graph.flags.fill(1);
                        }
                        let fresh = CandidateSets::build_from_graph(&mut fresh_graph, width, &mut fresh_rng);
                        let (reused, _) = CandidateSets::build_from_graph_reusing(
                            &mut reused_graph, width, &mut reused_rng, &mut workspace,
                        );
                        assert_eq!(fresh.new_indices, reused.new_indices);
                        assert_eq!(fresh.old_indices, reused.old_indices);
                        assert_eq!(fresh_graph.flags, reused_graph.flags);
                        assert_eq!(fresh_graph.indices, reused_graph.indices);
                        assert_eq!(fresh_graph.distances, reused_graph.distances);
                        assert_eq!(fresh_rng.next_u64(), reused_rng.next_u64());
                        let index_pointer = reused.new_indices.as_ptr();
                        workspace.recycle(reused);
                        assert_eq!(workspace.new_indices.as_ptr(), index_pointer);
                    }
                }
            });
        }
    }

    #[test]
    fn vector_membership_preserves_candidate_heaps() {
        let mut rng = FastRng::new(381);
        for width in [1, 3, 7, 8, 15, 16, 30, 31, 32, 60, 65] {
            let mut scalar_priorities = vec![f32::INFINITY; width + 1];
            let mut vector_priorities = scalar_priorities.clone();
            let mut scalar_indices = vec![-1; width + 1];
            let mut vector_indices = scalar_indices.clone();
            for step in 0..4000 {
                let index = match step % 19 {
                    0 => i32::MAX,
                    1 => -1,
                    _ => (rng.next_u64() % (width as u64 * 3)) as i32,
                };
                let priority = match step % 29 {
                    0 => f32::INFINITY,
                    1 => 0.0,
                    _ => rng.next_float(),
                };
                checked_heap_push_flat_with::<false>(&mut scalar_priorities[1..], &mut scalar_indices[1..], priority, index);
                checked_heap_push_flat_with::<true>(&mut vector_priorities[1..], &mut vector_indices[1..], priority, index);
                assert_eq!(scalar_indices, vector_indices);
                assert_eq!(scalar_priorities, vector_priorities);
            }
        }
    }

    #[test]
    fn candidate_profiling_preserves_outputs_flags_and_rng() {
        for threads in [1, 4, 8] {
            rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
                for points in [0, 71, 513] {
                    for width in [1, 15, 30, 60] {
                        let mut initial = NeighborHeap::new(points, 30);
                        for point in 0..points {
                            for slot in 0..30 {
                                if (point + slot) % 11 != 0 {
                                    initial.checked_flagged_push(point, ((point * 17 + slot * 7) % points) as i32,
                                        (slot + 1) as f32, (point + slot) % 3 == 0);
                                }
                            }
                        }
                        let mut plain_graph = initial.clone();
                        let mut timed_graph = initial;
                        let mut plain_rng = FastRng::new(42);
                        let mut timed_rng = FastRng::new(42);
                        let plain = CandidateSets::build_from_graph(&mut plain_graph, width, &mut plain_rng);
                        let started = Instant::now();
                        let (timed, stats) = CandidateSets::build_from_graph_profiled(&mut timed_graph, width, &mut timed_rng);
                        let elapsed = started.elapsed().as_secs_f64();
                        assert_eq!(plain.new_indices, timed.new_indices);
                        assert_eq!(plain.old_indices, timed.old_indices);
                        assert_eq!(plain_graph.flags, timed_graph.flags);
                        assert_eq!(plain_graph.indices, timed_graph.indices);
                        assert_eq!(plain_graph.distances, timed_graph.distances);
                        assert_eq!(plain_rng.next_u64(), timed_rng.next_u64());
                        let phases = [stats.initialization_seconds, stats.forward_seconds,
                            stats.reverse_seconds, stats.mark_seconds, stats.release_seconds];
                        assert!(phases.iter().all(|value| value.is_finite() && *value >= 0.0));
                        assert!(phases.iter().sum::<f64>() <= elapsed);
                    }
                }
            });
        }
    }

    #[cfg(feature = "active-candidates")]
    #[test]
    fn active_candidates_match_default_for_active_rows() {
        for threads in [2, 4, 8] {
            rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
                for (points, width, new_period) in [(513, 30, 3), (513, 15, 17), (1031, 30, 97), (300, 60, 1), (257, 1, 5)] {
                    let mut initial = NeighborHeap::new(points, 30);
                    for point in 0..points {
                        for slot in 0..30 {
                            if (point + slot) % 11 != 0 {
                                initial.checked_flagged_push(point, ((point * 17 + slot * 7) % points) as i32,
                                    (slot + 1) as f32, (point * 7 + slot) % new_period == 0);
                            }
                        }
                    }
                    let mut default_graph = initial.clone();
                    let mut active_graph = initial;
                    let mut default_rng = FastRng::new(42);
                    let mut active_rng = FastRng::new(42);
                    let mut workspace = CandidateWorkspace::default();
                    let (expected, _) = CandidateSets::build_from_graph_parallel::<false, false>(
                        &mut default_graph, width, &mut default_rng, threads, &mut workspace,
                    );
                    let (actual, stats) = CandidateSets::build_from_graph_parallel_active::<true, false>(
                        &mut active_graph, width, &mut active_rng, threads, &mut workspace,
                    );
                    assert_eq!(expected.new_indices, actual.new_indices);
                    assert_eq!(default_graph.flags, active_graph.flags);
                    assert_eq!(default_rng.next_u64(), active_rng.next_u64());
                    let mut active = 0;
                    for vertex in 0..points {
                        if expected.get_new(vertex).iter().any(|&candidate| candidate >= 0) {
                            active += 1;
                            assert_eq!(expected.get_old(vertex), actual.get_old(vertex));
                        } else {
                            assert!(actual.get_old(vertex).iter().all(|&candidate| candidate < 0));
                        }
                    }
                    assert_eq!(stats.active_vertices, active);
                }
            });
        }
    }

    #[test]
    fn reverse_candidate_round_trip() {
        for destination in [0, 19, i32::MAX] {
            for candidate in [0, 31, i32::MAX] {
                for is_new in [false, true] {
                    let record = reverse_candidate(destination, candidate, 0.625, is_new);
                    assert_eq!(unpack_reverse_candidate(record), (destination as usize, candidate, 0.625, is_new));
                }
            }
        }
        #[cfg(feature = "compact-candidates")]
        assert_eq!(std::mem::size_of::<ReverseCandidate>(), 12);
    }

    #[test]
    fn test_checked_heap_push_flat_basic() {
        let mut priorities = vec![f32::INFINITY; 3];
        let mut indices = vec![-1; 3];

        checked_heap_push_flat(&mut priorities, &mut indices, 0.5, 1);
        checked_heap_push_flat(&mut priorities, &mut indices, 0.3, 2);
        checked_heap_push_flat(&mut priorities, &mut indices, 0.7, 3);

        // All three should be in
        assert!(indices.contains(&1));
        assert!(indices.contains(&2));
        assert!(indices.contains(&3));
    }

    #[test]
    fn test_checked_heap_push_flat_duplicate() {
        let mut priorities = vec![f32::INFINITY; 3];
        let mut indices = vec![-1; 3];

        checked_heap_push_flat(&mut priorities, &mut indices, 0.5, 1);
        checked_heap_push_flat(&mut priorities, &mut indices, 0.3, 1); // Duplicate - should be rejected
        checked_heap_push_flat(&mut priorities, &mut indices, 0.7, 1); // Duplicate - should be rejected

        // Only one entry should have index 1
        assert_eq!(indices.iter().filter(|&&x| x == 1).count(), 1);
    }

    #[test]
    fn test_checked_heap_push_flat_overflow() {
        let mut priorities = vec![f32::INFINITY; 3];
        let mut indices = vec![-1; 3];

        checked_heap_push_flat(&mut priorities, &mut indices, 0.9, 1);
        checked_heap_push_flat(&mut priorities, &mut indices, 0.8, 2);
        checked_heap_push_flat(&mut priorities, &mut indices, 0.7, 3);
        // Heap full with priorities [0.9, 0.8, 0.7]

        checked_heap_push_flat(&mut priorities, &mut indices, 0.5, 4); // Should replace 0.9

        // 4 should be in heap, 1 should not
        assert!(indices.contains(&4));
        assert!(!indices.contains(&1));
    }

    #[test]
    fn test_build_candidates() {
        let mut graph = NeighborHeap::new(5, 3);

        graph.unchecked_flagged_push(0, 1, 0.1, true);
        graph.unchecked_flagged_push(0, 2, 0.2, false);
        graph.unchecked_flagged_push(0, 3, 0.3, false);
        graph.unchecked_flagged_push(1, 0, 0.1, true);
        graph.unchecked_flagged_push(1, 2, 0.15, true);

        let mut rng = FastRng::new(42);
        let candidates = CandidateSets::build_from_graph(&mut graph, 10, &mut rng);

        // Check point 0's candidates include 1 as new
        let new_0 = candidates.get_new(0);
        assert!(new_0.contains(&1));

        // Check point 1's candidates include 0 and 2 as new
        let new_1 = candidates.get_new(1);
        assert!(new_1.contains(&0));
        assert!(new_1.contains(&2));
    }

    #[test]
    fn test_reverse_neighbors() {
        let mut graph = NeighborHeap::new(3, 2);

        graph.unchecked_flagged_push(0, 1, 0.1, true);
        graph.unchecked_flagged_push(1, 2, 0.2, true);

        let mut rng = FastRng::new(42);
        let candidates = CandidateSets::build_from_graph(&mut graph, 10, &mut rng);

        // Point 1 should have 0 as a reverse neighbor
        let new_1 = candidates.get_new(1);
        assert!(new_1.contains(&0));

        // Point 2 should have 1 as a reverse neighbor
        let new_2 = candidates.get_new(2);
        assert!(new_2.contains(&1));
    }

    #[test]
    fn test_max_candidates_limit() {
        let mut graph = NeighborHeap::new(10, 5);

        for i in 1..10 {
            graph.unchecked_flagged_push(0, i as i32, i as f32 * 0.1, true);
        }

        let mut rng = FastRng::new(42);
        let candidates = CandidateSets::build_from_graph(&mut graph, 3, &mut rng);

        // Point 0 should have at most 3 new candidates
        let new_0 = candidates.get_new(0);
        let count = new_0.iter().filter(|&&x| x >= 0).count();
        assert!(count <= 3);
    }
}
