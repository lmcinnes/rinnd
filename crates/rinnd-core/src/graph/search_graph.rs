//! CSR (Compressed Sparse Row) search graph for efficient traversal.

use crate::distance::Distance;
use rand::Rng;

/// Search graph in CSR format for efficient neighbor traversal.
///
/// This format is memory-efficient and cache-friendly for graph search
/// operations where we iterate over neighbors of each vertex.
#[derive(Clone, Debug)]
pub struct SearchGraph {
    /// Row pointers: indptr[i] to indptr[i+1] gives the range of neighbors for vertex i
    pub indptr: Vec<i32>,
    /// Column indices: the neighbor indices
    pub indices: Vec<i32>,
    /// Number of vertices
    pub n_vertices: usize,
}

impl SearchGraph {
    /// Create a new empty search graph.
    pub fn new(n_vertices: usize) -> Self {
        Self {
            indptr: vec![0; n_vertices + 1],
            indices: Vec::new(),
            n_vertices,
        }
    }

    /// Create from dense neighbor arrays (after diversification/pruning).
    ///
    /// Takes row-major neighbor indices where -1 indicates no neighbor.
    pub fn from_dense(neighbor_indices: &[i32], n_points: usize, k: usize) -> Self {
        let mut indptr = Vec::with_capacity(n_points + 1);
        let mut indices = Vec::new();

        indptr.push(0);

        for point in 0..n_points {
            let start = point * k;
            let end = start + k;

            for &neighbor in &neighbor_indices[start..end] {
                if neighbor >= 0 {
                    indices.push(neighbor);
                }
            }

            indptr.push(indices.len() as i32);
        }

        Self {
            indptr,
            indices,
            n_vertices: n_points,
        }
    }

    /// Build a bidirectional search graph from dense neighbor indices.
    ///
    /// For each edge (i -> j), also adds (j -> i). Duplicate edges are removed.
    /// This matches PyNNDescent's approach of unioning forward + reverse graphs.
    pub fn from_dense_bidirectional(neighbor_indices: &[i32], n_points: usize, k: usize) -> Self {
        // First pass: count edges per vertex (forward + reverse)
        let mut adj: Vec<Vec<i32>> = vec![Vec::new(); n_points];

        for point in 0..n_points {
            let start = point * k;
            let end = start + k;
            for &neighbor in &neighbor_indices[start..end] {
                if neighbor >= 0 {
                    let n = neighbor as usize;
                    adj[point].push(neighbor);
                    adj[n].push(point as i32);
                }
            }
        }

        // Sort and dedup each row, then build CSR
        let mut indptr = Vec::with_capacity(n_points + 1);
        let mut indices = Vec::new();
        indptr.push(0);

        for row in &mut adj {
            row.sort_unstable();
            row.dedup();
            indices.extend_from_slice(row);
            indptr.push(indices.len() as i32);
        }

        Self {
            indptr,
            indices,
            n_vertices: n_points,
        }
    }

    /// Get the neighbors of a vertex.
    #[inline]
    pub fn neighbors(&self, vertex: usize) -> &[i32] {
        let start = self.indptr[vertex] as usize;
        let end = self.indptr[vertex + 1] as usize;
        &self.indices[start..end]
    }

    /// Get the degree (number of neighbors) of a vertex.
    #[inline]
    pub fn degree(&self, vertex: usize) -> usize {
        (self.indptr[vertex + 1] - self.indptr[vertex]) as usize
    }

    /// Get total number of edges.
    #[inline]
    pub fn num_edges(&self) -> usize {
        self.indices.len()
    }

    /// Compute the union with another graph (for undirected graph from directed).
    pub fn union_with(&self, other: &SearchGraph) -> SearchGraph {
        use std::collections::BTreeSet;

        let n = self.n_vertices;
        let mut adjacency: Vec<BTreeSet<i32>> = vec![BTreeSet::new(); n];

        // Add edges from self
        for v in 0..n {
            for &neighbor in self.neighbors(v) {
                adjacency[v].insert(neighbor);
            }
        }

        // Add edges from other
        for v in 0..n {
            for &neighbor in other.neighbors(v) {
                adjacency[v].insert(neighbor);
            }
        }

        // Convert back to CSR
        let mut indptr = Vec::with_capacity(n + 1);
        let mut indices = Vec::new();

        indptr.push(0);
        for neighbors in &adjacency {
            for &neighbor in neighbors {
                indices.push(neighbor);
            }
            indptr.push(indices.len() as i32);
        }

        SearchGraph {
            indptr,
            indices,
            n_vertices: n,
        }
    }

    /// Transpose the graph (reverse all edges).
    pub fn transpose(&self) -> SearchGraph {
        let n = self.n_vertices;

        // Count incoming edges for each vertex
        let mut in_degree = vec![0i32; n];
        for &neighbor in &self.indices {
            if neighbor >= 0 {
                in_degree[neighbor as usize] += 1;
            }
        }

        // Build indptr for transposed graph
        let mut indptr = Vec::with_capacity(n + 1);
        indptr.push(0);
        for &deg in &in_degree {
            indptr.push(indptr.last().unwrap() + deg);
        }

        // Fill indices
        let mut indices = vec![0i32; self.indices.len()];
        let mut current_pos = indptr[..n].to_vec();

        for v in 0..n {
            for &neighbor in self.neighbors(v) {
                if neighbor >= 0 {
                    let pos = current_pos[neighbor as usize] as usize;
                    indices[pos] = v as i32;
                    current_pos[neighbor as usize] += 1;
                }
            }
        }

        SearchGraph {
            indptr,
            indices,
            n_vertices: n,
        }
    }

    /// Reorder the graph according to a permutation.
    pub fn reorder(&mut self, order: &[usize]) {
        let n = self.n_vertices;

        // Create inverse mapping
        let mut inverse = vec![0usize; n];
        for (new_pos, &old_pos) in order.iter().enumerate() {
            inverse[old_pos] = new_pos;
        }

        // Build new graph
        let mut new_indptr = Vec::with_capacity(n + 1);
        let mut new_indices = Vec::new();

        new_indptr.push(0);
        for new_v in 0..n {
            let old_v = order[new_v];

            // Map neighbors to new indices
            for &old_neighbor in self.neighbors(old_v) {
                if old_neighbor >= 0 {
                    let new_neighbor = inverse[old_neighbor as usize] as i32;
                    new_indices.push(new_neighbor);
                }
            }

            new_indptr.push(new_indices.len() as i32);
        }

        self.indptr = new_indptr;
        self.indices = new_indices;
    }

    /// Prune edges to limit maximum degree.
    pub fn prune_degree(&mut self, max_degree: usize) {
        let n = self.n_vertices;

        let mut new_indptr = Vec::with_capacity(n + 1);
        let mut new_indices = Vec::new();

        new_indptr.push(0);
        for v in 0..n {
            let neighbors = self.neighbors(v);
            let keep = neighbors.len().min(max_degree);

            for &neighbor in &neighbors[..keep] {
                new_indices.push(neighbor);
            }

            new_indptr.push(new_indices.len() as i32);
        }

        self.indptr = new_indptr;
        self.indices = new_indices;
    }

    /// Build a diversified + pruned search graph matching PyNNDescent's pipeline.
    ///
    /// Steps:
    /// 1. Forward diversify: relative-neighborhood pruning on sorted k-NN
    /// 2. Build forward CSR graph
    /// 3. Reverse diversify: transpose, sort rows by distance, diversify
    /// 4. Union: forward ∪ diversified_reverse
    /// 5. Remove self-loops
    /// 6. Degree prune: cap max vertex degree
    pub fn from_dense_diversified<D: Distance<f32>>(
        neighbor_indices: &[i32],
        neighbor_distances: &[f32],
        data: &[f32],
        n_points: usize,
        k: usize,
        dim: usize,
        dist: &D,
        diversify_prob: f32,
        pruning_degree_multiplier: f32,
    ) -> (Self, f32) {
        use rayon::prelude::*;

        let max_degree = (pruning_degree_multiplier * k as f32).round() as usize;

        // Step 1: Forward diversify (parallel over points)
        // Each point's neighbors are already sorted by ascending distance after sort_all().
        let diversified: Vec<Vec<(i32, f32)>> = (0..n_points)
            .into_par_iter()
            .map(|i| {
                let row_start = i * k;
                let row_indices = &neighbor_indices[row_start..row_start + k];
                let row_dists = &neighbor_distances[row_start..row_start + k];
                diversify_row(
                    row_indices,
                    row_dists,
                    data,
                    dim,
                    dist,
                    diversify_prob,
                    max_degree,
                )
            })
            .collect();

        // Step 2: Build forward CSR graph (with distances for reverse diversification)
        let mut fwd_indptr = Vec::with_capacity(n_points + 1);
        let mut fwd_indices = Vec::new();
        let mut fwd_data = Vec::new();
        fwd_indptr.push(0i32);
        for row in &diversified {
            for &(idx, d) in row {
                fwd_indices.push(idx);
                fwd_data.push(d);
            }
            fwd_indptr.push(fwd_indices.len() as i32);
        }

        // Step 3: Transpose to get reverse graph, then diversify reverse edges
        let (rev_indptr, rev_indices, rev_data) =
            transpose_csr(&fwd_indptr, &fwd_indices, &fwd_data, n_points);

        // Diversify reverse graph rows (parallel)
        let rev_diversified: Vec<Vec<(i32, f32)>> = (0..n_points)
            .into_par_iter()
            .map(|i| {
                let start = rev_indptr[i] as usize;
                let end = rev_indptr[i + 1] as usize;
                if start == end {
                    return Vec::new();
                }
                let row_indices = &rev_indices[start..end];
                let row_data = &rev_data[start..end];

                // Sort by distance for diversification
                let mut order: Vec<usize> = (0..row_indices.len()).collect();
                order.sort_unstable_by(|&a, &b| {
                    row_data[a]
                        .partial_cmp(&row_data[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                diversify_sorted_csr(
                    row_indices,
                    row_data,
                    &order,
                    data,
                    dim,
                    dist,
                    diversify_prob,
                    max_degree,
                )
            })
            .collect();

        // Step 4-6: weighted union, self-loop removal, and degree pruning.
        // Edge weights must survive until pruning: selecting by vertex ID here
        // can discard the nearest edges and materially weaken search quality.
        let graph = weighted_union_pruned(&diversified, &rev_diversified, n_points, max_degree);

        let min_distance = fwd_data
            .iter()
            .map(|&d| if d == 0.0 { f32::EPSILON } else { d })
            .filter(|&d| d.is_finite())
            .fold(f32::INFINITY, f32::min);

        (graph, min_distance)
    }
}

/// Union weighted forward and reverse rows and retain the nearest edges.
///
/// PyNNDescent combines duplicate sparse edges with `maximum`, then performs
/// degree pruning using the retained edge weights. The returned binary CSR is
/// sorted by vertex ID only after the nearest edges have been selected.
fn weighted_union_pruned(
    forward: &[Vec<(i32, f32)>],
    reverse: &[Vec<(i32, f32)>],
    n_points: usize,
    max_degree: usize,
) -> SearchGraph {
    let mut indptr = Vec::with_capacity(n_points + 1);
    let mut indices = Vec::new();
    indptr.push(0);

    for vertex in 0..n_points {
        let mut row = Vec::with_capacity(forward[vertex].len() + reverse[vertex].len());
        row.extend(
            forward[vertex].iter().copied().filter(|&(idx, distance)| {
                idx >= 0 && idx != vertex as i32 && distance.is_finite()
            }),
        );
        row.extend(
            reverse[vertex].iter().copied().filter(|&(idx, distance)| {
                idx >= 0 && idx != vertex as i32 && distance.is_finite()
            }),
        );

        // Group duplicate vertex IDs and match scipy sparse `maximum` used by
        // PyNNDescent when unioning forward and reverse search graphs.
        row.sort_unstable_by_key(|&(idx, _)| idx);
        let mut deduplicated: Vec<(i32, f32)> = Vec::with_capacity(row.len());
        for (idx, distance) in row {
            if let Some((last_idx, last_distance)) = deduplicated.last_mut() {
                if *last_idx == idx {
                    *last_distance = last_distance.max(distance);
                    continue;
                }
            }
            deduplicated.push((idx, distance));
        }

        // Select by distance, with vertex ID providing deterministic tie order.
        deduplicated.sort_unstable_by(
            |&(left_idx, left_distance), &(right_idx, right_distance)| {
                left_distance
                    .total_cmp(&right_distance)
                    .then_with(|| left_idx.cmp(&right_idx))
            },
        );
        deduplicated.truncate(max_degree);

        // Search CSR rows follow PyNNDescent's binary CSR column order.
        deduplicated.sort_unstable_by_key(|&(idx, _)| idx);
        indices.extend(deduplicated.into_iter().map(|(idx, _)| idx));
        indptr.push(indices.len() as i32);
    }

    SearchGraph {
        indptr,
        indices,
        n_vertices: n_points,
    }
}

/// Diversify a single row of sorted neighbors (for forward diversification).
///
/// Implements relative-neighborhood graph pruning: a neighbor j is kept only
/// if no previously-retained neighbor c is closer to j than the query is to j.
fn diversify_row<D: Distance<f32>>(
    row_indices: &[i32],
    row_dists: &[f32],
    data: &[f32],
    dim: usize,
    dist: &D,
    prune_probability: f32,
    max_degree: usize,
) -> Vec<(i32, f32)> {
    let mut retained = Vec::new();

    for pos in 0..row_indices.len() {
        let j = row_indices[pos];
        if j < 0 {
            break;
        }
        let d_ij = row_dists[pos];

        let mut keep = true;
        for &(c, _d_ic) in &retained {
            // Check if retained neighbor c is closer to j than i is to j
            let c_start = (c as usize) * dim;
            let j_start = (j as usize) * dim;
            let d_cj = dist.distance(&data[c_start..c_start + dim], &data[j_start..j_start + dim]);
            if _d_ic > f32::MIN_POSITIVE && d_cj < d_ij {
                if prune_probability >= 1.0 || rand::thread_rng().gen::<f32>() < prune_probability {
                    keep = false;
                    break;
                }
            }
        }

        if keep {
            retained.push((j, d_ij));
            if retained.len() >= max_degree {
                break;
            }
        }
    }

    retained
}

/// Diversify a CSR row given a sorted order (for reverse graph diversification).
///
/// Returns retained neighbor indices and their original edge distances.
fn diversify_sorted_csr<D: Distance<f32>>(
    row_indices: &[i32],
    row_data: &[f32],
    sorted_order: &[usize],
    data: &[f32],
    dim: usize,
    dist: &D,
    prune_probability: f32,
    max_degree: usize,
) -> Vec<(i32, f32)> {
    let mut retained_indices = Vec::new();
    let mut retained_data = Vec::new();

    for &pos in sorted_order {
        let j = row_indices[pos];
        if j < 0 {
            continue;
        }
        let d_ij = row_data[pos];

        let mut keep = true;
        for idx in 0..retained_indices.len() {
            let c = retained_indices[idx];
            let d_ic: f32 = retained_data[idx];
            let c_start = (c as usize) * dim;
            let j_start = (j as usize) * dim;
            let d_cj = dist.distance(&data[c_start..c_start + dim], &data[j_start..j_start + dim]);
            if d_ic > f32::MIN_POSITIVE && d_cj < d_ij {
                if prune_probability >= 1.0 || rand::thread_rng().gen::<f32>() < prune_probability {
                    keep = false;
                    break;
                }
            }
        }

        if keep {
            retained_indices.push(j);
            retained_data.push(d_ij);
            if retained_indices.len() >= max_degree {
                break;
            }
        }
    }

    retained_indices.into_iter().zip(retained_data).collect()
}

/// Transpose a CSR graph, producing a new CSR with distances.
fn transpose_csr(
    indptr: &[i32],
    indices: &[i32],
    data: &[f32],
    n: usize,
) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    // Count incoming edges per vertex
    let mut in_degree = vec![0i32; n];
    for &neighbor in indices {
        if neighbor >= 0 {
            in_degree[neighbor as usize] += 1;
        }
    }

    // Build transposed indptr
    let mut t_indptr = Vec::with_capacity(n + 1);
    t_indptr.push(0i32);
    for &deg in &in_degree {
        t_indptr.push(t_indptr.last().unwrap() + deg);
    }

    let total_edges = *t_indptr.last().unwrap() as usize;
    let mut t_indices = vec![0i32; total_edges];
    let mut t_data = vec![0.0f32; total_edges];
    let mut current_pos: Vec<i32> = t_indptr[..n].to_vec();

    for v in 0..(indptr.len() - 1) {
        let start = indptr[v] as usize;
        let end = indptr[v + 1] as usize;
        for idx in start..end {
            let neighbor = indices[idx];
            if neighbor >= 0 {
                let pos = current_pos[neighbor as usize] as usize;
                t_indices[pos] = v as i32;
                t_data[pos] = data[idx];
                current_pos[neighbor as usize] += 1;
            }
        }
    }

    (t_indptr, t_indices, t_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_dense() {
        // 3 points, k=2
        let neighbors = vec![1, 2, 0, 2, 0, 1];
        let graph = SearchGraph::from_dense(&neighbors, 3, 2);

        assert_eq!(graph.n_vertices, 3);
        assert_eq!(graph.neighbors(0), &[1, 2]);
        assert_eq!(graph.neighbors(1), &[0, 2]);
        assert_eq!(graph.neighbors(2), &[0, 1]);
    }

    #[test]
    fn test_from_dense_with_missing() {
        // Some -1 entries
        let neighbors = vec![1, -1, 0, 2, -1, -1];
        let graph = SearchGraph::from_dense(&neighbors, 3, 2);

        assert_eq!(graph.neighbors(0), &[1]);
        assert_eq!(graph.neighbors(1), &[0, 2]);
        assert_eq!(graph.neighbors(2).len(), 0);
    }

    #[test]
    fn test_transpose() {
        let neighbors = vec![1, 2, 2, -1, -1, -1];
        let graph = SearchGraph::from_dense(&neighbors, 3, 2);
        let transposed = graph.transpose();

        // Original: 0 -> [1, 2], 1 -> [2]
        // Transposed: 1 <- [0], 2 <- [0, 1]
        assert_eq!(transposed.neighbors(0).len(), 0);
        assert_eq!(transposed.neighbors(1), &[0]);
        assert!(transposed.neighbors(2).contains(&0) && transposed.neighbors(2).contains(&1));
    }

    #[test]
    fn test_union() {
        let neighbors1 = vec![1, -1, 0, -1, -1, -1];
        let neighbors2 = vec![-1, 2, -1, -1, 0, 1];

        let g1 = SearchGraph::from_dense(&neighbors1, 3, 2);
        let g2 = SearchGraph::from_dense(&neighbors2, 3, 2);

        let union = g1.union_with(&g2);

        assert_eq!(union.neighbors(0).len(), 2); // 1 from g1, 2 from g2
        assert_eq!(union.neighbors(2).len(), 2); // 0 and 1 from g2
    }

    #[test]
    fn test_prune_degree() {
        let neighbors = vec![1, 2, 0, 2, 0, 1];
        let mut graph = SearchGraph::from_dense(&neighbors, 3, 2);

        graph.prune_degree(1);

        assert_eq!(graph.degree(0), 1);
        assert_eq!(graph.degree(1), 1);
        assert_eq!(graph.degree(2), 1);
    }

    #[test]
    fn test_weighted_union_prunes_by_distance_not_vertex_id() {
        let forward = vec![
            vec![(4, 0.1), (1, 0.4), (2, 0.3)],
            vec![],
            vec![],
            vec![],
            vec![],
        ];
        let reverse = vec![vec![(3, 0.2), (4, 0.9)], vec![], vec![], vec![], vec![]];

        let graph = weighted_union_pruned(&forward, &reverse, 5, 2);

        // Duplicate edge 4 takes max(0.1, 0.9), matching scipy `maximum`,
        // leaving vertices 3 and 2 as the nearest two edges.
        assert_eq!(graph.neighbors(0), &[2, 3]);
    }

    #[test]
    fn test_weighted_union_filters_edges_and_breaks_ties_by_id() {
        let forward = vec![
            vec![(0, 0.0), (-1, 0.1), (3, 0.2), (2, 0.2)],
            vec![],
            vec![],
            vec![],
        ];
        let reverse = vec![vec![(1, f32::INFINITY)], vec![], vec![], vec![]];

        let graph = weighted_union_pruned(&forward, &reverse, 4, 1);

        assert_eq!(graph.neighbors(0), &[2]);
        assert_eq!(graph.num_edges(), 1);
    }
}
