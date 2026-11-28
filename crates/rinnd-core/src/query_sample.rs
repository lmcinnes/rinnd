//! Search indexes adapted to a sample of the queries they will answer.
//!
//! When queries come from a different distribution than the indexed data (text
//! queries against image embeddings, say), much of a search's effort can go
//! into reaching the region of the graph that holds a query's neighbours, and
//! graph edges built from the data alone may not connect the points those
//! neighbours lie among. Given example queries and their nearest neighbours in
//! the data, an index can be adapted to them (see
//! [`crate::NNDescentIndex::prepare_with_query_sample`]):
//!
//! * query-induced edges ([`query_induced_edges`]): each sample query's few
//!   nearest data points (its "pivots") get edges to the query's other near
//!   neighbours, pruned among themselves as relative neighbourhood graph edges
//!   are, so that the points queries' results lie among are linked to each
//!   other (as in RoarGraph's projection of a query-data bipartite graph). The
//!   edges join the diversified k-NN edges before the search graph is
//!   symmetrised and degree pruned;
//! * a query search tree ([`query_search_tree`]): searches start from the
//!   leaves of a random projection tree built over the sample queries rather
//!   than the data, each leaf listing the nearest data points of the sample
//!   queries that fall in it.

use crate::distance::Distance;
use crate::rng::FastRng;
use crate::tree::{build_rp_tree, FlatTree};
use rayon::prelude::*;

/// Sample queries (`n_queries x dim`, row-major) and the indices of their
/// nearest data points, nearest first (`n_queries x width`, -1 padded).
#[derive(Clone, Copy, Debug)]
pub struct QuerySample<'a> {
    pub queries: &'a [f32],
    pub neighbors: &'a [i32],
    pub n_queries: usize,
    pub width: usize,
}

impl QuerySample<'_> {
    #[inline]
    fn neighbors_of(&self, query: usize) -> &[i32] {
        &self.neighbors[query * self.width..(query + 1) * self.width]
    }
}

/// How a query sample adapts an index.
#[derive(Clone, Copy, Debug)]
pub struct QuerySampleOptions {
    /// Query-induced edges kept per point (0: none).
    pub edges: usize,
    /// The nearest data points of each sample query offered edges.
    pub pivots: usize,
    /// The neighbours of each sample query edges are drawn from.
    pub candidates: usize,
    /// Sample queries per query search tree leaf (0: no query tree).
    pub tree_leaf_size: usize,
    /// The most data points a query tree leaf lists.
    pub tree_entries: usize,
    /// How many of each sample query's nearest data points leaves draw from.
    pub tree_ranks: usize,
}

impl Default for QuerySampleOptions {
    fn default() -> Self {
        Self {
            edges: 16,
            pivots: 4,
            candidates: 64,
            tree_leaf_size: 30,
            tree_entries: 32,
            tree_ranks: 10,
        }
    }
}

/// Edges among the data points the sample queries' neighbours lie among.
///
/// Each of a query's `pivots` nearest data points is offered edges to the
/// query's other neighbours among its first `candidates`; a point keeps, of all
/// the edges it is offered, at most `edges`, closest first, dropping any that a
/// kept edge's endpoint is closer to (relative neighbourhood pruning).
///
/// Returns each data point's edges as `(neighbor, distance)`, nearest first.
pub fn query_induced_edges<D: Distance<f32>>(
    data: &[f32],
    dim: usize,
    n_points: usize,
    distance: &D,
    sample: &QuerySample,
    options: &QuerySampleOptions,
) -> Vec<Vec<(i32, f32)>> {
    let mut out = vec![Vec::new(); n_points];
    if options.edges == 0 || sample.n_queries == 0 || sample.width == 0 {
        return out;
    }
    let pivots = options.pivots.min(sample.width);
    let candidates = options.candidates.min(sample.width);
    // (pivot, query) pairs, grouped by pivot.
    let mut pairs: Vec<(i32, u32)> = Vec::with_capacity(sample.n_queries * pivots);
    for query in 0..sample.n_queries {
        for &pivot in &sample.neighbors_of(query)[..pivots] {
            if pivot >= 0 && (pivot as usize) < n_points {
                pairs.push((pivot, query as u32));
            }
        }
    }
    pairs.sort_unstable();
    let mut groups: Vec<(i32, usize, usize)> = Vec::new();
    let mut start = 0;
    while start < pairs.len() {
        let pivot = pairs[start].0;
        let mut end = start + 1;
        while end < pairs.len() && pairs[end].0 == pivot {
            end += 1;
        }
        groups.push((pivot, start, end));
        start = end;
    }
    let row = |index: i32| &data[index as usize * dim..(index as usize + 1) * dim];
    let edges: Vec<(i32, Vec<(i32, f32)>)> = groups
        .par_iter()
        .map(|&(pivot, start, end)| {
            let mut offered: Vec<(f32, i32)> = Vec::with_capacity((end - start) * candidates);
            for &(_, query) in &pairs[start..end] {
                for &candidate in &sample.neighbors_of(query as usize)[..candidates] {
                    if candidate < 0 || candidate == pivot || candidate as usize >= n_points {
                        continue;
                    }
                    offered.push((distance.distance(row(pivot), row(candidate)), candidate));
                }
            }
            offered.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let mut kept: Vec<(i32, f32)> = Vec::with_capacity(options.edges);
            let mut last = -1;
            for (d, candidate) in offered {
                if kept.len() >= options.edges {
                    break;
                }
                if candidate == last {
                    continue;
                }
                last = candidate;
                let pruned = kept.iter().any(|&(other, _)| {
                    other == candidate || distance.distance(row(other), row(candidate)) < d
                });
                if !pruned {
                    kept.push((candidate, d));
                }
            }
            (pivot, kept)
        })
        .collect();
    for (pivot, kept) in edges {
        out[pivot as usize] = kept;
    }
    out
}

/// A random projection tree over the sample queries for finding search entry
/// points.
///
/// Each leaf lists (up to `tree_entries` of) the nearest data points of the
/// queries in it, taking every query's nearest first, then every query's
/// second nearest, and so on, from each query's `tree_ranks` nearest; the
/// leaves list at most `n_points` points in all.
pub fn query_search_tree(
    sample: &QuerySample,
    dim: usize,
    n_points: usize,
    options: &QuerySampleOptions,
    angular: bool,
    rng: &mut FastRng,
) -> FlatTree {
    let leaf_size = options.tree_leaf_size.max(1);
    let mut tree = build_rp_tree(sample.queries, sample.n_queries, dim, leaf_size, rng, angular, 200);
    let leaves: Vec<usize> = (0..tree.children.len())
        .filter(|&node| tree.children[node][0] <= 0)
        .collect();
    let per_leaf = options.tree_entries.min(n_points / leaves.len().max(1)).max(1);
    let ranks = options.tree_ranks.min(sample.width).max(1);
    let fallback = sample.neighbors.iter().copied().find(|&n| n >= 0).unwrap_or(0);
    let mut indices: Vec<i32> = Vec::with_capacity(leaves.len() * per_leaf);
    let mut chosen: Vec<i32> = Vec::with_capacity(per_leaf);
    for node in leaves {
        let (start, end) = (-tree.children[node][0] as usize, -tree.children[node][1] as usize);
        let members = &tree.indices[start..end];
        chosen.clear();
        'ranks: for rank in 0..ranks {
            for &query in members {
                let candidate = sample.neighbors_of(query as usize)[rank];
                if candidate >= 0 && (candidate as usize) < n_points && !chosen.contains(&candidate) {
                    chosen.push(candidate);
                    if chosen.len() >= per_leaf {
                        break 'ranks;
                    }
                }
            }
        }
        if chosen.is_empty() {
            chosen.push(fallback);
        }
        let position = indices.len() as i32;
        indices.extend_from_slice(&chosen);
        tree.children[node] = [-position, -(indices.len() as i32)];
    }
    tree.indices = indices;
    tree
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::SquaredEuclidean;

    fn exact_neighbors(data: &[f32], queries: &[f32], dim: usize, width: usize) -> Vec<i32> {
        let n = data.len() / dim;
        let mut out = Vec::new();
        for query in queries.chunks_exact(dim) {
            let mut order: Vec<(f32, i32)> = (0..n)
                .map(|i| (SquaredEuclidean.distance(query, &data[i * dim..(i + 1) * dim]), i as i32))
                .collect();
            order.sort_by(|a, b| a.0.total_cmp(&b.0));
            out.extend(order[..width].iter().map(|&(_, i)| i));
        }
        out
    }

    fn sample_data(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = FastRng::new(seed);
        (0..n * dim).map(|_| rng.next_float() * 2.0 - 1.0).collect()
    }

    #[test]
    fn query_edges_link_pivots_to_query_neighbours() {
        let (n, dim, width) = (500, 8, 20);
        let data = sample_data(n, dim, 1);
        let queries = sample_data(50, dim, 2);
        let neighbors = exact_neighbors(&data, &queries, dim, width);
        let sample = QuerySample { queries: &queries, neighbors: &neighbors, n_queries: 50, width };
        let options = QuerySampleOptions { edges: 4, pivots: 2, candidates: 10, ..Default::default() };
        let edges = query_induced_edges(&data, dim, n, &SquaredEuclidean, &sample, &options);
        let pivots: std::collections::HashSet<i32> =
            (0..50).flat_map(|q| neighbors[q * width..q * width + 2].to_vec()).collect();
        let mut total = 0;
        for (point, kept) in edges.iter().enumerate() {
            if !pivots.contains(&(point as i32)) {
                assert!(kept.is_empty());
                continue;
            }
            assert!(!kept.is_empty() && kept.len() <= 4);
            total += kept.len();
            for window in kept.windows(2) {
                assert!(window[0].1 <= window[1].1);
            }
            for &(other, d) in kept {
                assert_ne!(other, point as i32);
                let exact = SquaredEuclidean.distance(&data[point * dim..(point + 1) * dim], &data[other as usize * dim..(other as usize + 1) * dim]);
                assert!((exact - d).abs() < 1e-5);
            }
        }
        assert!(total > 0);
    }

    #[test]
    fn query_tree_leaves_list_query_neighbours() {
        let (n, dim, width) = (500, 8, 20);
        let data = sample_data(n, dim, 3);
        let queries = sample_data(200, dim, 4);
        let neighbors = exact_neighbors(&data, &queries, dim, width);
        let sample = QuerySample { queries: &queries, neighbors: &neighbors, n_queries: 200, width };
        // Leaves hold at most 20 queries, so 32 entries list every query's nearest point.
        let options = QuerySampleOptions { tree_leaf_size: 20, tree_entries: 32, tree_ranks: 5, ..Default::default() };
        let tree = query_search_tree(&sample, dim, n, &options, false, &mut FastRng::new(5));
        assert!(tree.indices.len() <= n);
        let mut rng = FastRng::new(6);
        let (mut frontier, mut leaves) = (Vec::new(), Vec::new());
        for q in 0..200 {
            tree.search_leaves(&queries[q * dim..(q + 1) * dim], &mut rng, 1, &mut frontier, &mut leaves);
            let (start, end) = leaves[0];
            assert!(end > start && end - start <= 32);
            for &idx in &tree.indices[start..end] {
                assert!(idx >= 0 && (idx as usize) < n);
            }
            // A sample query's leaf lists its nearest point.
            assert!(tree.indices[start..end].contains(&neighbors[q * width]));
        }
    }
}
