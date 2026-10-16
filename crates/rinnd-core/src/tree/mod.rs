//! Random projection tree structures and construction.

mod breadth_first;
#[cfg(feature = "hybrid-forest")]
mod hybrid_forest;
mod builder;
mod flat_tree;

#[cfg(not(feature = "hybrid-forest"))]
pub(crate) use breadth_first::build_breadth_first_leaf_forest;
#[cfg(feature = "hybrid-forest")]
pub(crate) use hybrid_forest::{build_hybrid_forest, build_hybrid_leaf_forest, local_points_for};
pub use builder::{build_rp_forest, build_rp_tree, rptree_leaf_array};
pub(crate) use builder::{build_rp_leaf_forest, build_rp_leaf_tree};
pub use flat_tree::{FlatTree, QuantizedFlatTree};

#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub mod benchmark {
    use super::breadth_first::build_breadth_first_leaf_forest;
    use super::build_rp_leaf_forest;
    use crate::rng::FastRng;

    pub fn build_depth_first_leaf_forest(
        data: &[f32],
        n_points: usize,
        dim: usize,
        n_trees: usize,
        leaf_size: usize,
        seed: u64,
        angular: bool,
        max_depth: usize,
    ) -> Vec<Vec<i32>> {
        build_rp_leaf_forest(
            data,
            n_points,
            dim,
            n_trees,
            leaf_size,
            &mut FastRng::new(seed),
            angular,
            max_depth,
        )
    }

    pub fn build_breadth_first_leaf_forest_batched(
        data: &[f32],
        n_points: usize,
        dim: usize,
        n_trees: usize,
        leaf_size: usize,
        seed: u64,
        angular: bool,
        max_depth: usize,
        tree_batch_width: usize,
    ) -> Vec<Vec<i32>> {
        build_breadth_first_leaf_forest(
            data,
            n_points,
            dim,
            n_trees,
            leaf_size,
            &mut FastRng::new(seed),
            angular,
            max_depth,
            tree_batch_width,
        )
    }
}
