//! Random projection tree structures and construction.

mod builder;
mod flat_tree;

pub use builder::{build_rp_forest, build_rp_tree, rptree_leaf_array};
pub use flat_tree::{FlatTree, QuantizedFlatTree};
