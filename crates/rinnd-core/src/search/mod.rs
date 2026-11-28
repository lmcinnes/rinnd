//! Search algorithms for querying the k-NN graph.

mod greedy;

pub use greedy::{
    greedy_search, greedy_search_affine_into, greedy_search_affine_with_workspace_initialized,
    greedy_search_indices_with_workspace,
    greedy_search_indices_with_workspace_initialized,
    greedy_search_quantized_custom_with_workspace_initialized,
    greedy_search_quantized_i8_with_workspace,
    greedy_search_quantized_i8_with_workspace_initialized, greedy_search_with_workspace,
    greedy_search_with_workspace_initialized, SearchWorkspace,
};
pub(crate) use greedy::prefetch_row;
