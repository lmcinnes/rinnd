//! Heap data structures for neighbor management.
//!
//! This module provides the `NeighborHeap` structure which maintains k-nearest
//! neighbors for each point. It uses a max-heap structure to efficiently track
//! the k smallest distances, with support for the "new/old" flag tracking used
//! in NN-Descent.

mod candidate_heap;
mod neighbor_heap;

pub use candidate_heap::{BoundedHeap, CandidateHeap};
pub use neighbor_heap::NeighborHeap;
