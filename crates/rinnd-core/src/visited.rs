//! Epoch-stamped visited set for efficient tracking during graph search.

/// A dense epoch-stamped set for tracking visited nodes during search.
///
/// Clearing normally increments the current epoch instead of writing the full
/// allocation. The backing storage is reset only when the `u16` epoch wraps.
#[derive(Clone)]
pub struct VisitedSet {
    epochs: Vec<u16>,
    epoch: u16,
    n_elements: usize,
}

impl VisitedSet {
    /// Create a new visited set that can track `n` elements.
    pub fn new(n: usize) -> Self {
        Self {
            epochs: vec![0; n],
            epoch: 1,
            n_elements: n,
        }
    }

    /// Check if an index has been visited.
    #[inline]
    pub fn is_visited(&self, idx: i32) -> bool {
        debug_assert!(idx >= 0 && (idx as usize) < self.n_elements);
        self.epochs[idx as usize] == self.epoch
    }

    /// Mark an index as visited.
    #[inline]
    pub fn mark(&mut self, idx: i32) {
        debug_assert!(idx >= 0 && (idx as usize) < self.n_elements);
        self.epochs[idx as usize] = self.epoch;
    }

    /// Check if visited and mark in one operation.
    /// Returns `true` if the index was already visited, `false` otherwise.
    #[inline]
    pub fn check_and_mark(&mut self, idx: i32) -> bool {
        debug_assert!(idx >= 0 && (idx as usize) < self.n_elements);
        let entry = &mut self.epochs[idx as usize];
        let was_visited = *entry == self.epoch;
        *entry = self.epoch;
        was_visited
    }

    /// Start a new empty generation in constant time except on epoch overflow.
    #[inline]
    pub fn clear(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.epochs.fill(0);
            self.epoch = 1;
        }
    }

    /// Get the number of elements this set can track.
    pub fn capacity(&self) -> usize {
        self.n_elements
    }
}

impl std::fmt::Debug for VisitedSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VisitedSet")
            .field("n_elements", &self.n_elements)
            .field("epoch", &self.epoch)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut visited = VisitedSet::new(100);

        assert!(!visited.is_visited(0));
        assert!(!visited.is_visited(50));
        assert!(!visited.is_visited(99));

        visited.mark(50);
        assert!(!visited.is_visited(0));
        assert!(visited.is_visited(50));
        assert!(!visited.is_visited(99));
    }

    #[test]
    fn test_check_and_mark() {
        let mut visited = VisitedSet::new(100);

        // First check should return false and mark
        assert!(!visited.check_and_mark(42));

        // Second check should return true (already visited)
        assert!(visited.check_and_mark(42));

        // Different index should return false
        assert!(!visited.check_and_mark(43));
    }

    #[test]
    fn test_clear() {
        let mut visited = VisitedSet::new(100);

        visited.mark(10);
        visited.mark(20);
        visited.mark(30);

        assert!(visited.is_visited(10));
        assert!(visited.is_visited(20));
        assert!(visited.is_visited(30));

        visited.clear();

        assert!(!visited.is_visited(10));
        assert!(!visited.is_visited(20));
        assert!(!visited.is_visited(30));
    }

    #[test]
    fn test_boundary_indices() {
        let mut visited = VisitedSet::new(256);

        // Test byte boundaries
        for i in [0, 7, 8, 15, 16, 255] {
            assert!(!visited.check_and_mark(i));
            assert!(visited.is_visited(i));
        }
    }

    #[test]
    fn test_clear_handles_epoch_overflow() {
        let mut visited = VisitedSet::new(16);
        visited.mark(7);
        visited.epoch = u16::MAX;
        visited.epochs[3] = u16::MAX;

        visited.clear();

        assert_eq!(visited.epoch, 1);
        assert!(visited.epochs.iter().all(|&epoch| epoch == 0));
        assert!(!visited.is_visited(3));
        assert!(!visited.is_visited(7));
    }

    #[test]
    fn test_large_set() {
        let mut visited = VisitedSet::new(1_000_000);

        // Mark every 1000th element
        for i in (0..1_000_000).step_by(1000) {
            visited.mark(i as i32);
        }

        // Verify
        for i in 0..1_000_000 {
            let expected = i % 1000 == 0;
            assert_eq!(visited.is_visited(i as i32), expected);
        }
    }
}
