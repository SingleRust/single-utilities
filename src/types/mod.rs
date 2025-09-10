use std::hash::Hash;

/// Represents the direction of operations in matrix or array computations.
/// 
/// This enum is used to specify whether operations should be performed
/// along rows or columns of a data structure.
pub enum Direction {
    /// Operations performed along columns (vertical direction)
    COLUMN,
    /// Operations performed along rows (horizontal direction)
    ROW,
}

impl Clone for Direction {
    fn clone(&self) -> Self {
        match self {
            Self::ROW => Self::ROW,
            Self::COLUMN => Self::COLUMN,
        }
    }
}

impl Direction {
    /// Checks if the direction is row-wise.
    /// 
    /// # Returns
    /// `true` if the direction is `ROW`, `false` if it's `COLUMN`
    pub fn is_row(&self) -> bool {
        match self {
            Self::ROW => true,
            Self::COLUMN => false,
        }
    }
}

/// A trait for types that can serve as batch identifiers.
/// 
/// This trait is used to identify and group data in batch processing operations.
/// Types implementing this trait must be cloneable, comparable for equality,
/// and hashable for efficient lookup operations.
pub trait BatchIdentifier: Clone + Eq + Hash {}

// Implement BatchIdentifier for common types
impl BatchIdentifier for String {}
impl BatchIdentifier for &str {}
impl BatchIdentifier for i32 {}
impl BatchIdentifier for u32 {}
impl BatchIdentifier for usize {}

/// Enumeration of distance metrics for mathematical computations.
/// 
/// This enum defines common distance metrics used in machine learning,
/// clustering, and similarity calculations. Each variant represents
/// a different approach to measuring the distance between points or vectors.
#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    /// Euclidean distance (L2 norm) - straight-line distance between points
    Euclidean,
    /// Manhattan distance (L1 norm) - sum of absolute differences along each dimension
    Manhattan,
    /// Cosine distance - measures the cosine of the angle between vectors
    Cosine,
}
