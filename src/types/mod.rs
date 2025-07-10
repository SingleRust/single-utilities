use std::hash::Hash;

pub enum Direction {
    COLUMN,
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
    pub fn is_row(&self) -> bool {
        match self {
            Self::ROW => true,
            Self::COLUMN => false,
        }
    }
}

pub trait BatchIdentifier: Clone + Eq + Hash {}

// Implement BatchIdentifier for common types
impl BatchIdentifier for String {}
impl BatchIdentifier for &str {}
impl BatchIdentifier for i32 {}
impl BatchIdentifier for u32 {}
impl BatchIdentifier for usize {}

#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    /// Euclidean distance
    Euclidean,
    /// Manhattan distance
    Manhattan,
    /// Cosine distance
    Cosine,
}
