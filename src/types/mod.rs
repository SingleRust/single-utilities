use crate::utils::validate_net;
use std::collections::HashMap;
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

pub struct PathwayNetwork {
    names: Vec<String>,  // name of pathways
    starts: Vec<usize>,  // start of pathway
    offsets: Vec<usize>, // length of pathway
    cnct: Vec<usize>,    // gene index of pathway
    weights: Vec<f32>,   // weight of each gene in the pathway
}

impl PathwayNetwork {
    pub fn new(
        names: Vec<String>,
        starts: Vec<usize>,
        offsets: Vec<usize>,
        cnct: Vec<usize>,
        weights: Vec<f32>,
    ) -> Self {
        Self {
            names,
            starts,
            offsets,
            cnct,
            weights,
        }
    }

    pub fn new_wo_weights(
        names: Vec<String>,
        starts: Vec<usize>,
        offsets: Vec<usize>,
        cnct: Vec<usize>,
    ) -> Self {
        let weights = vec![1f32; cnct.len()];
        Self {
            names,
            starts,
            offsets,
            cnct,
            weights,
        }
    }

    pub fn new_from_vec(
        sources: Vec<String>,
        targets: Vec<String>,
        weights: Option<Vec<f32>>,
        features: Vec<String>,
        tmin: u32,
    ) -> Self {
        let res = validate_net(sources, targets, weights, false).unwrap();
        let tmin = tmin as usize;
        let filtered: HashMap<String, Vec<(String, f32)>> = res
            .into_iter()
            .filter_map(|(k, v)| if v.len() >= tmin { Some((k, v)) } else { None })
            .collect();

        let name_to_id: HashMap<String, usize> = features
            .iter()
            .enumerate()
            .map(|(idx, name)| (name.clone(), idx))
            .collect();

        let total_lengths = filtered.values().fold(0usize, |v, a| v + a.len());
        let num_pathways = filtered.len();

        let mut names: Vec<String> = Vec::with_capacity(num_pathways);
        let mut starts: Vec<usize> = Vec::with_capacity(num_pathways);
        let mut offsets: Vec<usize> = Vec::with_capacity(num_pathways);
        let mut cnct: Vec<usize> = Vec::with_capacity(total_lengths);
        let mut weights_vec: Vec<f32> = Vec::with_capacity(total_lengths);

        let mut i = 0usize;

        for (k, v) in filtered.into_iter() {
            let len = v.len();

            for (g_name, g_weight) in v {
                let g_idx = name_to_id.get(&g_name).unwrap();
                cnct.push(*g_idx);
                weights_vec.push(g_weight);
            }

            names.push(k);
            starts.push(i);
            offsets.push(len);
            i += len;
        }

        Self {
            names,
            starts,
            offsets,
            cnct,
            weights: weights_vec,
        }
    }

    pub fn get_pathway_name(&self, idx: usize) -> &str {
        self.names[idx].as_str()
    }

    pub fn get_pathway_features(&self, idx: usize) -> &[usize] {
        let srt = self.starts[idx];
        let off = srt + self.offsets[idx];
        &self.cnct[srt..off]
    }

    pub fn get_pathway_features_and_weights(&self, idx: usize) -> (&[usize], &[f32]) {
        let srt = self.starts[idx];
        let off = srt + self.offsets[idx];
        (&self.cnct[srt..off], &self.weights[srt..off])
    }

    pub fn get_num_pathways(&self) -> usize {
        self.names.len()
    }
}
