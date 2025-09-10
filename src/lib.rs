//! # Single Utilities
//!
//! A comprehensive utilities library designed for the SingleRust ecosystem and beyond.
//! This crate provides fundamental building blocks for mathematical computations,
//! data processing, and algorithmic operations commonly needed in scientific computing,
//! machine learning, and data analysis applications.
//!
//! ## Overview
//!
//! Single Utilities offers a collection of traits and types that abstract common
//! patterns in numerical computing while maintaining flexibility and performance.
//! While primarily developed for the SingleRust ecosystem, the library is designed
//! with modularity and interoperability in mind, making it suitable for use with
//! other Rust libraries and frameworks.
//!
//! ## Features
//!
//! ### Traits Module
//! - **Numeric Operations**: Comprehensive traits for basic and advanced numeric operations
//! - **Thread Safety**: Thread-safe variants for concurrent and parallel computations
//! - **SIMD Support**: Optional SIMD-accelerated operations when the "simd" feature is enabled
//! - **Type Constraints**: Flexible trait bounds for generic mathematical algorithms
//!
//! ### Types Module
//! - **Direction Handling**: Utilities for row/column-oriented operations
//! - **Distance Metrics**: Common distance functions for similarity calculations
//! - **Batch Processing**: Identifiers and utilities for batch operations
//!
//! ## Usage
//!
//! ```rust
//! use single_utilities::traits::NumericOps;
//! use single_utilities::types::{Direction, DistanceMetric};
//!
//! // Use traits to constrain generic functions
//! fn process_data<T: NumericOps>(data: &[T]) -> T {
//!     // Your numeric processing logic here
//!     T::zero()
//! }
//!
//! // Use direction for matrix operations
//! let direction = Direction::ROW;
//! if direction.is_row() {
//!     // Process row-wise
//! }
//! ```
//!
//! ## Feature Flags
//!
//! - `simd`: Enables SIMD-accelerated operations using the `simba` crate
//!
//! ## Compatibility
//!
//! This library is designed to work seamlessly with:
//! - The broader SingleRust ecosystem
//! - Standard numeric libraries like `num-traits`
//! - SIMD libraries like `simba` (when feature is enabled)
//! - Custom mathematical and scientific computing libraries

pub mod traits;

pub mod types;
