use num_traits::float::FloatCore;
use num_traits::{Bounded, FromPrimitive, NumCast, One, ToPrimitive, Unsigned, Zero};
#[cfg(feature = "simd")]
use simba::{scalar::RealField, simd::SimdRealField};
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, AddAssign, MulAssign, SubAssign};

/// A trait defining fundamental numeric operations and constraints.
/// 
/// This trait bundles common numeric operations required for mathematical computations,
/// including zero/one elements, numeric casting, assignment operations, ordering, 
/// bounds checking, and basic arithmetic. Types implementing this trait can be used
/// in generic numeric algorithms.
pub trait NumericOps:
    Zero
    + One
    + NumCast
    + Copy
    + AddAssign
    + MulAssign
    + SubAssign
    + PartialOrd
    + Bounded
    + Add<Output = Self>
    + Sum
    + Debug
    + Default
{
}
impl<
    T: Zero
        + One
        + NumCast
        + Copy
        + AddAssign
        + MulAssign
        + SubAssign
        + PartialOrd
        + Bounded
        + Add<Output = Self>
        + Sum
        + Debug
        + Default,
> NumericOps for T
{
}

/// A thread-safe version of `NumericOps`.
/// 
/// This trait extends `NumericOps` with `Send + Sync` bounds, making it suitable
/// for use in concurrent and parallel computations where data needs to be 
/// safely shared across threads.
pub trait NumericOpsTS: NumericOps + Send + Sync {}

impl<T: NumericOps + Send + Sync> NumericOpsTS for T {}

/// A trait for floating-point numeric operations.
/// 
/// Extends `NumericOps` with floating-point specific operations from the `num_traits`
/// crate, including floating-point arithmetic, primitive conversions, and core 
/// floating-point functionality. This trait is designed for types that represent
/// real numbers with decimal precision.
pub trait FloatOps:
    NumericOps + num_traits::Float + FromPrimitive + ToPrimitive + FloatCore
{
}

impl<T: NumericOps + num_traits::Float + FromPrimitive + ToPrimitive + FloatCore> FloatOps for T {}

/// A thread-safe version of `FloatOps`.
/// 
/// This trait extends `FloatOps` with `Send + Sync` bounds for safe use in
/// concurrent floating-point computations across multiple threads.
pub trait FloatOpsTS: FloatOps + Sync + Send {}

impl<T: FloatOps + Send + Sync> FloatOpsTS for T {}

#[cfg(feature = "simd")]
/// A SIMD-enabled floating-point operations trait.
/// 
/// Extends `FloatOpsTS` with SIMD (Single Instruction, Multiple Data) capabilities
/// from the `simba` crate. This enables vectorized floating-point operations for
/// improved performance in mathematical computations.
/// 
/// Only available when the "simd" feature is enabled.
pub trait FloatOpsTSSimba: FloatOpsTS + SimdRealField + RealField {}

#[cfg(feature = "simd")]
impl<T: FloatOpsTS + SimdRealField + RealField> FloatOpsTSSimba for T {}

/// A trait for numeric types suitable for normalization operations.
/// 
/// This trait combines floating-point arithmetic with assignment operations,
/// summation capabilities, and numeric casting. It's specifically designed
/// for types that need to participate in normalization algorithms where
/// values are scaled or adjusted relative to some total or maximum.
pub trait NumericNormalize:
    num_traits::Float + std::ops::AddAssign + std::iter::Sum + num_traits::NumCast
{
}

// Blanket implementation for any type that satisfies the bounds
impl<T> NumericNormalize for T where
    T: num_traits::Float + std::ops::AddAssign + std::iter::Sum + num_traits::NumCast
{
}

/// A trait for vectors that can be resized and filled with default values.
/// 
/// Provides functionality to clear a vector and resize it to a specific length,
/// filling all positions with the default value of the element type. This is
/// useful for initializing or resetting vectors to a known state.
pub trait ZeroVec {
    /// Clears the vector and resizes it to the specified length, filling with default values.
    /// 
    /// # Arguments
    /// * `len` - The desired length of the vector
    fn zero_len(&mut self, len: usize);
}

impl<T: Default + Clone> ZeroVec for Vec<T> {
    fn zero_len(&mut self, len: usize) {
        self.clear();
        self.reserve(len);
        self.extend(std::iter::repeat_n(T::default(), len));
    }
}

/// A trait for unsigned integer types suitable for indexing operations.
/// 
/// This trait combines unsigned integer properties with zero/one elements,
/// ordering capabilities, and conversion to/from `usize`. It's designed for
/// types that can be safely used as array or vector indices while providing
/// mathematical operations and bounds checking.
pub trait UIndex:
    Unsigned + Zero + One + Copy + Eq + Ord + PartialOrd + From<usize> + Into<usize> + Bounded
{
}

impl<I: Unsigned + Zero + One + Copy + Eq + Ord + PartialOrd + From<usize> + Into<usize> + Bounded>
    UIndex for I
{
}

/// A trait for scalar values used in mathematical computations.
/// 
/// This trait defines the minimal requirements for types that can be used as
/// scalar values in mathematical operations. It requires static lifetime,
/// cloning capability, equality comparison, and debug formatting. This is
/// typically used as a constraint for generic mathematical algorithms.
pub trait Scalar: 'static + Clone + PartialEq + Debug {}

impl<T: 'static + Clone + PartialEq + Debug> Scalar for T {}
