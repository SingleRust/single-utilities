use nalgebra::{Dim, Dyn, U1};
use ndarray::ShapeBuilder;
use num_traits::float::FloatCore;
use num_traits::{Bounded, FromPrimitive, NumCast, One, ToPrimitive, Unsigned, Zero};
#[cfg(feature = "simd")]
use simba::{scalar::RealField, simd::SimdRealField};
use std::fmt::Debug;
use std::hash::Hash;
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
    Unsigned + Zero + One + Copy + Eq + Ord + PartialOrd + From<usize> + Into<usize> + Bounded + Hash
{
}

impl<
    I: Unsigned
        + Zero
        + One
        + Copy
        + Eq
        + Ord
        + PartialOrd
        + From<usize>
        + Into<usize>
        + Bounded
        + Hash,
> UIndex for I
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

#[cfg(feature = "convert")]
pub trait IntoNalgebra {
    type Out;

    fn into_nalgebra(self) -> Self::Out;
}

#[cfg(feature = "convert")]
pub trait IntoNdarray1 {
    type Out;

    fn into_ndarray1(self) -> Self::Out;
}

#[cfg(feature = "convert")]
pub trait IntoNdarray2 {
    type Out;

    fn into_ndarray2(self) -> Self::Out;
}

#[cfg(feature = "convert")]
impl<'a, N: Scalar, R: Dim, RStride: Dim, CStride: Dim> IntoNdarray1
    for nalgebra::Matrix<N, R, U1, nalgebra::ViewStorageMut<'a, N, R, U1, RStride, CStride>>
{
    type Out = ndarray::ArrayViewMut1<'a, N>;

    fn into_ndarray1(self) -> Self::Out {
        unsafe {
            ndarray::ArrayViewMut1::from_shape_ptr(
                (self.shape().0,).strides((self.strides().0,)),
                self.as_ptr() as *mut N,
            )
        }
    }
}

impl<'a, N: Scalar, R: Dim, C: Dim, RStride: Dim, CStride: Dim> IntoNdarray2
    for nalgebra::Matrix<N, R, C, nalgebra::ViewStorage<'a, N, R, C, RStride, CStride>>
{
    type Out = ndarray::ArrayView2<'a, N>;

    fn into_ndarray2(self) -> Self::Out {
        unsafe {
            ndarray::ArrayView2::from_shape_ptr(self.shape().strides(self.strides()), self.as_ptr())
        }
    }
}

#[cfg(feature = "convert")]
impl<N: Scalar> IntoNdarray2 for nalgebra::Matrix<N, Dyn, Dyn, nalgebra::VecStorage<N, Dyn, Dyn>>
where
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<Dyn, Dyn, Buffer<N> = nalgebra::VecStorage<N, Dyn, Dyn>>,
{
    type Out = ndarray::Array2<N>;

    fn into_ndarray2(self) -> Self::Out {
        ndarray::Array2::from_shape_vec(self.shape().strides(self.strides()), self.data.into())
            .unwrap()
    }
}

#[cfg(feature = "convert")]
impl<N: Scalar> IntoNdarray1 for nalgebra::DVector<N> {
    type Out = ndarray::Array1<N>;

    fn into_ndarray1(self) -> Self::Out {
        ndarray::Array1::from_shape_vec((self.shape().0,), self.data.into()).unwrap()
    }
}

#[cfg(feature = "convert")]
impl<T> IntoNalgebra for ndarray::Array1<T>
where
    T: nalgebra::Scalar,
{
    type Out = nalgebra::DVector<T>;
    fn into_nalgebra(self) -> Self::Out {
        let len = Dyn(self.len());
        // There is no method to give nalgebra the vector directly where it isn't allocated. If you call
        // from_vec_generic, it simply calls from_iterator_generic which uses Iterator::collect(). Due to this,
        // the simplest solution is to just pass an iterator over the values. If you come across this because you
        // have a performance issue, I would recommend creating the owned data using naglebra and borrowing it with
        // ndarray to perform operations on it instead of the other way around.
        Self::Out::from_iterator_generic(len, nalgebra::Const::<1>, self.iter().cloned())
    }
}

#[cfg(feature = "convert")]
impl<T> IntoNalgebra for ndarray::Array2<T>
where
    T: nalgebra::Scalar,
{
    type Out = nalgebra::DMatrix<T>;
    fn into_nalgebra(self) -> Self::Out {
        let nrows = Dyn(self.nrows());
        let ncols = Dyn(self.ncols());
        Self::Out::from_iterator_generic(nrows, ncols, self.t().iter().cloned())
    }
}
