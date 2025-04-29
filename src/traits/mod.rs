use std::fmt::Debug;
use std::iter::Sum;
use num_traits::{Bounded, FromPrimitive, NumCast, One, ToPrimitive, Zero};
use std::ops::{Add, AddAssign, MulAssign, SubAssign};
use num_traits::float::FloatCore;

pub trait NumericOps:
    Zero + One + NumCast + Copy + AddAssign + MulAssign + SubAssign + PartialOrd + Bounded + Add<Output = Self> + Sum + Debug + Default
{
}
impl<
    T: Zero + One + NumCast + Copy + AddAssign + MulAssign + SubAssign + PartialOrd + Bounded + Add<Output = Self> + Sum + Debug + Default,
> NumericOps for T
{
}

pub trait NumericOpsTS: NumericOps + Send + Sync {}

impl<T: NumericOps + Send + Sync> NumericOpsTS for T {}

pub trait FloatOps: NumericOps + num_traits::Float + FromPrimitive + ToPrimitive + FloatCore {}

impl<T: NumericOps + num_traits::Float + FromPrimitive + ToPrimitive + FloatCore> FloatOps for T {}

pub trait FloatOpsTS: FloatOps + Sync + Send {}

impl<T: FloatOps + Send + Sync> FloatOpsTS for T {}

// Define a type alias for our numeric constraints
pub trait NumericNormalize:
    num_traits::Float + std::ops::AddAssign + std::iter::Sum + num_traits::NumCast
{
}

// Blanket implementation for any type that satisfies the bounds
impl<T> NumericNormalize for T where
    T: num_traits::Float + std::ops::AddAssign + std::iter::Sum + num_traits::NumCast
{
}

pub trait ZeroVec {
    fn zero_len(&mut self, len: usize);
}

impl<T: Default + Clone> ZeroVec for Vec<T> {
    fn zero_len(&mut self, len: usize) {
        self.clear();
        self.reserve(len);
        self.extend(std::iter::repeat_n(T::default(), len));
    }
}
