use num_traits::{Bounded, NumCast, One, Zero};
use std::ops::Add;

pub trait NumericOps:
    Zero + One + NumCast + Copy + std::ops::AddAssign + PartialOrd + Bounded + Add<Output = Self>
{
}
impl<
    T: Zero + One + NumCast + Copy + std::ops::AddAssign + PartialOrd + Bounded + Add<Output = Self>,
> NumericOps for T
{
}

pub trait FloatOps: NumericOps + num_traits::Float {}
impl<T: NumericOps + num_traits::Float> FloatOps for T {}

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
