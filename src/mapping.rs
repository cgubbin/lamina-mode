pub trait ConformalMapping {
    type Complex;

    fn apply(&self, value: Self::Complex) -> Self::Complex;
    fn first_derivative(&self, value: Self::Complex) -> Self::Complex;
    fn second_derivative(&self, value: Self::Complex) -> Self::Complex;
}

pub trait InverseConformalMapping: ConformalMapping {
    fn inverse(&self, value: Self::Complex) -> Self::Complex;
    fn inverse_first_derivative(&self, value: Self::Complex) -> Self::Complex;
    fn inverse_second_derivative(&self, value: Self::Complex) -> Self::Complex;
}
