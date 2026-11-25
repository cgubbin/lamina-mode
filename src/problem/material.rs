//! Stack regions and the constitutive entries each polarisation sees.
//!
//! A region is isotropic, or carries ε and μ diagonal in the stack frame
//! (principal axes on x, y, z). TE (Ey, Hx, Hz) sees μ_xx, μ_zz and ε_yy; TM
//! (Hy, Ex, Ez) sees ε_xx, ε_zz and μ_yy. Regions with off-diagonal entries
//! are refused.
//!
//! Constant exteriors are construction evidence: erased handles and sampled
//! zero derivatives do not prove constancy, so spectral constructors take a
//! [`ConstantExterior`].
use super::C;
use lamina_core::{
    AnalyticalMaterialHandle, AnalyticalMaterialRegion, AnalyticalTensorMaterialHandle,
    AnalyticalTensorStack, ComplexPlane, ComplexScalar, ConstitutiveClass, ConstitutiveError,
    ConstitutiveRates, ConstitutiveTensor, DerivativeOrder, DifferentiableMaterial,
    DifferentiableMeromorphicMaterial, DifferentiableMeromorphicTensorMaterial,
    DifferentiableTensorMaterial, EvaluateDifferentiableMeromorphicMaterial,
    EvaluateMeromorphicMaterial, FiniteLayerIndex, Material, MeromorphicMaterial,
    MeromorphicTensorMaterial, ModeJet1, Polarisation, Rotation, Sampled, Scalar, SeedJet,
    StackRegion, TensorConstitutiveLift, TensorMaterial,
};
use ndarray::{Ix0, arr0};

/// A stack region: an isotropic material, or a tensor material oriented in
/// the stack frame.
pub type Region = AnalyticalMaterialRegion<C>;
/// A stack of [`Region`]s; scalar stacks convert with `From`.
pub type ModeStack = AnalyticalTensorStack<C>;

/// Stack-frame ε or μ of an isotropic or diagonal region.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Principal<T> {
    Isotropic(T),
    /// The diagonal `(xx, yy, zz)`.
    Diagonal([T; 3]),
}

impl<T: Copy> Principal<T> {
    /// The diagonal `(xx, yy, zz)`.
    pub fn entries(&self) -> [T; 3] {
        match *self {
            Self::Isotropic(value) => [value; 3],
            Self::Diagonal(entries) => entries,
        }
    }

    pub fn map<U>(self, f: impl Fn(T) -> U) -> Principal<U> {
        match self {
            Self::Isotropic(value) => Principal::Isotropic(f(value)),
            Self::Diagonal(entries) => Principal::Diagonal(entries.map(f)),
        }
    }

    /// `Σ_i w_ii v_i`, formed as `w (v_x + v_y + v_z)` for an isotropic `w`.
    pub fn weigh<V>(&self, values: [V; 3]) -> V
    where
        V: Copy + std::ops::Add<Output = V> + std::ops::Mul<T, Output = V>,
    {
        match *self {
            Self::Isotropic(w) => (values[0] + values[1] + values[2]) * w,
            Self::Diagonal([x, y, z]) => values[0] * x + values[1] * y + values[2] * z,
        }
    }
}

/// `[ε, μ]` of a region in the stack frame.
pub(crate) type Tensors<T = C> = [Principal<T>; 2];

/// A region's stack-frame `[ε', μ']` along a direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Rates(pub Tensors);

impl Rates {
    pub const ZERO: Self = Self([Principal::Isotropic(C::ZERO); 2]);
}

impl ConstitutiveRates for Rates {
    fn stack_frame(&self) -> [ConstitutiveTensor<C>; 2] {
        self.0.map(|tensor| match tensor {
            Principal::Isotropic(value) => ConstitutiveTensor::Isotropic(value),
            Principal::Diagonal(entries) => ConstitutiveTensor::Diagonal(entries),
        })
    }
}

/// The entries of ε and μ one polarisation sees. Its primary field ψ (Ey for
/// TE, Z₀Hy for TM) obeys `(1/t) ψ″ + (v k0² − β²/n) ψ = 0`, so
///
/// ```text
/// κ² = t (v k0² − β²/n) = s (χ k0² − β²),   χ = v n,   s = t/n
/// ```
///
/// with branch points at `β² = χ k0²`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Channel {
    /// `t`: μ_xx (TE) or ε_xx (TM), the weight of the normal derivative.
    pub transverse: C,
    /// `n`: μ_zz (TE) or ε_zz (TM).
    pub normal: C,
    /// `v`: ε_yy (TE) or μ_yy (TM).
    pub invariant: C,
}

impl Channel {
    pub fn of(tensors: &Tensors, polarisation: Polarisation) -> Self {
        let [epsilon, mu] = tensors.map(|t| t.entries());
        let (weights, invariant) = match polarisation {
            Polarisation::TransverseElectric => (mu, epsilon[1]),
            Polarisation::TransverseMagnetic => (epsilon, mu[1]),
        };
        Self {
            transverse: weights[0],
            normal: weights[2],
            invariant,
        }
    }

    /// `χ = (line/k0)²`.
    pub fn chi(&self) -> C {
        self.invariant * self.normal
    }

    /// `s = t/n`, absent when `t = n`.
    pub fn scale(&self) -> Option<C> {
        (self.transverse != self.normal).then(|| self.transverse / self.normal)
    }

    /// `κ² = s (χ k0² − β²)`.
    pub fn kappa_squared(&self, k0: C, beta: C) -> C {
        let q = self.chi() * k0 * k0 - beta * beta;
        self.scale().map_or(q, |s| s * q)
    }

    /// The rate of `κ²` along a direction in which the entries change at
    /// `rate`, `k0` at `k0_rate` and `β` at `beta_rate`.
    pub fn kappa_squared_rate(&self, rate: &Self, k0: C, k0_rate: f64, beta: C, beta_rate: C) -> C {
        let chi = self.chi();
        let chi_rate = rate.invariant * self.normal + self.invariant * rate.normal;
        let q_rate = chi_rate * k0 * k0 + 2.0 * chi * k0 * k0_rate - 2.0 * beta * beta_rate;
        if self.transverse == self.normal && rate.transverse == rate.normal {
            return q_rate;
        }
        let s = self.transverse / self.normal;
        let s_rate = (rate.transverse - s * rate.normal) / self.normal;
        s * q_rate + s_rate * (chi * k0 * k0 - beta * beta)
    }

    /// The entries seen: `t`, `n`, `v`.
    pub fn entries(&self) -> [C; 3] {
        [self.transverse, self.normal, self.invariant]
    }

    /// `(ε, μ)` with product `χ`: `(ε_yy, μ_zz)` for TE, `(ε_zz, μ_yy)` for
    /// TM. Without a [`scale`](Self::scale) they are the scalars of an
    /// isotropic medium presenting the same channel.
    pub fn factors(&self, polarisation: Polarisation) -> (C, C) {
        match polarisation {
            Polarisation::TransverseElectric => (self.invariant, self.normal),
            Polarisation::TransverseMagnetic => (self.normal, self.invariant),
        }
    }
}

/// The region of index `index` (left exterior 0, layers, then the right
/// exterior) of a stack with `layers` finite layers.
pub(crate) fn region_id(index: usize, layers: usize) -> StackRegion {
    match index {
        0 => StackRegion::LeftExterior,
        i if i > layers => StackRegion::RightExterior,
        i => StackRegion::Layer(FiniteLayerIndex::new(i - 1)),
    }
}

/// Each region of `stack` with its identifier: left exterior, layers, right
/// exterior.
pub(crate) fn regions(stack: &ModeStack) -> impl Iterator<Item = (StackRegion, &Region)> {
    let layers = stack.len();
    std::iter::once(stack.left_exterior())
        .chain(stack.iter().map(|layer| layer.material()))
        .chain(std::iter::once(stack.right_exterior()))
        .enumerate()
        .map(move |(index, region)| (region_id(index, layers), region))
}

/// Refuse a region whose ε or μ may have off-diagonal stack-frame entries.
pub(crate) fn check_regions(stack: &ModeStack) -> Result<(), ConstitutiveError> {
    match regions(stack).find(|(_, region)| region.class() == ConstitutiveClass::Full) {
        Some((region, _)) => Err(ConstitutiveError::Unsupported {
            region,
            class: ConstitutiveClass::Full,
        }),
        None => Ok(()),
    }
}

type J = ModeJet1<C, Ix0>;

/// `[ε, μ]` and `[∂ε/∂k0, ∂μ/∂k0]` of a region at complex `k0`. An
/// isotropic region evaluates its scalar material directly.
pub(crate) fn tensors_and_rates(
    region: StackRegion,
    material: &Region,
    k0: C,
) -> Result<[Tensors; 2], ConstitutiveError> {
    if let Some(scalar) = material.scalar() {
        let (at, first) = (Scalar(k0), DerivativeOrder::First);
        return Ok([
            [
                Principal::Isotropic(scalar.evaluate_relative_permittivity_complex(at)),
                Principal::Isotropic(scalar.evaluate_relative_permeability_complex(at)),
            ],
            [
                Principal::Isotropic(
                    scalar.evaluate_relative_permittivity_complex_derivative(at, first),
                ),
                Principal::Isotropic(
                    scalar.evaluate_relative_permeability_complex_derivative(at, first),
                ),
            ],
        ]);
    }
    let k0 = J::variable(arr0(k0), 0).map_err(|_| ConstitutiveError::NonFinite { region })?;
    let jets = <J as TensorConstitutiveLift<ComplexPlane, Region>>::constitutive_tensors(
        region, material, &k0,
    )?;
    let mut out = [[Principal::Isotropic(C::ZERO); 2]; 2];
    for (index, tensor) in jets.into_iter().enumerate() {
        let part = |jet: &J, order: usize| {
            if order == 0 {
                jet.value()[()]
            } else {
                jet.first()[()]
            }
        };
        for (order, slot) in out.iter_mut().enumerate() {
            slot[index] = match &tensor {
                ConstitutiveTensor::Isotropic(jet) => Principal::Isotropic(part(jet, order)),
                ConstitutiveTensor::Diagonal(entries) => {
                    Principal::Diagonal(entries.each_ref().map(|jet| part(jet, order)))
                }
                ConstitutiveTensor::Full(_) => {
                    return Err(ConstitutiveError::Unsupported {
                        region,
                        class: ConstitutiveClass::Full,
                    });
                }
            };
        }
    }
    Ok(out)
}

/// A 3×3 stack-frame tensor, row-major.
pub(crate) type Tensor = [[C; 3]; 3];

/// All nine stack-frame entries of `[ε, μ]` and of `[∂ε/∂k0, ∂μ/∂k0]` of a
/// region of any class at complex `k0`.
pub(crate) fn full_tensors(
    region: StackRegion,
    material: &Region,
    k0: C,
) -> Result<[[Tensor; 2]; 2], ConstitutiveError> {
    let diagonal = |entries: [C; 3]| {
        std::array::from_fn(|row| {
            std::array::from_fn(|column| if row == column { entries[row] } else { C::ZERO })
        })
    };
    if material.class() != ConstitutiveClass::Full {
        return tensors_and_rates(region, material, k0)
            .map(|parts| parts.map(|pair| pair.map(|t| diagonal(t.entries()))));
    }
    let k0 = J::variable(arr0(k0), 0).map_err(|_| ConstitutiveError::NonFinite { region })?;
    let jets = <J as TensorConstitutiveLift<ComplexPlane, Region>>::constitutive_tensors(
        region, material, &k0,
    )?;
    let mut out = [[[[C::ZERO; 3]; 3]; 2]; 2];
    for (index, tensor) in jets.iter().enumerate() {
        for (order, slot) in out.iter_mut().enumerate() {
            let part = |jet: &J| {
                if order == 0 {
                    jet.value()[()]
                } else {
                    jet.first()[()]
                }
            };
            slot[index] = match tensor {
                ConstitutiveTensor::Isotropic(jet) => diagonal([part(jet); 3]),
                ConstitutiveTensor::Diagonal(entries) => diagonal(entries.each_ref().map(part)),
                ConstitutiveTensor::Full(entries) => {
                    entries.each_ref().map(|r| r.each_ref().map(part))
                }
            };
        }
    }
    Ok(out)
}

/// `[ε, μ]` of a region at complex `k0`.
pub(crate) fn tensors(
    region: StackRegion,
    material: &Region,
    k0: C,
) -> Result<Tensors, ConstitutiveError> {
    if let Some(scalar) = material.scalar() {
        let at = Scalar(k0);
        return Ok([
            Principal::Isotropic(scalar.evaluate_relative_permittivity_complex(at)),
            Principal::Isotropic(scalar.evaluate_relative_permeability_complex(at)),
        ]);
    }
    tensors_and_rates(region, material, k0).map(|[values, _]| values)
}

/// The channel each exterior presents to `polarisation` at complex `k0`,
/// left then right.
pub(crate) fn exterior_channels(
    stack: &ModeStack,
    k0: C,
    polarisation: Polarisation,
) -> Result<[Channel; 2], ConstitutiveError> {
    let channel =
        |region, material| tensors(region, material, k0).map(|t| Channel::of(&t, polarisation));
    Ok([
        channel(StackRegion::LeftExterior, stack.left_exterior())?,
        channel(StackRegion::RightExterior, stack.right_exterior())?,
    ])
}

/// The scalar materials of the exteriors, left then right, refusing an
/// exterior that is not isotropic.
pub(crate) fn isotropic_exteriors(
    stack: &ModeStack,
) -> Result<[AnalyticalMaterialHandle<C>; 2], ConstitutiveError> {
    let scalar = |region, material: &Region| {
        material
            .scalar()
            .cloned()
            .ok_or(ConstitutiveError::Unsupported {
                region,
                class: material.class(),
            })
    };
    Ok([
        scalar(StackRegion::LeftExterior, stack.left_exterior())?,
        scalar(StackRegion::RightExterior, stack.right_exterior())?,
    ])
}

/// A constant exterior: isotropic, or diagonal in the stack frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstantExterior {
    epsilon: [C; 3],
    mu: [C; 3],
    diagonal: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
#[error("constant exterior factors must be finite and nonzero")]
pub struct InvalidConstantExterior;
impl ConstantExterior {
    /// An isotropic exterior.
    pub fn new(epsilon: C, mu: C) -> Result<Self, InvalidConstantExterior> {
        Self::validated([epsilon; 3], [mu; 3], false)
    }
    /// An exterior whose ε and μ are diagonal in the stack frame, with
    /// entries `(xx, yy, zz)`.
    pub fn diagonal(epsilon: [C; 3], mu: [C; 3]) -> Result<Self, InvalidConstantExterior> {
        Self::validated(epsilon, mu, true)
    }
    fn validated(
        epsilon: [C; 3],
        mu: [C; 3],
        diagonal: bool,
    ) -> Result<Self, InvalidConstantExterior> {
        if epsilon
            .iter()
            .chain(&mu)
            .any(|v| !v.re.is_finite() || !v.im.is_finite() || *v == C::new(0.0, 0.0))
        {
            return Err(InvalidConstantExterior);
        }
        Ok(Self {
            epsilon,
            mu,
            diagonal,
        })
    }
    /// The diagonal `(xx, yy, zz)` of ε.
    pub fn epsilon(&self) -> [C; 3] {
        self.epsilon
    }
    /// The diagonal `(xx, yy, zz)` of μ.
    pub fn mu(&self) -> [C; 3] {
        self.mu
    }
    /// The channel this exterior presents to `polarisation`.
    pub(crate) fn channel(&self, polarisation: Polarisation) -> Channel {
        let tensor = |entries: [C; 3]| {
            if self.diagonal {
                Principal::Diagonal(entries)
            } else {
                Principal::Isotropic(entries[0])
            }
        };
        Channel::of(&[tensor(self.epsilon), tensor(self.mu)], polarisation)
    }
    /// The exterior as a stack region.
    pub(crate) fn region(&self) -> Region {
        if self.diagonal {
            Region::oriented(
                AnalyticalTensorMaterialHandle::new(Uniform {
                    epsilon: self.epsilon,
                    mu: self.mu,
                }),
                Rotation::identity(),
            )
        } else {
            AnalyticalMaterialHandle::new(Uniform {
                epsilon: self.epsilon[0],
                mu: self.mu[0],
            })
            .into()
        }
    }
}

/// A material with constant ε and μ: scalars, or stack-frame diagonals.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Uniform<T> {
    epsilon: T,
    mu: T,
}
fn convert<T: ComplexScalar<RealField = f64>>(v: C) -> T {
    T::from_real(v.re) + T::i() * T::from_real(v.im)
}
fn diagonal<T: ComplexScalar<RealField = f64>>(entries: [C; 3]) -> [[T; 3]; 3] {
    let mut tensor = [[T::from_real(0.0); 3]; 3];
    for (axis, entry) in entries.into_iter().enumerate() {
        tensor[axis][axis] = convert(entry);
    }
    tensor
}
impl Material for Uniform<C> {
    type Real = f64;
    fn relative_permittivity<I, T>(&self, k: I) -> I::Mapped<T>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = f64>,
    {
        k.map(|_| convert(self.epsilon))
    }
    fn relative_permeability<I, T>(&self, k: I) -> I::Mapped<T>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = f64>,
    {
        k.map(|_| convert(self.mu))
    }
}
impl DifferentiableMaterial for Uniform<C> {
    fn relative_permittivity_derivative<I, T>(&self, k: I, _: DerivativeOrder) -> I::Mapped<T>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = f64>,
    {
        k.map(|_| T::from_real(0.0))
    }
    fn relative_permeability_derivative<I, T>(&self, k: I, _: DerivativeOrder) -> I::Mapped<T>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = f64>,
    {
        k.map(|_| T::from_real(0.0))
    }
}
impl MeromorphicMaterial for Uniform<C> {
    fn relative_permittivity_complex<I, T>(&self, k: I) -> I::Mapped<T>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = T>,
    {
        k.map(|_| convert(self.epsilon))
    }
    fn relative_permeability_complex<I, T>(&self, k: I) -> I::Mapped<T>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = T>,
    {
        k.map(|_| convert(self.mu))
    }
}
impl DifferentiableMeromorphicMaterial for Uniform<C> {
    fn relative_permittivity_complex_derivative<I, T>(
        &self,
        k: I,
        _: DerivativeOrder,
    ) -> I::Mapped<T>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = T>,
    {
        k.map(|_| T::from_real(0.0))
    }
    fn relative_permeability_complex_derivative<I, T>(
        &self,
        k: I,
        _: DerivativeOrder,
    ) -> I::Mapped<T>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = T>,
    {
        k.map(|_| T::from_real(0.0))
    }
}
impl TensorMaterial for Uniform<[C; 3]> {
    type Real = f64;
    const PRINCIPAL_AXES: bool = true;
    fn relative_permittivity_tensor<I, T>(&self, k: I) -> I::Mapped<[[T; 3]; 3]>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = f64>,
    {
        k.map(|_| diagonal(self.epsilon))
    }
    fn relative_permeability_tensor<I, T>(&self, k: I) -> I::Mapped<[[T; 3]; 3]>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = f64>,
    {
        k.map(|_| diagonal(self.mu))
    }
}
impl DifferentiableTensorMaterial for Uniform<[C; 3]> {
    fn relative_permittivity_tensor_derivative<I, T>(
        &self,
        k: I,
        _: DerivativeOrder,
    ) -> I::Mapped<[[T; 3]; 3]>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = f64>,
    {
        k.map(|_| [[T::from_real(0.0); 3]; 3])
    }
}
impl MeromorphicTensorMaterial for Uniform<[C; 3]> {
    fn relative_permittivity_tensor_complex<I, T>(&self, k: I) -> I::Mapped<[[T; 3]; 3]>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = T>,
    {
        k.map(|_| diagonal(self.epsilon))
    }
    fn relative_permeability_tensor_complex<I, T>(&self, k: I) -> I::Mapped<[[T; 3]; 3]>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = T>,
    {
        k.map(|_| diagonal(self.mu))
    }
}
impl DifferentiableMeromorphicTensorMaterial for Uniform<[C; 3]> {
    fn relative_permittivity_tensor_complex_derivative<I, T>(
        &self,
        k: I,
        _: DerivativeOrder,
    ) -> I::Mapped<[[T; 3]; 3]>
    where
        T: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = T>,
    {
        k.map(|_| [[T::from_real(0.0); 3]; 3])
    }
}
