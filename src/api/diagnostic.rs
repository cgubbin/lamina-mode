use super::*;
use std::{error::Error, fmt, sync::Arc};
/// Stage that produced a diagnostic. Chart failures can coexist with successes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Stage {
    Construction,
    Domain,
    Discovery,
    Refinement,
    Evaluation,
    Interpretation,
}
/// Structured checked-optical category. The original error remains a source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpticalFailure {
    Input,
    Backend,
    Output,
}
/// Source-preserving public failure. `frequency_error` retains continuation
/// side/kind/work; `optical_failure` distinguishes checked input/backend/output.
#[derive(Clone, Debug)]
pub struct Diagnostic {
    stage: Stage,
    pub(super) foreign: bool,
    optical: Option<OpticalFailure>,
    source: Arc<dyn Error + Send + Sync + 'static>,
}
impl Diagnostic {
    pub fn is_foreign_association(&self) -> bool {
        self.foreign
    }
    pub fn stage(&self) -> Stage {
        self.stage
    }
    pub fn optical_failure(&self) -> Option<OpticalFailure> {
        self.optical
    }
    pub fn frequency_error(&self) -> Option<&FrequencyError> {
        self.find_source()
    }
    pub fn find_source<E: Error + 'static>(&self) -> Option<&E> {
        let mut e: &(dyn Error + 'static) = self.source.as_ref();
        loop {
            if let Some(v) = e.downcast_ref() {
                return Some(v);
            }
            e = e.source()?;
        }
    }
    pub(super) fn new<E: Error + Send + Sync + 'static>(stage: Stage, e: E) -> Self {
        let mut diagnostic = Self {
            stage,
            foreign: false,
            optical: None,
            source: Arc::new(e),
        };
        diagnostic.foreign = matches!(
            diagnostic.find_source::<PlanError>(),
            Some(PlanError::ForeignDomain | PlanError::ForeignBatch)
        ) || matches!(
            diagnostic.find_source::<RefinementInputError>(),
            Some(RefinementInputError::ForeignProblem | RefinementInputError::ForeignDomain)
        ) || matches!(
            diagnostic.find_source::<InterpretationError>(),
            Some(InterpretationError::ForeignProblem | InterpretationError::ForeignDomain)
        );
        diagnostic
    }
    pub(super) fn checked<E: Error + Send + Sync + 'static, BE: Error + 'static>(
        stage: Stage,
        e: E,
    ) -> Self {
        let mut d = Self::new(stage, e);
        d.optical = d
            .find_source::<lamina_core::DeterminantEvaluationError<BE>>()
            .map(|e| match e {
                lamina_core::DeterminantEvaluationError::NonFiniteInput => OpticalFailure::Input,
                lamina_core::DeterminantEvaluationError::Backend { .. } => OpticalFailure::Backend,
                lamina_core::DeterminantEvaluationError::NonFiniteOutput => OpticalFailure::Output,
            });
        d
    }
    pub(super) fn at_stage(&self, stage: Stage) -> Self {
        Self {
            stage,
            ..self.clone()
        }
    }
}
impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.stage, self.source)
    }
}
impl Error for Diagnostic {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(self.source.as_ref())
    }
}
#[derive(Clone, Copy, Debug, thiserror::Error)]
#[error("domain strategy does not match the physical problem")]
pub struct StrategyMismatch;
#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Clone, Debug, thiserror::Error)]
    #[error("backend fault")]
    struct Fault;
    #[test]
    fn checked_categories_and_original_sources_are_public() {
        use lamina_core::DeterminantEvaluationError as E;
        for (error, kind) in [
            (E::<Fault>::NonFiniteInput, OpticalFailure::Input),
            (E::Backend { source: Fault }, OpticalFailure::Backend),
            (E::NonFiniteOutput, OpticalFailure::Output),
        ] {
            let d = Diagnostic::checked::<_, Fault>(Stage::Evaluation, error);
            assert_eq!(d.optical_failure(), Some(kind));
            assert!(d.find_source::<E<Fault>>().is_some());
            if kind == OpticalFailure::Backend {
                assert!(d.find_source::<Fault>().is_some());
            }
        }
    }
}
