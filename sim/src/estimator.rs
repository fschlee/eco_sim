use rand::{Rng};
use super::MentalState;
use super::{Action, Observation};
use crate::entity::Source;

pub trait Estimator: std::fmt::Display {
    fn sample<R: Rng + ?Sized>(& self, scale: f32, rng: & mut R) -> MentalState;
    fn update_seen<'a>(& 'a mut self, action: Option<Action>, others: impl Source<'a, MentalState>, observation: impl Observation);
    fn update_unseen<'a>(& 'a mut self, others: impl Source<'a, MentalState>, observation: impl Observation);
    fn into_ms(&self) -> MentalState {
        let mut rng : rand_xorshift::XorShiftRng = rand::SeedableRng::seed_from_u64(0);
        self.sample(0.0, &mut rng)
    }
}
