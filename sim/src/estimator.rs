use rand::{Rng};
use super::MentalState;
use super::{Action, Observation};

pub trait Estimator: std::fmt::Display {
    fn sample<R: Rng + ?Sized>(& self, scale: f32, rng: & mut R) -> MentalState;
    fn update_seen(& mut self, action: Option<Action>, observation: impl Observation);
    fn update_unseen(& mut self, observation: impl Observation);
}