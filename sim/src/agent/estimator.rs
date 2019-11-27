use rand::{Rng};
use super::MentalState;
use crate::{World, Action, Observation, WorldEntity, Position, StorageSlice};
use crate::entity::{Entity, Storage, Source};
use crate::entity_type::EntityType;


pub trait MentalStateRep: std::fmt::Display + Sized {
    fn sample<R: Rng + ?Sized>(& self, scale: f32, rng: & mut R) -> MentalState;
    fn update_seen<'a>(& 'a mut self, action: Option<Action>, others: impl Source<'a, MentalState>, observation: impl Observation);
    fn update_unseen<'a>(& 'a mut self, others: impl Source<'a, MentalState>, observation: impl Observation);
    fn into_ms(&self) -> MentalState {
        let mut rng : rand_xorshift::XorShiftRng = rand::SeedableRng::seed_from_u64(0);
        self.sample(0.0, &mut rng)
    }
    fn default(we: &WorldEntity) -> Self;
    fn from_aggregate<B>(we: &WorldEntity, iter: impl Iterator<Item=B>) -> Self where B: std::borrow::Borrow<Self>;
    fn get_type(&self) -> EntityType;
}


pub trait Estimator {
    fn invoke(&self, we: WorldEntity) -> Option<MentalState>;
    fn invoke_sampled<R: rand::Rng + ?Sized>(& self, we: WorldEntity,rng: & mut R) -> Option<MentalState>;
    fn learn(& mut self, action: Option<Action>, other: WorldEntity, other_pos: Position, world: & World);
}
#[derive(Clone, Debug)]
pub struct LearningEstimator<E : MentalStateRep> {
    pub agents: Vec<(WorldEntity, u32)>,
    pub estimators: Storage<E>,
}

impl<E: MentalStateRep> LearningEstimator<E> {
    fn get_estimate_or_init(& mut self, entity: &WorldEntity, observation: impl Observation) -> &mut E {

        self.estimators.get_or_insert_with(entity, ||
            MentalStateRep::default(entity)
            // MentalStateRep::from_aggregate(entity, self.estimators.into_iter().filter(|e| e.get_type() == entity.e_type()) )
        )
    }
    pub fn new(agents: Vec<(WorldEntity, u32)>) -> Self {
        Self {
            agents,
            estimators: Storage::new(),
        }
    }
}

impl<E: MentalStateRep + 'static> Estimator for LearningEstimator<E> {
    fn invoke(&self, entity: WorldEntity) -> Option<MentalState> {
        self.estimators.get(entity).map(MentalStateRep::into_ms)
    }

    fn invoke_sampled<R: rand::Rng + ?Sized>(&self, entity: WorldEntity, rng: &mut R) -> Option<MentalState> {
        self.estimators.get(entity).map(|e| e.sample(1.0, rng))
    }
    fn learn(& mut self, action: Option<Action>, other: WorldEntity, other_pos: Position, world: & World) {
        for (agent, sight) in self.agents.clone() {
            if let (Some(ms), Some(own_pos)) =
            (self.estimators.get_mut(agent), world.positions.get(agent)) {
                let observation = world.observe_in_radius(&agent, sight); // &(*world);//
                let dist = own_pos.distance(&other_pos);
                if dist <= sight {
                    self.get_estimate_or_init(&other, observation.borrow());
                }
                if let Some((es, sc)) = self.estimators.split_out_mut(other) {
                    if dist <= sight {
                        es.update_seen(action, &sc, observation);
                    }
                    else {
                        es.update_unseen(&sc, observation);
                    }
                }
            }
        }
    }
}

impl<'a, T: MentalStateRep + Sized + 'static> Estimator for StorageSlice<'a, T> {
    fn invoke(&self, entity: WorldEntity) -> Option<MentalState> {
        self.get(entity).map(MentalStateRep::into_ms)
    }

    fn invoke_sampled<R: rand::Rng + ?Sized>(& self, we: WorldEntity,rng: & mut R) -> Option<MentalState> {
        self.get(we).map(|e: &T| e.sample(1.0, rng))
    }
    fn learn(& mut self, action: Option<Action>, other: WorldEntity, other_pos: Position, world: & World) {
        unimplemented!()
    }
}