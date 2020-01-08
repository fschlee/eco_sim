use super::estimate::PointEstimateRep;
use super::MentalState;
use crate::entity::{Entity, Source, Storage, WorldEntity};
use crate::entity_type::EntityType;
use crate::position::Coord;
use crate::{Action, Cell, Event, Observation, Occupancy, Position, StorageSlice, World};

use rand::Rng;
use rayon::prelude::*;
use std::collections::{hash_map::RandomState, HashMap};
use crate::agent::estimate::ParticleFilterRep;

pub trait MentalStateRep: std::fmt::Display + Sized {
    fn sample<R: Rng + ?Sized>(&self, scale: f32, rng: &mut R) -> MentalState;
    fn update_seen<'a>(
        &'a mut self,
        action: Action,
        others: &impl Estimator,
        observation: &impl Observation,
    );
    fn update_on_events<'a>(
        &'a mut self,
        events: impl IntoIterator<Item = &'a Event> + Copy,
        world_model: Option<&'a World<Occupancy>>,
    );
    fn update_unseen<'a>(&'a mut self, others: &impl Estimator, observation: &impl Observation);
    fn into_ms(&self) -> MentalState {
        let mut rng: rand_xorshift::XorShiftRng = rand::SeedableRng::seed_from_u64(0);
        self.sample(0.0, &mut rng)
    }
    fn default(we: &WorldEntity) -> Self;
    fn from_aggregate<B>(we: &WorldEntity, iter: impl Iterator<Item = B>) -> Self
    where
        B: std::borrow::Borrow<Self>;
    fn get_type(&self) -> EntityType;
}

pub trait Estimator {
    fn invoke(&self, we: WorldEntity) -> Option<MentalState>;
    fn update_on_events<'a>(
        &'a mut self,
        events: impl IntoIterator<Item = &'a Event> + Copy,
        world_model: Option<&'a World<Occupancy>>,
    );
    type Rep: MentalStateRep;
    fn invoke_sampled<'a, R: rand::Rng + Sized>(
        &'a self,
        we: WorldEntity,
        rng: &'a mut R,
        n: usize,
    ) -> InvokeIter<'a, Self::Rep, R>;
    fn learn<'a, C: Cell>(
        &mut self,
        action: Action,
        other: WorldEntity,
        other_pos: Position,
        world: &World<C>,
        world_models: &'a impl Source<'a, &'a World<Occupancy>>,
    );
}

#[derive(Clone, Debug)]
pub struct LearningEstimator<E: MentalStateRep> {
    pub agents: Vec<(WorldEntity, Coord)>,
    pub estimators: Storage<E>,
}

pub enum InvokeIter<'a, E: MentalStateRep, R: Rng> {
    Rep {
        count: usize,
        rep: &'a E,
        rng: &'a mut R,
    },
    Empty,
}
impl<'a, E: MentalStateRep, R: Rng> Iterator for InvokeIter<'a, E, R> {
    type Item = MentalState;
    fn next(&mut self) -> Option<Self::Item> {
        let ms = match self {
            Self::Empty => None,
            Self::Rep {
                ref mut count,
                rep,
                rng,
            } => {
                if *count == 0 {
                    None
                } else {
                    *count -= 1;
                    Some(rep.sample(1.0, rng))
                }
            }
        };
        if ms.is_none() {
            *self = Self::Empty;
        }
        ms
    }
}

impl<E: MentalStateRep + 'static> LearningEstimator<E> {
    fn assure_init(&mut self, entity: &WorldEntity, observation: &impl Observation) {
        if self.estimators.get(entity).is_some() {
            return;
        }
        {
            let rep = MentalStateRep::from_aggregate(
                entity,
                self.estimators
                    .into_iter()
                    .filter(|e| e.get_type() == entity.e_type()),
            );
            self.estimators.insert(entity, rep);
        }
    }
    pub fn new(agents: Vec<(WorldEntity, Coord)>) -> Self {
        Self {
            agents,
            estimators: Storage::new(),
        }
    }
    pub fn replace(&mut self, old: WorldEntity, new: WorldEntity) {
        if let Some(tpl) = self.agents.iter_mut().find(|(id, _)| id.id() == old.id()) {
            tpl.0 = new;
        }
    }
    fn learn_helper(
        &mut self,
        _agent: WorldEntity,
        other: WorldEntity,
        action: Action,
        sight: Coord,
        own_pos: &Position,
        other_pos: Position,
        observation: impl Observation,
    ) {
        let dist = own_pos.distance(&other_pos);
        if dist <= sight {
            self.assure_init(&other, &observation);
        }
        if let Some((es, sc)) = self.estimators.split_out_mut(other) {
            if dist <= sight {
                es.update_seen(action, &sc, &observation);
            } else {
                es.update_unseen(&sc, &observation);
            }
        }
    }
}

impl<E: MentalStateRep + 'static> Estimator for LearningEstimator<E> {
    fn invoke(&self, entity: WorldEntity) -> Option<MentalState> {
        self.estimators.get(entity).map(MentalStateRep::into_ms)
    }
    type Rep = E;
    fn invoke_sampled<'a, R: rand::Rng + Sized>(
        &'a self,
        entity: WorldEntity,
        rng: &'a mut R,
        n: usize,
    ) -> InvokeIter<'a, E, R> {
        if let Some(rep) = self.estimators.get(entity) {
            InvokeIter::Rep { rng, count: n, rep }
        } else {
            InvokeIter::Empty
        }

        // self.estimators.get(entity).iter().flat_map(|e| e.sample(1.0, rng)).fuse()
    }
    fn learn<'a, C: Cell>(
        &mut self,
        action: Action,
        other: WorldEntity,
        other_pos: Position,
        world: &World<C>,
        world_models: &'a impl Source<'a, &'a World<Occupancy>>,
    ) {
        for (agent, sight) in self.agents.clone() {
            if let Some(own_pos) = world.positions.get(agent) {
                if let Some(wm) = world_models.get(agent.into()) {
                    self.learn_helper(agent, other, action, sight, own_pos, other_pos, wm);
                } else {
                    self.learn_helper(
                        agent,
                        other,
                        action,
                        sight,
                        own_pos,
                        other_pos,
                        world.observe_in_radius(&agent, sight),
                    );
                }
            }
        }
    }
    fn update_on_events<'a>(
        &'a mut self,
        events: impl IntoIterator<Item = &'a Event> + Copy,
        world_model: Option<&'a World<Occupancy>>,
    ) {
        for est in self.estimators.iter_mut() {
            est.update_on_events(events, world_model)
        }
    }
}

impl<'c, T: MentalStateRep + Sized + 'static> Estimator for StorageSlice<'c, T> {
    type Rep = T;
    fn invoke(&self, entity: WorldEntity) -> Option<MentalState> {
        self.get(entity.into()).map(MentalStateRep::into_ms)
    }

    fn invoke_sampled<'b, R: rand::Rng + Sized>(
        &'b self,
        we: WorldEntity,
        rng: &'b mut R,
        n: usize,
    ) -> InvokeIter<'b, Self::Rep, R> {
        InvokeIter::Empty
        // self.get(we).iter().flat_map(|e: &T| e.sample(1.0, rng)).fuse()
    }
    fn learn<'a, C: Cell>(
        &mut self,
        action: Action,
        other: WorldEntity,
        other_pos: Position,
        world: &World<C>,
        world_models: &impl Source<'a, &'a World<Occupancy>>,
    ) {
        unimplemented!()
    }
    fn update_on_events<'a>(
        &'a mut self,
        events: impl IntoIterator<Item = &'a Event> + Copy,
        world_model: Option<&'a World<Occupancy>>,
    ) {
        unimplemented!()
    }
}

type EstimateRep = ParticleFilterRep;
pub type EstimatorT = LearningEstimator<EstimateRep>;

#[derive(Clone, Debug, Default)]
pub struct EstimatorMap {
    pub estimators: Vec<LearningEstimator<EstimateRep>>,
    pub estimator_map: HashMap<Entity, usize, RandomState>,
}

impl EstimatorMap {
    pub fn insert(&mut self, ms: &MentalState) {
        self.estimator_map
            .insert(ms.id.into(), self.estimators.len());
        self.estimators
            .push(LearningEstimator::new(vec![(ms.id, ms.sight_radius)]));
    }
    pub fn get(&self, entity: Entity) -> Option<&EstimatorT> {
        if let Some(i) = self.estimator_map.get(&entity) {
            return self.estimators.get(*i);
        }
        None
    }
    pub fn get_representation_source<'a>(
        &'a self,
        entity: Entity,
    ) -> Option<impl Iterator<Item = &impl MentalStateRep> + 'a> {
        self.get(entity).map(|r| r.estimators.into_iter())
    }
    pub fn rebind_estimator(&mut self, old: WorldEntity, new: WorldEntity) {
        if let Some(idx) = self.estimator_map.remove(&old.into()) {
            self.estimators[idx].replace(old, new);
            self.estimator_map.insert(new.into(), idx);
        }
    }
    pub fn par_iter_mut<'a>(&'a mut self) -> impl ParallelIterator<Item = &'a mut EstimatorT> + 'a {
        self.estimators.par_iter_mut()
    }
}
