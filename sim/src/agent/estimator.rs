use rand::{Rng};
use super::MentalState;
use crate::{World, Action, Observation, WorldEntity, Position, StorageSlice, Cell};
use crate::entity::{Entity, Storage, Source};
use crate::entity_type::EntityType;


pub trait MentalStateRep: std::fmt::Display + Sized {
    fn sample<R: Rng + ?Sized>(& self, scale: f32, rng: & mut R) -> MentalState;
    fn update_seen<'a>(& 'a mut self, action: Action, others: & impl Estimator, observation: &impl Observation);
    fn update_unseen<'a>(& 'a mut self, others: & impl Estimator, observation: & impl Observation);
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
    type Rep: MentalStateRep;
    fn invoke_sampled<'a, R: rand::Rng + Sized>(& 'a self, we: WorldEntity,rng: & 'a mut R, n: usize) -> InvokeIter<'a, Self::Rep, R>;
    fn learn<C: Cell>(& mut self, action: Action, other: WorldEntity, other_pos: Position, world: & World<C>);
}

#[derive(Clone, Debug)]
pub struct LearningEstimator<E : MentalStateRep> {
    pub agents: Vec<(WorldEntity, u32)>,
    pub estimators: Storage<E>,
}

pub enum InvokeIter<'a, E: MentalStateRep, R: Rng> {
    Rep {
        count: usize,
        rep: & 'a E,
        rng: & 'a mut R,
    },
    Empty
}
impl<'a, E: MentalStateRep, R: Rng>  Iterator for InvokeIter<'a, E, R> {

    type Item = MentalState;
    fn next(&mut self) -> Option<Self::Item> {
        let ms = match self {
            Self::Empty => None,
            Self::Rep{ ref mut count, rep, rng} => {
                if *count == 0 {
                    None

                } else {
                    *count -= 1;
                    Some(rep.sample(1.0, rng))
                }
            }
        };
        if ms.is_none() {
            *self = Self::Empty;;
        }
        ms
    }
}

impl<E: MentalStateRep> LearningEstimator<E> {
    fn assure_init(& mut self, entity: &WorldEntity, observation: &impl Observation) {
        if self.estimators.get(entity).is_some() {
           return;
        }
        {
            let rep = MentalStateRep::from_aggregate(entity, self.estimators.into_iter().filter(|e| e.get_type() == entity.e_type()));
            self.estimators.insert(entity, rep);
        }
    }
    pub fn new(agents: Vec<(WorldEntity, u32)>) -> Self {
        Self {
            agents,
            estimators: Storage::new(),
        }
    }
    pub fn replace(& mut self, old: WorldEntity, new: WorldEntity){
        if let Some(tpl) = self.agents.iter_mut().find(|(id,_)| id.id() == old.id()){
            tpl.0 = new;
        }
    }
}

impl<E: MentalStateRep + 'static> Estimator for LearningEstimator<E> {
    fn invoke(&self, entity: WorldEntity) -> Option<MentalState> {
        self.estimators.get(entity).map(MentalStateRep::into_ms)
    }
    type Rep = E;
    fn invoke_sampled<'a, R: rand::Rng + Sized>(&'a self, entity: WorldEntity, rng: & 'a mut R, n: usize) -> InvokeIter<'a, E, R> {
        if let Some(rep) = self.estimators.get(entity) {
            InvokeIter::Rep { rng, count: n, rep }
        }
        else {
            InvokeIter::Empty
        }

        // self.estimators.get(entity).iter().flat_map(|e| e.sample(1.0, rng)).fuse()
    }
    fn learn<C: Cell>(& mut self, action: Action, other: WorldEntity, other_pos: Position, world: & World<C>) {
        for (agent, sight) in self.agents.clone() {
            if let Some(own_pos) = world.positions.get(agent) {
                let observation = world.observe_in_radius(&agent, sight); // &(*world);//
                let dist = own_pos.distance(&other_pos);
                if dist <= sight {
                    self.assure_init(&other, &observation);
                }
                if let Some((es, sc)) = self.estimators.split_out_mut(other) {
                    if dist <= sight {
                        es.update_seen(action, &sc, &observation);
                    }
                    else {
                        es.update_unseen(&sc, &observation);
                    }
                }
            }
        }
    }
}

impl<'a, T: MentalStateRep + Sized + 'static> Estimator for StorageSlice<'a, T> {
    type Rep = T;
    fn invoke(&self, entity: WorldEntity) -> Option<MentalState> {

       self.get(entity).map(MentalStateRep::into_ms)
    }

    fn invoke_sampled<'b, R: rand::Rng + Sized>(& 'b self, we: WorldEntity,rng: & 'b mut R, n: usize) -> InvokeIter<'b, Self::Rep, R>{
        InvokeIter::Empty
        // self.get(we).iter().flat_map(|e: &T| e.sample(1.0, rng)).fuse()
    }
    fn learn<C: Cell>(& mut self, action: Action, other: WorldEntity, other_pos: Position, world: & World<C>) {
        unimplemented!()
    }
}