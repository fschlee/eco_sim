use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use std::borrow::Borrow;

use super::estimator::MentalStateRep;
use crate::agent::estimator::Estimator;
use crate::agent::{Behavior, Hunger, MentalState};
use crate::entity::WorldEntity;
use crate::entity_type::EntityType;
use crate::position::Coord;
use crate::util::f32_cmp;
use crate::world::{Action, Event, Health, Observation, Occupancy, PhysicalState, Speed, World};
use crate::{EmotionalState, Prob};
use lazysort::{SortedBy, SortedPartial};
use rand_xorshift::XorShiftRng;
use smallvec::SmallVec;
use std::iter::repeat;

#[derive(Clone, Debug)]
pub struct PointEstimateRep {
    pub id: WorldEntity,
    pub physical_state: PhysicalState,
    pub emotional_state: EmotionalState,
    pub current_action: Action,
    pub current_behavior: Option<Behavior>,
    pub sight_radius: Coord,
    pub use_mdp: bool,
}
impl PointEstimateRep {
    pub fn update(&mut self, ms: &MentalState) {
        self.emotional_state += &ms.emotional_state;
        self.current_behavior = ms.current_behavior.clone();
        self.current_action = ms.current_action;
    }
}

impl MentalStateRep for PointEstimateRep {
    fn sample<R: Rng + ?Sized>(&self, scale: f32, rng: &mut R) -> MentalState {
        let mut sample: MentalState = self.into();
        sample_ms(&mut sample, scale, rng);
        sample
    }
    fn update_seen<'a>(
        &'a mut self,
        action: Action,
        others: &impl Estimator,
        observation: &impl Observation,
    ) {
        if let Some(phys) = observation.observed_physical_state(&self.id) {
            self.physical_state = phys.clone();
        }
        if let Some(pos) = observation.observed_position(&self.id) {
            let mut ms: MentalState = (&(*self)).into();
            let threat_map = ms.threat_map(observation, others);
            ms.decide_simple(
                &(self.physical_state),
                pos,
                observation,
                others,
                &threat_map,
            );
            if action == ms.current_action {
                self.update(&ms);
            } else {
                let mut rng: rand_xorshift::XorShiftRng =
                    rand::SeedableRng::seed_from_u64(self.id.id() as u64);
                let max_tries = 255;
                // let mut valid = Vec::new();
                for i in (0..max_tries).rev() {
                    let scale = (1.0 + i as f32).log2() / (max_tries as f32).log2();
                    let mut sample = self.sample(scale, &mut rng);
                    sample.decide_simple(
                        &(self.physical_state),
                        pos,
                        observation,
                        others,
                        &threat_map,
                    );
                    if action == sample.current_action {
                        // valid.push(sample);
                        self.update(&sample);
                        break;
                    }
                }
                /*
                if valid.len() > 0 {
                    self.update(&valid[0]);
                    self.emotional_state = EmotionalState::average(valid.drain(..).map(|ms| ms.emotional_state))
                }
                */
            }
        }
    }
    fn update_unseen<'a>(&'a mut self, others: &impl Estimator, observation: &impl Observation) {

        //TODO
    }
    fn into_ms(&self) -> MentalState {
        MentalState {
            id: self.id,
            emotional_state: self.emotional_state.clone(),
            current_action: self.current_action,
            current_behavior: self.current_behavior.clone(),
            sight_radius: self.sight_radius,
            rng: rand::SeedableRng::seed_from_u64(self.id.id() as u64),
            use_mdp: false,
            world_model: None,
            score: 0.0,
        }
    }
    fn from_aggregate<B>(we: &WorldEntity, iter: impl Iterator<Item = B>) -> Self
    where
        B: Borrow<Self>,
    {
        let mut def = Self::default(we);
        {
            def.emotional_state =
                EmotionalState::average(iter.map(|b| b.borrow().emotional_state.clone()));
        }
        def
    }
    fn default(entity: &WorldEntity) -> Self {
        PointEstimateRep {
            id: entity.clone(),
            physical_state: entity
                .e_type()
                .typical_physical_state()
                .unwrap_or(PhysicalState::new(Health(50.0), Speed(0.2), None)),
            emotional_state: EmotionalState::new(
                EntityType::iter()
                    .filter_map(|other| {
                        if entity.e_type().can_eat(&other) {
                            Some((other, 0.5))
                        } else {
                            None
                        }
                    })
                    .collect(),
            ),
            current_action: Action::default(),
            current_behavior: None,
            sight_radius: 5,
            use_mdp: false,
        }
    }
    fn get_type(&self) -> EntityType {
        self.id.e_type()
    }
    fn update_on_events<'a>(
        &'a mut self,
        events: impl IntoIterator<Item = &'a Event> + Copy,
        _world_model: Option<&'a World<Occupancy>>,
    ) {
        {
            let mut new: MentalState = (&(*self)).into();
            new.update_on_events(events);
            self.update(&new);
        }
    }
}

impl std::fmt::Display for PointEstimateRep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        let ms: MentalState = self.into();
        writeln!(f, "{:?} ({})", self.id.e_type(), self.id.id())?;
        writeln!(f, "Hunger: ({})", self.emotional_state.hunger().0)?;
        writeln!(f, "Preferences:")?;
        for (t, p) in ms.food_preferences() {
            writeln!(f, "{:?}: {}", t, p)?;
        }
        writeln!(f, "Behavior:")?;
        writeln!(f, "{}", Behavior::fmt(&self.current_behavior))?;
        writeln!(f, "Action:")?;
        self.current_action.fmt(f)
    }
}

impl Borrow<EmotionalState> for PointEstimateRep {
    fn borrow(&self) -> &EmotionalState {
        &self.emotional_state
    }
}

const PARTICLE_COUNT: usize = 10;

#[derive(Clone, Debug)]
pub struct ParticleFilterRep {
    id: WorldEntity,
    physical_state: PhysicalState,
    particles: SmallVec<[(MentalState, Prob); PARTICLE_COUNT]>,
    /// Probability of action missprediction given approximately correct mental state.
    p_wrong_action: Prob,
}

impl MentalStateRep for ParticleFilterRep {
    fn sample<R: Rng + ?Sized>(&self, scale: f32, rng: &mut R) -> MentalState {
        let distr = rand_distr::Uniform::new(0.0f32, 1.0);
        let mut pick = rng.sample(distr);
        let mut ms = self
            .particles
            .iter()
            .find(|(ms, p)| {
                pick -= *p;
                pick <= 0.0
            })
            .unwrap_or(&self.particles[0])
            .0
            .clone();
        sample_ms(&mut ms, scale, rng);
        ms
    }

    fn update_seen<'a>(
        &'a mut self,
        action: Action,
        others: &impl Estimator,
        observation: &impl Observation,
    ) {
        if let Some(phys) = observation.observed_physical_state(&self.id) {
            self.physical_state = phys.clone();
        }
        if let Some(pos) = observation.observed_position(&self.id) {
            let threat_map = self.particles[0].0.threat_map(observation, others);
            let mut rng = self.particles[0].0.rng.clone();
            // Low variance resampling
            let inc = 1.0 / PARTICLE_COUNT as f32;
            let mut target = rng.sample(rand_distr::Uniform::new(0.0, inc));
            let mut acc = 0.0;
            let mut next = SmallVec::new();
            let mut p_right_action = 0.0;
            for (ms, p) in self.particles.iter() {
                acc += *p;
                if acc > target {
                    let n =
                        (((acc - target) / inc).ceil() as usize).max(PARTICLE_COUNT - next.len());
                    target += n as f32 * inc;
                    for i in 0..n {
                        let mut new_ms = ms.clone();
                        sample_ms(&mut new_ms, 0.1 * i as f32, &mut rng);
                        new_ms.decide_simple(
                            &self.physical_state,
                            pos,
                            observation,
                            others,
                            &threat_map,
                        );
                        let mut new_p = *p / n as f32;
                        if new_ms.current_action == action {
                            p_right_action += new_p;
                            new_p *= 1.0 - self.p_wrong_action;
                        } else {
                            new_p *= self.p_wrong_action;
                        }
                        next.push((new_ms, new_p));
                    }
                }
            }
            debug_assert!(next.len() == PARTICLE_COUNT);
            let p_right: f32 = next.iter().map(|(_, p)| *p).sum();
            let inv = 1.0 / p_right;
            for (_, p) in next.iter_mut() {
                *p *= inv;
            }
            next.sort_by(|(_, p0), (_, p1)| f32_cmp(p1, p0));
            let p_right_action_given_right: f32 = p_right_action / p_right;
            self.p_wrong_action = 0.9f32 * self.p_wrong_action
                + 0.1f32 * (1.0f32 - p_right_action_given_right).min(0.01).max(0.4);
            std::mem::swap(&mut self.particles, &mut next);
        }
    }

    fn update_on_events<'a>(
        &'a mut self,
        events: impl IntoIterator<Item = &'a Event> + Copy,
        world_model: Option<&'a World<Occupancy>>,
    ) {
        for (ms, _) in self.particles.iter_mut() {
            ms.update_on_events(events);
        }
    }

    fn update_unseen<'a>(&'a mut self, others: &impl Estimator, observation: &impl Observation) {
        // Todo
    }

    fn default(we: &WorldEntity) -> Self {
        let mut rng: XorShiftRng = rand::SeedableRng::seed_from_u64(we.id() as u64);
        let distr = rand_distr::Uniform::new(0.0f32, 1.0);
        let particles = std::iter::repeat_with(|| {
            let v: Vec<_> = EntityType::iter()
                .filter_map(|et| {
                    if we.e_type().can_eat(&et) {
                        let pref = rng.sample(distr);
                        Some((et, pref))
                    } else {
                        None
                    }
                })
                .collect();
            (
                MentalState::new(*we, v, false, false),
                1.0 / PARTICLE_COUNT as f32,
            )
        })
        .take(PARTICLE_COUNT)
        .collect();
        ParticleFilterRep {
            id: we.clone(),
            physical_state: we
                .e_type()
                .typical_physical_state()
                .unwrap_or(PhysicalState::new(Health(50.0), Speed(0.2), None)),
            particles,
            p_wrong_action: 0.2,
        }
    }

    fn from_aggregate<B>(we: &WorldEntity, iter: impl Iterator<Item = B>) -> Self
    where
        B: std::borrow::Borrow<Self>,
    {
        let mut rng: XorShiftRng = rand::SeedableRng::seed_from_u64(we.id() as u64);
        let distr = rand_distr::Uniform::new(0.0f32, 1.0);
        use lazysort::{SortedBy, SortedPartial};
        let mut particles: SmallVec<_> = iter
            .flat_map(|b| {
                let v: Vec<_> = b.borrow().particles.iter().cloned().collect();
                v.into_iter()
            })
            .sorted_by(|(_, p0), (_, p1)| f32_cmp(p1, p0))
            .chain(std::iter::repeat_with(|| {
                let v: Vec<_> = EntityType::iter()
                    .filter_map(|et| {
                        if we.e_type().can_eat(&et) {
                            let pref = rng.sample(distr);
                            Some((et, pref))
                        } else {
                            None
                        }
                    })
                    .collect();
                (MentalState::new(*we, v, false, false), 0.01)
            }))
            .take(PARTICLE_COUNT)
            .collect(); // for_each(|(ms, p)| { particles.push(ms); probs.push(p); });
        normalize(&mut particles, |(_, p)| p);
        // let particles = unsafe { std::mem::transmute(arr) };
        ParticleFilterRep {
            id: *we,
            physical_state: we.e_type().typical_physical_state().unwrap(),
            particles,

            p_wrong_action: 0.2,
        }
    }

    fn get_type(&self) -> EntityType {
        self.id.e_type()
    }
}

fn normalize<T, F: Fn(&mut T) -> &mut f32>(probs: &mut [T], accessor: F) {
    let s: f32 = probs.iter_mut().map(|f| *((&accessor)(f))).sum();
    let inv: f32 = 1.0f32 / s;
    for p in probs.iter_mut() {
        *(accessor(p)) *= inv;
    }
}

fn sample_ms<R: Rng + ?Sized>(sample: &mut MentalState, scale: f32, rng: &mut R) {
    let hunger_sample: f32 = StandardNormal.sample(rng);
    sample.emotional_state += Hunger(hunger_sample * scale);
    for et in EntityType::iter() {
        if sample.id.e_type().can_eat(&et) {
            let pref_sample: f32 = StandardNormal.sample(rng);
            sample
                .emotional_state
                .set_preference(et, sample.emotional_state.pref(et).0 + pref_sample * scale)
        }
    }
    let bhv_sample: f32 = StandardNormal.sample(rng);
    if bhv_sample * scale > 0.25f32 {
        sample.current_behavior = None;
    }
    sample.rng = rand::SeedableRng::seed_from_u64(rng.gen());
}

impl std::fmt::Display for ParticleFilterRep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        writeln!(f, "{:?} ({})", self.id.e_type(), self.id.id())?;
        for (ms, prob) in self.particles.iter() {
            writeln!(f, "Probability {:.2}%:", prob * 100.0);
            writeln!(f, "Hunger: ({})", ms.emotional_state.hunger().0)?;
            writeln!(f, "Preferences:")?;
            for (t, p) in ms.food_preferences() {
                writeln!(f, "{:?}: {}", t, p)?;
            }
            writeln!(f, "Behavior:")?;
            writeln!(f, "{}", Behavior::fmt(&ms.current_behavior))?;
            writeln!(f, "Action:")?;
            ms.current_action.fmt(f)?;
        }
        Ok(())
    }
}
