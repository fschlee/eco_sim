use rand::{Rng};
use rand_distr::{Normal, StandardNormal, Distribution};
use std::borrow::Borrow;

use crate::agent::{MentalState, Behavior, Hunger, Reward};
use super::estimator::MentalStateRep;
use crate::entity::{WorldEntity, Storage, Source};
use crate::world::{Action, PhysicalState, Health, Meat, Satiation, Speed, MoveProgress, Observation};
use crate::entity_type::{EntityType};
use crate::agent::estimator::Estimator;
use crate::util::clip;
use crate::EmotionalState;

#[derive(Clone, Debug)]
pub struct PointEstimateRep {
    pub id: WorldEntity,
    pub physical_state: PhysicalState,
    pub emotional_state: EmotionalState,
    pub current_action: Option<Action>,
    pub current_behavior: Option<Behavior>,
    pub sight_radius: u32,
    pub use_mdp: bool,
}
impl PointEstimateRep {
    pub fn update(&mut self, ms: &MentalState){
        self.emotional_state = ms.emotional_state.clone();
        self.current_behavior = ms.current_behavior.clone();
        self.current_action = ms.current_action;
    }
}

impl MentalStateRep for PointEstimateRep {
    fn sample<R: Rng + ?Sized>(&self, scale: f32, rng: &mut R) -> MentalState {
        let mut sample : MentalState = self.into();
        let hunger_sample : f32 = StandardNormal.sample(rng);
        sample.emotional_state += Hunger(hunger_sample * scale);
        for et in EntityType::iter() {
            if self.id.e_type().can_eat(&et) {
                let pref_sample : f32 = StandardNormal.sample(rng);
                sample.emotional_state.set_preference(et, sample.emotional_state.pref(et).0 + pref_sample * scale)
            }
        }
        let bhv_sample : f32  = StandardNormal.sample(rng);
        if  bhv_sample * scale > 0.25f32 {
            sample.current_behavior = None;
        }
        sample.rng = rand::SeedableRng::seed_from_u64(rng.gen());
        sample
    }
    fn update_seen<'a>(& 'a mut self, action: Option<Action>, others: &impl Estimator, observation: & impl Observation) {

        if let Some(pos) = observation.observed_position(&self.id) {
            let mut ms : MentalState= (&(*self)).into();

            if action == ms.decide( &(self.physical_state), pos, observation, others){
                self.update(&ms);
            }
            else {
                let mut rng : rand_xorshift::XorShiftRng = rand::SeedableRng::seed_from_u64(self.id.id() as u64);
                let max_tries = 255;
                for i in 0..max_tries {
                    let scale = (1.0 + i as f32).log2() / 256f32.log2();
                    let mut sample = self.sample(scale, & mut rng);
                    if action == sample.decide( &(self.physical_state), pos, observation, others){
                        self.update(&sample);
                    }
                }
            }
        }
    }
    fn update_unseen<'a>(& 'a mut self, others: & impl Estimator, observation: &impl Observation){
        // fn update_unseen(&mut self, others: impl Source<'_, MentalState>, observation: impl Observation) {

        //TODO
    }
    fn into_ms(&self) -> MentalState {
        MentalState {
            id: self.id,
            emotional_state: self.emotional_state.clone(),
            current_action: self.current_action,
            current_behavior: self.current_behavior.clone(),
            sight_radius: self.sight_radius,
            use_mdp: false,
            rng: rand::SeedableRng::seed_from_u64(self.id.id() as u64),
        }
    }
    fn from_aggregate<B>(we: &WorldEntity, iter: impl Iterator<Item=B>) -> Self where B: Borrow<Self> {

        let mut def = Self::default(we);
        {
            def.emotional_state = EmotionalState::average(iter.map(|b | b.borrow().emotional_state.clone()));
        }
        def
    }
    fn default(entity: &WorldEntity) -> Self {
        PointEstimateRep {
            id: entity.clone(),
            physical_state: entity.e_type().typical_physical_state().unwrap_or(PhysicalState::new(
                Health(50.0),
                Speed(0.2),
                None )
            ),
            emotional_state: EmotionalState::new(EntityType::iter().filter_map(|other| {
                if entity.e_type().can_eat(&other) {
                    Some((other, 0.5))
                }
                else {
                    None
                }
            }).collect()),
            current_action: None,
            current_behavior: None,
            sight_radius: 5,
            use_mdp: false
        }
    }
    fn get_type(&self) -> EntityType{
        self.id.e_type()
    }
}

impl std::fmt::Display for PointEstimateRep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        let ms : MentalState = self.into();
        writeln!(f, "{:?} ({})", self.id.e_type(), self.id.id())?;
        writeln!(f, "Hunger: ({})", self.emotional_state.hunger().0)?;
        writeln!(f, "Preferences:")?;
        for (t, p) in ms.food_preferences() {
            writeln!(f, "{:?}: {}", t, p)?;
        }
        writeln!(f, "Behavior:")?;
        writeln!(f, "{}", Behavior::fmt(&self.current_behavior))?;
        writeln!(f, "Action:")?;
        writeln!(f, "{}", Action::fmt(&self.current_action))
    }
}

impl Borrow<EmotionalState> for PointEstimateRep {
    fn borrow(&self) -> &EmotionalState {
        &self.emotional_state
    }
}