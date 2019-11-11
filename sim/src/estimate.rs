use rand::{Rng, thread_rng};

use super::agent::{MentalState, Behavior, Hunger, Reward};
use super::entity::{Entity, Storage};
use super::world::{Action, PhysicalState, Health, Meat, Satiation, Observation};
use super::entity_type::{EntityType, ENTITY_TYPES};

#[derive(Clone, Debug)]
pub struct Estimate {
    pub id: Entity,
    pub physical_state: PhysicalState,
    pub hunger: Hunger,
    pub food_preferences: Vec<(EntityType, Reward)>,
    pub current_action: Option<Action>,
    pub current_behavior: Option<Behavior>,
    pub sight_radius: u32,
    pub use_mdp: bool,
}
impl Estimate {
    pub fn update(&mut self, ms: &MentalState){
        self.hunger = ms.hunger;
        self.food_preferences = ms.food_preferences.clone();
        self.current_behavior = ms.current_behavior.clone();
        self.current_action = ms.current_action;
    }
    pub fn sample(&self, seed: u64) -> MentalState {
        let mut sample : MentalState = self.into();
        let mut rng : rand_xorshift::XorShiftRng = rand::SeedableRng::seed_from_u64(seed);
        sample.hunger.0 =  10.0f32.min(0.0f32.max( sample.hunger.0 + rng.gen_range(-10.0, 10.0)));
        for (_, pref) in sample.food_preferences.iter_mut() {
            *pref = 1.0f32.min(0.0f32.max( *pref + rng.gen_range(-0.5, 0.5)));
        }
        sample.rng = rng;
        sample
    }
}

impl Into<MentalState> for &Estimate {
    fn into(self) -> MentalState {
        MentalState {
            id: self.id,
            hunger: self.hunger,
            food_preferences: self.food_preferences.clone(),
            current_action: self.current_action,
            current_behavior: self.current_behavior.clone(),
            sight_radius: self.sight_radius,
            use_mdp: false,
            rng: rand::SeedableRng::seed_from_u64(self.id.id as u64),
            estimates: Storage::new()
        }
    }
}
impl Estimate {
    pub fn updated(&self, observation: impl Observation, action: Option<Action>) -> Option<Estimate> {
        if let Some(pos) = observation.observed_position(&self.id){
            let mut ms : MentalState= self.into();
            if action == ms.decide( &(self.physical_state), pos, observation.borrow()){
                return None;
            }
            else {
                let max_tries = 20;
                for i in 0..max_tries {
                    let mut sample = self.sample(i);
                    if action == sample.decide( &(self.physical_state), pos, observation.borrow()){
                        let mut est = self.clone();
                        est.update(&sample);
                        return Some(est);
                    }
                }


                //Todo
            }
        }
        None
    }
}
pub fn default_estimate(entity: & Entity) -> Estimate {
    Estimate{
        id: entity.clone(),
        physical_state: entity.e_type.typical_physical_state().unwrap_or(PhysicalState {
            health: Health(0.0),
            meat: Meat(0.0),
            attack: None,
            satiation: Satiation(0.0)
        }),
        hunger: Default::default(),
        food_preferences: ENTITY_TYPES.iter().filter_map(|other| {
            if entity.e_type.can_eat(other) {
                Some((*other, 0.5))
            }
            else {
                None
            }
        }).collect(),
        current_action: None,
        current_behavior: None,
        sight_radius: 5,
        use_mdp: false
    }
}