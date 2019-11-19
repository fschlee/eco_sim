use rand::{Rng};
use rand_distr::{Normal, StandardNormal, Distribution};

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
    pub fn sample(&self, index: u8) -> MentalState {
        let mut sample : MentalState = self.into();
        let mut rng : rand_xorshift::XorShiftRng = rand::SeedableRng::seed_from_u64(index as u64);
        let scale = (1.0 + index as f32).log2() / 256f32.log2();

        let mut hunger_sample = rng.sample(Normal::new(self.hunger.0, scale * 10.0).unwrap()); // can only fail if std_dev < 0 or nan;
        sample.hunger.0 = clip(hunger_sample, 0.0, 10.0);
        for (_, pref) in sample.food_preferences.iter_mut() {
            let pref_sample : f32 = StandardNormal.sample(& mut rng);
            *pref = clip(*pref +  pref_sample * scale, 0.0, 1.0);
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
impl std::fmt::Display for Estimate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        writeln!(f, "{:?} ({})", self.id.e_type, self.id.id)?;
        writeln!(f, "Hunger: ({})", self.hunger.0)?;
        writeln!(f, "Preferences:")?;
        for (t, p) in &self.food_preferences {
            writeln!(f, "{:?}: {}", t, p)?;
        }
        writeln!(f, "Behavior:")?;
        writeln!(f, "{}", Behavior::fmt(&self.current_behavior))?;
        writeln!(f, "Action:")?;
        writeln!(f, "{}", Action::fmt(&self.current_action))
    }
}

fn clip(val: f32, min: f32, max: f32) -> f32 {
    max.min(min.max(val))
}