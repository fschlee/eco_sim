#![feature(vec_remove_item)]
#![feature(proc_macro_hygiene)]

pub mod entity;
pub mod entity_type;
pub mod world;

pub mod agent;
pub mod util;

pub use entity::*;
pub use world::*;
pub use position::*;
pub use crate::agent::*;
pub use crate::entity_type::EntityType;

use std::ops::Range;
use rand::{SeedableRng};
use rand_xorshift::XorShiftRng;

#[derive(Clone)]
pub struct SimState {
    pub world: World<DefCell>,
    pub agent_system: AgentSystem,
    entity_manager: EntityManager,
    sim_step: f32,
    time_acc: f32,
}

impl SimState {
    pub fn advance(&mut self, time_step: f32) {
        self.time_acc += time_step;
        while self.time_acc >= self.sim_step {
            self.time_acc -= self.sim_step;
            self.agent_system.advance(&mut self.world, &mut self.entity_manager);
            self.world.advance();
        }
    }
    pub fn new(time_step: f32) -> Self {
        Self::new_with_seed(time_step, 0)
    }
    pub fn new_with_seed(time_step: f32, seed: u64) -> Self {
        let mut entity_manager = EntityManager::default();
        let rng = XorShiftRng::seed_from_u64(seed);
        let (world, agents) = World::init(rng.clone(), &mut entity_manager);
        let agent_system = AgentSystem::init(agents, &world, false, rng);
        Self {
            time_acc: 0.0,
            sim_step: time_step,
            agent_system,
            world: world,
            entity_manager,
        }
    }
    pub fn get_view(
        &self,
        x: Range<usize>,
        y: Range<usize>,
    ) -> impl Iterator<Item = (usize, usize, &[ViewData])> + '_ {
        self.world.get_view(x, y)
    }
    pub fn entities_at(&self, position: Position) -> &[WorldEntity] {
        (&self.world).entities_at(position)
    }
    pub fn update_mental_state(& mut self, mental_state: MentalState) {
        self.agent_system.mental_states.insert(&mental_state.id.clone(), mental_state);
    }
    pub fn get_mental_state(&self, entity :&WorldEntity) -> Option<&MentalState> {
        self.agent_system.mental_states.get(entity)
    }
    pub fn get_mental_model(&  self, entity: &WorldEntity) -> Option<impl Iterator<Item = & impl agent::estimator::MentalStateRep>> {
        self.agent_system.get_representation_source(entity.into())
    }
    pub fn get_physical_state(&self, entity: &WorldEntity) -> Option<&PhysicalState> {
        self.world.physical_states.get(entity)
    }
    pub fn get_visibility(& self, entity: &WorldEntity) -> impl Iterator<Item=Position> {
        let pos = self.world.positions.get(entity);
        let ms = self.agent_system.mental_states.get(entity);
        match (pos, ms) {
            (Some(pos), Some(ms)) => {
                let radius = ms.sight_radius;
                PositionWalker::new(*pos, radius)
            }
            _ => PositionWalker::empty()
        }
    }
    pub fn threat_map(&self, we: &WorldEntity) -> Vec<f32> {
        self.agent_system.threat_map(we, &self.world)
    }
}
