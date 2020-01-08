#![feature(vec_remove_item)]
#![feature(proc_macro_hygiene)]
#![feature(try_blocks)]
#![feature(const_fn)]
#![feature(const_loop)]
#![feature(const_if_match)]
#![feature(const_in_array_repeat_expressions)]
#![feature(const_generics)]
#![feature(type_alias_impl_trait)]
#![feature(generic_associated_types)]
#![feature(try_trait)]

pub mod entity;
pub mod entity_type;
pub mod world;

pub mod agent;

#[cfg(any(feature = "torch", feature = "reinforce"))]
pub mod rl_env_helper;

pub mod util;

pub use crate::agent::*;
pub use crate::entity_type::EntityType;
pub use entity::*;
pub use position::*;
pub use world::*;

use rand::SeedableRng;
use rand_xorshift::XorShiftRng;
use std::ops::Range;

#[derive(Clone)]
pub struct SimState {
    pub world: World<DefCell>,
    pub agent_system: AgentSystem,
    entity_manager: EntityManager,
    sim_step: f32,
    time_acc: f32,
    pub next_step_count: u64,
}

impl SimState {
    pub fn advance(
        &mut self,
        time_step: f32,
        overridden: impl IntoIterator<Item = (WorldEntity, Result<Action, FailReason>)>,
    ) {
        self.time_acc += time_step;
        if self.time_acc >= self.sim_step {
            self.time_acc -= self.sim_step;
            self.agent_system.override_actions(overridden);
            self.agent_system.infer(&self.world);
            self.world.events.clear();

            #[cfg(feature = "torch")]
            {
                let ms: &MentalState = self.agent_system.mental_states.iter().next().unwrap();
                let w = rl_env_helper::ObsvWriter::new(&self.world, &[]);
                let mut arr = ndarray::Array4::zeros((MAP_HEIGHT, MAP_WIDTH, 8, 9));
                w.encode_observation(&mut arr);
            }
            self.world.act(&self.agent_system.actions);
            self.world.advance();
            self.agent_system.process_feedback(&self.world.events);
            self.respawn_killed();
            self.agent_system.decide(&self.world);
            self.next_step_count += 1;
        }
    }
    fn respawn_killed(&mut self) {
        let Self {
            ref mut world,
            ref mut entity_manager,
            agent_system:
                AgentSystem {
                    ref mut estimator_map,
                    ref mut mental_states,
                    ref killed,
                    ..
                },
            ..
        } = self;
        for entity in killed {
            if let Some(mut ms) = mental_states.remove(entity) {
                let new_e = world.respawn(&entity, &mut ms, entity_manager);
                debug_assert_eq!(ms.id, new_e);
                mental_states.insert(&new_e, ms);
                estimator_map.rebind_estimator(*entity, new_e);
            }
        }
    }
    pub fn new(time_step: f32) -> Self {
        Self::new_with_seed(time_step, 0)
    }
    pub fn new_with_seed(time_step: f32, seed: u64) -> Self {
        #[cfg(feature = "torch")]
        println!("using torch");
        let mut entity_manager = EntityManager::default();
        let rng = XorShiftRng::seed_from_u64(seed);
        let (world, agents) = World::init(rng.clone(), &mut entity_manager);
        let agent_system = AgentSystem::init(agents, &world, false, true, rng);
        let mut this = Self {
            time_acc: 0.0,
            sim_step: time_step,
            agent_system,
            world: world,
            entity_manager,
            next_step_count: 1,
        };
        this.agent_system.decide(&this.world);
        this
    }
    pub fn get_view(
        &self,
        x: Range<usize>,
        y: Range<usize>,
    ) -> impl Iterator<Item = (usize, usize, &[ViewData])> + '_ {
        self.world.get_view(x, y)
    }
    pub fn get_world_knowledge_view(
        &self,
        x: Range<usize>,
        y: Range<usize>,
        we: WorldEntity,
    ) -> impl Iterator<Item = (usize, usize, &[ViewData])> + '_ {
        self.agent_system
            .mental_states
            .get(we)
            .and_then(|ms| ms.world_model.as_deref())
            .into_iter()
            .flat_map(move |w| w.get_view(x.clone(), y.clone()))
    }
    pub fn entities_at(&self, position: Position) -> &[WorldEntity] {
        (&self.world).entities_at(position)
    }
    pub fn update_mental_state(&mut self, id: WorldEntity, mut f: impl FnMut(&mut MentalState)) {
        self.agent_system.mental_states.get_mut(id).map(|ms| f(ms));
    }
    pub fn get_mental_state(&self, entity: &WorldEntity) -> Option<&MentalState> {
        self.agent_system.mental_states.get(entity)
    }
    pub fn get_mental_model(
        &self,
        entity: &WorldEntity,
    ) -> Option<impl Iterator<Item = &impl agent::estimator::MentalStateRep>> {
        self.agent_system.get_representation_source(entity.into())
    }
    pub fn get_physical_state(&self, entity: &WorldEntity) -> Option<&PhysicalState> {
        self.world.physical_states.get(entity)
    }
    pub fn get_visibility(&self, entity: &WorldEntity) -> impl Iterator<Item = Position> {
        let pos = self.world.positions.get(entity);
        let ms = self.agent_system.mental_states.get(entity);
        match (pos, ms) {
            (Some(pos), Some(ms)) => {
                let radius = ms.sight_radius;
                PositionWalker::new(*pos, radius)
            }
            _ => PositionWalker::empty(),
        }
    }
    pub fn threat_map(&self, we: &WorldEntity) -> Vec<f32> {
        self.agent_system.threat_map(we, &self.world)
    }
}
