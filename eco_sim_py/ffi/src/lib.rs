use numpy::{IntoPyArray, PyArray4};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use eco_sim::{
    rl_env_helper::{ObsvWriter, ENTITY_REP_SIZE, MAX_REP_PER_CELL},
    Action, Cell, Coord, SimState, World, WorldEntity, MAP_HEIGHT, MAP_WIDTH,
};
use itertools::Itertools;
use ndarray::{Array4, ArrayBase};
use pyo3::types::PyType;
use std::ops::Deref;
use std::sync::{Arc, RwLock};

type EnvObservation = Py<PyArray4<f32>>;

type EnvAction = usize;
type Reward = f32;

#[pyclass]
struct Environment {
    sim: Arc<RwLock<SimState>>,
    agent: WorldEntity,
    old_reward: f32,
    sight: Coord,
}

#[pymethods]
impl Environment {
    #[new]
    fn new(obj: &PyRawObject, seed: u64, start_gui: bool) {
        let sim = SimState::new_with_seed(1.0, seed);
        let ms = (&sim.agent_system.mental_states)
            .into_iter()
            .next()
            .unwrap();
        let agent = ms.id;
        let sight = ms.sight_radius;
        let sim = Arc::new(RwLock::new(sim));

        obj.init(Environment {
            sim,
            agent,
            old_reward: 0.0,
            sight,
        });
    }
    fn reset(&mut self, seed: u64) {
        let mut sim = self.sim.write().unwrap();
        *sim = SimState::new_with_seed(1.0, seed);
        let ms = (&sim.agent_system.mental_states)
            .into_iter()
            .next()
            .unwrap();
        self.agent = ms.id;
        self.sight = ms.sight_radius;
        self.old_reward = ms.score
    }
    fn state(&self, py: Python<'_>) -> PyResult<(EnvObservation, Reward, EnvAction, bool)> {
        let sim = self.sim.read().unwrap();
        let world = &sim.world;
        let (reward, act, done) = if let (Some(ms), Some(pos)) = (
            &sim.get_mental_state(&self.agent),
            world.positions.get(self.agent),
        ) {
            let new_reward = ms.score;
            let reward = new_reward - self.old_reward;
            let suggested = ms.current_action;
            (reward, suggested, false)
        } else {
            (-1000.0, Action::Idle, true)
        };

        let obsv_writer =
            ObsvWriter::new(world, *world.positions.get(self.agent).unwrap(), self.sight);
        let suggested = obsv_writer.encode_action(act);
        let mut obs = Array4::zeros((MAP_HEIGHT, MAP_WIDTH, MAX_REP_PER_CELL, ENTITY_REP_SIZE));
        obsv_writer.encode_observation(&mut obs);
        Ok((obs.into_pyarray(py).to_owned(), reward, suggested, done))
    }
    fn step(
        &mut self,
        py: Python<'_>,
        action: EnvAction,
    ) -> PyResult<(EnvObservation, Reward, EnvAction, bool)> {
        {
            let world = &self.sim.read().unwrap().world;
            let obsv_writer =
                ObsvWriter::new(world, *world.positions.get(self.agent).unwrap(), self.sight);
            let action_to_take = obsv_writer.decode_action(action);
        }
        py.allow_threads(|| self.sim.write().unwrap().advance(1.0));
        self.state(py)
    }
    #[staticmethod]
    fn action_space_size() -> usize {
        ObsvWriter::ACTION_SPACE_SIZE
    }
    #[staticmethod]
    fn map_width() -> usize {
        MAP_WIDTH
    }
    #[staticmethod]
    fn map_height() -> usize {
        MAP_HEIGHT
    }
    #[staticmethod]
    fn max_reps_per_square() -> usize {
        MAX_REP_PER_CELL
    }
    #[staticmethod]
    fn rep_size() -> usize {
        ENTITY_REP_SIZE
    }
}

#[pymodule]
fn eco_sim(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Environment>()?;
    Ok(())
}
