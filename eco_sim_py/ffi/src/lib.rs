use numpy::{IntoPyArray, PyArray2, PyArray4};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use eco_sim::{
    entity::Source,
    entity_type::Count,
    rl_env_helper::{
        ObsvWriter, ENTITY_REP_SIZE, MAX_REP_PER_CELL, MENTAL_REP_SIZE, PHYS_REP_SIZE,
    },
    Action, Cell, Coord, MentalState, SimState, World, WorldEntity, MAP_HEIGHT, MAP_WIDTH,
};
use itertools::Itertools;
use ndarray::{Array4, ArrayBase};
use pyo3::types::PyType;
use std::ops::Deref;
use std::sync::{Arc, RwLock};

type EnvObservation = PyArray4<f32>;

type EnvAction = usize;
type Reward = f32;

#[pyclass]
struct Environment {
    sim: Arc<RwLock<SimState>>,
    agents: Vec<(WorldEntity, Coord)>,
}

#[pymethods]
impl Environment {
    #[new]
    fn new(obj: &PyRawObject, seed: u64, start_gui: bool) {
        let sim = SimState::new_with_seed(1.0, seed);
        let sim = Arc::new(RwLock::new(sim));

        obj.init(Environment {
            sim,
            agents: Vec::new(),
        });
    }
    fn reset(&mut self, seed: u64) {
        let mut sim = self.sim.write().unwrap();
        *sim = SimState::new_with_seed(1.0, seed);
        self.agents.clear()
    }
    fn register_agents(&mut self, number: Option<usize>) -> Vec<u8> {
        let Self {
            ref mut agents,
            ref mut sim,
        } = self;
        let sim = sim.write().unwrap();
        let mut old = Vec::new();
        agents.retain(|(a, _)| {
            let ret = sim
                .world
                .get_physical_state(a)
                .map_or(false, |p| !p.is_dead())
                && sim.agent_system.mental_states.get(a).is_some();
            if ret {
                old.push(*a);
            }
            ret
        });
        let n = number.unwrap_or(1);
        let mut c = 0;
        for ms in (&sim.agent_system.mental_states).into_iter() {
            if old.contains(&ms.id) {
                continue;
            }
            match sim.world.get_physical_state(&ms.id) {
                Some(phys) if !phys.is_dead() => {
                    agents.push((ms.id, ms.sight_radius));
                    c += 1;
                    if c >= n {
                        break;
                    }
                }
                _ => (),
            }
        }
        agents
            .iter()
            .map(|(we, _c)| we.e_type().idx() as u8)
            .collect()
    }
    fn state<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Vec<&'py EnvObservation>,
        Vec<Reward>,
        Vec<EnvAction>,
        &'py PyArray2<f32>,
        &'py PyArray2<f32>,
        Vec<Vec<bool>>,
        Vec<(usize, Option<usize>)>,
        bool,
    )> {
        let sim = self.sim.read().unwrap();
        let world = &sim.world;
        let obsv_writer = ObsvWriter::new(world, &self.agents);
        let mut positions = Vec::with_capacity(self.agents.len());
        let mut observations = Vec::with_capacity(self.agents.len());
        let mut rewards = Vec::with_capacity(self.agents.len());
        let mut suggested = Vec::with_capacity(self.agents.len());
        let mut remappings = Vec::new();
        let mut shifted = 0;
        for (i, (agent, _c)) in self.agents.iter().enumerate() {
            let mut pyarr = PyArray4::zeros(
                py,
                (MAP_HEIGHT, MAP_WIDTH, MAX_REP_PER_CELL, ENTITY_REP_SIZE),
                false,
            );
            let killed = false;
            observations.push(pyarr);

            let (reward, act, killed) = match (
                &sim.get_mental_state(agent),
                world.get_physical_state(agent),
            ) {
                (Some(ms), Some(phys)) if !phys.is_dead() => {
                    let reward = ms.score;
                    let suggested = ms.current_action;
                    (reward, suggested, false)
                }
                _ => (-1000.0, Action::Idle, true),
            };
            positions.push(world.positions.get(agent));
            suggested.push(obsv_writer.encode_action(*agent, act).unwrap());
            rewards.push(reward);
            if killed {
                remappings.push((i, None));
                shifted += 1;
            } else if shifted > 0 {
                remappings.push((i, Some(i - shifted)))
            }
        }
        let visibility = self
            .agents
            .iter()
            .enumerate()
            .map(|(i, (a, s))| {
                let v: Vec<bool> = positions
                    .iter()
                    .map(|op| match (op, positions[i]) {
                        (Some(other_pos), Some(pos)) if pos.distance(other_pos) <= *s => true,
                        _ => false,
                    })
                    .collect();
                v
            })
            .collect();
        let mut phys = PyArray2::zeros(py, (self.agents.len(), PHYS_REP_SIZE), false);
        let mut mental = PyArray2::zeros(py, (self.agents.len(), MENTAL_REP_SIZE), false);
        //  let mut obs = Array4::zeros((MAP_HEIGHT, MAP_WIDTH, MAX_REP_PER_CELL, ENTITY_REP_SIZE));

        {
            let mut ones: Vec<_> = observations
                .iter_mut()
                .map(|pyarr| pyarr.as_array_mut())
                .collect();
            obsv_writer.encode_views(&mut ones);
            obsv_writer.encode_agents(
                &mut phys.as_array_mut(),
                &mut mental.as_array_mut(),
                &sim.agent_system.mental_states,
            );
        }
        Ok((
            observations,
            rewards,
            suggested,
            phys,
            mental,
            visibility,
            remappings,
            false,
        ))
    }
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: Vec<EnvAction>,
    ) -> PyResult<(
        Vec<&'py EnvObservation>,
        Vec<Reward>,
        Vec<EnvAction>,
        &'py PyArray2<f32>,
        &'py PyArray2<f32>,
        Vec<Vec<bool>>,
        Vec<(usize, Option<usize>)>,
        bool,
    )> {
        let actions_to_take: Vec<_> = {
            let sim = self.sim.read().unwrap();
            let world = &sim.world;
            self.agents.retain(|(a, _)| {
                world.get_physical_state(a).map_or(false, |p| !p.is_dead())
                    && sim.agent_system.mental_states.get(a).is_some()
            });
            let obsv_writer = ObsvWriter::new(world, &self.agents);
            self.agents
                .iter()
                .zip(actions.iter())
                .filter_map(|((we, _c), a)| obsv_writer.decode_action(*we, *a).map(|a| ((*we, a))))
                .collect()
        };
        py.allow_threads(|| self.sim.write().unwrap().advance(1.0, actions_to_take));
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
