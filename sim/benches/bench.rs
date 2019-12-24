#![feature(const_in_array_repeat_expressions)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use eco_sim::agent::{AgentSystem, MentalState};
use eco_sim::entity::{EntityManager, WorldEntity};
use eco_sim::position::Position;
use eco_sim::world::{Observation, Occupancy, World};
use rand::{thread_rng, Rng, RngCore};

fn sim(c: &mut Criterion) {
    let mut sim = eco_sim::SimState::new_with_seed(1.0, 0);
    /*
    c.bench_function("array of arrays", |b| {
        b.iter(|| {
            let mut arr = [[0; 11]; 11];
            let s = black_box(0);
            for p in Position::iter() {
                arr[p.y as usize][p.x as usize] += s;
            }
            arr[black_box(0)][black_box(0)]
        })
    });
    c.bench_function("array", |b| {
        b.iter(|| {
            let mut arr = [0; 11 *  11];
            let s = black_box(0);
            for p in Position::iter() {
                arr[p.idx()] += s;
            }
            arr[black_box(0)]
        })
    });
    c.bench_function("vec", |b| {
        b.iter(|| {
            let mut arr = vec![0; 11 *  11];
            let s = black_box(0);
            for p in Position::iter() {
                arr[p.idx()] += s;
            }
            arr[black_box(0)]
        })
    });
    */
    c.bench_function("sim", |b| {
        b.iter(|| {
            sim.advance(1.0);
        })
    });
    let agents: Vec<_> = sim
        .agent_system
        .mental_states
        .into_iter()
        .map(|ms| ms.id)
        .collect();
    /*
    c.bench_function("threat map 1",    |b|{
        b.iter(||{
            let AgentSystem { ref mental_states, ref estimator_map, .. } = &sim.agent_system;
            for ms in mental_states.into_iter() {
                ms.threat_map(&&**ms.world_model.as_ref().unwrap(), estimator_map.get(ms.id.into()).unwrap());
            }
        })
    }); */
    c.bench_function("threat map 1", |b| {
        b.iter(|| {
            for a in &agents {
                sim.threat_map(a);
            }
        })
    });
    c.bench_function("pathing with observation", |b| {
        b.iter(|| {
            for agent in &agents {
                let observation = sim.world.observe_in_radius(agent, 6);
                let start = sim.world.positions.get(agent).unwrap();
                for goal in Position::iter() {
                    observation.path_as(*start, goal, agent);
                }
            }
        })
    });
    c.bench_function("pathing on world", |b| {
        b.iter(|| {
            for agent in &agents {
                let observation = &sim.world;
                let start = sim.world.positions.get(agent).unwrap();
                for goal in Position::iter() {
                    observation.path_as(*start, goal, agent);
                }
            }
        })
    });
}

criterion_group!(benches, sim);
criterion_main!(benches);
