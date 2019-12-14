
use eco_sim::world::{World, Observation, Occupancy};
use eco_sim::position::Position;
use eco_sim::entity::{WorldEntity, EntityManager};
use rand::{thread_rng, Rng, RngCore};
use criterion::{Criterion, criterion_group, criterion_main, black_box};


fn sim(c: &mut Criterion) {
    let mut sim = eco_sim::SimState::new_with_seed(1.0, 0);
    c.bench_function("sim", |b| b.iter(|| {
        sim.advance(1.0);
    }));
    let agents : Vec<_> = sim.agent_system.mental_states.into_iter().map(|ms| ms.id).collect();
    c.bench_function("pathing with observation", |b| b.iter(|| {
        for agent in &agents {
            let observation = sim.world.observe_in_radius(agent,6);
            let start = sim.world.positions.get(agent).unwrap();
            for goal in Position::iter() {
                observation.path_as(*start, goal, agent);
            }
        }
    }));
    c.bench_function("pathing on world", |b| b.iter(|| {
        for agent in &agents {
            let observation = &sim.world;
            let start = sim.world.positions.get(agent).unwrap();
            for goal in Position::iter() {
                observation.path_as(*start, goal, agent);
            }
        }
    }));
}

criterion_group!(benches, sim);
criterion_main!(benches);