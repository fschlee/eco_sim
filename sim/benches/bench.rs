
use eco_sim::world::{World, Observation, Position, Occupancy};
use eco_sim::entity::{WorldEntity, EntityManager};
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;
use criterion::{Criterion, criterion_group, criterion_main, black_box};


fn sim(c: &mut Criterion) {
    let mut sim = eco_sim::SimState::new(1.0);
    c.bench_function("sim", |b| b.iter(|| {
        sim.advance(1.0);
    }));
    c.bench_function("pathing with observation", |b| b.iter(|| {
        for agent in &sim.agent_system.agents {
            let observation = sim.world.observe_in_radius(agent,6);
            let start = sim.world.positions.get(agent).unwrap();
            for goal in Position::iter() {
                observation.path_as(*start, goal, agent);
            }
        }
    }));
    c.bench_function("pathing on world", |b| b.iter(|| {
        for agent in &sim.agent_system.agents {
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