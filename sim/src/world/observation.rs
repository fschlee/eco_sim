use rand::Rng;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::cell::Cell;
use super::position::{Coord, Dir, Position, PositionMap};
use super::{PhysicalState, World, MAP_HEIGHT, MAP_WIDTH};
use crate::agent::AgentSystem;
use crate::entity::{EntityManager, Storage, WorldEntity};
use crate::Prob;

type Cost = i32;
#[derive(Eq, PartialEq, Clone, Debug)]
pub struct PathNode {
    pub pos: Position,
    pub exp_cost: Cost,
}

impl Ord for PathNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other.exp_cost.cmp(&self.exp_cost)
    }
}
impl PartialOrd for PathNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.exp_cost.partial_cmp(&self.exp_cost)
    }
}
pub trait Observation: Clone {
    type B: Observation;
    type CellType: Cell;
    fn borrow<'a>(&'a self) -> Self::B;
    fn find_closest<'a>(
        &'a self,
        starting_point: Position,
        predicate: impl Fn(&WorldEntity, &World<Self::CellType>) -> bool + 'a,
    ) -> Box<dyn Iterator<Item = (WorldEntity, Position)> + 'a>;
    fn known_can_pass(&self, entity: &WorldEntity, position: Position) -> Option<bool>;
    fn can_pass_prob(&self, entity: &WorldEntity, position: Position) -> Prob {
        match self.known_can_pass(entity, position) {
            Some(true) => 1.0,
            Some(false) => 0.0,
            None => entity.e_type().pass_rate(),
        }
    }
    fn entities_at(&self, position: Position) -> &[WorldEntity];
    fn observed_physical_state(&self, entity: &WorldEntity) -> Option<&PhysicalState>;
    fn observed_position(&self, entity: &WorldEntity) -> Option<Position>;
    fn into_expected<C: Cell>(
        &self,
        filler: impl Fn(Position) -> C,
        mut rng: impl Rng,
    ) -> (EntityManager, World<C>, AgentSystem) {
        let mut cells = C::empty_init();
        let mut entity_manager = EntityManager::default();
        let mut physical_states = Storage::new();
        let mut positions = Storage::new();
        let mut agents = Vec::new();
        let mut insert_cell = |e, Position { x, y }| {
            cells[y as usize][x as usize].push(e);
        };
        let pos = Position {
            x: (MAP_WIDTH / 2) as Coord,
            y: (MAP_HEIGHT / 2) as Coord,
        };
        let radius = std::cmp::max(MAP_HEIGHT, MAP_WIDTH) as u32;
        for (e, p) in self.find_closest(pos, |_, _| true) {
            if let Ok(ent) = entity_manager.put(e) {
                let new_e = WorldEntity::new(ent, e.e_type());
                positions.insert(&new_e, p);
                insert_cell(new_e, p);
                if let Some(ps) = e.e_type().typical_physical_state() {
                    physical_states.insert(&new_e, ps);
                    agents.push(new_e);
                }
            }
        }
        let world = World {
            cells,
            physical_states,
            positions,
            events: Vec::new(),
            move_list: Vec::new(),
        };
        let agent_system = AgentSystem::init(agents, &world, false, false, rng);
        (entity_manager, world, agent_system)
    }
    fn path_as(&self, start: Position, goal: Position, entity: &WorldEntity) -> Option<Vec<Dir>> {
        if self.known_can_pass(entity, goal) == Some(false) {
            return None;
        }
        let mut came_from: PositionMap<Dir> = PositionMap::new();
        let mut cost = PositionMap::new();
        let mut queue = BinaryHeap::new();
        cost.insert(start, 0 as Cost);
        queue.push(PathNode {
            pos: start,
            exp_cost: start.distance(&goal) as Cost,
        });
        while let Some(PathNode { pos, exp_cost }) = queue.pop() {
            let base_cost = *cost.get(&pos).unwrap();
            for d in pos.dir(&goal).closest() {
                if let Some(n) = pos.step(*d) {
                    if self.known_can_pass(entity, n) == Some(false) {
                        continue;
                    }
                    if n == goal {
                        let mut v = vec![*d];
                        let mut current = pos;
                        while let Some(from) = came_from.get(&current) {
                            v.push(*from);
                            current = current.step(from.opposite()).unwrap();
                        }
                        v.reverse();
                        return Some(v);
                    } else {
                        let mut insert = true;
                        if let Some(c) = cost.get(&n) {
                            if *c <= base_cost + 1 {
                                insert = false;
                            }
                        }
                        if insert {
                            cost.insert(n, base_cost + 1);
                            came_from.insert(n, *d);
                            queue.push(PathNode {
                                pos: n,
                                exp_cost: base_cost + 1 + n.distance(&goal) as Cost,
                            })
                        }
                    }
                }
            }
        }
        None
    }
    fn iter<'a>(&'a self) -> Box<dyn Iterator<Item = (Position, Option<&'a Self::CellType>)> + 'a>;
    fn is_observed(&self, pos: &Position) -> bool;
}

impl<'b, C: Cell> Observation for &'b World<C> {
    type B = &'b World<C>;
    type CellType = C;
    fn borrow<'a>(&'a self) -> Self {
        self.clone()
    }
    fn find_closest<'a>(
        &'a self,
        starting_point: Position,
        predicate: impl Fn(&WorldEntity, Self) -> bool + 'a,
    ) -> Box<dyn Iterator<Item = (WorldEntity, Position)> + 'a> {
        Box::new(
            EntityWalker::new(
                self,
                starting_point,
                std::cmp::max(MAP_HEIGHT, MAP_WIDTH) as Coord,
            )
            .filter(move |(e, p)| predicate(e, self)),
        )
    }
    fn known_can_pass(&self, entity: &WorldEntity, position: Position) -> Option<bool> {
        Some(self.can_pass(entity, position))
    }
    fn entities_at(&self, position: Position) -> &[WorldEntity] {
        World::entities_at(*self, position)
    }
    fn observed_physical_state(&self, entity: &WorldEntity) -> Option<&PhysicalState> {
        self.get_physical_state(entity)
    }
    fn observed_position(&self, entity: &WorldEntity) -> Option<Position> {
        self.positions.get(entity).copied()
    }
    fn iter<'a>(&'a self) -> Box<dyn Iterator<Item = (Position, Option<&'a Self::CellType>)> + 'a> {
        Box::new(self.iter_cells().map(|(p, c)| (p, Some(c))))
    }
    fn is_observed(&self, pos: &Position) -> bool {
        true
    }
}
#[derive(Clone, Debug)]
pub struct RadiusObservation<'a, C: Cell> {
    radius: Coord,
    center: Position,
    world: &'a World<C>,
}
impl<'a, C: Cell> RadiusObservation<'a, C> {
    pub fn new(radius: Coord, center: Position, world: &'a World<C>) -> Self {
        Self {
            radius,
            center,
            world,
        }
    }
}
impl<'b, C: Cell> Observation for RadiusObservation<'b, C> {
    type B = RadiusObservation<'b, C>;
    type CellType = C;
    fn borrow<'a>(&'a self) -> Self {
        self.clone()
    }
    fn find_closest<'a>(
        &'a self,
        starting_point: Position,
        predicate: impl Fn(&WorldEntity, &World<C>) -> bool + 'a,
    ) -> Box<dyn Iterator<Item = (WorldEntity, Position)> + 'a> {
        Box::new(
            EntityWalker::new(self.world, starting_point, self.radius).filter(move |(e, p)| {
                self.center.distance(p) <= self.radius && predicate(e, self.world)
            }),
        )
    }
    fn known_can_pass(&self, entity: &WorldEntity, position: Position) -> Option<bool> {
        if self.center.distance(&position) > self.radius {
            return None;
        }
        Some(self.world.can_pass(entity, position))
    }
    fn observed_physical_state(&self, entity: &WorldEntity) -> Option<&PhysicalState> {
        if let Some(pos) = self.world.positions.get(entity) {
            if pos.distance(&self.center) <= self.radius {
                return self.world.get_physical_state(entity);
            }
        }
        None
    }
    fn observed_position(&self, entity: &WorldEntity) -> Option<Position> {
        if let Some(pos) = self.world.positions.get(entity) {
            if pos.distance(&self.center) <= self.radius {
                return Some(*pos);
            }
        }
        None
    }
    fn entities_at(&self, position: Position) -> &[WorldEntity] {
        if self.center.distance(&position) <= self.radius {
            return self.world.entities_at(position);
        }
        &[]
    }
    fn iter<'a>(&'a self) -> Box<dyn Iterator<Item = (Position, Option<&'a Self::CellType>)> + 'a> {
        Box::new(self.world.iter_cells().map(move |(p, c)| {
            if self.center.distance(&p) <= self.radius {
                (p, Some(c))
            } else {
                (p, None)
            }
        }))
    }
    fn is_observed(&self, pos: &Position) -> bool {
        self.center.distance(pos) <= self.radius
    }
}
struct EntityWalker<'a, C: Cell> {
    position_walker: PositionWalker,
    world: &'a World<C>,
    current: &'a [WorldEntity],
    current_pos: Position,
    subindex: usize,
}
impl<'a, C: Cell> EntityWalker<'a, C> {
    pub fn new(world: &'a World<C>, center: Position, max_radius: Coord) -> Self {
        let position_walker = PositionWalker::new(center, max_radius);
        Self {
            position_walker,
            world,
            current: world.entities_at(center),
            current_pos: center.clone(),
            subindex: 0,
        }
    }
}
impl<'a, C: Cell> Iterator for EntityWalker<'a, C> {
    type Item = (WorldEntity, Position);
    fn next(&mut self) -> Option<Self::Item> {
        if self.subindex < self.current.len() {
            let item = self.current[self.subindex].clone();
            self.subindex += 1;
            return Some((item, self.current_pos));
        }
        if let Some(pos) = self.position_walker.next() {
            self.current = self.world.entities_at(pos);
            self.current_pos = pos;
            self.subindex = 0;
            return self.next();
        }
        None
    }
}

pub struct PositionWalker {
    center: Position,
    current_pos: Option<Position>,
    radius: Coord,
    max_radius: Coord,
    delta: Coord,
}
impl PositionWalker {
    pub fn new(center: Position, max_radius: Coord) -> Self {
        Self {
            center,
            current_pos: Some(center.clone()),
            radius: 0,
            max_radius,
            delta: 0,
        }
    }
    pub fn empty() -> Self {
        Self {
            center: Position { x: 0, y: 0 },
            current_pos: None,
            radius: 1,
            max_radius: 0,
            delta: 0,
        }
    }
    pub fn get_current(&self) -> Option<Position> {
        self.current_pos
    }
}
impl Iterator for PositionWalker {
    type Item = Position;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(current) = self.current_pos {
            if current.x > self.center.x {
                let x = self.center.x + self.delta - self.radius;
                if x >= 0 {
                    self.current_pos = Some(Position { x: x, y: current.y });
                    return Some(current);
                }
            }
            if current.y > self.center.y {
                let x = self.center.x + self.radius - self.delta;
                let y = self.center.y - self.delta;
                if x >= 0 && y >= 0 {
                    self.current_pos = Some(Position { x, y });
                    return Some(current);
                }
            }
            if self.delta < self.radius {
                self.delta += 1;
                self.current_pos = Some(Position {
                    x: self.center.x + self.radius - self.delta,
                    y: self.center.y + self.delta,
                });
                return Some(current);
            }
            let v = self.center.x + self.center.y;
            if self.radius < self.max_radius
                && (self.radius < v || self.radius < (MAP_WIDTH + MAP_HEIGHT) as Coord - v)
            {
                self.radius += 1;
                self.delta = 0;
                self.current_pos = Some(Position {
                    x: self.center.x + self.radius,
                    y: self.center.y,
                });
                return Some(current);
            }
        }
        None
    }
}
