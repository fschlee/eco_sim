use rand::Rng;

use super::cell::Cell;
use super::position::{Coord, Dir, Position, PositionMap, PositionWalker};
use super::{PhysicalState, World, MAP_HEIGHT, MAP_WIDTH};
use crate::agent::AgentSystem;
use crate::entity::{EntityManager, Storage, WorldEntity};
use crate::Prob;
use std::iter::Filter;

use finder::{find_helper, Finder};
use std::ops::Try;

pub trait Observation: Clone {
    type B: Observation;
    type CellType: Cell;
    type Iter<'a>: Iterator<Item = (Position, Option<&'a Self::CellType>)>;
    fn borrow<'a>(&'a self) -> Self::B;
    fn find_closest<'a, F: Fn(&WorldEntity, &World<Self::CellType>) -> bool>(
        &'a self,
        starting_point: Position,
        predicate: F,
    ) -> Finder<'a, F, Self::CellType>;
    fn entities_at(&self, position: Position) -> &[WorldEntity];
    fn cell_at(&self, pos: Position) -> Option<&Self::CellType>;
    fn iter<'a>(&'a self) -> Self::Iter<'a>;
    fn observed_physical_state(&self, entity: &WorldEntity) -> Option<&PhysicalState>;
    fn observed_position(&self, entity: &WorldEntity) -> Option<Position>;
    fn is_observed(&self, pos: &Position) -> bool;
    fn known_can_pass(&self, entity: &WorldEntity, position: Position) -> Option<bool>;
    fn can_pass_prob(&self, entity: &WorldEntity, position: Position) -> Prob {
        match self.known_can_pass(entity, position) {
            Some(true) => 1.0,
            Some(false) => 0.0,
            None => entity.e_type().pass_rate(),
        }
    }
    fn into_expected<C: Cell>(
        &self,
        _filler: impl Fn(Position) -> C,
        rng: impl Rng,
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
}

impl<'b, C: Cell> Observation for &'b World<C> {
    type B = &'b World<C>;
    type CellType = C;
    type Iter<'a> = impl Iterator<Item = (Position, Option<&'a Self::CellType>)>;

    fn borrow<'a>(&'a self) -> Self {
        self.clone()
    }
    fn find_closest<'a, F: Fn(&WorldEntity, &World<Self::CellType>) -> bool>(
        &'a self,
        starting_point: Position,
        predicate: F,
    ) -> Finder<'a, F, C> {
        find_helper(
            EntityWalker::new(
                self,
                starting_point,
                std::cmp::max(MAP_HEIGHT, MAP_WIDTH) as Coord,
            ),
            predicate,
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
    fn iter<'a>(&'a self) -> Self::Iter<'a> {
        self.iter_cells().map(|(p, c)| (p, Some(c)))
    }
    fn is_observed(&self, _pos: &Position) -> bool {
        true
    }
    fn cell_at(&self, pos: Position) -> Option<&Self::CellType> {
        Some(&self[pos])
    }
}

impl<C: Cell> std::ops::Index<Position> for &World<C> {
    type Output = C;

    fn index(&self, index: Position) -> &Self::Output {
        &self.cells[index.y as usize][index.x as usize]
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
impl<'b, C: Cell + 'b> Observation for RadiusObservation<'b, C> {
    type B = RadiusObservation<'b, C>;
    type CellType = C;
    type Iter<'a> = impl Iterator<Item = (Position, Option<&'a Self::CellType>)>;
    fn borrow<'a>(&'a self) -> Self {
        self.clone()
    }
    fn find_closest<'a, F: Fn(&WorldEntity, &World<Self::CellType>) -> bool>(
        &'a self,
        starting_point: Position,
        predicate: F,
    ) -> Finder<'a, F, C> {
        find_helper(
            EntityWalker::new(self.world, starting_point, self.radius),
            predicate,
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
    fn iter<'a>(&'a self) -> Self::Iter<'a> {
        self.world.iter_cells().map(move |(p, c)| {
            if self.center.distance(&p) <= self.radius {
                (p, Some(c))
            } else {
                (p, None)
            }
        })
    }
    fn is_observed(&self, pos: &Position) -> bool {
        self.center.distance(pos) <= self.radius
    }
    fn cell_at(&self, pos: Position) -> Option<&Self::CellType> {
        if self.is_observed(&pos) {
            Some(&self.world[pos])
        } else {
            None
        }
    }
}

struct EntityWalker<'a, C: Cell> {
    position_walker: PositionWalker,
    pub(self) world: &'a World<C>,
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

    fn try_fold<B, F: FnMut(B, Self::Item) -> R, R: Try<Ok = B>>(
        &mut self,
        init: B,
        mut f: F,
    ) -> R {
        let between = self.current[self.subindex..]
            .iter()
            .try_fold(init, |i, e| f(i, (*e, self.current_pos)))?;
        let Self {
            ref world,
            ref mut position_walker,
            ..
        } = self;
        position_walker.try_fold(between, |init, pos| {
            world
                .entities_at(pos)
                .iter()
                .try_fold(init, |i, e| f(i, (*e, pos)))
        })
    }
}

mod finder {
    use super::*;
    pub type Finder<'a, F, C> = impl Iterator<Item = (WorldEntity, Position)> + 'a;

    pub(super) fn find_helper<'a, C: 'a + Cell, F: 'a + Fn(&WorldEntity, &World<C>) -> bool>(
        walker: EntityWalker<'a, C>,
        predicate: F,
    ) -> Finder<'a, F, C> {
        let world = walker.world;
        walker.filter(move |(e, _)| predicate(e, world))
    }
}
