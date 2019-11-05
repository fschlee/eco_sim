use log::error;
use std::ops::Range;
use rand::{Rng};

use super::entity::*;
use super::agent::AgentSystem;
use crate::{MentalState, Hunger};
use std::collections::HashMap;
use std::collections::hash_map::Entry;


#[derive(Ord, PartialOrd, Eq, PartialEq, Copy, Clone, Debug)]
pub enum EntityType {
    Rock,
    Tree,
    Grass,
    Clover,
    Rabbit,
    Deer,
    Wolf,
}
impl EntityType {
    pub fn can_eat(&self, other: &Self) -> bool {
        use EntityType::*;
        match (self, other) {
            (Rabbit, Grass) => true,
            (Rabbit, Clover) => true,
            (Deer, Grass) => true,
            (Deer, Clover) => true,
            (Deer, Tree) => true,
            (Wolf, Rabbit) => true,
            (Wolf, Deer) => true,
            _ => false,
        }
    }
    pub fn can_pass(&self, other: &Self) -> bool {
        use EntityType::*;
        match (self, other) {
            (Rabbit, Grass) => true,
            (Rabbit, Clover) => true,
            (Rabbit, Tree) => true,
            (Deer, Grass) => true,
            (Deer, Clover) => true,
            (Deer, Tree) => true,
            (Wolf, Grass) => true,
            (Wolf, Clover) => true,
            (Wolf, Tree) => true,
            (Wolf, Rabbit) => true,
            (Wolf, Deer) => true,
            _ => false,
        }
    }
    pub fn typical_physical_state(&self)  -> Option<PhysicalState> {
        use EntityType::*;
        match self {
            Rabbit=> Some(
                PhysicalState{
                    health: Health(50.0),
                    meat: Meat(40.0),
                    attack: None,
                    satiation: Satiation(10.0)
                }),
            Deer => Some(
                PhysicalState{
                    health: Health(300.0),
                    meat: Meat(100.0),
                    attack: None,
                    satiation: Satiation(10.0)
                }),
            Wolf => Some(
                PhysicalState{
                    health: Health(300.0),
                    meat: Meat(100.0),
                    attack: Some(Attack(60.0)),
                    satiation: Satiation(10.0)
                }),
            _ => None
        }
    }
}
impl Default for EntityType {
    fn default() -> Self {
        Self::Rock
    }
}
pub const ENTITY_TYPES : [EntityType; ENTITY_TYPE_COUNT] = [
    EntityType::Rock,
    EntityType::Tree,
    EntityType::Grass,
    EntityType::Clover,
    EntityType::Rabbit,
    EntityType::Deer,
    EntityType::Wolf,
];

pub fn et_idx(et : EntityType) -> usize {
    match et {
        EntityType::Rock => 0,
        EntityType::Tree => 1,
        EntityType::Grass => 2,
        EntityType::Clover => 3,
        EntityType::Rabbit => 4,
        EntityType::Deer => 5,
        EntityType::Wolf => 6,
    }
}

pub const ENTITY_TYPE_COUNT: usize = 7;

#[derive(Ord, PartialOrd, Eq, PartialEq, Copy, Clone, Debug)]
pub enum Action {
    Move(Position),
    Eat(Entity),
    Attack(Entity),
}


#[derive(PartialOrd, PartialEq, Copy, Clone, Debug)]
pub struct Health(pub f32);

impl Health {
    pub fn suffer(&mut self, attack: Attack) {
        self.0 -= attack.0
    }
}

#[derive(PartialOrd, PartialEq, Copy, Clone, Debug)]
pub struct Meat(pub f32);

#[derive(PartialOrd, PartialEq, Copy, Clone, Debug)]
pub struct Attack(pub f32);

#[derive(PartialOrd, PartialEq, Copy, Clone, Debug)]
pub struct Satiation(pub f32);

#[derive(Clone, Debug)]
pub struct PhysicalState {
    pub health: Health,
    pub meat: Meat,
    pub attack: Option<Attack>,
    pub satiation: Satiation,
}
impl PhysicalState {
    pub fn is_dead(&self) -> bool {
        self.health.0 <= 0.0
    }
}

pub const MAP_HEIGHT: usize = 11;
pub const MAP_WIDTH: usize = 11;

#[derive(Ord, PartialOrd, Eq, PartialEq, Hash, Copy, Clone, Debug)]
pub struct Position {
    pub x: u32,
    pub y: u32,
}

impl Position {
    pub fn is_neighbour(&self, other: &Position) -> bool {
        self != other
            && ((self.x as i64) - (other.x as i64)).abs() <= 1
            && ((self.y as i64) - (other.y as i64)).abs() <= 1
    }
    pub fn neighbours(&self) -> impl IntoIterator<Item=Position> {
        let mut neighbours = vec![
            Position{ x: self.x + 1, y: self.y },
            Position{ x: self.x, y: self.y + 1 },
        ];
        if self.x > 0 {
            neighbours.push(Position{x: self.x -1, y: self.y});
        }
        if self.y > 0 {
            neighbours.push(Position{x: self.x, y: self.y -1});
        }
        neighbours
    }
    pub fn distance(&self, other: & Position) -> u32 {
        ((self.x as i32 - other.x as i32).abs() + (self.y as i32 - other.y as i32).abs()) as u32
    }
    pub fn within_bounds(& self) -> bool {
        self.x < MAP_WIDTH as u32 && self.y < MAP_HEIGHT as u32
    }
}

pub struct PositionMap<T> {
    map : HashMap<Position, T>
}
impl<T> PositionMap<T> {
    pub fn new() -> Self {
        Self{map: HashMap::new()}
    }
    pub fn get(&self, k: &Position) -> Option<&T> {
        self.map.get(k)
    }
    pub fn insert(&mut self, k: Position, v: T) -> Option<T> {
        self.map.insert(k, v)
    }
    pub fn entry(&mut self, k: Position) -> Entry<Position, T> {
        self.map.entry(k)
    }

}
pub type ViewData = Option<Vec<EntityType>>;

#[derive(Clone, Debug)]
pub enum Occupancy {
    Empty,
    Filled(Vec<Entity>),
    ExpectedFilled(Vec<Entity>),
    ExpectedEmpty,
    Unknown,
}

#[derive(Clone, Debug, Default)]
pub struct World {
    cells: [[Option<Vec<Entity>>; MAP_WIDTH]; MAP_HEIGHT],
    pub entity_types: Storage<EntityType>,
    pub physical_states: Storage<PhysicalState>,
    pub positions: Storage<Position>,
}

impl World {
    pub fn init(mut rng: impl Rng, entity_manager: &mut EntityManager) -> (Self, Vec<Entity>) {
        let mut cells: [[Option<Vec<Entity>>; MAP_WIDTH]; MAP_HEIGHT] = Default::default();
        let mut entity_types = Storage::new();
        let mut physical_states =  Storage::new();
        let mut positions = Storage::new();
        let mut inserter = |entity_type: EntityType, count| {
            let mut c = 0;
            let mut inserted = Vec::new();
            while c < count {
                let x = rng.gen_range(0, MAP_WIDTH);
                let y = rng.gen_range(0, MAP_HEIGHT);
                if let None = cells[y][x] {
                    let entity = entity_manager.fresh(entity_type);
                    positions.insert(&entity, Position{x : x as u32, y: y as u32});
                    entity_types.insert(&entity, entity_type);
                    if let Some(phys_state) = entity_type.typical_physical_state() {
                        physical_states.insert(&entity, phys_state);
                    }
                    cells[y][x] = Some(vec![entity]);
                    c+= 1;
                    inserted.push(entity);
                }
            }
            inserted
        };
        inserter(EntityType::Tree, MAP_WIDTH * MAP_HEIGHT / 8);
        inserter(EntityType::Rock, MAP_WIDTH * MAP_HEIGHT / 16);
        inserter(EntityType::Grass, MAP_WIDTH * MAP_HEIGHT / 6);
        inserter(EntityType::Clover, MAP_WIDTH * MAP_HEIGHT / 6);
        let mut agents = inserter(EntityType::Rabbit, 2);
        agents.append(&mut inserter(EntityType::Wolf, 1));
        agents.append( & mut inserter(EntityType::Deer, 2));

        (
            Self {
                cells,
                entity_types,
                physical_states,
                positions,
            },
            agents
        )
    }
    pub fn respawn(&mut self, entity: &Entity, mental_state: & mut  MentalState, entity_manager: & mut EntityManager) -> Entity {
        let new_e = *entity;

        if let Some(et) = self.entity_types.get(entity) {
            let mut random_pos = || {
                let x = mental_state.rng.gen_range(0, MAP_WIDTH);
                let y = mental_state.rng.gen_range(0, MAP_HEIGHT);
                Position {x: x as u32, y : y as u32}
            };
            let mut pos = random_pos();
            while !self.type_can_pass(et, pos) {
                pos = random_pos();
            }
            self.positions.insert(&new_e, pos);
            mental_state.respawn_as(&new_e);
            if let Some(phys_state) = et.typical_physical_state() {
                self.physical_states.insert(&new_e, phys_state);
            }
            self.entity_types.insert(&new_e, et.clone());
        }
        new_e
    }

    pub fn act(&mut self, entity: &Entity, action: Action) -> Result<(), &str> {
        if let Some(own_pos) = self.positions.get(entity) {
            match action {
                Action::Move(pos) => {
                    if own_pos.is_neighbour(&pos) && self.can_pass(entity, pos) {
                        self.move_unchecked(entity, pos);
                        Ok(())
                    } else {
                        Err("invalid move")
                    }
                }
                Action::Eat(target) => {
                    if self.positions.get(&target) == Some(own_pos) {
                        let own_type = self
                            .entity_types
                            .get(entity)
                            .ok_or("unknown eater entity")?;
                        let target_type = self
                            .entity_types
                            .get(&target)
                            .ok_or("unknown food entity")?;
                        if own_type.can_eat(target_type) {
                            self.eat_unchecked(entity, &target);
                            Ok(())
                        } else {
                            Err("entity can't eat that")
                        }
                    } else {
                        Err("can only eat things in the same tile")
                    }
                }
                Action::Attack(opponent) => {
                    if self.positions.get(&opponent) == Some(own_pos) {
                        if let Some(attack) = self.physical_states.get(entity).and_then(|phys| phys.attack).clone() {
                            if let Some(phys_target) = self.physical_states.get_mut(&opponent) {
                                phys_target.health.suffer(attack);
                                Ok(())
                            } else {
                                Err("opponent has no physical state")
                            }
                        } else {
                            Err("entity incapable of attacking")
                        }


                    }else {
                        Err("cannot attack targets in the same tile")
                    }
                }
            }
        } else {
            error!(
                "Entity {:?} has no known position but tries to do {:?}",
                entity, action
            );
            Err("Entity with unknown position acting")
        }
    }
    fn can_pass(&self, entity: &Entity, position: Position) -> bool {
        if !position.within_bounds() {
            return false
        }
        if let Some(mover) = self.entity_types.get(entity) {
            return self.type_can_pass(mover, position);
        }
        false
    }
    pub fn type_can_pass(&self, entity_type: & EntityType, position: Position) -> bool {
        if !position.within_bounds() {
            return false
        }
        self.entities_at(position)
                .iter()
                .all(|e| match self.entity_types.get(e) {
                    Some(e) => entity_type.can_pass(e),
                    None => true,
                })
    }
    pub fn observe_as(&self, entity: &Entity) -> impl Observation + '_ {
        let radius = std::cmp::max(MAP_HEIGHT, MAP_WIDTH) as u32;
        self.observe_in_radius(entity, radius)
    }
    pub fn observe_in_radius(&self, entity: &Entity, radius: u32) -> impl Observation +'_ {
        let pos = match self.positions.get(entity) {
            Some(pos) => pos.clone(),
            None => Position { x: (MAP_WIDTH / 2) as u32, y: (MAP_HEIGHT / 2) as u32 },
        };
        RadiusObservation::new(radius, pos, self)

    }
    pub fn get_physical_state(&self, entity: &Entity) -> Option<&PhysicalState> {
        self.physical_states.get(entity)
    }
    pub fn advance(&mut self) {}
    pub fn entities_at(&self, position: Position) -> &[Entity] {
        let x = position.x as usize;
        let y = position.y as usize;
        if x < MAP_WIDTH && y < MAP_HEIGHT {
            if let Some(ents) = &self.cells[y][x] {
                return ents;
            }
        }
        return &[];
    }
    fn entities_at_mut(&mut self, position: Position) -> &mut Vec<Entity> {
        let x = position.x as usize;
        let y = position.y as usize;
        let entry = &mut self.cells[y][x];
        entry.get_or_insert(Vec::new())
    }
    fn move_unchecked(&mut self, entity: &Entity, new_position: Position) {
        if let Some((_, old_pos)) = self.positions.insert(entity, new_position) {
            self.entities_at_mut(old_pos).retain(|e| e != entity);
        }
        self.entities_at_mut(new_position).push(entity.clone());
    }
    fn eat_unchecked(&mut self, eater: &Entity, eaten: &Entity) {
        let decrement = 5.0;
        self.physical_states
            .get_mut(eater)
            .map(|ps| ps.satiation.0 += decrement);
        let remove = {
            if let Some(phys) = self.physical_states.get_mut(eaten) {
                phys.meat.0 -= decrement;
                phys.meat.0 <= 0.0
            }
            else {
                false
            }
        };
        if remove {
            self.physical_states.remove(eaten);
            self.positions.remove(eaten);
        }
    }
    pub fn get_view(
        &self,
        x_range: Range<usize>,
        y_range: Range<usize>,
    ) -> impl Iterator<Item = (usize, usize, ViewData)> + '_ {
        self.cells[y_range.clone()]
            .iter()
            .zip(y_range.clone())
            .flat_map(move |(v, y)| {
                v[x_range.clone()]
                    .iter()
                    .zip(x_range.clone())
                    .map(move |(oes, x)| {
                        let res = match oes {
                            Some(es) => Some(
                                es.iter()
                                    .filter_map(|e| self.entity_types.get(e).cloned())
                                    .collect::<Vec<_>>(),
                            ),
                            None => None,
                        };
                        (x, y.clone(), res)
                    })
            })
    }

}

pub trait Observation: Clone {
    type B: Observation;
    fn borrow<'a>(& 'a self) -> Self::B;
    fn find_closest<'a>(& 'a self, starting_point: Position, predicate: impl Fn(&Entity, &World) -> bool + 'a) -> Box<dyn Iterator<Item=(Entity, Position)> + 'a>;
    fn known_can_pass(&self, entity: &Entity, position: Position) -> Option<bool>;
    fn get_type(& self, entity: & Entity) -> Option<EntityType>;
    fn entities_at(&self, position: Position) -> &[Entity];
    fn observed_physical_state(&self, entity: &Entity) -> Option<&PhysicalState>;
    fn observed_position(& self, entity: &Entity) -> Option<Position>;
    fn into_expected(& self, filler: impl Fn(Position) -> Option<Vec<EntityType>>, mut rng: impl Rng) -> (EntityManager, World, AgentSystem) {

        let mut cells: [[Option<Vec<Entity>>; MAP_WIDTH]; MAP_HEIGHT] = Default::default();
        let mut entity_manager = EntityManager::default();
        let mut entity_types = Storage::new();
        let mut physical_states = Storage::new();
        let mut positions = Storage::new();
        let mut agents = Vec::new();
        let mut insert_cell = |e, Position {x, y}| {
            cells[y as usize][x as usize].get_or_insert(Vec::new()).push(e);
        };
        let pos = Position { x: (MAP_WIDTH / 2) as u32, y: (MAP_HEIGHT / 2) as u32 };
        let radius = std::cmp::max(MAP_HEIGHT, MAP_WIDTH) as u32;
        for (e, p) in self.find_closest(pos, |e, w| true) {
            if let Ok(new_e) = entity_manager.put(e) {
                positions.insert(&new_e, p);
                insert_cell(new_e, p);
                if let Some(tp) = self.get_type(&e) {
                    entity_types.insert(&new_e, tp);
                    if let Some(ps) = tp.typical_physical_state() {
                        physical_states.insert(&new_e, ps);
                        agents.push(new_e);
                    }
                }
            }
        }
        let world = World{cells, entity_types, physical_states, positions };
        let agent_system = AgentSystem::init(agents, &world, false, rng);
        (entity_manager, world, agent_system)

    }
}

impl<'b> Observation for & 'b World {
    type B = & 'b World;
    fn borrow<'a>(& 'a self) -> Self {
        self.clone()
    }
    fn find_closest<'a>(& 'a self, starting_point: Position, predicate: impl Fn(&Entity, Self) -> bool + 'a) -> Box<dyn Iterator<Item=(Entity, Position)> + 'a> {
        Box::new(EntityWalker::new(self, starting_point, std::cmp::max(MAP_HEIGHT, MAP_WIDTH) as u32).filter(move |(e, p)| predicate(e, self)))
    }
    fn known_can_pass(&self, entity: &Entity, position: Position) -> Option<bool> {
        Some(self.can_pass(entity, position))
    }
    fn get_type(& self, entity: & Entity) -> Option<EntityType> {
        self.entity_types.get(entity).copied()
    }
    fn entities_at(&self, position: Position) -> &[Entity] {
        let x = position.x as usize;
        let y = position.y as usize;
        if x < MAP_WIDTH && y < MAP_HEIGHT {
            if let Some(ents) = &self.cells[y][x] {
                return ents;
            }
        }
        return &[];
    }
    fn observed_physical_state(&self, entity: &Entity) -> Option<&PhysicalState> {
        self.get_physical_state(entity)
    }
    fn observed_position(& self, entity: &Entity) -> Option<Position>{
        self.positions.get(entity).copied()
    }

}
#[derive(Clone, Debug)]
pub struct RadiusObservation<'a> {
    radius: u32,
    center: Position,
    world: & 'a World,
}
impl<'a> RadiusObservation<'a> {
    pub fn new(radius: u32, center: Position, world: & 'a World) -> Self {
        Self{ radius, center, world}
    }
}
impl<'b> Observation for RadiusObservation<'b> {
    type B =  RadiusObservation<'b>;
    fn borrow<'a>(& 'a self) -> Self {
        self.clone()
    }
    fn find_closest<'a>(& 'a self, starting_point: Position, predicate: impl Fn(&Entity, &World) -> bool + 'a) -> Box<dyn Iterator<Item=(Entity, Position)> + 'a> {
        Box::new(EntityWalker::new(self.world, starting_point, self.radius).filter(
            move |(e, p)| self.center.distance(p) <= self.radius &&  predicate(e, self.world)))
    }
    fn known_can_pass(&self, entity: &Entity, position: Position) -> Option<bool> {
        if self.center.distance(&position) > self.radius {
            return None;
        }
        Some(self.world.can_pass(entity, position))
    }
    fn get_type(& self, entity: & Entity) -> Option<EntityType> {
        if let Some(pos) = self.world.positions.get(entity) {
            if self.center.distance(pos) <= self.radius {
                return self.world.get_type(entity);
            }
        }
        None
    }
    fn observed_physical_state(&self, entity: &Entity) -> Option<&PhysicalState> {
        if let Some(pos)= self.world.positions.get(entity) {
            if pos.distance(&self.center) <= self.radius {
                return self.world.get_physical_state(entity)
            }
        }
        None
    }
    fn observed_position(&self, entity: &Entity) -> Option<Position> {
        if let Some(pos)= self.world.positions.get(entity) {
            if pos.distance(&self.center) <= self.radius {
                return Some(*pos)
            }
        }
        None
    }
    fn entities_at(&self, position: Position) -> &[Entity] {
        if self.center.distance(&position) <= self.radius {
            return self.world.entities_at(position);
        }
        &[]
    }
}
struct EntityWalker<'a> {
    position_walker: PositionWalker,
    world: &'a World,
    current: &'a [Entity],
    current_pos: Position,
    subindex: usize,
}
impl<'a> EntityWalker<'a> {
    pub fn new(world: &'a World, center: Position, max_radius: u32) -> Self {
        let position_walker = PositionWalker::new(center, max_radius);
        Self{
            position_walker,
            world,
            current: world.entities_at(center),
            current_pos: center.clone(),
            subindex: 0,
        }
    }
}
impl<'a> Iterator for EntityWalker<'a> {
    type Item = (Entity, Position);
    fn next(& mut self) -> Option<Self::Item> {
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
    radius: u32,
    max_radius: u32,
    delta: u32,
}
impl PositionWalker {
    pub fn new(center: Position, max_radius: u32) -> Self {
        Self{
            center,
            current_pos: Some(center.clone()),
            radius: 0,
            max_radius,
            delta: 0,
        }
    }
    pub fn empty() -> Self {
        Self{
            center: Position{x:0, y: 0},
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
    fn next(& mut self) -> Option<Self::Item> {
        if let Some(current) = self.current_pos {

            if current.x > self.center.x {
                let x = self.center.x as i32 + self.delta as i32 - self.radius as i32;
                if x >= 0 {
                    self.current_pos = Some(Position{x: x as u32, y : current.y});
                    return Some(current);
                }

            }
            if current.y > self.center.y {
                let x = self.center.x as i32 + self.radius as i32 - self.delta as i32;
                let y = self.center.y as i32 - self.delta as i32;
                if x >= 0 && y >= 0 {
                    self.current_pos = Some(Position{x: x as u32, y: y as u32 });
                    return Some(current);
                }

            }
            if self.delta < self.radius {
                self.delta += 1;
                self.current_pos = Some(Position { x: self.center.x + self.radius - self.delta, y: self.center.y + self.delta });
                return Some(current);
            }
            let v = self.center.x + self.center.y;
            if self.radius < self.max_radius && (self.radius < v || self.radius < (MAP_WIDTH + MAP_HEIGHT) as u32 - v) {
                self.radius += 1;
                self.delta = 0;
                self.current_pos = Some(Position{ x : self.center.x + self.radius, y: self.center.y });
                return Some(current);
            }
        }
        None
    }
}

