use log::error;
use std::ops::Range;
use rand::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;
use std::collections::HashMap;

#[derive(Ord, PartialOrd, Eq, PartialEq, Copy, Clone, Debug)]
pub struct Entity {
    pub id: u32,
    pub gen: i32,
}

#[derive(Clone, Debug, Default)]
pub struct EntityManager {
    generations: Vec<i32>,
    valid: Vec<bool>,
    full_to: usize,
    deleted: u32,
}

impl EntityManager {

    pub fn destroy(&mut self, entity: Entity) -> Result<(), &str> {
        debug_assert!(entity.id < self.generations.len() as u32);
        let id = entity.id as usize;
        if self.valid[id] && self.generations[id] == entity.gen {
            self.valid[id] = false;
            self.deleted += 1;
            if (self.full_to > id) {
                self.full_to = id;
            }
            Ok(())
        } else {
            Err("Entity to be destroyed does not exist")
        }
    }
    pub fn fresh(&mut self) -> Entity {
        if self.deleted > 0 {
            for i in (self.full_to + 1)..self.valid.len() {
                if !self.valid[i] {
                    let gen = self.generations[i] + 1;
                    self.generations[i] = gen;
                    self.valid[i] = true;
                    self.deleted -= 1;
                    self.full_to = i;
                    return Entity { id: i as u32, gen };
                }
            }
        }
        debug_assert!(self.deleted == 0);
        let len = self.valid.len();
        let gen = 0;
        self.generations.push(gen);
        self.valid.push(true);
        self.full_to = len;
        Entity {
            id: len as u32,
            gen,
        }
    }
}

pub struct EntityIter<'a> {
    em: &'a EntityManager,
    idx: usize,
}

impl<'a> Iterator for EntityIter<'a> {
    type Item = Entity;
    fn next(&mut self) -> Option<Self::Item> {
        while self.idx + 1 < self.em.valid.len() {
            self.idx += 1;
            if self.em.valid[self.idx] {
                return Some(Entity {
                    id: self.idx as u32,
                    gen: self.em.generations[self.idx],
                });
            }
        }
        None
    }
}

impl<'a> IntoIterator for &'a EntityManager {
    type Item = Entity;
    type IntoIter = EntityIter<'a>;
    fn into_iter(self) -> Self::IntoIter {
        EntityIter { em: self, idx: 0 }
    }
}

#[derive(Clone, Debug)]
pub struct Storage<T> {
    content: Vec<Option<T>>,
    generations: Vec<i32>,
}

impl<T> Storage<T> {
    pub fn get(&self, entity: &Entity) -> Option<&T> {
        let Entity { id, gen } = entity;
        let id = *id as usize;
        if let Some(stored_gen) = self.generations.get(id) {
            if stored_gen == gen {
                if let Some(opt) = self.content.get(id) {
                    return opt.as_ref();
                }
            }
        }
        None
    }
    pub fn get_mut(&mut self, entity: &Entity) -> Option<&mut T> {
        let Entity { id, gen } = entity;
        let id = *id as usize;
        if let Some(stored_gen) = self.generations.get(id) {
            if stored_gen == gen {
                if let Some(opt) = self.content.get_mut(id) {
                    return opt.as_mut();
                }
            }
        }
        None
    }
    pub fn insert(&mut self, entity: &Entity, val: T) -> Option<(i32, T)> {
        let Entity { id, gen } = entity;
        let id = *id as usize;
        let end = self.generations.len();
        if id >= end {
            for _ in end..=id {
                self.generations.push(-1);
                self.content.push(None);
            }
        }
        let old_gen = self.generations[id];
        let mut old_cont = Some(val);
        std::mem::swap(&mut self.content[id], &mut old_cont);
        self.generations[id] = *gen;
        if old_gen >= 0 && old_cont.is_some() {
            return Some((old_gen, old_cont.unwrap()));
        }
        None
    }
    pub fn new() -> Self {
        Self {
            content: Vec::new(),
            generations: Vec::new(),
        }
    }
}
impl<T> Default for Storage<T> {
    fn default() -> Self {
        Self::new()
    }
}

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
                    health: Health(100.0),
                    attack: None,
                    satiation: Satiation(10.0)
                }),
            _ => None
        }
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
}
#[derive(PartialOrd, PartialEq, Copy, Clone, Debug, Default)]
pub struct Hunger(pub f32);

const HUNGER_THRESHOLD : Hunger = Hunger(1.0);

const HUNGER_INCREMENT : f32 = 0.0001;

type Reward = f32;

#[derive(Clone, Debug)]
pub struct MentalState {
    pub id: Entity,
    pub hunger: Hunger,
    pub food_preferences: Vec<(EntityType, Reward)>,
    pub current_action: Option<Action>,
    pub sight_radius: u32,
}

impl MentalState {
    pub fn new(entity: Entity, food_preferences: Vec<(EntityType, Reward)>) -> Self {
        assert!(food_preferences.len() > 0);
        Self{
            id: entity,
            hunger: Hunger::default(),
            food_preferences,
            current_action: None,
            sight_radius: 5,
        }
    }
    pub fn decide(
        &mut self,
        own_type: EntityType,
        physical_state: &PhysicalState,
        own_position: Position,
        observation: impl Observation,
    ) -> Option<Action> {
        self.update(physical_state, own_position);
        self.decide_mdp(own_type, physical_state, own_position, observation);
        self.current_action

    }
    fn update(& mut self, physical_state: &PhysicalState, own_position: Position,) {
        self.hunger.0 += (20.0 -  physical_state.satiation.0) * HUNGER_INCREMENT;
        match self.current_action {
            Some(Action::Eat(food)) => {
                if self.hunger.0 <= 0.0 {
                    self.current_action = None;
                }
            },
            Some(Action::Move(goal)) => {
                if own_position == goal {
                    self.current_action = None;
                }
            },
            None => (),

        }
    }
    fn  decide_simple(
        &mut self,
        own_type: EntityType,
        physical_state: &PhysicalState,
        own_position: Position,
        observation: impl Observation,
    ) {

        if self.hunger > HUNGER_THRESHOLD {
            match observation.find_closest(own_position, |e, w| {
                match w.entity_types.get(e) {
                    Some(other_type) => own_type.can_eat(other_type),
                    None => false
                }
            }).next() {
                Some((entity, position)) => {
                    if position == own_position {
                        self.current_action = Some(Action::Eat(entity));
                    }
                    else {
                        if own_position.is_neighbour(&position) {
                            self.current_action = Some(Action::Move(position));
                        }
                        else{
                            if let Some(step) = self.path(own_position, position, observation.clone()){
                                self.current_action = Some(Action::Move(step));
                            }
                        }
                    }
                }
                None => println!("no food"),
            }
        }
    }
    fn  decide_mdp(
        &mut self,
        own_type: EntityType,
        physical_state: &PhysicalState,
        own_position: Position,
        observation: impl Observation,
    ) {
        let world_expectation = &World::from_observation(observation);
        let direct_reward = | action, hunger | {
            match action {
                Action::Move(_) => {
                    return 0.1;
                },
                Action::Eat(food) => {
                    if let Some(food_type) = world_expectation.get_type(&food) {
                        if hunger > HUNGER_THRESHOLD {
                            if let Some((_, pref)) = self.food_preferences.iter().find(|(et, pref)| et == &food_type) {
                                return 0.2 + pref;
                            }
                        }
                    }
                    return 0.0;
                }

            }
        };
        const TIME_DEPTH : usize = 10;
        let id = self.id.clone();
        let mut reward_expectations = [[[0.0; MAP_WIDTH]; MAP_HEIGHT]; TIME_DEPTH];
        let mut policy =  [[[None; MAP_WIDTH]; MAP_HEIGHT]; TIME_DEPTH];

        let mut advance = |depth: usize, ps : PhysicalState, pos, ms : MentalState| {
            let Position{x, y} = pos;
            let mut updated_action = None;
            let mut max_reward : f32 = reward_expectations[depth][y as usize][x as usize];

            for food in  world_expectation.entities_at(pos).iter(){
                let mut reward = direct_reward(Action::Eat(*food), ms.hunger);
                if depth < TIME_DEPTH - 1 {
                    reward += reward_expectations[depth+1][y as usize][x as usize];
                }
                if reward > max_reward {
                    max_reward = reward;
                    updated_action = Some(Action::Eat(*food));
                }
            }

            for neighbor in pos.neighbours() {
                let Position{x, y} = neighbor;
                if neighbor.within_bounds() && world_expectation.known_can_pass(&id, neighbor) != Some(false) {
                    let mut reward = direct_reward(Action::Move(neighbor), ms.hunger);
                    if depth < TIME_DEPTH - 1 {
                        reward += reward_expectations[depth + 1][y as usize][x as usize];
                    }
                    if reward > max_reward {
                        max_reward = reward;
                        updated_action = Some(Action::Move(neighbor));
                    }
                }
            }
            if updated_action.is_some() {
                policy[depth][y as usize][x as usize] = updated_action;
                reward_expectations[depth][y as usize][x as usize] = max_reward;
                assert_eq!(max_reward, reward_expectations[depth][y as usize][x as usize]);
                return true;
            }
            false
        };
        let mut converged = false;
        let mut runs = 0;
        while !converged && runs < 100 {
            runs += 1;
            converged = true;
            let ps = physical_state.clone();
            let ms = self.clone();
            for t in (0 .. TIME_DEPTH).rev() {
                for x in 0..MAP_WIDTH {
                    for y in 0..MAP_HEIGHT {
                        let pos = Position{ x: x as u32, y: y as u32};
                        if advance(t, ps.clone(), pos, ms.clone()) {
                            converged = false;
                        }
                    }
                }
            }
        }
        println!("expectation {:?}", reward_expectations[0][own_position.y as usize][own_position.x as usize]);
        let act = policy[0][own_position.y as usize][own_position.x as usize];
        if act.is_some() {
            self.current_action = act;
        }
    }
    pub fn path(&self, current: Position, goal: Position, observation: impl Observation) -> Option<Position> {
        let d = current.distance(&goal);
        for n in current.neighbours(){
            if current.distance(&n) < d && Some(true) == observation.known_can_pass(&self.id, n) {
                return Some(n);
            }
        }
        None
    }
}

#[derive(PartialOrd, PartialEq, Copy, Clone, Debug)]
pub struct Health(f32);

impl Health {
    pub fn suffer(&mut self, attack: Attack) {
        self.0 -= attack.0
    }
}
#[derive(PartialOrd, PartialEq, Copy, Clone, Debug)]
pub struct Attack(f32);

#[derive(PartialOrd, PartialEq, Copy, Clone, Debug)]
pub struct Satiation(f32);

#[derive(Clone, Debug)]
pub struct PhysicalState {
    pub health: Health,
    pub attack: Option<Attack>,
    pub satiation: Satiation,
}

pub const MAP_HEIGHT: usize = 10;
pub const MAP_WIDTH: usize = 10;

#[derive(Ord, PartialOrd, Eq, PartialEq, Copy, Clone, Debug)]
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

#[derive(Clone, Debug, Default)]
pub struct World {
    cells: [[Option<Vec<Entity>>; MAP_WIDTH]; MAP_HEIGHT],
    entity_types: Storage<EntityType>,
    physical_states: Storage<PhysicalState>,
    positions: Storage<Position>,
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
                    let entity = entity_manager.fresh();
                    positions.insert(&entity, Position{x : x as u32, y: y as u32});
                    entity_types.insert(&entity, entity_type);
                    if let Some(phys_state) = entity_type.typical_physical_state() {
                        physical_states.insert(&entity, phys_state.clone());
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
    pub fn from_observation(observation: impl Observation) -> Self {
        let mut cells: [[Option<Vec<Entity>>; MAP_WIDTH]; MAP_HEIGHT] = Default::default();
        let mut entity_types = Storage::new();
        let mut physical_states = Storage::new();
        let mut positions = Storage::new();
        let pos = Position { x: (MAP_WIDTH / 2) as u32, y: (MAP_HEIGHT / 2) as u32 };
        let radius = std::cmp::max(MAP_HEIGHT, MAP_WIDTH) as u32;
        for (e, p) in observation.find_closest(pos, |e, w| true) {
            positions.insert(&e, p);
            if let Some(tp) = observation.get_type(&e) {
                entity_types.insert(&e, tp);
                if let Some(ps) = tp.typical_physical_state() {
                    physical_states.insert(&e, ps);
                }
            }
        }
        Self{cells, entity_types, physical_states, positions }
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
        if let Some(mover) = self.entity_types.get(entity) {
            return self
                .entities_at(position)
                .iter()
                .all(|e| match self.entity_types.get(e) {
                    Some(e) => mover.can_pass(e),
                    None => true,
                });
        }
        false
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
        self.physical_states
            .get_mut(eater)
            .map(|ps| ps.satiation.0 += 5.0);
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
    fn find_closest<'a>(& 'a self, starting_point: Position, predicate: impl Fn(&Entity, &World) -> bool + 'a) -> Box<Iterator<Item=(Entity, Position)> + 'a>;
    fn known_can_pass(&self, entity: &Entity, position: Position) -> Option<bool>;
    fn get_type(& self, entity: & Entity) -> Option<EntityType>;
    fn entities_at(&self, position: Position) -> &[Entity];
}

impl<'b> Observation for & 'b World {
    fn find_closest<'a>(& 'a self, starting_point: Position, predicate: impl Fn(&Entity, Self) -> bool + 'a) -> Box<Iterator<Item=(Entity, Position)> + 'a> {
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
    fn find_closest<'a>(& 'a self, starting_point: Position, predicate: impl Fn(&Entity, &World) -> bool + 'a) -> Box<Iterator<Item=(Entity, Position)> + 'a> {
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

struct PositionWalker {
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

pub type ViewData = Option<Vec<EntityType>>;



#[derive(Clone, Debug, Default)]
struct AgentSystem {
    agents: Vec<Entity>,
    mental_states: Storage<MentalState>,
}

impl AgentSystem {
    pub fn advance(&mut self, world: &mut World) {
        for entity in &self.agents {
            let opt_action = match (
                self.mental_states.get_mut(entity),
                world.get_physical_state(entity),
                world.entity_types.get(entity),
                world.positions.get(entity),
            ) {
                (Some(mental_state), Some(physical_state), Some(et), Some(position)) => {
                    mental_state.decide(*et, physical_state, *position, world.observe_in_radius(entity, mental_state.sight_radius))
                }
                _ => None,
            };

            if let Some(action) = opt_action {
                world.act(entity, action);
            }
        }
    }
    pub fn init(agents: Vec<Entity>, world: &World, mut rng: impl Rng) -> Self {
        let mut mental_states = Storage::new();
        for agent in &agents {
            if let Some(et) = world.get_type(agent) {
                let food_prefs = ENTITY_TYPES.iter().filter(|e|et.can_eat(e)).map(|e| (e.clone(), rng.gen_range(0.0, 1.0))).collect();
                mental_states.insert(agent, MentalState::new(agent.clone(), food_prefs));
            }

        }
        Self{ agents, mental_states}
    }
}

#[derive(Clone, Debug, Default)]
pub struct SimState {
    world: World,
    agent_system: AgentSystem,
    entity_manager: EntityManager,
    sim_step: f32,
    time_acc: f32,
}

impl SimState {
    pub fn advance(&mut self, time_step: f32) {
        self.time_acc += time_step;
        while self.time_acc >= self.sim_step {
            self.time_acc -= self.sim_step;
            self.agent_system.advance(&mut self.world);
            self.world.advance();
        }
    }
    pub fn new(time_step: f32) -> Self {
        let mut entity_manager = EntityManager::default();
        let rng = XorShiftRng::from_seed([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        let (world, agents) = World::init(rng.clone(), &mut entity_manager);
        let agent_system = AgentSystem::init(agents, &world, rng);

        Self {
            time_acc: 0.0,
            sim_step: time_step,
            agent_system,
            world,
            entity_manager,
        }
    }
    pub fn get_view(
        &self,
        x: Range<usize>,
        y: Range<usize>,
    ) -> impl Iterator<Item = (usize, usize, ViewData)> + '_ {
        self.world.get_view(x, y)
    }
    pub fn entities_at(&self, position: Position) -> &[Entity] {
        (&self.world).entities_at(position)
    }
    pub fn update_mental_state(& mut self, mental_state: MentalState) {
        self.agent_system.mental_states.insert(&mental_state.id.clone(), mental_state);
    }
    pub fn get_mental_state(&self, entity :&Entity) -> Option<&MentalState> {
        self.agent_system.mental_states.get(entity)
    }
    pub fn get_type(& self, entity: & Entity) -> Option<EntityType> {
        self.world.entity_types.get(entity).copied()
    }
    pub fn get_visibility(& self, entity: &Entity) -> impl Iterator<Item=Position> {
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
}
