use log::error;
use std::ops::Range;
use rand::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;

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
    Rabbit,
}
impl EntityType {
    pub fn can_eat(&self, other: &Self) -> bool {
        use EntityType::*;
        match (self, other) {
            (Rabbit, Grass) => true,
            _ => false,
        }
    }
    pub fn can_pass(&self, other: &Self) -> bool {
        use EntityType::*;
        match (self, other) {
            (Rabbit, Grass) => true,
            (Rabbit, Tree) => true,
            _ => false,
        }
    }
}
#[derive(Ord, PartialOrd, Eq, PartialEq, Copy, Clone, Debug)]
pub enum Action {
    Move(Position),
    Eat(Entity),
}
#[derive(PartialOrd, PartialEq, Copy, Clone, Debug, Default)]
struct Hunger(f32);

#[derive(Clone, Debug)]
struct MentalState {
    id: Entity,
    hunger: Hunger,
    current_action: Option<Action>,
}
const HUNGER_THRESHOLD : Hunger = Hunger(1.0);

const HUNGER_INCREMENT : f32 = 0.01;
impl MentalState {
    pub fn new(entity: Entity) -> Self {
        Self{
            id: entity,
            hunger: Hunger::default(),
            current_action: None,
        }
    }
    pub fn decide(
        &mut self,
        own_type: EntityType,
        physical_state: &PhysicalState,
        own_position: Position,
        observation: Observation,
    ) -> Option<Action> {
        self.hunger.0 += HUNGER_INCREMENT;
        // println!("decide");
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
            None => {
                if self.hunger > HUNGER_THRESHOLD {
                    println!("hungry");
                    match observation.find_closest(own_position, |e, w| {
                        match w.entity_types.get(e) {
                            Some(other_type) => own_type.can_eat(other_type),
                            None => false
                        }
                    }).next() {
                        Some((entity, position)) => {
                            if position == own_position {
                                println!("at food source {:?}", position);
                                self.current_action = Some(Action::Eat(entity));
                            }
                            else {
                                println!("moving towards food source {:?}", position);
                                if own_position.is_neighbour(&position) {
                                    self.current_action = Some(Action::Move(position));
                                }
                                else{
                                    if let Some(step) = self.path(own_position, position, &observation){
                                        self.current_action = Some(Action::Move(step));
                                    }

                                }
                            }
                        }
                        None => println!("no food"),
                    }
                }
                else {
                    println!("not hungry yet");
                }
            }
        }
        self.current_action
    }
    pub fn path(&self, current: Position, goal: Position, observation: &Observation) -> Option<Position> {
        let d = current.distance(&goal);
        for n in current.neighbours(){
            if current.distance(&n) < d && observation.can_pass(&self.id, n) {
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
    x: u32,
    y: u32,
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
        let mut inserter = |entity_type, count, opt_phys_state : Option<PhysicalState>| {
            let mut c = 0;
            let mut inserted = Vec::new();
            while c < count {
                let x = rng.gen_range(0, MAP_WIDTH);
                let y = rng.gen_range(0, MAP_HEIGHT);
                if let None = cells[y][x] {
                    let entity = entity_manager.fresh();
                    positions.insert(&entity, Position{x : x as u32, y: y as u32});
                    entity_types.insert(&entity, entity_type);
                    if let Some(phys_state) = &opt_phys_state {
                        physical_states.insert(&entity, phys_state.clone());
                    }
                    cells[y][x] = Some(vec![entity]);
                    c+= 1;
                    inserted.push(entity);
                }
            }
            inserted
        };
        inserter(EntityType::Tree, MAP_WIDTH * MAP_HEIGHT / 8, None);
        inserter(EntityType::Rock, MAP_WIDTH * MAP_HEIGHT / 16, None);
        inserter(EntityType::Grass, MAP_WIDTH * MAP_HEIGHT / 4, None);
        let agents = inserter(EntityType::Rabbit, 1, Some(
            PhysicalState{
                health: Health(100.0),
                attack: None,
                satiation: Satiation(10.0)
            }));

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
    pub fn observe_as(&self, entity: &Entity) -> Observation {
        self
    }
    pub fn get_physical_state(&self, entity: &Entity) -> Option<&PhysicalState> {
        self.physical_states.get(entity)
    }
    pub fn advance(&mut self) {}
    pub fn can_pass(&self, entity: &Entity, position: Position) -> bool {
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
    pub fn find_closest<'a>(& 'a self, starting_point: Position, predicate: impl Fn(&Entity, &Self) -> bool + 'a) -> impl Iterator<Item=(Entity, Position)> + 'a {
        EntityWalker::new(self, starting_point).filter(move |(e, p)| predicate(e, self))
    }
}
struct EntityWalker<'a> {
    world: &'a World,
    center: Position,
    current_pos: Position,
    current: &'a [Entity],
    radius: i32,
    delta: i32,
    subindex: usize,
}
impl<'a> EntityWalker<'a> {
    pub fn new(world: &'a World, center: Position) -> Self {
        let current = world.entities_at(center);
        Self{
            world,
            center,
            current_pos: center.clone(),
            current,
            radius: 0,
            delta: 0,
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
        if self.current_pos.x > self.center.x {
            let x = self.center.x as i32 + self.delta - self.radius;
            if x >= 0 {
                self.current_pos = Position{x: x as u32, y : self.current_pos.y};
                self.current = self.world.entities_at(self.current_pos);
                self.subindex = 0;
                return self.next();
            }

        }
        if self.current_pos.y > self.center.y {
            let x = self.center.x as i32 + self.radius - self.delta;
            let y = self.center.y as i32 - self.delta;
            if x >= 0 && y >= 0 {
                self.current_pos = Position{x: x as u32, y: y as u32 };
                self.current = self.world.entities_at(self.current_pos);
                self.subindex = 0;
                return self.next();
            }

        }
        if self.delta < self.radius {
            self.delta += 1;
            self.current_pos = Position { x: (self.center.x as i32 + self.radius - self.delta) as u32, y: self.center.y + self.delta as u32 };
            let Position{x, y} = self.current_pos;
            self.current = self.world.entities_at(self.current_pos);
            self.subindex = 0;
            return self.next();
        }
        let v = self.center.x + self.center.y;
        if (self.radius as u32) < v || (self.radius as u32) < (MAP_WIDTH + MAP_HEIGHT) as u32 - v {
            self.radius += 1;
            self.delta = 0;
            self.current_pos = Position{ x : self.center.x + self.radius as u32, y: self.center.y };
            self.current = self.world.entities_at(self.current_pos);
            self.subindex = 0;
            return self.next();
        }
        None
    }
}

pub type ViewData = Option<Vec<EntityType>>;
type Observation<'a> = &'a World;

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
                    mental_state.decide(*et, physical_state, *position, world.observe_as(entity))
                }
                _ => None,
            };

            if let Some(action) = opt_action {
                world.act(entity, action);
            }
        }
    }
    pub fn init(agents: Vec<Entity>, world: &World) -> Self {
        let mut mental_states = Storage::new();
        for agent in &agents {
            mental_states.insert(agent, MentalState::new(agent.clone()));
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
        let (world, agents) = World::init(rng, &mut entity_manager);
        let agent_system = AgentSystem::init(agents, &world);

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
}
