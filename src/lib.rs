use log::error;
use arr_macro::arr;
use std::ops::Range;

#[derive(Ord, PartialOrd, Eq, PartialEq, Copy, Clone, Debug)]
pub struct Entity {
    pub id : u32,
    pub gen: i32,
}

#[derive(Clone, Debug)]
pub struct EntityManager {
    generations: Vec<i32>,
    valid: Vec<bool>,
    full_to: usize,
    deleted: u32,
}
impl EntityManager{
    pub fn destroy(& mut self, entity: Entity) -> Result<(), &str> {
        debug_assert!(entity.id < self.generations.len() as u32);
        let id = entity.id as usize;
        if self.valid[id] && self.generations[id] == entity.gen {
            self.valid[id] =  false;
            self.deleted += 1;
            if (self.full_to > id) {
                self.full_to = id;
            }
            Ok(())
        } else {
            Err("Entity to be destroyed does not exist")
        }
    }
    pub fn fresh(& mut self) -> Entity {
        if self.deleted > 0 {
            for i in (self.full_to +1) .. self.valid.len() {
                if !self.valid[i] {
                    let gen = self.generations[i] + 1;
                    self.generations[i] = gen;
                    self.valid[i] = true;
                    self.deleted -= 1;
                    self.full_to = i;
                    return Entity { id : i as u32, gen }
                }
            }
        }
        debug_assert!(self.deleted == 0);
        let len = self.valid.len();
        let gen = 0;
        self.generations.push(gen);
        self.valid.push(true);
        self.full_to = len;
        Entity { id: len as u32, gen}
    }
}
pub struct EntityIter<'a> {
    em: &'a EntityManager,
    idx: usize,
}
impl<'a> Iterator for EntityIter<'a> {
    type Item = Entity;
    fn next(&mut self)-> Option<Self::Item> {
        while self.idx + 1 < self.em.valid.len() {
            self.idx += 1;
            if self.em.valid[self.idx] {
                return Some(Entity{ id : self.idx as u32, gen: self.em.generations[self.idx]})
            }
        }
        None
    }
}
impl<'a> IntoIterator for & 'a EntityManager {
    type Item = Entity;
    type IntoIter = EntityIter<'a>;
    fn into_iter(self) -> Self::IntoIter {
        EntityIter{ em: self, idx: 0}
    }
}

#[derive(Clone, Debug)]
pub struct Storage<T> {
    content: Vec<Option<T>>,
    generations: Vec<i32>
}
impl<T> Storage<T> {
    pub fn get(&self, entity: &Entity) -> Option<&T> {
        let Entity{id, gen} = entity;
        let id = *id as usize;
        if let Some(stored_gen) = self.generations.get(id) {
            if stored_gen == gen {
                if let Some(opt) = self.content.get(id){
                    return opt.as_ref();
                }
            }
        }
        None
    }
    pub fn get_mut(&mut self, entity: &Entity) -> Option<&mut T> {
        let Entity{id, gen} = entity;
        let id = *id as usize;
        if let Some(stored_gen) = self.generations.get(id) {
            if stored_gen == gen {
                if let Some(opt) = self.content.get_mut(id){
                    return opt.as_mut();
                }
            }
        }
        None
    }
    pub fn insert(& mut self, entity: &Entity, val: T) -> Option<(i32, T)> {
        let Entity{id, gen} = entity;
        let id = *id as usize;
        let end = self.generations.len();
        if id >= end {
            for _ in end ..=id {
                self.generations.push(-1);
                self.content.push(None);
            }
        }
        let old_gen = self.generations[id];
        let mut old_cont = Some(val);
        std::mem::swap(& mut self.content[id], & mut old_cont);
        self.generations[id] = *gen;
        if old_gen >= 0 && old_cont.is_some() {
            return Some((old_gen, old_cont.unwrap()));
        }
        None
    }
    pub fn new() -> Self {
        Self{ content: Vec::new(), generations: Vec::new() }
    }
}
impl<T> Default for Storage<T> {
    fn default() -> Self {
        Self::new()
    }
}
#[derive(Ord, PartialOrd, Eq, PartialEq, Copy, Clone, Debug)]
enum EntityType{
    Rock,
    Tree,
    Grass,
    Rabbit,
}
impl EntityType{
    pub fn can_eat(&self, other: &Self)-> bool {
        use EntityType::*;
        match (self, other) {
            (Rabbit, Grass) => true,
            _ => false
        }
    }
    pub fn can_pass(&self, other: &Self)-> bool {
        use EntityType::*;
        match (self, other) {
            (Rabbit, Grass) => true,
            (Rabbit, Tree) => true,
            _ => false
        }
    }
}
#[derive(Ord, PartialOrd, Eq, PartialEq, Copy, Clone, Debug)]
pub enum Action{
    Move(Position),
    Eat(Entity),
}
#[derive(PartialOrd, PartialEq, Copy, Clone, Debug)]
struct Hunger(f32);
#[derive(Clone, Debug)]
struct MentalState {
    hunger: Hunger,
}
impl MentalState {
    pub fn decide(&mut self, physical_state: &PhysicalState, observation: Observation) -> Option<Action> {

        None
    }
}

#[derive(PartialOrd, PartialEq, Copy, Clone, Debug)]
struct Health(f32);
impl Health{
    pub fn suffer(&mut self, attack: Attack){
        self.0 -= attack.0
    }
}
#[derive(PartialOrd, PartialEq, Copy, Clone, Debug)]
struct Attack(f32);

#[derive(PartialOrd, PartialEq, Copy, Clone, Debug)]
struct Satiation(f32);
#[derive(Clone, Debug)]
pub struct PhysicalState {
    health: Health,
    attack: Option<Attack>,
    satiation: Hunger,
}


const MAP_HEIGHT: usize= 10;
const MAP_WIDTH: usize = 10;

#[derive(Ord, PartialOrd, Eq, PartialEq, Copy, Clone, Debug)]
pub struct Position {
    x: u32,
    y: u32,
}
impl Position {
    pub fn neighbours(&self, other: &Position)-> bool {
        self != other
            && ((self.x as i64) - (other.x as i64)).abs() <= 1
            && ((self.y as i64) - (other.y as i64)).abs() <= 1
    }
}
#[derive(Clone, Debug, Default)]
pub struct World {
    cells : [[Option<Vec<Entity>>; MAP_WIDTH]; MAP_HEIGHT],
    entity_types: Storage<EntityType>,
    physical_states: Storage<PhysicalState>,
    positions: Storage<Position>,
}
impl World {
    pub fn new() -> Self {
        Self{ cells: Default::default(), entity_types: Storage::new(), physical_states: Storage::new(), positions: Storage::new() }
    }
    pub fn act(& mut self, entity: & Entity, action: Action)-> Result<(), &str>{
        if let Some(own_pos) = self.positions.get(entity){
            match action {
                Action::Move(pos) => {
                    if own_pos.neighbours(&pos) && self.can_pass(entity, pos) {
                        self.move_unchecked(entity, pos);
                        Ok(())
                    }
                    else {
                        Err("invalid move")
                    }
                },
                Action::Eat(entity) => {
                    Ok(())
                }
            }
        } else {
            error!("Entity {:?} has no known position but tries to do {:?}", entity, action);
            Err("Entity with unknown position acting")
        }
    }
    pub fn observe_as(&self, entity: & Entity) -> Observation {
        self
    }
    pub fn get_physical_state(&self, entity: & Entity) -> Option<&PhysicalState> {
        self.physical_states.get(entity)
    }
    pub fn advance(& mut self){

    }
    pub fn can_pass(&self, entity: & Entity, position: Position) -> bool {
        if let Some(mover) = self.entity_types.get(entity) {
            return self.entities_at(position).iter().all(|e| {
                match self.entity_types.get(e) {
                    Some(e) => mover.can_pass(e),
                    None => true,
                }
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
        return &[]
    }
    fn entities_at_mut(&mut self, position: Position) -> &mut Vec<Entity> {
        let x = position.x as usize;
        let y = position.y as usize;
        let entry = &mut self.cells[y][x];
        entry.get_or_insert( Vec::new())
    }
    fn move_unchecked(& mut self, entity: &Entity, new_position: Position) {
        if let Some((_, old_pos)) = self.positions.insert(entity, new_position) {
            self.entities_at_mut(old_pos).retain(|e| e != entity );
        }
        self.entities_at_mut(new_position).push(entity.clone());
    }
    pub fn get_view(&self, x_range: Range<usize>, y_range: Range<usize>) -> impl Iterator< Item = (usize, usize, ViewData)> + '_{
        self.cells[y_range.clone()].iter().zip(y_range.clone())
            .flat_map(move |(v, y)| v[x_range.clone()].iter().zip(x_range.clone())
                .map(move |(oes, x)|{
                    let res = match oes {
                        Some(es) => Some(es.iter().filter_map(|e| self.entity_types.get(e).cloned()).collect::<Vec<_>>()),
                        None => None
                    };
                    (x, y.clone(), res)
        }))
    }
}
pub type ViewData = Option<Vec<EntityType>>;
type Observation<'a> = &'a World;

#[derive(Clone, Debug, Default)]
struct AgentSystem {
    agents: Vec<Entity>,
    mental_states: Storage<MentalState>
}
impl AgentSystem {
    pub fn advance(& mut self, world: & mut World){
        for entity in &self.agents {
            let opt_action =
                match (self.mental_states.get_mut(entity), world.get_physical_state(entity)) {
                    (Some(mental_state), Some(physical_state)) =>
                        mental_state.decide(physical_state, world.observe_as(entity)),
                    _ => None
                };

            if let Some(action) = opt_action {
                world.act(entity, action);
            }
        }
    }
}
#[derive(Clone, Debug, Default)]
pub struct SimState {
    world: World,
    agent_system: AgentSystem,
    sim_step: f32,
    time_acc: f32,
}

impl SimState {
    pub fn advance(& mut self, time_step: f32){
        self.time_acc += time_step;
        while self.time_acc >= self.sim_step {
            self.time_acc -= self.sim_step;
            self.agent_system.advance(& mut self.world);
            self.world.advance();
        }
    }
    pub fn new(time_step: f32) -> Self {
        Self{
            time_acc: 0.0,
            sim_step: time_step,
            agent_system: AgentSystem::default(),
            world: World::new(),
        }
    }
    pub fn get_view(&self, x: Range<usize>, y: Range<usize>) -> impl Iterator< Item = (usize, usize, ViewData)> + '_ {
        self.world.get_view(x, y)
    }
}