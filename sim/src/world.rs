use log::error;
use std::ops::{Range};
use rand::{Rng};
use std::collections::{BinaryHeap};
use std::cmp::Ordering;
use std::mem::{size_of, MaybeUninit};
use smallvec::SmallVec;

use super::entity::*;
use super::entity_type::*;
use super::agent::AgentSystem;
use crate::{MentalState};
use crate::position::{Position, Dir, PositionMap, Coord};

#[derive(Ord, PartialOrd, Eq, PartialEq, Copy, Clone, Debug)]
pub enum Action {
    Idle,
    Move(Dir),
    Eat(WorldEntity),
    Attack(WorldEntity),
}
impl std::fmt::Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error>  {
        use Action::*;
        match self {
            Idle => write!(f, "Idle"),
            Eat(food) => write!(f, "Eating {:?}", food.e_type()),
            Move(dir) => write!(f, "Moving {}", dir),
            Attack(target) => {
                write!(f, "Attacking {:?}", target.e_type())
            }
        }
    }
}
impl Default for Action {
    fn default() -> Self {
        Action::Idle
    }
}


#[derive(PartialEq, Copy, Clone, Debug)]
pub enum Outcome {
    Incomplete,
    Moved(Dir),
    Consumed(Meat, EntityType),
    Hurt{ damage: Damage, target: WorldEntity, lethal: bool},
    Rested,
}
#[derive(PartialEq, Copy, Clone, Debug)]
pub struct Event {
    pub actor: WorldEntity,
    pub outcome: Outcome,
}
#[derive(PartialOrd, PartialEq, Copy, Clone, Debug)]
pub struct Health(pub f32);

type Damage = Health;
impl Health {
    pub fn suffer(&mut self, attack: Attack) -> Damage {
        self.0 -= attack.0;
        Health(attack.0)
    }
}

#[derive(PartialOrd, PartialEq, Copy, Clone, Debug)]
pub struct Meat(pub f32);

#[derive(PartialOrd, PartialEq, Copy, Clone, Debug)]
pub struct Attack(pub f32);

#[derive(PartialOrd, PartialEq, Copy, Clone, Debug)]
pub struct Satiation(pub f32);

#[derive(PartialOrd, PartialEq, Copy, Clone, Debug)]
pub struct Speed(pub f32);

impl std::ops::Mul<f32> for Speed {
    type Output = Speed;
    fn mul(self, rhs: f32)-> Self::Output {
        Speed(self.0 * rhs)
    }
}

#[derive(PartialOrd, PartialEq, Copy, Clone, Debug, Default)]
pub struct MoveProgress(pub f32);

impl std::ops::AddAssign<Speed> for MoveProgress {
    fn add_assign(&mut self, rhs: Speed) {
        self.0 += rhs.0
    }
}

#[derive(Clone, Debug)]
pub struct PhysicalState {
    pub health: Health,
    pub max_health: Health,
    pub meat: Meat,
    pub speed: Speed,
    pub move_progress: MoveProgress,
    pub move_target: Option<Position>,
    pub attack: Option<Attack>,
    pub satiation: Satiation,
}

impl std::fmt::Display for PhysicalState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        if self.is_dead() {
            writeln!(f, "{:.2} meat remaining", self.meat.0)?;
        }
        else {
            writeln!(f, "Health: {:.2}/{:.2}", self.health.0, self.max_health.0)?;
            writeln!(f, "Speed : {:.}", self.speed.0)?;
            if let Some(att) = self.attack {
                writeln!(f, "Attack: {:.}", att.0)?;
            }
        }
        Ok(())
    }
}

impl PhysicalState {
    pub fn is_dead(&self) -> bool {
        self.health.0 <= 0.0
    }
    pub fn partial_move(&mut self, goal: Position) -> MoveProgress {
        if Some(goal) != self.move_target {
            self.move_target = Some(goal);
        }
        self.move_progress += self.speed * (self.health.0 / self.max_health.0);
        self.move_progress
    }
    pub fn new(max_health: Health, speed: Speed, attack: Option<Attack>) -> Self {
        Self {
            health : max_health,
            max_health,
            meat: Meat(max_health.0 * 0.5),
            speed,
            move_progress: MoveProgress::default(),
            move_target: None,
            attack,
            satiation: Satiation(10.0)
        }
    }
}


pub const MAP_HEIGHT: usize = 11;
pub const MAP_WIDTH: usize = 11;

pub type ViewData = WorldEntity;

pub trait Cell : std::ops::Deref<Target=[WorldEntity]> + Sized + Clone + Sync {
    fn empty_init() ->  [[Self; MAP_WIDTH]; MAP_HEIGHT];
    fn retain<F : FnMut(&WorldEntity)-> bool>(& mut self, f: F);
    fn push(& mut self, we: WorldEntity);
    fn unknown() -> Self;
    fn is_empty(&self) -> bool;
}

pub type DefCell = SmallVec<[WorldEntity; 3]>;

impl Cell for DefCell {
    fn empty_init() -> [[Self; 11]; 11] {
        empty_initialize()
    }

    fn retain<F: FnMut(&WorldEntity) -> bool>(&mut self, mut f: F) {
        self.retain(|we| f(&*we))
    }

    fn push(&mut self, we: WorldEntity) {
        self.push(we)
    }

    fn unknown() -> Self {
        SmallVec::new()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }
}

#[derive(Clone, Debug)]
pub enum Occupancy {
    Empty,
    Filled(Vec<WorldEntity>),
    ExpectedFilled(Vec<WorldEntity>, Vec<f32>),
    ExpectedEmpty,
    Unknown,
}
impl std::ops::Deref for Occupancy {
    type Target = [WorldEntity];

    fn deref(&self) -> &Self::Target {
        use Occupancy::*;
        match self {
            Empty | ExpectedEmpty | Unknown => &[],
            Filled(v) => &*v,
            ExpectedFilled(wes, probs) => &*wes,
        }
    }
}
impl Cell for Occupancy {
    fn empty_init() -> [[Self; 11]; 11] {
        Occupancy::initialize(Occupancy::Empty)
    }

    fn retain<F: FnMut(&WorldEntity) -> bool>(&mut self, mut f: F) {
        use Occupancy::*;
        match self {
            Empty | ExpectedEmpty | Unknown => (),
            ExpectedFilled(wes, probs ) => {
                let mut i = 0;
                wes.retain(|we : & WorldEntity | {
                    i += 1;
                    if f(we) {
                        probs.remove(i);
                        true
                    }
                    else {
                        false
                    }
                })
            }
            Filled(v) => v.retain(f),

        }
    }

    fn push(&mut self, we: WorldEntity) {
        use Occupancy::*;
        match self {
            Empty | ExpectedEmpty | Unknown | ExpectedFilled(_, _) => {
                *self = Filled(vec![we]);
            }
            Filled(v) => v.push(we),

        }
    }

    fn unknown() -> Self {
        Occupancy::Unknown
    }

    fn is_empty(&self) -> bool {
        use Occupancy::*;
        match self {
            Empty | ExpectedEmpty => true,
            _ => false,
        }
    }
}
impl Occupancy {
    fn initialize(none: Occupancy) -> [[Occupancy; MAP_WIDTH]; MAP_HEIGHT] {
        unsafe {
            let bytes = std::slice::from_raw_parts(&none as * const _ as * const u8, size_of::<Occupancy>());
            if bytes.iter().all(|b| *b == 0u8) {
                return MaybeUninit::zeroed().assume_init();
            }
        }
        let mut a : [[MaybeUninit<Occupancy>; MAP_WIDTH]; MAP_HEIGHT] = unsafe { MaybeUninit::uninit().assume_init() };
        for row in &mut a[..] {
            for elem in & mut row[..] {
                *elem = MaybeUninit::new(none.clone());
            }
        }
        unsafe {
            std::mem::transmute(a)
        }
    }
}

#[derive(Clone)]
pub struct World<C: Cell> {
    cells: [[C; MAP_WIDTH]; MAP_HEIGHT],
    pub physical_states: Storage<PhysicalState>,
    pub positions: Storage<Position>,
    pub events: Vec<Event>,
}

impl<C: Cell> World<C> {
    pub fn init(mut rng: impl Rng, entity_manager: &mut EntityManager) -> (Self, Vec<WorldEntity>) {
        let area = MAP_WIDTH * MAP_HEIGHT;
        let mut cells = C::empty_init();
        let mut physical_states =  Storage::new();
        let mut positions = Storage::new();
        let mut agents = Vec::new();
        let mut inserter = |entity_type: EntityType| {
            let count = area / entity_type.rate();
            let mut c = 0;
            while c < count {
                let x = rng.gen_range(0, MAP_WIDTH);
                let y = rng.gen_range(0, MAP_HEIGHT);
                let cell_v = & mut cells[y][x];
                if cell_v.into_iter().all(|other | entity_type.can_pass(&other.e_type())) {
                    let entity = WorldEntity::new(entity_manager.fresh(), entity_type);
                    positions.insert(&entity, Position{x : x as Coord, y: y as Coord});
                    if let Some(phys_state) = entity_type.typical_physical_state() {
                        physical_states.insert(&entity, phys_state);
                        agents.push(entity);
                    }
                    cell_v.push(entity);
                    c+= 1;
                }
            }
        };

        for et in EntityType::iter() {
            inserter(et);
        }
        (
            Self {
                cells,
                physical_states,
                positions,
                events: Vec::new(),
            },
            agents
        )
    }
    pub fn respawn(&mut self, entity: &WorldEntity, mental_state: & mut  MentalState, entity_manager: & mut EntityManager) -> WorldEntity {
        let new_e = WorldEntity::new(entity_manager.fresh(), entity.e_type());


        let mut random_pos = || {
            let x = mental_state.rng.gen_range(0, MAP_WIDTH);
            let y = mental_state.rng.gen_range(0, MAP_HEIGHT);
            Position {x: x as Coord, y : y as Coord}
        };
        let mut pos = random_pos();
        while !self.type_can_pass(&entity.e_type(), pos) {
            pos = random_pos();
        }
        self.move_unchecked(&new_e, pos);
        mental_state.respawn_as(&new_e);
        if let Some(phys_state) = entity.e_type().typical_physical_state() {
            self.physical_states.insert(&new_e, phys_state);
        }
        new_e
    }
    pub fn act(&mut self, actions: impl Iterator<Item=(WorldEntity, Action)>) {
        let mut move_list = Vec::new();
        for (actor, action) in actions {
            match self.act_one(&actor, action) {
                Err(err) => error!("Action of {} failed: {}", actor, err),
                Ok(Outcome::Moved(dir)) => {
                    move_list.push((actor, dir));
                    self.events.push(Event { actor, outcome: Outcome::Moved(dir)});
                },
                Ok(outcome) => self.events.push(Event{ actor, outcome }),
            }
        }
        for (actor, dir) in move_list {
            if let Some(pos) = self.positions.get(actor).and_then(|p| p.step(dir)) {
                self.move_unchecked(&actor, pos);

            }
            else {
                error!("Processing invalid move {} by {}", dir, actor);
            }
        }
    }
    fn act_one(&mut self, entity: &WorldEntity, action: Action) -> Result<Outcome, &str> {
        let own_pos = self.positions.get(entity)
            .ok_or("Entity has no known position but tries to act")?;

        match action {
            Action::Move(dir) => {
                match own_pos.step(dir) {
                    Some(pos) if self.can_pass(entity, pos) => {
                        let phys = self.physical_states.get_mut(entity)
                            .ok_or("Entity has no physical state but tries to move")?;
                        if phys.partial_move(pos) >= MoveProgress(1.0) {
                            phys.move_target = None;
                            phys.move_progress = MoveProgress::default();
                            Ok(Outcome::Moved(dir))

                        } else {
                            Ok(Outcome::Incomplete)
                        }
                    },
                    Some(_) => Err("blocked move"),
                    None => Err("Invalid move"),
                }
            }
            Action::Eat(target) => {
                if self.positions.get(&target) == Some(own_pos) {
                    if entity.e_type().can_eat(&target.e_type()) {
                        Ok(self.eat_unchecked(entity, &target))
                    } else {
                        Err("entity can't eat that")
                    }
                } else {
                    Err("can only eat things in the same tile")
                }
            },
            Action::Attack(opponent) => {
                let pos = self.positions.get(&opponent)
                    .ok_or("Entity tries to attack opponent with no known position")?;
                if pos != own_pos && !(self.can_pass(entity, *pos) && own_pos.is_neighbour(pos)) {
                    return Err("Cannot reach attacked opponent")
                }
                let phys = self.physical_states.get_mut(entity)
                    .ok_or("Entity has no physical state but tries to attack")?;
                if let Some(attack) = phys.attack {
                    if let Some(phys_target) = self.physical_states.get_mut(&opponent) {
                        let damage = phys_target.health.suffer(attack);
                        Ok(Outcome::Hurt { damage, target: opponent, lethal: phys_target.is_dead() })
                    } else {
                        Err("opponent has no physical state")
                    }
                } else {
                    Err("entity incapable of attacking")
                }
            },
            Action::Idle => Ok(Outcome::Rested),
        }
    }
    fn can_pass(&self, entity: &WorldEntity, position: Position) -> bool {
        if !position.within_bounds() {
            return false
        }
        self.type_can_pass(&entity.e_type(), position)

    }
    pub fn type_can_pass(&self, entity_type: & EntityType, position: Position) -> bool {
        if !position.within_bounds() {
            return false
        }
        self.entities_at(position)
                .iter()
                .all(|e| entity_type.can_pass(&e.e_type()))
    }
    pub fn observe_as(&self, entity: &WorldEntity) -> impl Observation + '_ {
        let radius = std::cmp::max(MAP_HEIGHT, MAP_WIDTH) as Coord;
        self.observe_in_radius(entity, radius)
    }
    pub fn observe_in_radius(&self, entity: &WorldEntity, radius: Coord) -> impl Observation +'_ {
        let pos = match self.positions.get(entity) {
            Some(pos) => pos.clone(),
            None => Position { x: (MAP_WIDTH / 2) as Coord, y: (MAP_HEIGHT / 2) as Coord },
        };
        RadiusObservation::new(radius, pos, self)

    }
    pub fn get_physical_state(&self, entity: &WorldEntity) -> Option<&PhysicalState> {
        self.physical_states.get(entity)
    }
    pub fn advance(&mut self) {}
    pub fn entities_at(&self, position: Position) -> &[WorldEntity] {
        let x = position.x as usize;
        let y = position.y as usize;
        if x < MAP_WIDTH && y < MAP_HEIGHT {
            return &self.cells[y][x][..]
        }
        return &[];
    }
    fn entities_at_mut(&mut self, position: Position) -> &mut C{
        let x = position.x as usize;
        let y = position.y as usize;
        &mut self.cells[y][x]
    }
    fn move_unchecked(&mut self, entity: &WorldEntity, new_position: Position) {
        if let Some((_, old_pos)) = self.positions.insert(entity, new_position) {
            self.entities_at_mut(old_pos).retain(|e| e != entity);
        }
        self.entities_at_mut(new_position).push(entity.clone());
    }
    fn eat_unchecked(&mut self, eater: &WorldEntity, eaten: &WorldEntity) -> Outcome {
        let mut decrement = 5.0;
        let mut remove = false;

        if let Some(phys) = self.physical_states.get_mut(eaten) {
            if phys.meat.0 <= decrement {
                    remove =  true;
                    decrement = phys.meat.0;

            }
            phys.meat.0 -= decrement;
        }
        self.physical_states
            .get_mut(eater)
            .map(|ps| ps.satiation.0 += decrement);
        if remove {
            self.physical_states.remove(eaten);
            if let Some(pos) = self.positions.remove(eaten) {
                self.entities_at_mut(pos).retain(|e| e!= eaten)
            }
        }
        Outcome::Consumed(Meat(decrement), eaten.e_type())
    }
    pub fn get_view(
        &self,
        x_range: Range<usize>,
        y_range: Range<usize>,
    ) -> impl Iterator<Item = (usize, usize, &[WorldEntity])> + '_ {
        self.cells[y_range.clone()]
            .iter()
            .zip(y_range.clone())
            .flat_map(move |(v, y)| {
                v[x_range.clone()]
                    .iter()
                    .zip(x_range.clone())
                    .map(move |(res, x)| {

                        (x, y.clone(), &**res)
                    })
            })
    }
}
impl<C: Cell> std::fmt::Debug for World<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "World {{ cells : _, physical_states: {:?}, positions: {:?} }}", self.physical_states, self.positions)
    }
}

type Cost = i32;
#[derive(Eq, PartialEq, Clone, Debug)]
pub struct PathNode {
    pub pos : Position,
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
    fn borrow<'a>(& 'a self) -> Self::B;
    fn find_closest<'a>(& 'a self, starting_point: Position, predicate: impl Fn(&WorldEntity, &World<Self::CellType>) -> bool + 'a) -> Box<dyn Iterator<Item=(WorldEntity, Position)> + 'a>;
    fn known_can_pass(&self, entity: &WorldEntity, position: Position) -> Option<bool>;
    fn entities_at(&self, position: Position) -> &[WorldEntity];
    fn observed_physical_state(&self, entity: &WorldEntity) -> Option<&PhysicalState>;
    fn observed_position(& self, entity: &WorldEntity) -> Option<Position>;
    fn into_expected<C: Cell>(& self, filler: impl Fn(Position) -> C, mut rng: impl Rng) -> (EntityManager, World<C>, AgentSystem) {

        let mut cells  = C::empty_init();
        let mut entity_manager = EntityManager::default();
        let mut physical_states = Storage::new();
        let mut positions = Storage::new();
        let mut agents = Vec::new();
        let mut insert_cell = |e, Position {x, y}| {
            cells[y as usize][x as usize].push(e);
        };
        let pos = Position { x: (MAP_WIDTH / 2) as Coord, y: (MAP_HEIGHT / 2) as Coord };
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
        let world = World{cells, physical_states, positions, events: Vec::new() };
        let agent_system = AgentSystem::init(agents, &world, false, rng);
        (entity_manager, world, agent_system)
    }
    fn path_as(&self, start: Position, goal: Position, entity: &WorldEntity) -> Option<Vec<Dir>> {
        if self.known_can_pass(entity, goal) == Some(false) {
            return None
        }
        let mut came_from : PositionMap<Dir> = PositionMap::new();
        let mut cost = PositionMap::new();
        let mut queue = BinaryHeap::new();
        cost.insert(start, 0 as Cost);
        queue.push(PathNode { pos: start, exp_cost: start.distance(&goal) as Cost });
        while let Some(PathNode{pos, exp_cost}) = queue.pop() {
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
                    }
                    else {
                        let mut insert = true;
                        if let Some(c) =cost.get(&n) {
                            if *c <= base_cost + 1 {
                                insert = false;
                            }
                        }
                        if insert {
                            cost.insert(n,base_cost + 1);
                            came_from.insert(n, *d);
                            queue.push(PathNode{pos: n, exp_cost : base_cost + 1 + n.distance(&goal) as Cost })
                        }
                    }
                }

            }
        }
        None
    }
}

impl<'b, C: Cell> Observation for & 'b World<C> {
    type B = & 'b World<C>;
    type CellType = C;
    fn borrow<'a>(& 'a self) -> Self {
        self.clone()
    }
    fn find_closest<'a>(& 'a self, starting_point: Position, predicate: impl Fn(&WorldEntity, Self) -> bool + 'a) -> Box<dyn Iterator<Item=(WorldEntity, Position)> + 'a> {
        Box::new(EntityWalker::new(self, starting_point, std::cmp::max(MAP_HEIGHT, MAP_WIDTH) as Coord).filter(move |(e, p)| predicate(e, self)))
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
    fn observed_position(& self, entity: &WorldEntity) -> Option<Position>{
        self.positions.get(entity).copied()
    }

}
#[derive(Clone, Debug)]
pub struct RadiusObservation<'a, C: Cell> {
    radius: Coord,
    center: Position,
    world: & 'a World<C>,
}
impl<'a, C: Cell> RadiusObservation<'a, C> {
    pub fn new(radius: Coord, center: Position, world: & 'a World<C>) -> Self {
        Self{ radius, center, world}
    }
}
impl<'b, C: Cell> Observation for RadiusObservation<'b, C> {
    type B =  RadiusObservation<'b, C>;
    type CellType = C;
    fn borrow<'a>(& 'a self) -> Self {
        self.clone()
    }
    fn find_closest<'a>(& 'a self, starting_point: Position, predicate: impl Fn(&WorldEntity, &World<C>) -> bool + 'a) -> Box<dyn Iterator<Item=(WorldEntity, Position)> + 'a> {
        Box::new(EntityWalker::new(self.world, starting_point, self.radius).filter(
            move |(e, p)| self.center.distance(p) <= self.radius &&  predicate(e, self.world)))
    }
    fn known_can_pass(&self, entity: &WorldEntity, position: Position) -> Option<bool> {
        if self.center.distance(&position) > self.radius {
            return None;
        }
        Some(self.world.can_pass(entity, position))
    }
    fn observed_physical_state(&self, entity: &WorldEntity) -> Option<&PhysicalState> {
        if let Some(pos)= self.world.positions.get(entity) {
            if pos.distance(&self.center) <= self.radius {
                return self.world.get_physical_state(entity)
            }
        }
        None
    }
    fn observed_position(&self, entity: &WorldEntity) -> Option<Position> {
        if let Some(pos)= self.world.positions.get(entity) {
            if pos.distance(&self.center) <= self.radius {
                return Some(*pos)
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
}
struct EntityWalker<'a, C : Cell> {
    position_walker: PositionWalker,
    world: &'a World<C>,
    current: &'a [WorldEntity],
    current_pos: Position,
    subindex: usize,
}
impl<'a, C: Cell> EntityWalker<'a, C> {
    pub fn new(world: &'a World<C>, center: Position, max_radius: Coord) -> Self {
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
impl<'a, C: Cell> Iterator for EntityWalker<'a, C> {
    type Item = (WorldEntity, Position);
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
    radius: Coord,
    max_radius: Coord,
    delta: Coord,
}
impl PositionWalker {
    pub fn new(center: Position, max_radius: Coord) -> Self {
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
                let x = self.center.x  + self.delta  - self.radius ;
                if x >= 0 {
                    self.current_pos = Some(Position{x: x , y : current.y});
                    return Some(current);
                }

            }
            if current.y > self.center.y {
                let x = self.center.x  + self.radius  - self.delta ;
                let y = self.center.y  - self.delta;
                if x >= 0 && y >= 0 {
                    self.current_pos = Some(Position{x, y });
                    return Some(current);
                }

            }
            if self.delta < self.radius {
                self.delta += 1;
                self.current_pos = Some(Position { x: self.center.x + self.radius - self.delta, y: self.center.y + self.delta });
                return Some(current);
            }
            let v = self.center.x + self.center.y;
            if self.radius < self.max_radius && (self.radius < v || self.radius < (MAP_WIDTH + MAP_HEIGHT) as Coord - v) {
                self.radius += 1;
                self.delta = 0;
                self.current_pos = Some(Position{ x : self.center.x + self.radius, y: self.center.y });
                return Some(current);
            }
        }
        None
    }
}

fn none_initialize<T>() -> [[Option<Vec<T>>; MAP_WIDTH]; MAP_HEIGHT] {
    let none = None::<Vec<T>>;
    unsafe {
        let bytes = std::slice::from_raw_parts(&none as * const _ as * const u8, size_of::<Option<Vec<T>>>());
        if bytes.iter().all(|b| *b == 0u8) {
            return MaybeUninit::zeroed().assume_init();
        }
    }
    let mut a : [[MaybeUninit<Option<Vec<T>>>; MAP_WIDTH]; MAP_HEIGHT] = unsafe { MaybeUninit::uninit().assume_init() };
    for row in &mut a[..] {
        for elem in & mut row[..] {
            *elem = MaybeUninit::new(None);
        }
    }
    unsafe {
        std::mem::transmute(a)
    }
}
fn empty_initialize<A: smallvec::Array>() -> [[SmallVec<A>; MAP_WIDTH]; MAP_HEIGHT] {
    unsafe {
        MaybeUninit::zeroed().assume_init()
    }
}