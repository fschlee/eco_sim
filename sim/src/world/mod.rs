pub mod body;
pub mod cell;
pub mod observation;
pub mod position;

use log::{error, warn};
use rand::Rng;
use std::ops::Range;

pub use body::*;
pub use cell::*;
pub use observation::*;
pub use position::*;

use super::entity::*;
use super::entity_type::*;
use crate::position::{Coord, Dir, Position};
use crate::MentalState;

#[derive(Ord, PartialOrd, Eq, PartialEq, Copy, Clone, Debug)]
pub enum Action {
    Idle,
    Move(Dir),
    Eat(WorldEntity),
    Attack(WorldEntity),
}
impl std::fmt::Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        use Action::*;
        match self {
            Idle => write!(f, "Idle"),
            Eat(food) => write!(f, "Eating {:?}", food.e_type()),
            Move(dir) => write!(f, "Moving {}", dir),
            Attack(target) => write!(f, "Attacking {:?}", target.e_type()),
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
    Hurt {
        damage: Damage,
        target: WorldEntity,
        lethal: bool,
    },
    Rested,
}
#[derive(PartialEq, Copy, Clone, Debug)]
pub struct Event {
    pub actor: WorldEntity,
    pub outcome: Outcome,
}

pub const MAP_HEIGHT: usize = 11;
pub const MAP_WIDTH: usize = 11;

pub type ViewData = WorldEntity;

#[derive(Clone)]
pub struct World<C: Cell> {
    pub(crate) cells: [[C; MAP_WIDTH]; MAP_HEIGHT],
    pub physical_states: Storage<PhysicalState>,
    pub positions: Storage<Position>,
    pub events: Vec<Event>,
}

impl<C: Cell> World<C> {
    pub const EATING_DECREMENT: f32 = 5.0;
    pub fn init(mut rng: impl Rng, entity_manager: &mut EntityManager) -> (Self, Vec<WorldEntity>) {
        let area = MAP_WIDTH * MAP_HEIGHT;
        let mut cells = C::empty_init();
        let mut physical_states = Storage::new();
        let mut positions = Storage::new();
        let mut agents = Vec::new();
        let mut inserter = |entity_type: EntityType| {
            let count = area / entity_type.rate();
            let mut c = 0;
            while c < count {
                let x = rng.gen_range(0, MAP_WIDTH);
                let y = rng.gen_range(0, MAP_HEIGHT);
                let cell_v = &mut cells[y][x];
                if cell_v
                    .into_iter()
                    .all(|other| entity_type.can_pass(&other.e_type()))
                {
                    let entity = WorldEntity::new(entity_manager.fresh(), entity_type);
                    positions.insert(
                        &entity,
                        Position {
                            x: x as Coord,
                            y: y as Coord,
                        },
                    );
                    if let Some(phys_state) = entity_type.typical_physical_state() {
                        physical_states.insert(&entity, phys_state);
                        agents.push(entity);
                    }
                    cell_v.push(entity);
                    c += 1;
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
            agents,
        )
    }
    pub fn respawn(
        &mut self,
        entity: &WorldEntity,
        mental_state: &mut MentalState,
        entity_manager: &mut EntityManager,
    ) -> WorldEntity {
        let new_e = WorldEntity::new(entity_manager.fresh(), entity.e_type());

        let mut random_pos = || {
            let x = mental_state.rng.gen_range(0, MAP_WIDTH);
            let y = mental_state.rng.gen_range(0, MAP_HEIGHT);
            Position {
                x: x as Coord,
                y: y as Coord,
            }
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
    pub fn act<'a>(&'a mut self, actions: impl IntoIterator<Item = &'a (WorldEntity, Action)>) {
        let mut move_list = Vec::new();
        for &(actor, action) in actions {
            match self.act_one(&actor, action) {
                Err(err) => error!("Action of {} failed: {}", actor, err),
                Ok(Outcome::Moved(dir)) => {
                    move_list.push((actor, dir));
                    self.events.push(Event {
                        actor,
                        outcome: Outcome::Moved(dir),
                    });
                }
                Ok(outcome) => self.events.push(Event { actor, outcome }),
            }
        }
        for (actor, dir) in move_list {
            if let Some(pos) = self.positions.get(actor).and_then(|p| p.step(dir)) {
                self.move_unchecked(&actor, pos);
            } else {
                error!("Processing invalid move {:?} by {}", dir, actor);
            }
        }
    }
    fn act_one(&mut self, entity: &WorldEntity, action: Action) -> Result<Outcome, String> {
        let own_pos = self
            .positions
            .get(entity)
            .ok_or("Entity has no known position but tries to act")?;

        match action {
            Action::Move(dir) => match own_pos.step(dir) {
                Some(pos) if self.can_pass(entity, pos) => {
                    let phys = self
                        .physical_states
                        .get_mut(entity)
                        .ok_or("Entity has no physical state but tries to move")?;
                    if phys.partial_move(pos) >= MoveProgress(1.0) {
                        phys.move_target = None;
                        phys.move_progress = MoveProgress::default();
                        Ok(Outcome::Moved(dir))
                    } else {
                        Ok(Outcome::Incomplete)
                    }
                }
                Some(pos) => Err(format!(
                    "Move from {:?} to {:?} blocked by {:?}",
                    own_pos,
                    pos,
                    self.entities_at(pos)
                        .iter()
                        .find(|&e| !entity.e_type().can_pass(&e.e_type()))
                        .unwrap()
                )),
                None => Err("Invalid move".to_owned()),
            },
            Action::Eat(target) => {
                if self.positions.get(&target) == Some(own_pos) {
                    if entity.e_type().can_eat(&target.e_type()) {
                        Ok(self.eat_unchecked(entity, &target, Self::EATING_DECREMENT))
                    } else {
                        Err(format!("Can't eat {} ", target))
                    }
                } else {
                    Err("Can only eat things in the same tile".to_owned())
                }
            }
            Action::Attack(opponent) => {
                let pos = self
                    .positions
                    .get(&opponent)
                    .ok_or("Entity tries to attack opponent with no known position")?;
                if pos != own_pos && !(self.can_pass(entity, *pos) && own_pos.is_neighbour(pos)) {
                    return Err(format!(
                        "Cannot reach attacked opponent {} at {:?} from {:?}",
                        opponent, pos, own_pos
                    ));
                }
                let phys = self
                    .physical_states
                    .get_mut(entity)
                    .ok_or("Entity has no physical state but tries to attack")?;
                if let Some(attack) = phys.attack {
                    if let Some(phys_target) = self.physical_states.get_mut(&opponent) {
                        let damage = phys_target.health.suffer(attack);
                        Ok(Outcome::Hurt {
                            damage,
                            target: opponent,
                            lethal: phys_target.is_dead(),
                        })
                    } else {
                        Err(format!(
                            "Attacked opponent {} has no physical state",
                            opponent
                        ))
                    }
                } else {
                    Err(format!("Incapable of attacking {}", opponent))
                }
            }
            Action::Idle => Ok(Outcome::Rested),
        }
    }
    fn can_pass(&self, entity: &WorldEntity, position: Position) -> bool {
        if !position.within_bounds() {
            return false;
        }
        self.type_can_pass(&entity.e_type(), position)
    }
    pub fn type_can_pass(&self, entity_type: &EntityType, position: Position) -> bool {
        if !position.within_bounds() {
            return false;
        }
        self.entities_at(position)
            .iter()
            .all(|e| entity_type.can_pass(&e.e_type()))
    }
    pub fn observe_as(&self, entity: &WorldEntity) -> impl Observation + '_ {
        let radius = std::cmp::max(MAP_HEIGHT, MAP_WIDTH) as Coord;
        self.observe_in_radius(entity, radius)
    }
    pub fn observe_in_radius(&self, entity: &WorldEntity, radius: Coord) -> impl Observation + '_ {
        let pos = match self.positions.get(entity) {
            Some(pos) => pos.clone(),
            None => Position {
                x: (MAP_WIDTH / 2) as Coord,
                y: (MAP_HEIGHT / 2) as Coord,
            },
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
            return &self.cells[y][x][..];
        }
        return &[];
    }
    fn entities_at_mut(&mut self, position: Position) -> &mut C {
        let x = position.x as usize;
        let y = position.y as usize;
        &mut self.cells[y][x]
    }
    pub(crate) fn move_unchecked(&mut self, entity: &WorldEntity, new_position: Position) {
        if let Some((_, old_pos)) = self.positions.insert(entity, new_position) {
            self.entities_at_mut(old_pos).retain(|e| e != entity);
        }
        self.entities_at_mut(new_position).push(entity.clone());
    }
    fn eat_unchecked(
        &mut self,
        eater: &WorldEntity,
        eaten: &WorldEntity,
        mut decrement: f32,
    ) -> Outcome {
        let mut remove = false;
        if let Some(phys) = self.physical_states.get_mut(eaten) {
            if phys.meat.0 <= decrement {
                remove = true;
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
                self.entities_at_mut(pos).retain(|e| e != eaten)
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
                    .map(move |(res, x)| (x, y.clone(), &**res))
            })
    }
    pub fn iter_cells<'a>(&'a self) -> impl Iterator<Item = (Position, &'a C)> + 'a {
        self.cells.iter().enumerate().flat_map(|(y, row)| {
            row.iter().enumerate().map(move |(x, c)| {
                (
                    Position {
                        x: x as Coord,
                        y: y as Coord,
                    },
                    c,
                )
            })
        })
    }
}
impl World<Occupancy> {
    pub fn empty() -> Self {
        Self {
            cells: Occupancy::empty_init(),
            positions: Storage::new(),
            physical_states: Storage::new(),
            events: Vec::new(),
        }
    }
    pub fn confident_act<'a>(
        &'a mut self,
        actions: impl IntoIterator<Item = &'a (WorldEntity, Action)>,
        observer: WorldEntity,
    ) {
        let mut move_list = Vec::new();
        for &(actor, action) in actions {
            match self.act_one(&actor, action) {
                Err(err) => error!(
                    "Observed action of {} as modelled by {} failed: {}",
                    actor, observer, err
                ),
                Ok(Outcome::Moved(dir)) => {
                    move_list.push((actor, dir));
                    self.events.push(Event {
                        actor,
                        outcome: Outcome::Moved(dir),
                    });
                }
                Ok(outcome) => self.events.push(Event { actor, outcome }),
            }
        }
        for (actor, dir) in move_list {
            if let Some(pos) = self.positions.get(actor).and_then(|p| p.step(dir)) {
                self.move_unchecked(&actor, pos);
            } else {
                warn!("Processing invalid move {:?} by {}", dir, actor);
            }
        }
    }
    pub fn act_uncertain<'a>(
        &'a mut self,
        action: Action,
        entity: WorldEntity,
        position: Position,
        mut prob: f32,
    ) -> impl Iterator<Item = (Event, f32)> + 'a {
        assert!(prob > 0.0 && prob <= 1.0);
        if (prob > 0.0) {
            let res: Result<Result<_, _>, _> = try {
                match action {
                    Action::Move(dir) => match position.step(dir) {
                        Some(pos) if self.can_pass(&entity, pos) => {
                            prob *= (&*self).can_pass_prob(&entity, pos);
                            let phys = self
                                .physical_states
                                .get_mut(entity)
                                .ok_or("Entity has no physical state but tries to move")?;
                            if phys.partial_move(pos) >= MoveProgress(1.0) {
                                phys.move_target = None;
                                phys.move_progress = MoveProgress::default();
                                self.move_uncertain(&entity, pos, prob);
                                Ok(Outcome::Moved(dir))
                            } else {
                                Ok(Outcome::Incomplete)
                            }
                        }
                        Some(_) => Err("blocked move"),
                        None => Err("Invalid move"),
                    },
                    Action::Eat(target) => {
                        if self.positions.get(&target) == Some(&position) {
                            if entity.e_type().can_eat(&target.e_type()) {
                                Ok(self.eat_unchecked(
                                    &entity,
                                    &target,
                                    Self::EATING_DECREMENT * prob,
                                ))
                            } else {
                                Err("entity can't eat that")
                            }
                        } else {
                            Err("can only eat things in the same tile")
                        }
                    }
                    Action::Attack(opponent) => {
                        let pos = self
                            .positions
                            .get(&opponent)
                            .ok_or("Entity tries to attack opponent with no known position")?;
                        if pos != &position
                            && !(self.can_pass(&entity, *pos) && position.is_neighbour(pos))
                        {
                            Err("Cannot reach attacked opponent")?;
                        }
                        let phys = self
                            .physical_states
                            .get_mut(entity)
                            .ok_or("Entity has no physical state but tries to attack")?;
                        if let Some(mut attack) = phys.attack {
                            attack.0 *= prob;
                            if let Some(phys_target) = self.physical_states.get_mut(&opponent) {
                                let damage = phys_target.health.suffer(attack);
                                Ok(Outcome::Hurt {
                                    damage,
                                    target: opponent,
                                    lethal: phys_target.is_dead(),
                                })
                            } else {
                                Err("opponent has no physical state")
                            }
                        } else {
                            Err("entity incapable of attacking")
                        }
                    }
                    Action::Idle => Ok(Outcome::Rested),
                }
            };
            let outcome = res.and_then(std::convert::identity).ok();
            return outcome
                .map(|outcome| {
                    (
                        Event {
                            actor: entity,
                            outcome,
                        },
                        prob,
                    )
                })
                .into_iter();
        }
        None.into_iter()
    }
    fn move_uncertain(&mut self, entity: &WorldEntity, new_position: Position, prob: f32) {
        if let Some((_, old_pos)) = self.positions.insert(entity, new_position) {
            if self.entities_at_mut(old_pos).reduce_prob(entity, prob) > prob {
                self.positions.insert(entity, old_pos);
            }
        }
        self.entities_at_mut(new_position)
            .push_prob(entity.clone(), prob);
    }
}
impl<C: Cell> std::fmt::Debug for World<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(
            f,
            "World {{ cells : _, physical_states: {:?}, positions: {:?} }}",
            self.physical_states, self.positions
        )
    }
}
