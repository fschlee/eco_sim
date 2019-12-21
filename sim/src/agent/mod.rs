pub mod emotion;
pub mod estimate;
pub mod estimator;

use log::{error, info};
use rand::Rng;
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::{HashMap, LinkedList};

use super::entity::*;
use super::entity_type::EntityType;
use super::world::*;
use crate::position::{Coord, Dir, Position};
use crate::util::f32_cmp;
use crate::world::cell::Cell;
use crate::Action::Eat;
use crate::Behavior::Partake;

use crate::agent::estimator::{Estimator, LearningEstimator};
pub use emotion::{EmotionalState, Hunger};
use estimate::PointEstimateRep;
use estimator::{EstimatorMap, MentalStateRep};

use std::collections::hash_map::RandomState;

#[derive(Ord, PartialOrd, Eq, PartialEq, Clone, Debug)]
pub enum Behavior {
    Search(Vec<EntityType>),
    Travel(Position),
    FleeFrom(WorldEntity),
    Hunt(WorldEntity),
    Partake(WorldEntity),
}
impl Behavior {
    pub fn fmt(bhv: &std::option::Option<Behavior>) -> String {
        match bhv {
            None => format!("Undecided"),
            Some(Behavior::Travel(goal)) => format!("traveling towards {:?}", goal),
            Some(Behavior::FleeFrom(enemy)) => format!("fleeing from {:?}", enemy.e_type()),
            Some(Behavior::Hunt(prey)) => format!("hunting {:?} ", prey.e_type()),
            Some(Behavior::Partake(food)) => format!("partaking of  {:?} ", food.e_type()),
            Some(Behavior::Search(target)) => format!("searching for {:?}", target),
        }
    }
}

const HUNGER_THRESHOLD: Hunger = Hunger(0.2);

const HUNGER_INCREMENT: f32 = 0.00001;

const FLEE_THREAT: f32 = 5.0;

pub type Reward = f32;

pub type Threat = f32;

#[derive(Clone, Debug)]
pub struct MentalState {
    pub id: WorldEntity,
    pub emotional_state: EmotionalState,
    pub current_action: Action,
    pub current_behavior: Option<Behavior>,
    pub sight_radius: Coord,
    pub use_mdp: bool,
    pub rng: rand_xorshift::XorShiftRng,
    pub score: Reward,
    pub world_model: Option<Box<World<Occupancy>>>,
}

impl<R: MentalStateRep> From<&R> for MentalState {
    fn from(rep: &R) -> Self {
        rep.into_ms()
    }
}

impl MentalState {
    const UNSEEN_PROB_MULTIPLIER: f32 = 0.9;
    const PROB_REMOVAL_THRESHOLD: f32 = 0.1;
    pub fn new(
        entity: WorldEntity,
        food_preferences: Vec<(EntityType, Reward)>,
        use_mdp: bool,
        have_world_model: bool,
    ) -> Self {
        debug_assert!(food_preferences.len() > 0);
        let emotional_state = EmotionalState::new(food_preferences);
        let world_model = if have_world_model {
            Some(Box::new(World::empty()))
        } else {
            None
        };
        Self {
            id: entity,
            emotional_state,
            current_action: Action::Idle,
            current_behavior: None,
            sight_radius: 5,
            use_mdp,
            rng: rand::SeedableRng::seed_from_u64(entity.id() as u64),
            score: 0.0,
            world_model,
        }
    }
    pub fn decide(
        &mut self,
        physical_state: &PhysicalState,
        own_position: Position,
        observation: &impl Observation,
        estimator: &impl Estimator,
    ) -> Action {
        if let Some(world_model) = self.world_model.take() {
            let observation = &&*world_model;
            {
                self.update(physical_state, own_position, observation);
            }
            if self.use_mdp {
                unimplemented!()
                // self.decide_mdp(physical_state, own_position, observation, estimator);
            } else {
                self.decide_simple(physical_state, own_position, observation, estimator);
            }
            self.world_model = Some(world_model);
        } else {
            {
                self.update(physical_state, own_position, observation);
            }
            if self.use_mdp {
                unimplemented!()
                // self.decide_mdp(physical_state, own_position, observation, estimator);
            } else {
                self.decide_simple(physical_state, own_position, observation, estimator);
            }
        }
        self.current_action
    }
    fn update(
        &mut self,
        physical_state: &PhysicalState,
        own_position: Position,
        observation: &impl Observation,
    ) {
        self.emotional_state += Hunger((20.0 - physical_state.satiation.0) * HUNGER_INCREMENT);
        match self.current_action {
            Action::Eat(food) => {
                if self.emotional_state.hunger().0 <= 0.0
                    || !observation.entities_at(own_position).contains(&food)
                {
                    self.current_action = Action::Idle;
                }
            }
            Action::Attack(opponent) => {
                let can_attack = {
                    if let (Some(pos), Some(phys)) = (
                        observation.observed_position(&opponent),
                        observation.observed_physical_state(&opponent),
                    ) {
                        pos == own_position && phys.health.0 > 0.0
                    } else {
                        false
                    }
                };
                if !can_attack {
                    self.current_action = Action::Idle;
                }
            }
            Action::Move(d) => if let Some(pos)= own_position.step(d) {
                if observation.known_can_pass(&self.id, pos) == Some(false) {
                    self.current_action = Action::Idle;
                }
            } else {
                self.current_action = Action::Idle;
            }
            Action::Idle =>  (),
        }
    }
    fn update_on_events<'a>(& mut self, events: impl IntoIterator<Item=& 'a Event> + Copy) -> Reward {
        use Outcome::*;
        let mut score = 0.0;
        for Event { actor, outcome } in events {
            if *actor == self.id {
                match outcome {
                    Rested | Incomplete => (),
                    Moved(_) => self.current_action = Action::Idle,
                    Consumed(food, tp) => {
                        if let Some(r) = self.lookup_preference(*tp) {
                            score += r * food.0 * self.emotional_state.hunger().0
                        }
                    }
                    Hurt {
                        damage,
                        target,
                        lethal,
                    } => {
                        if *lethal {
                            self.current_action = Action::Idle
                        }
                    }
                }
            } else {
                match outcome {
                    Hurt {
                        damage,
                        target,
                        lethal,
                    } if *target == self.id => {
                        score -= damage.0 * 100.0;
                        if *lethal {
                            score -= 10e5;
                        }
                    }
                    _ => (),
                }
            }
        }
        self.score += score;
        return score;
    }
    pub fn update_world_model<'a>(
        &'a mut self,
        actions: impl IntoIterator<Item = itertools::Either<WorldEntity, &'a (WorldEntity, Action)>>,
        observation: &'a impl Observation,
        estimator: &impl Estimator,
    ) {
        use itertools::Itertools;
        if let Some(ref mut wm) = self.world_model {
            let (unseen, mut confident_actions): (Vec<WorldEntity>, Vec<(WorldEntity, Action)>) =
                actions.into_iter().partition_map(std::convert::identity);

            let mut expected_actions = Vec::new();
            for (pos, c) in wm.iter_cells().filter(|(p, c) |observation.is_observed(p)) {
                use Occupancy::*;
                match c {
                    Empty=> (),
                    Unknown => (),
                    Filled(v) =>
                        v.iter().filter(|e  |e.e_type().is_mobile() && unseen.contains(e)).for_each(|e|
                            if let (Some(phys), Some(mut ms)) = (wm.physical_states.get(e), estimator.invoke(*e)) {
                                confident_actions.push((*e, ms.decide(phys, pos, &&**wm, estimator)))
                            }
                        ),
                    ExpectedFilled(vs, ps) =>
                        vs.iter().zip(ps.iter()).filter(|(e, p)| **p > Self::PROB_REMOVAL_THRESHOLD && e.e_type().is_mobile() && unseen.contains(e)).for_each(|(e, p)|{
                            if let (Some(phys), Some(mut ms)) = (wm.physical_states.get(e), estimator.invoke(*e)) {
                                let action = ms.decide(phys, pos, &&**wm, estimator);
                                if *p > 0.9 {
                                    confident_actions.push((*e, action));
                                } else if *p > 0.1 {
                                    expected_actions.push((*e, action, pos, *p));
                                }

                            }
                        })
                }
            }
            wm.confident_act(&confident_actions, self.id);
            for (we, a ,pos,  p) in expected_actions {
                wm.act_uncertain(a, we, pos, p).count();
            }
            wm.advance();

            for (pos, opt_cell) in observation.iter() {
                if let Some(cell) = opt_cell {
                    wm.cells[pos.y as usize][pos.x as usize] = Occupancy::Empty;
                    for e in (**cell).iter() {
                        wm.move_unchecked(e, pos);
                        observation
                            .observed_physical_state(e)
                            .map(|p| wm.physical_states.insert(e, p.clone()));
                    }
                } else {
                    let mut clear = false;
                    match  &mut wm.cells[pos.y as usize][pos.x as usize] {
                        Occupancy::Empty => (),
                        Occupancy::Unknown => (),
                        o @ Occupancy::Filled(_) => {
                            let mut tmp = Occupancy::Unknown;
                            std::mem::swap(o, &mut tmp);
                            if let Occupancy::Filled(v) = tmp {
                                let probs = v
                                    .iter()
                                    .map(|e| {
                                        if e.e_type().is_mobile() {
                                            Self::UNSEEN_PROB_MULTIPLIER
                                        } else {
                                            1.0
                                        }
                                    })
                                    .collect();
                                *o = Occupancy::ExpectedFilled(v, probs);
                            }
                        }
                        Occupancy::ExpectedFilled(ws, ps) => {
                            let mut i = 0;
                            ws.retain(|w| {
                                if w.e_type().is_mobile() {
                                    i += 1;
                                    true
                                } else {
                                    ps[i] *= Self::UNSEEN_PROB_MULTIPLIER;
                                    // also filters out NaN
                                    if ps[i] >= Self::PROB_REMOVAL_THRESHOLD {
                                        i += 1;
                                        true
                                    } else {
                                        ps.remove(i);
                                        false
                                    }
                                }
                            });
                            if ws.len() < 1 {
                                clear = true;
                            }
                        }
                    }
                    if clear {
                        wm.cells[pos.y as usize][pos.x as usize] = Occupancy::Unknown;
                    }

                }
            }
        }
    }
    fn decide_simple(
        &mut self,
        physical_state: &PhysicalState,
        own_position: Position,
        observation: &impl Observation,
        estimator: &impl Estimator,
    ) {
        let own_type = self.id.e_type();
        let possible_threat =
            max_threat(self.calculate_threat(own_position, observation, estimator));
        if let Some((predator, threat)) = possible_threat {
            if threat > FLEE_THREAT {
                self.current_behavior = Some(Behavior::FleeFrom(predator.clone()))
            }
        }
        use lazysort::SortedBy;
        if self.current_behavior.is_none() && self.emotional_state.hunger() > HUNGER_THRESHOLD {
            if let Some((reward, food, position)) = observation
                .find_closest(own_position, |e, w| self.id.e_type().can_eat(&e.e_type()))
                .filter_map(|(e, p)| {
                    if let Some(rw) = self.lookup_preference(e.e_type()) {
                        let dist = own_position.distance(&p) as f32 * 0.05;
                        Some((rw - dist, e, p))
                    } else {
                        None
                    }
                })
                .sorted_by(
                    |(rw1, _, _), (rw2, _, _)| f32_cmp(rw2, rw1), // descending sort
                )
                .filter(|(_, _, p)| {
                    match max_threat(self.calculate_threat(*p, observation, estimator)) {
                        Some((_, threat)) if threat > FLEE_THREAT => false,
                        _ => true,
                    }
                })
                .next()
            {
                match observation.observed_physical_state(&food) {
                    Some(ps)
                        if ps.health > Health(0.0)
                            && observation.known_can_pass(&self.id, position) == Some(true) =>
                    {
                        self.current_behavior = Some(Behavior::Hunt(food));
                    }
                    _ => {
                        self.current_behavior = Some(Behavior::Partake(food));
                    }
                }
            } else {
                self.current_behavior = Some(Behavior::Search(
                    self.food_preferences().map(|(f, r)| f.clone()).collect(),
                ));
            }
        }
        match self.current_behavior.clone() {
            None => (),
            Some(Behavior::FleeFrom(predator)) => {
                let mut escaped = false;
                if observation.known_can_pass(&predator, own_position) != Some(false) {
                    if let Some(pos) = observation.observed_position(&predator) {
                        let d = pos.distance(&own_position);
                        let mut new_pos = own_position;
                        let mut value = 0;
                        for n in own_position.neighbours() {
                            if observation.known_can_pass(&self.id, n) == Some(true) {
                                let v = match (
                                    observation.known_can_pass(&predator, n),
                                    pos.distance(&n).cmp(&d),
                                ) {
                                    (Some(false), _) => 10,
                                    (None, Ordering::Greater) => 4,
                                    (None, Ordering::Equal) => 2,
                                    (None, Ordering::Less) => 1,
                                    (Some(true), Ordering::Greater) => 3,
                                    (Some(true), Ordering::Equal) => 1,
                                    (Some(true), Ordering::Less) => 0,
                                };
                                if v > value {
                                    value = v;
                                    new_pos = n;
                                }
                            }
                        }
                        if new_pos != own_position {
                            self.current_action = Action::Move(own_position.dir(&new_pos));
                        }
                    } else {
                        escaped = true;
                    }
                } else {
                    escaped = true;
                }
                if escaped {
                    self.current_behavior = None;
                }
            }
            Some(Behavior::Hunt(prey)) => match observation.observed_position(&prey) {
                Some(pos) if observation.known_can_pass(&self.id, pos) != Some(false) => {
                    if pos == own_position || own_position.is_neighbour(&pos) {
                        match observation.observed_physical_state(&prey) {
                            Some(ps) if ps.is_dead() => {
                                self.current_behavior = Some(Partake(prey));
                                if pos == own_position {
                                    self.current_action = Eat(prey);
                                }
                            }
                            _ => self.current_action = Action::Attack(prey),
                        }
                    } else if let Some(path) = self.path(own_position, pos, observation) {
                        self.current_action = Action::Move(path[0]);
                    } else {
                        self.current_behavior = None;
                    }
                }
                _ => self.current_behavior = None,
            },
            Some(Behavior::Partake(food)) => match observation.observed_position(&food) {
                Some(pos) if observation.known_can_pass(&self.id, pos) != Some(false) => {
                    if pos == own_position {
                        self.current_action = Eat(food);
                    } else if let Some(path) = self.path(own_position, pos, observation) {
                        self.current_action = Action::Move(path[0]);
                    } else {
                        self.current_behavior = None;
                    }
                }
                _ => self.current_behavior = None,
            },
            Some(Behavior::Search(list)) => {
                if let Some(_) = observation
                    .find_closest(own_position, |e, w| list.contains(&e.e_type()))
                    .next()
                {
                    self.current_behavior = None;
                } else {
                    self.random_step(own_position, observation);
                }
            }
            Some(Behavior::Travel(goal)) => {
                if let Action::Move(_) = self.current_action {
                } else {
                    if let Some(path) = self.path(own_position, goal, observation) {
                        self.current_action = Action::Move(path[0]);
                    } else {
                        self.current_behavior = None;
                    }
                }
            }
        }
        match self.current_action.clone() {
            Action::Move(dir) => {
                if let Some(Behavior::FleeFrom(_)) = self.current_behavior {
                } else {
                    match own_position
                        .step(dir)
                        .and_then(|p| max_threat(self.calculate_threat(p, observation, estimator)))
                    {
                        Some((_, threat)) if threat > FLEE_THREAT => {
                            self.current_action = Action::Idle;
                            self.current_behavior = None;
                        }
                        _ => (),
                    }
                }
            }
            _ => (),
        }
    }
    /*
    fn decide_mdp(
        &mut self,
        physical_state: &PhysicalState,
        own_position: Position,
        observation: &impl Observation,
        estimator: &impl Estimator,
    ) {
        let (_, expected_world, _) = observation.into_expected(|_| Occupancy::unknown(), thread_rng());
        let world_expectation = &expected_world;
        let direct_reward = | action, hunger | {
            match action {
                Action::Move(_) => {
                    return 0.1;
                },
                Action::Eat(food) => {
                    if hunger > HUNGER_THRESHOLD {
                        if let Some((_, pref)) = self.food_preferences().find(|(et, pref)| et == &food.e_type()) {
                            return 0.2 + pref;
                        }
                    }
                    return 0.0;
                },
                Action::Attack(_) => return 0.0,
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
                let mut reward = direct_reward(Action::Eat(*food), ms.emotional_state.hunger());
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
                    let mut reward = direct_reward(Action::Move(neighbor), ms.emotional_state.hunger());
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
        // println!("expectation {:?}", reward_expectations[0][own_position.y as usize][own_position.x as usize]);
        let act = policy[0][own_position.y as usize][own_position.x as usize];
        if act.is_some() {
            self.current_action = act;
        }
    } */
    /*
        fn decide_hierarchical(
            &mut self,
            physical_state: &PhysicalState,
            own_position: Position,
            observation: & impl Observation,
            estimator: &impl Estimator,
        ) {
            let threat = self.calculate_threat( own_position, observation, estimator );

            // todo switch behaviors
            match &self.current_behavior.clone() {
                None => (),
                Some(Behavior::Hunt(prey)) => (),
                Some(Behavior::Search(foods)) => (),
                Some(Behavior::FleeFrom(predator)) => (),
                Some(Behavior::Partake(food)) => (),
            }

            // todo action based on behavior
            match &self.current_behavior {
                None => (),
                Some(Behavior::Hunt(prey)) => (),
                Some(Behavior::Search(foods)) => (),
                Some(Behavior::FleeFrom(predator)) => (),
                Some(Behavior::Partake(food)) => (),
            };
        }
    */

    // Threats are unordered
    fn calculate_threat<'a>(
        &'a self,
        //  physical_state: &PhysicalState,
        own_position: Position,
        observation: &'a impl Observation,
        estimator: &'a impl Estimator,
    ) -> impl Iterator<Item = (WorldEntity, Threat)> + 'a {
        let mut rng = self.rng.clone();
        observation
            .find_closest(own_position, move |e, w| {
                e.e_type().can_eat(&self.id.e_type())
            })
            .filter_map(move |(entity, position)| {
                match observation.path_as(position, own_position, &entity) {
                    Some(v) => {
                        let mut total = 0.0;
                        let inv_dist = 1.0 / v.len() as f32;
                        debug_assert!(!inv_dist.is_nan());

                        for pred_ms in estimator.invoke_sampled(entity, &mut rng, 10) {
                            pred_ms.lookup_preference(self.id.e_type()).map(|pref| {
                                let mut score =
                                    50.0 * pred_ms.emotional_state.hunger().0 * pref * inv_dist;
                                if let Some(Behavior::Hunt(prey)) = pred_ms.current_behavior {
                                    if prey == self.id {
                                        score += 100.0 * inv_dist;
                                    } else {
                                        score *= 0.5;
                                    }
                                }
                                total += score;
                            });
                        }

                        Some((entity, total))
                    }
                    _ => None,
                }
            })
    }
    pub fn path(
        &self,
        current: Position,
        goal: Position,
        observation: &impl Observation,
    ) -> Option<Vec<Dir>> {
        observation.path_as(current, goal, &self.id)
    }

    pub fn random_step(&mut self, current: Position, observation: &impl Observation) {
        use rand::seq::SliceRandom;
        if let Some(step) = current
            .neighbours()
            .into_iter()
            .filter(|p| observation.known_can_pass(&self.id, *p) == Some(true))
            .collect::<Vec<Position>>()
            .choose(&mut self.rng)
        {
            self.current_action = Action::Move(current.dir(step));
        }
    }
    pub fn lookup_preference(&self, entity_type: EntityType) -> Option<Reward> {
        if !self.id.e_type().can_eat(&entity_type) {
            return None;
        }
        let pref = self.emotional_state.pref(entity_type).0;
        debug_assert!(
            !pref.is_nan(),
            "{} pref for {:?} is NaN",
            self.id,
            entity_type
        );
        Some(pref)
    }
    pub fn respawn_as(&mut self, entity: &WorldEntity) {
        self.id = entity.clone();
        let fp = self.food_preferences().collect();
        self.emotional_state = EmotionalState::new(fp);
        self.current_action = Action::Idle;
        self.current_behavior = None;
    }
    pub fn food_preferences<'a>(&'a self) -> impl Iterator<Item = (EntityType, f32)> + 'a {
        let own_et = self.id.e_type();
        EntityType::iter()
            .zip(self.emotional_state.preferences())
            .filter_map(move |(et, r)| {
                if own_et.can_eat(&et) {
                    Some((et, (*r).0))
                } else {
                    None
                }
            })
    }
    // pub fn set_preference()
}

#[derive(Clone, Debug, Default)]
pub struct AgentSystem {
    pub mental_states: Storage<MentalState>,
    pub estimator_map: EstimatorMap,
    pub actions: Vec<(WorldEntity, Action)>,
    pub killed: Vec<WorldEntity>,
    old_positions: Storage<Position>,
}

impl AgentSystem {
    pub fn advance<C: Cell>(&mut self, world: &mut World<C>, entity_manager: &mut EntityManager) {
        use itertools::{Either, Itertools};
        use rayon::prelude::*;
        let mut res: LinkedList<_> = {
            let Self {
                ref mut mental_states,
                ref estimator_map,
                ref actions,
                ref old_positions,
                ..
            } = self;
            mental_states.par_iter_mut().map(|mental_state|  {
                let entity = mental_state.id;
                match (
                    world.get_physical_state(&entity),
                    world.positions.get(&entity),
                    estimator_map.get(entity.into()),
                ) {
                    (Some(physical_state), Some(position), Some(estimator)) => {
                        if physical_state.is_dead() {
                            info!("Agent {:?} died", entity);
                            Either::Right(entity.clone())
                        } else {
                            let sight = mental_state.sight_radius;
                            let observation = world.observe_in_radius(&entity, sight);
                            let observed_actions = actions.iter().map(|t | {
                                match (old_positions.get(t.0), old_positions.get(entity)) {
                                    (Some(p), Some(old_pos)) if old_pos.distance(p) <= sight => Either::Right(t),
                                    _ => Either::Left(t.0)
                                }
                            });
                            mental_state.update_world_model(observed_actions, &observation, estimator);
                            let action = mental_state.decide(physical_state, *position, &observation, estimator);
                            Either::Left((entity, action))
                        }
                    }
                    (ps, p, _) => {
                        error!("Agent {:?} killed due to incomplete data:physical state {:?}, postion {:?}", entity, ps.is_some(), p.is_some());
                        Either::Right(entity.clone())
                    },
                }
            }).collect()
        };
        self.actions.clear();
        self.killed.clear();
        for either in res {
            match either {
                Either::Left(a) => self.actions.push(a),
                Either::Right(k) => self.killed.push(k),
            }
        }
        {
            let Self {
                ref mut estimator_map,
                ref actions,
                ref mental_states,
                ..
            } = self;
            fn adapter(ms: &MentalState) -> Option<&World<Occupancy>> {
                ms.world_model.as_deref()
            }
            let world_models = StorageAdapter::new(mental_states, Box::new(adapter));
            estimator_map.par_iter_mut().for_each(|est| {
                for (entity, action) in actions {
                    if let Some(other_pos) = world.positions.get(entity) {
                        est.learn(*action, *entity, *other_pos, world, &world_models);
                    }
                }
            });
        }

        world.act(&self.actions);
        for ms in self.mental_states.iter_mut() {
            ms.update_on_events(&world.events);
        }
        {
            self.estimator_map.par_iter_mut().for_each(|est|{
                est.update_on_events(&world.events, None);
            });
        }
        {
            let Self {
                ref mut estimator_map,
                ref mut mental_states,
                ref killed,
                ..
            } = self;
            for entity in killed {
                if let Some(mut ms) = mental_states.remove(entity) {
                    let new_e = world.respawn(&entity, &mut ms, entity_manager);
                    debug_assert_eq!(ms.id, new_e);
                    mental_states.insert(&new_e, ms);
                    estimator_map.rebind_estimator(*entity, new_e);
                }
            }
        }
    }
    pub fn get_representation_source<'a>(
        &'a self,
        entity: Entity,
    ) -> Option<impl Iterator<Item = &impl MentalStateRep> + 'a> {
        self.estimator_map.get_representation_source(entity)
    }
    pub fn init<C: Cell>(
        agents: Vec<WorldEntity>,
        world: &World<C>,
        use_mdp: bool,
        have_world_model: bool,
        mut rng: impl Rng,
    ) -> Self {
        let mut estimator_map = EstimatorMap::default();
        let mut mental_states = Storage::new();
        for agent in &agents {
            let food_prefs = EntityType::iter()
                .filter(|e| agent.e_type().can_eat(e))
                .map(|e| (e.clone(), rng.gen_range(0.0, 1.0)))
                .collect();
            let ms = MentalState::new(agent.clone(), food_prefs, use_mdp, have_world_model);
            estimator_map.insert(&ms);
            mental_states.insert(agent, ms);
        }
        Self {
            mental_states,
            estimator_map,
            actions: Vec::new(),
            killed: Vec::new(),
            old_positions: Storage::new(),
        }
    }
    pub fn threat_map<C: Cell>(&self, we: &WorldEntity, world: &World<C>) -> Vec<f32> {
        let mut vec = Vec::with_capacity(MAP_HEIGHT * MAP_WIDTH);
        if let (Some(ms), Some(est), Some(current_pos)) = (
            self.mental_states.get(we),
            self.estimator_map.get(we.into()),
            world.positions.get(we),
        ) {
            let observation = RadiusObservation::new(ms.sight_radius, *current_pos, world);
            for y in 0..MAP_HEIGHT as Coord {
                for x in 0..MAP_WIDTH as Coord {
                    let mut sum = 0.0;
                    for (_, threat) in ms.calculate_threat(Position { x, y }, &observation, est) {
                        sum += threat;
                    }
                    vec.push(sum);
                }
            }
        }
        vec
    }
}
#[inline]
fn max_threat(
    threats: impl Iterator<Item = (WorldEntity, Threat)>,
) -> Option<(WorldEntity, Threat)> {
    threats.max_by(|(_, t1), (_, t2)| f32_cmp(t1, t2))
}
