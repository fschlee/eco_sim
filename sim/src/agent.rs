
use rand::{Rng, thread_rng};
use std::cmp::Ordering;
use log::error;

use super::world::*;
use super::entity::*;

#[derive(Ord, PartialOrd, Eq, PartialEq, Clone, Debug)]
pub enum Behavior {
    Search(Vec<EntityType>),
    FleeFrom(Entity),
    Hunt(Entity),
}

#[derive(PartialOrd, PartialEq, Copy, Clone, Debug, Default)]
pub struct Hunger(pub f32);

const HUNGER_THRESHOLD : Hunger = Hunger(1.0);

const HUNGER_INCREMENT : f32 = 0.0001;

const FLEE_THRESHOLD : u32 = 4;

type Reward = f32;

#[derive(Clone, Debug)]
pub struct MentalState {
    pub id: Entity,
    pub hunger: Hunger,
    pub food_preferences: Vec<(EntityType, Reward)>,
    pub current_action: Option<Action>,
    pub current_behavior: Option<Behavior>,
    pub sight_radius: u32,
    pub use_mdp: bool,
    pub rng: rand_xorshift::XorShiftRng,
}

impl MentalState {
    pub fn new(entity: Entity, food_preferences: Vec<(EntityType, Reward)>, use_mdp: bool) -> Self {
        assert!(food_preferences.len() > 0);
        Self{
            id: entity,
            hunger: Hunger::default(),
            food_preferences,
            current_action: None,
            current_behavior: None,
            sight_radius: 5,
            use_mdp,
            rng: rand::SeedableRng::seed_from_u64(entity.id as u64)
        }
    }
    pub fn decide(
        &mut self,
        own_type: EntityType,
        physical_state: &PhysicalState,
        own_position: Position,
        observation: impl Observation,
    ) -> Option<Action> {
        self.update(physical_state, own_position, observation.clone());
        if self.use_mdp {
            self.decide_mdp(own_type, physical_state, own_position, observation);
        } else {
            self.decide_simple(own_type, physical_state, own_position, observation);
        }
        self.current_action

    }
    fn update(& mut self, physical_state: &PhysicalState, own_position: Position, observation: impl Observation,) {
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
            Some(Action::Attack(opponent)) => {
                let can_attack = {
                    if let (Some(pos), Some(phys)) =
                    (observation.observed_position(&opponent),
                     observation.observed_physical_state(&opponent)) {
                        pos == own_position && phys.health.0 > 0.0
                    } else {
                        false
                    }
                };
                if !can_attack {
                    self.current_action = None;
                }
            }
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
        if let Some((predator, pos)) = observation.find_closest(own_position, |e, w| {
            match w.entity_types.get(e) {
                Some(other_type) => other_type.can_eat(&own_type),
                None => false
            }
        }).next() {
            if pos.distance(&own_position) <= FLEE_THRESHOLD && observation.known_can_pass(&predator, own_position) != Some(false) {
                self.current_behavior = Some(Behavior::FleeFrom(predator))
            }
        }
        if self.current_behavior.is_none() && self.hunger > HUNGER_THRESHOLD {
            if let Some((reward, food, position)) = observation.find_closest(own_position, |e, w| {
                match w.entity_types.get(e) {
                    Some(other_type) => own_type.can_eat(other_type),
                    None => false
                }
            }).filter_map(|(e, p)| {
                if let Some(rw) = observation.get_type(&e).and_then(|et| self.lookup_preference(et))  {
                    let dist = own_position.distance(&p) as f32 * 0.05;
                    Some((rw - dist, e, p))
                } else {
                    None
                }
            }).max_by(|(rw1, _, _), (rw2, _, _)| {
                if rw1 < rw2 {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            }) {
                match observation.observed_physical_state(&food) {
                    Some(ps) if  ps.health > Health(0.0)  && observation.known_can_pass(&self.id, position) == Some(true) => {
                        self.current_behavior = Some(Behavior::Hunt(food));
                    }
                    _ => {
                        if position == own_position {
                            self.current_action = Some(Action::Eat(food));
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
                }

            }
            else {
                self.current_behavior = Some(Behavior::Search(self.food_preferences.iter().map(|(f, r)| f.clone()).collect()));
            }
        }
        match self.current_behavior.clone() {
            None => (),
            Some(Behavior::FleeFrom(predator)) => {
                let mut escaped = false;
                if observation.known_can_pass(&predator, own_position) != Some(false) {
                    if let Some(pos) = observation.observed_position(&predator) {
                        let d = pos.distance(&own_position);
                        let mut step = own_position;
                        let mut value = 0;
                        for n in own_position.neighbours() {
                            if observation.known_can_pass(&self.id, n) == Some(true) {
                                let v = match (observation.known_can_pass(&predator, n), pos.distance(&n).cmp( &d)) {
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
                                    step = n;
                                }
                            }
                        }
                        if step != own_position {
                            self.current_action = Some(Action::Move(step));
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
            },
            Some(Behavior::Hunt(prey)) => {
                match observation.observed_position(&prey) {
                    Some(pos) if observation.known_can_pass(&self.id, pos) != Some(false) => {
                        if pos == own_position {
                            self.current_action = Some(Action::Attack(prey));
                        }
                        else if let Some(step) = self.path(own_position, pos, observation.clone()){
                            self.current_action = Some(Action::Move(step));
                        }
                        else {
                            self.current_behavior = None;
                        }
                    }
                    _ => self.current_behavior = None,
                }
            }
            Some(Behavior::Search(list)) => {
                if let Some(_) = observation.find_closest(own_position, |e, w| {
                    match w.entity_types.get(e) {
                        Some(other_type) => list.contains(other_type),
                        None => false
                    }
                }).next() {
                    self.current_behavior = None;
                }
                else {
                    self.random_step(own_position, observation.clone());
                }
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
        let (_, expected_world, _) = observation.into_expected(|_| None, thread_rng());
        let world_expectation = &expected_world;
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
        // println!("expectation {:?}", reward_expectations[0][own_position.y as usize][own_position.x as usize]);
        let act = policy[0][own_position.y as usize][own_position.x as usize];
        if act.is_some() {
            self.current_action = act;
        }
    }
    pub fn path(&self, current: Position, goal: Position, observation: impl Observation) -> Option<Position> {
        let d = current.distance(&goal);
        for n in current.neighbours(){
            if n.distance(&goal) < d && Some(true) == observation.known_can_pass(&self.id, n) {
                return Some(n);
            }
        }
        None
    }
    pub fn random_step(& mut self, current: Position, observation: impl Observation) {
        use  rand::seq::SliceRandom;
        if let Some(step) = current.neighbours().into_iter()
            .filter(|p|
                observation.known_can_pass(&self.id, *p) == Some(true))
            .collect::<Vec<Position>>().choose(&mut self.rng) {
            self.current_action = Some(Action::Move(*step));
        }
    }
    pub fn lookup_preference(&self, entity_type: EntityType) -> Option<Reward> {
        self.food_preferences.iter().find_map(|(et, rw)| {
            if et == &entity_type
            {
                Some(*rw)
            } else {
                None
            }
        })
    }
    pub fn respawn_as(&mut self, entity: &Entity) {
        self.id = entity.clone();
        self.hunger = Hunger::default();
        self.current_action = None;
        self.current_behavior = None;
    }
}
#[derive(Clone, Debug)]
pub struct Estimate {
    pub id: Entity,
    pub physical_state: PhysicalState,
    pub entity_type: EntityType,
    pub hunger: Hunger,
    pub food_preferences: Vec<(EntityType, Reward)>,
    pub current_action: Option<Action>,
    pub current_behavior: Option<Behavior>,
    pub sight_radius: u32,
    pub use_mdp: bool,
}
impl Estimate {
    pub fn update(&mut self, ms: &MentalState){
        self.hunger = ms.hunger;
        self.food_preferences = ms.food_preferences.clone();
        self.current_behavior = ms.current_behavior.clone();
        self.current_action = ms.current_action;
    }
}

impl Into<MentalState> for &Estimate {
    fn into(self) -> MentalState {
        MentalState {
            id: self.id,
            hunger: self.hunger,
            food_preferences: self.food_preferences.clone(),
            current_action: self.current_action,
            current_behavior: self.current_behavior.clone(),
            sight_radius: self.sight_radius,
            use_mdp: false,
            rng: rand::SeedableRng::seed_from_u64(self.id.id as u64)
        }
    }
}


pub fn update_ms_estimate(observation: impl Observation, action: Action, old_ms: Estimate) -> Estimate {
    if let Some(pos) = observation.observed_position(&old_ms.id){
        let ms : MentalState= (&old_ms).into();
        let mut ms1 = ms.clone();
        if let Some(act) = ms1.decide(old_ms.entity_type, &(old_ms.physical_state), pos, observation){
            if act == action {
                let mut est = old_ms.clone();
                est.update(&ms1);
                return est;
            }
            //Todo
        }
    }
    return old_ms
}


#[derive(Clone, Debug, Default)]
pub struct AgentSystem {
    agents: Vec<Entity>,
    pub mental_states: Storage<MentalState>,
}

impl AgentSystem {
    pub fn advance(&mut self, world: &mut World, entity_manager: &mut EntityManager) {
        let mut killed = Vec::new();
        for entity in &self.agents{
            let opt_action = match (
                self.mental_states.get_mut(entity),
                world.get_physical_state(entity),
                world.entity_types.get(entity),
                world.positions.get(entity),
            ) {
                (Some(mental_state), Some(physical_state), Some(et), Some(position)) => {
                    if physical_state.is_dead() {
                        killed.push(entity.clone());
                        None
                }
                    else {
                        mental_state.decide(*et, physical_state, *position, world.observe_in_radius(entity, mental_state.sight_radius))
                    }
                }
                _ => {
                    killed.push(entity.clone());
                    error!("Agent {:?} killed due to incomplete data", entity);
                    None
                },
            };

            if let Some(action) = opt_action {
                world.act(entity, action);
            }

        }
        for entity in killed {
            self.agents.remove_item(&entity);

            if let Some(mut ms) = self.mental_states.remove(&entity) {
                let new_e = world.respawn(&entity, & mut ms, entity_manager);
                self.mental_states.insert(&new_e, ms);
                self.agents.push(new_e);
            }
        }
    }
    pub fn init(agents: Vec<Entity>, world: &World, use_mdp: bool, mut rng: impl Rng) -> Self {
        let mut mental_states = Storage::new();
        for agent in &agents {
            if let Some(et) = world.get_type(agent) {
                let food_prefs = ENTITY_TYPES.iter().filter(|e|et.can_eat(e)).map(|e| (e.clone(), rng.gen_range(0.0, 1.0))).collect();
                mental_states.insert(agent, MentalState::new(agent.clone(), food_prefs, use_mdp));
            }

        }
        Self{ agents, mental_states}
    }
}