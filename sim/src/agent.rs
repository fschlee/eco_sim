
use rand::{Rng, thread_rng};
use std::cmp::Ordering;
use log::error;

use super::world::*;
use super::entity::*;
use std::collections::BinaryHeap;

#[derive(Eq, PartialEq, Clone, Debug)]
pub struct PathNode {
    pub pos : Position,
    pub exp_cost: u32,
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

type Threat = f32;

#[derive(Clone, Debug)]
pub struct MentalState {
    pub id: Entity,
    pub e_type: EntityType,
    pub hunger: Hunger,
    pub food_preferences: Vec<(EntityType, Reward)>,
    pub current_action: Option<Action>,
    pub current_behavior: Option<Behavior>,
    pub sight_radius: u32,
    pub use_mdp: bool,
    pub rng: rand_xorshift::XorShiftRng,
    estimates: Storage<Estimate>
}

impl MentalState {
    pub fn new(entity: Entity, e_type: EntityType, food_preferences: Vec<(EntityType, Reward)>, use_mdp: bool) -> Self {
        assert!(food_preferences.len() > 0);
        Self{
            id: entity,
            e_type,
            hunger: Hunger::default(),
            food_preferences,
            current_action: None,
            current_behavior: None,
            sight_radius: 5,
            use_mdp,
            rng: rand::SeedableRng::seed_from_u64(entity.id as u64),
            estimates: Storage::new()
        }
    }
    pub fn decide(
        &mut self,
        physical_state: &PhysicalState,
        own_position: Position,
        observation: impl Observation,
    ) -> Option<Action> {
        self.update(physical_state, own_position, observation.clone());
        if self.use_mdp {
            self.decide_mdp(physical_state, own_position, observation);
        } else {
            self.decide_simple( physical_state, own_position, observation);
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
        physical_state: &PhysicalState,
        own_position: Position,
        observation: impl Observation,
    ) {
        let own_type = self.e_type.clone();
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
                    Some(other_type) => self.e_type.can_eat(other_type),
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
                                if let Some(path) = self.path(own_position, position, observation.clone()){
                                    self.current_action = Some(Action::Move(path[0]));
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
                        else if let Some(path) = self.path(own_position, pos, observation.borrow()){
                            self.current_action = Some(Action::Move(path[0]));
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
    fn decide_mdp(
        &mut self,
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
    fn decide_hierarchical(
        &mut self,
        physical_state: &PhysicalState,
        own_position: Position,
        observation: impl Observation,
    ) {
        let threat = self.calculate_threat(physical_state, own_position, observation );

        // todo switch behaviors
        match &self.current_behavior.clone() {
            None => (),
            Some(Behavior::Hunt(prey)) => (),
            Some(Behavior::Search(foods)) => (),
            Some(Behavior::FleeFrom(predator)) => (),
        }

        // todo action based on behavior
        match &self.current_behavior {
            None => (),
            Some(Behavior::Hunt(prey)) => (),
            Some(Behavior::Search(foods)) => (),
            Some(Behavior::FleeFrom(predator)) => (),
        };
    }


    // Threats are unordered
    fn  calculate_threat(
        &self,
        physical_state: &PhysicalState,
        own_position: Position,
        observation: impl Observation,
    ) -> Vec<(Entity, Threat)>{
        observation.find_closest(own_position, |e, w| {
            match w.entity_types.get(e) {
                Some(other_type) => other_type.can_eat(&self.e_type),
                None => false
            }
        }).filter_map(|(entity, position)| {
            match (observation.get_type(&entity), self.path_as(position, own_position, &entity, observation.borrow())){

                (Some(other_type), Some(v)) => {
                    Some((entity, 1.0 / v.len() as f32))
                },
                _ => None
            }
        }).collect()
    }
    pub fn path(&self, current: Position, goal: Position, observation: impl Observation) -> Option<Vec<Position>>  {
        self.path_as(current, goal, &self.id, observation)
    }

    pub fn path_as(&self, start: Position, goal: Position, entity: &Entity, observation: impl Observation) -> Option<Vec<Position>> {
        let mut came_from : PositionMap<Position> = PositionMap::new();
        let mut cost = PositionMap::new();
        let mut queue = BinaryHeap::new();
        cost.insert(start, 0u32);
        queue.push(PathNode { pos: start, exp_cost: start.distance(&goal)});
        while let Some(PathNode{pos, exp_cost}) = queue.pop() {
            let base_cost = *cost.get(&pos).unwrap();
            for n in pos.neighbours() {
                if n == goal {
                    let mut v = vec![pos, n];
                    let mut current = pos;
                    while let Some(from) = came_from.get(&current) {
                        v.push(from.clone());
                        current = *from;
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
                        came_from.insert(n, pos);
                        queue.push(PathNode{pos: n, exp_cost : base_cost + 1 + n.distance(&goal) })
                    }
                }
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
    fn estimate(& mut self, entity: & Entity, e_type: EntityType) -> &mut Estimate {
        self.estimates.get_or_insert_with(entity, ||
            Estimate{
            id: entity.clone(),
            physical_state: e_type.typical_physical_state().unwrap_or(PhysicalState {
                health: Health(0.0),
                meat: Meat(0.0),
                attack: None,
                satiation: Satiation(0.0)
            }),
            entity_type: e_type,
            hunger: Default::default(),
            food_preferences: ENTITY_TYPES.iter().filter_map(|other| {
                if e_type.can_eat(other) {
                    Some((*other, 0.5))
                }
                else {
                    None
                }
            }).collect(),
            current_action: None,
            current_behavior: None,
            sight_radius: 5,
            use_mdp: false
        })
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
    pub fn sample(&self, seed: u64) -> MentalState {
        let mut sample : MentalState = self.into();
        let mut rng : rand_xorshift::XorShiftRng = rand::SeedableRng::seed_from_u64(seed);
        sample.hunger.0 += rng.gen_range(-0.5, 0.5);
        for (_, pref) in sample.food_preferences.iter_mut() {
            *pref = 1.0f32.min(0.0f32.max( *pref + rng.gen_range(-0.5, 0.5)));
        }
        sample
    }
}

impl Into<MentalState> for &Estimate {
    fn into(self) -> MentalState {
        MentalState {
            id: self.id,
            e_type: self.entity_type,
            hunger: self.hunger,
            food_preferences: self.food_preferences.clone(),
            current_action: self.current_action,
            current_behavior: self.current_behavior.clone(),
            sight_radius: self.sight_radius,
            use_mdp: false,
            rng: rand::SeedableRng::seed_from_u64(self.id.id as u64),
            estimates: Storage::new()
        }
    }
}
impl Estimate {
    pub fn updated(&self, observation: impl Observation, action: Action) -> Option<Estimate> {
        if let Some(pos) = observation.observed_position(&self.id){
            let ms : MentalState= self.into();

            let mut ms1 = ms.clone();
            if let Some(act) = ms1.decide( &(self.physical_state), pos, observation.borrow()){
                if act == action {
                    return None;
                }
                else {
                    let max_tries = 20;
                    for i in 0..max_tries {
                        let mut sample = self.sample(i);
                        if let Some(act) = sample.decide( &(self.physical_state), pos, observation.borrow()){
                            if act == action
                            {
                                let mut est = self.clone();
                                est.update(&sample);
                                return Some(est);
                            }
                    }
                    }

                }
                //Todo
            }
        }
        None
    }

}



#[derive(Clone, Debug, Default)]
pub struct AgentSystem {
    agents: Vec<Entity>,
    pub mental_states: Storage<MentalState>,
}

impl AgentSystem {
    pub fn advance(&mut self, world: &mut World, entity_manager: &mut EntityManager) {
        let mut killed = Vec::new();
        let mut actions = Vec::new();
        for entity in &self.agents{
            let opt_action = match (
                self.mental_states.get_mut(entity),
                world.get_physical_state(entity),
                world.positions.get(entity),
            ) {
                (Some(mental_state), Some(physical_state),  Some(position)) => {
                    if physical_state.is_dead() {
                        killed.push(entity.clone());
                        None
                }
                    else {
                        mental_state.decide(physical_state, *position, world.observe_in_radius(entity, mental_state.sight_radius))
                    }
                }
                (ms, ps, p) => {
                    killed.push(entity.clone());
                    error!("Agent {:?} killed due to incomplete data: mental state {:?}, physical state {:?}, postion {:?}", entity, ms.is_some(), ps.is_some(), p.is_some());
                    None
                },
            };

            if let Some(action) = opt_action {
                world.act(entity, action);
                actions.push((entity.clone(), action));
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
        for agent in &self.agents {
            if let Some(ms) = self.mental_states.get_mut(agent) {
                for (entity, action) in &actions {
                    let e_type = world.entity_types.get(entity).unwrap();
                    let sight = ms.sight_radius;
                    let estimate = ms.estimate(entity, *e_type);
                    if let  Some(mut upd) = estimate.updated(world.observe_in_radius(agent, sight), *action) {
                        std::mem::swap(&mut upd, estimate);
                    }
                }
            }
        }
    }
    pub fn init(agents: Vec<Entity>, world: &World, use_mdp: bool, mut rng: impl Rng) -> Self {
        let mut mental_states = Storage::new();
        for agent in &agents {
            if let Some(et) = world.get_type(agent) {
                let food_prefs = ENTITY_TYPES.iter().filter(|e|et.can_eat(e)).map(|e| (e.clone(), rng.gen_range(0.0, 1.0))).collect();
                mental_states.insert(agent, MentalState::new(agent.clone(), et,food_prefs, use_mdp));
            }

        }
        Self{ agents, mental_states}
    }
}