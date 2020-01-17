use log::error;
use ndarray::{s, Array4, ArrayBase, ArrayViewMut, DataMut, Ix2, Ix4, RawData};
#[cfg(feature = "torch")]
use tch::{Device, IndexOp, Kind, Tensor};

use crate::agent::emotion::EmotionalState;
use crate::world::{DefCell, Dir, Position, World};
use crate::{
    Action, Coord, Entity, EntityType, FailReason, MentalState, Source, WorldEntity, MAP_HEIGHT,
    MAP_WIDTH,
};

pub const MAX_REP_PER_CELL: usize = 8;
pub const ENTITY_REP_SIZE: usize = 9;
pub const PHYS_REP_SIZE: usize = ENTITY_REP_SIZE + 3;
pub const MENTAL_REP_SIZE: usize = EmotionalState::SIZE;

const X_SCALE: f32 = 1.0 / MAP_WIDTH as f32;
const Y_SCALE: f32 = 1.0 / MAP_HEIGHT as f32;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum EncodingError {
    IndexMismatch,
    ShapeMismatch,
    InvalidAction,
    PositionNotFound,
    NonStandardContiguousAccess,
}

pub struct ObsvWriter<'a> {
    world: &'a World<DefCell>,
    views: &'a [(WorldEntity, Coord)],
    #[cfg(feature = "torch")]
    device: Device,
}
impl<'a> ObsvWriter<'a> {
    pub const ACTION_SPACE_SIZE: usize = 5 + 6 * MAX_REP_PER_CELL;
    pub const SHAPE: [usize; 4] = [MAP_HEIGHT, MAP_WIDTH, MAX_REP_PER_CELL, ENTITY_REP_SIZE];
    pub fn new(world: &'a World<DefCell>, views: &'a [(WorldEntity, Coord)]) -> Self {
        Self {
            world,
            views,
            #[cfg(feature = "torch")]
            device: Device::Cpu,
        }
    }
    pub fn encode_action(&self, we: WorldEntity, action: Action) -> Result<usize, EncodingError> {
        use Action::*;
        use Dir::*;
        let dir_match = |dir| match dir {
            R => 1,
            L => 2,
            U => 3,
            D => 4,
        };
        if let Some(pos) = self.world.positions.get(we) {
            match action {
                Idle => Ok(0),
                Move(d) => Ok(dir_match(d)),
                Attack(target) | Eat(target) => {
                    let act = if let Attack(_) = action {
                        5 + MAX_REP_PER_CELL
                    } else {
                        5
                    };
                    if let Some(other_pos) = self.world.positions.get(target) {
                        let idx = self
                            .world
                            .entities_at(*other_pos)
                            .iter()
                            .position(|we| *we == target)
                            .expect("position storage inconsistent with cells")
                            .min(MAX_REP_PER_CELL - 1);
                        let tgt = 1.0 - idx as f32 / (0.5 * MAX_REP_PER_CELL as f32);
                        if other_pos == pos {
                            Ok(act + idx)
                        } else if other_pos.is_neighbour(pos) {
                            Ok(act + dir_match(pos.dir(other_pos)) * MAX_REP_PER_CELL + idx)
                        } else {
                            Err(EncodingError::InvalidAction)
                        }
                    } else {
                        Err(EncodingError::PositionNotFound)
                    }
                }
            }
        } else {
            Err(EncodingError::PositionNotFound)
        }
    }
    pub fn decode_action(&self, we: WorldEntity, idx: usize) -> Result<Action, FailReason> {
        use Action::*;
        let dir_match = |i| match i {
            1 => Ok(Dir::R),
            2 => Ok(Dir::L),
            3 => Ok(Dir::U),
            4 => Ok(Dir::D),
            _ => Err(FailReason::Unknown),
        };
        let nth = |n, pos| {
            self.world
                .entities_at(pos)
                .get(n)
                .ok_or(FailReason::TargetNotThere(Some(pos)))
        };
        const LAST_EAT: usize = 5 + MAX_REP_PER_CELL;
        if let Some(pos) = self.world.positions.get(we) {
            match idx {
                0 => Ok(Idle),
                1..=4 => dir_match(idx).map(|d| Move(d)),
                5..=LAST_EAT => nth(idx - 5, *pos).map(|we| Eat(*we)),
                _ if idx < 5 + 6 * MAX_REP_PER_CELL => {
                    let n = (idx - 5) % MAX_REP_PER_CELL;
                    let d = (idx - 5) / MAX_REP_PER_CELL - 1;
                    let target_pos = if d == 0 {
                        Ok(*pos)
                    } else {
                        dir_match(d)
                            .and_then(|dir| pos.step(dir).ok_or(FailReason::TargetNotThere(None)))
                    };
                    target_pos.and_then(|p| nth(n, p)).map(|we| Attack(*we))
                }
                _ => unreachable!(),
            }
        } else {
            Err(FailReason::Unknown)
        }
    }
    #[cfg(feature = "torch")]
    pub fn encode_map_in_tensor(&self) -> Tensor {
        let size = [
            MAP_HEIGHT as i64,
            MAP_WIDTH as i64,
            MAX_REP_PER_CELL as i64,
            ENTITY_REP_SIZE as i64,
        ];
        let mut ten = Tensor::empty(&size, (Kind::Float, self.device));
        for (pos, cell) in self.world.iter_cells() {
            for (i, we) in cell.iter().enumerate().take(MAX_REP_PER_CELL) {
                let rep = self.encode_entity(*we);
                ten.index_put_(
                    &[&Tensor::of_slice(&[pos.y as i64, pos.x as i64, i as i64])],
                    &Tensor::of_slice(&rep),
                    false,
                );
            }
        }

        ten.print();
        ten
    }
    pub fn encode_views<S: DataMut + RawData<Elem = f32>>(
        &self,
        ones: &mut [ArrayBase<S, Ix4>],
    ) -> Result<(), EncodingError> {
        if ones.len() < self.views.len() {
            return Err(EncodingError::IndexMismatch);
        }
        let mut map = Array4::zeros(Self::SHAPE);
        self.encode_map(&mut map)?;
        for (ref mut target, (we, range)) in ones.iter_mut().zip(self.views.iter()) {
            if target.shape() != Self::SHAPE {
                return Err(EncodingError::ShapeMismatch);
            }
            if let Some(view_pos) = self.world.positions.get(we) {
                for pos in Position::iter() {
                    if view_pos.distance(&pos) <= *range {
                        let idx = s![pos.y as i32, pos.x as i32, .., ..];
                        let rep = map
                            .slice(idx)
                            .to_slice()
                            .ok_or(EncodingError::NonStandardContiguousAccess)?;
                        target
                            .slice_mut(idx)
                            .into_slice()
                            .ok_or(EncodingError::NonStandardContiguousAccess)?
                            .copy_from_slice(rep);
                    }
                }
            }
        }
        Ok(())
    }
    pub fn encode_map<S: DataMut + RawData<Elem = f32>>(
        &self,
        zeros: &mut ArrayBase<S, Ix4>,
    ) -> Result<(), EncodingError> {
        for (pos, cell) in self.world.iter_cells() {
            for (i, we) in cell.iter().enumerate().take(MAX_REP_PER_CELL) {
                let rep = self.encode_entity(*we);
                zeros
                    .slice_mut(s![pos.y as i32, pos.x as i32, i, ..])
                    .into_slice()
                    .ok_or(EncodingError::NonStandardContiguousAccess)?
                    .copy_from_slice(&rep);
            }
        }
        Ok(())
    }
    fn encode_position(pos: Position) -> [f32; 2] {
        [pos.x as f32 * X_SCALE, pos.y as f32 * Y_SCALE]
    }
    fn encode_mental_state(mental_state: &MentalState) -> &[f32; EmotionalState::SIZE] {
        mental_state.emotional_state.encode()
    }
    pub fn encode_agents<'b, S: DataMut + RawData<Elem = f32>>(
        &self,
        physical: &mut ArrayBase<S, Ix2>,
        mental: &'b mut ArrayBase<S, Ix2>,
        mental_states: &'b (impl Source<'b, &'b MentalState> + 'b),
    ) -> Result<(), EncodingError> {
        if physical.shape() != &[self.views.len(), PHYS_REP_SIZE]
            || mental.shape() != &[self.views.len(), MENTAL_REP_SIZE]
        {
            return Err(EncodingError::ShapeMismatch);
        }
        for (i, (agent, _)) in self.views.iter().enumerate() {
            physical
                .slice_mut(s![i, 3..])
                .as_slice_mut()
                .ok_or(EncodingError::NonStandardContiguousAccess)?
                .copy_from_slice(&self.encode_entity(*agent));
            physical[[i, 0]] = crate::entity::encode(agent.into());

            let [x, y] = if let Some(pos) = self.world.positions.get(agent) {
                Self::encode_position(*pos)
            } else {
                [-1.0f32, -1.0f32]
            };
            physical[[i, 1]] = x;
            physical[[i, 2]] = y;
            if let Some(ms) = mental_states.get(agent.into()) {
                mental
                    .slice_mut(s![i, ..])
                    .as_slice_mut()
                    .ok_or(EncodingError::NonStandardContiguousAccess)?
                    .copy_from_slice(Self::encode_mental_state(ms));
            }
        }
        Ok(())
    }
    fn encode_entity(&self, we: WorldEntity) -> [f32; ENTITY_REP_SIZE] {
        use EntityType::*;
        let (f0, f1) = match we.e_type() {
            Rock => (0.0, 1.0),
            Burrow => (0.0, 0.4),
            Tree => (0.5, 0.2),
            Clover => (0.9, 0.0),
            Grass => (0.8, 0.1),
            Wolf => (0.2, 0.7),
            Rabbit => (0.4, 0.2),
            Deer => (0.5, 0.1),
        };
        if let Some(phys) = self.world.get_physical_state(&we) {
            if phys.is_dead() {
                [f0, f1, 0.0, phys.meat.0 * 0.002, 0.0, 0.0, 0.0, 0.0, 0.0]
            } else {
                let h = phys.health.0 / phys.max_health.0;
                let m = phys.max_health.0 * 0.002;
                let a = phys.attack.map(|a| a.0 * 0.02).unwrap_or(0.0);
                let sp = phys.speed.0 * 2.0;
                let sa = phys.satiation.0 / phys.max_health.0;
                use Dir::*;
                let (mov_lr, mov_ud) = match phys.move_target {
                    None => (0.0, 0.0),
                    Some(R) => (phys.move_progress.0, 0.0),
                    Some(L) => (-phys.move_progress.0, 0.0),
                    Some(U) => (0.0, phys.move_progress.0),
                    Some(D) => (0.0, -phys.move_progress.0),
                };
                [f0, f1, h, m, a, sp, sa, mov_lr, mov_ud]
            }
        } else {
            [f0, f1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_action() {
        let mut sim = crate::SimState::new_with_seed(1.0, 1977);
        sim.step(None);
        let test = |sim: &crate::SimState| {
            let mut actions = Vec::new();
            let views: Vec<_> = sim
                .agent_system
                .mental_states
                .iter()
                .map(|ms: &MentalState| {
                    actions.push(ms.current_action);
                    (ms.id, ms.sight_radius)
                })
                .collect();
            let obsv = ObsvWriter {
                views: &views,
                world: &sim.world,
                #[cfg(feature = "torch")]
                device: Device::Cpu,
            };
            for (e, ra) in &sim.agent_system.actions {
                if let Ok(a) = ra {
                    let enc_a = obsv.encode_action(*e, *a);
                    assert!(enc_a.is_ok());
                    let dec_a = obsv.decode_action(*e, enc_a.unwrap());
                    assert!(dec_a.is_err() || *ra == dec_a);
                }
            }
        };
        test(&sim);
        for _ in 0..20 {
            sim.step(None);
            test(&sim);
        }
    }
}
