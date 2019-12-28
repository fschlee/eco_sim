use log::error;
use ndarray::{s, Array4, ArrayBase, ArrayViewMut, DataMut, Ix4, RawData};
#[cfg(feature = "torch")]
use tch::{Device, IndexOp, Kind, Tensor};

use crate::world::{DefCell, Dir, Position, World};
use crate::{Action, Coord, EntityType, WorldEntity, MAP_HEIGHT, MAP_WIDTH};

pub const MAX_REP_PER_CELL: usize = 8;
pub const ENTITY_REP_SIZE: usize = 9;

pub struct ObsvWriter<'a> {
    world: &'a World<DefCell>,
    pos: Position,
    radius: Coord,
    #[cfg(feature = "torch")]
    device: Device,
}
impl<'a> ObsvWriter<'a> {
    pub const ACTION_SPACE_SIZE: usize = 5 + 6 * MAX_REP_PER_CELL;
    pub fn new(world: &'a World<DefCell>, pos: Position, radius: Coord) -> Self {
        Self {
            world,
            pos,
            radius,
            #[cfg(feature = "torch")]
            device: Device::Cpu,
        }
    }
    pub fn encode_action(&self, action: Action) -> usize {
        use Action::*;
        use Dir::*;
        let dir_match = |dir| match dir {
            R => 1,
            L => 2,
            U => 3,
            D => 4,
        };
        match action {
            Idle => 0,
            Move(d) => dir_match(d),
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
                    if *other_pos == self.pos {
                        act + idx
                    } else if other_pos.is_neighbour(&self.pos) {
                        act + dir_match(self.pos.dir(other_pos)) * MAX_REP_PER_CELL + idx
                    } else {
                        error!("invalid action encoded");
                        0
                    }
                } else {
                    error!("invalid action encoded: target has no position");
                    0
                }
            }
        }
    }
    pub fn decode_action(&self, idx: usize) -> Option<Action> {
        use Action::*;
        let dir_match = |i| match i {
            1 => Some(Dir::R),
            2 => Some(Dir::L),
            3 => Some(Dir::U),
            4 => Some(Dir::D),
            _ => None,
        };
        let nth = |n, pos| self.world.entities_at(pos).get(n);
        const LAST_EAT: usize = 5 + MAX_REP_PER_CELL;
        match idx {
            0 => Some(Idle),
            1..=4 => dir_match(idx).map(|d| Move(d)),
            5..=LAST_EAT => nth(idx - 5, self.pos).map(|we| Eat(*we)),
            _ if idx < 5 + 6 * MAX_REP_PER_CELL => {
                let n = (idx - 5) % MAX_REP_PER_CELL;
                let d = (idx - 5) / MAX_REP_PER_CELL - 1;
                let pos = if d == 0 {
                    Some(self.pos)
                } else {
                    dir_match(d).and_then(|dir| self.pos.step(dir))
                };
                pos.and_then(|p| nth(n, p)).map(|we| Attack(*we))
            }
            _ => unreachable!(),
        }
    }
    #[cfg(feature = "torch")]
    pub fn encode_observation_in_tensor(&self) -> Tensor {
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
    pub fn encode_observation<S: DataMut + RawData<Elem = f32>>(
        &self,
        target: &mut ArrayBase<S, Ix4>,
    ) {
        for (pos, cell) in self.world.iter_cells() {
            for (i, we) in cell.iter().enumerate().take(MAX_REP_PER_CELL) {
                let rep = self.encode_entity(*we);
                target
                    .slice_mut(s![pos.y as i32, pos.x as i32, i, ..])
                    .into_slice()
                    .unwrap()
                    .copy_from_slice(&rep);
            }
        }
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
