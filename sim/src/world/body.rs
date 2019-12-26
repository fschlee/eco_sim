use super::position::{Dir, Position};
use crate::util::clip;

#[derive(PartialOrd, PartialEq, Copy, Clone, Debug)]
pub struct Health(pub f32);

pub(super) type Damage = Health;
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
    fn mul(self, rhs: f32) -> Self::Output {
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
    pub move_target: Option<Dir>,
    pub attack: Option<Attack>,
    pub satiation: Satiation,
}

impl std::fmt::Display for PhysicalState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        if self.is_dead() {
            writeln!(f, "{:.2} meat remaining", self.meat.0)?;
        } else {
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
    pub const SATIATION_DECR: Satiation = Satiation(0.1);
    pub fn is_dead(&self) -> bool {
        self.health.0 <= 0.0
    }
    pub fn partial_move(&mut self, dir: Dir) -> MoveProgress {
        if Some(dir) != self.move_target {
            self.move_target = Some(dir);
            self.move_progress = MoveProgress(0.0);
        }
        self.move_progress += self.speed * (self.health.0 / self.max_health.0);
        self.move_progress
    }
    pub fn new(max_health: Health, speed: Speed, attack: Option<Attack>) -> Self {
        Self {
            health: max_health,
            max_health,
            meat: Meat(max_health.0 * 0.5),
            speed,
            move_progress: MoveProgress::default(),
            move_target: None,
            attack,
            satiation: Satiation(10.0),
        }
    }
}

impl std::ops::AddAssign<Satiation> for PhysicalState {
    fn add_assign(&mut self, rhs: Satiation) {
        debug_assert!(!rhs.0.is_nan());
        let mut val = self.satiation.0 + rhs.0;
        if val < 0.0 {
            self.health.0 += 0.1 * val;
            val = 0.0;
        }
        self.satiation.0 = clip(val, 0.0, self.max_health.0);
    }
}
impl std::ops::SubAssign<Satiation> for PhysicalState {
    fn sub_assign(&mut self, rhs: Satiation) {
        *self += Satiation(-rhs.0)
    }
}
impl std::ops::SubAssign<f32> for Satiation {
    fn sub_assign(&mut self, rhs: f32) {
        self.0 = clip(self.0 - rhs, 0.0, 1.0);
    }
}
