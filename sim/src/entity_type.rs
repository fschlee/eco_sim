use crate::world::{Attack, Health, Meat, MoveProgress, PhysicalState, Satiation, Speed};

use enum_macros::{EnumCount, EnumIter};

pub trait Count {
    const COUNT: usize;
    fn idx(&self) -> usize;
}

pub type Rarity = usize;

#[derive(Ord, PartialOrd, Eq, PartialEq, Copy, Clone, Debug, EnumIter, EnumCount)]
pub enum EntityType {
    Rock,
    Burrow,
    Tree,
    Grass,
    Clover,
    Rabbit,
    Deer,
    Wolf,
}
impl EntityType {
    #[inline]
    pub const fn can_eat(&self, other: &Self) -> bool {
        use EntityType::*;
        match self {
            Rock | Burrow | Tree | Grass | Clover => false,
            Rabbit => match other {
                Grass | Clover => true,
                Rock | Burrow | Tree | Rabbit | Deer | Wolf => false,
            },
            Deer => match other {
                Grass | Clover | Tree => true,
                Rock | Burrow | Rabbit | Deer | Wolf => false,
            },
            Wolf => match other {
                Rabbit | Deer => true,
                Rock | Burrow | Tree | Grass | Clover | Wolf => false,
            },
        }
    }
    #[inline]
    pub const fn can_pass(&self, other: &Self) -> bool {
        use EntityType::*;

        match self {
            Rock | Burrow | Grass | Clover | Tree => false,
            Deer | Wolf => match other {
                Rock | Burrow => false,
                Grass | Clover | Tree => true,
                Rabbit | Deer | Wolf => true,
            },
            Rabbit => match other {
                Rock => false,
                Grass | Clover | Tree | Burrow => true,
                Rabbit | Deer | Wolf => true,
            },
        }
    }
    pub fn typical_physical_state(&self) -> Option<PhysicalState> {
        use EntityType::*;
        match self {
            Rabbit => Some(PhysicalState::new(Health(50.0), Speed(0.2), None)),
            Deer => Some(PhysicalState::new(Health(300.0), Speed(0.4), None)),
            Wolf => Some(PhysicalState::new(
                Health(300.0),
                Speed(0.3),
                Some(Attack(60.0)),
            )),
            Rock | Grass | Clover | Tree | Burrow => None,
        }
    }
    pub const fn rate(&self) -> Rarity {
        use EntityType::*;
        match self {
            Rock => 16,
            Burrow => 100,
            Tree => 8,
            Grass => 6,
            Clover => 6,
            Rabbit => 25,
            Deer => 30,
            Wolf => 50,
        }
    }
    #[inline]
    pub const fn is_mobile(&self) -> bool {
        use EntityType::*;
        match self {
            Rabbit | Deer | Wolf => true,
            Rock | Grass | Clover | Tree | Burrow => false,
        }
    }
    #[inline]
    pub const fn pass_rate(&self) -> f32 {
        let mut rate = 1.0;
        let mut i = 0;
        while i < EntityType::COUNT {
            let e = ITEMS[i];
            if !self.can_pass(&e) {
                rate *= 1.0 / e.rate() as f32;
            }
        }
        rate
    }
}
impl Default for EntityType {
    fn default() -> Self {
        Self::Rock
    }
}

const ITEMS: [EntityType; EntityType::COUNT] = [
    EntityType::Rock,
    EntityType::Burrow,
    EntityType::Tree,
    EntityType::Grass,
    EntityType::Clover,
    EntityType::Rabbit,
    EntityType::Deer,
    EntityType::Wolf,
];
