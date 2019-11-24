use super::world::{PhysicalState, Health, Meat, Satiation, Attack};

use enum_macros::{EnumCount, EnumIter};


pub trait Count {
    const COUNT : usize;
    fn idx(&self) -> usize;
}

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
    pub fn can_eat(&self, other: &Self) -> bool {
        use EntityType::*;
        match self {
            Rock | Burrow | Tree | Grass | Clover => false,
            Rabbit => match other {
                Grass | Clover => true,
                Rock | Burrow | Tree | Rabbit | Deer | Wolf => false
            },
            Deer => match other {
                Grass | Clover | Tree => true,
                Rock  | Burrow | Rabbit | Deer | Wolf => false
            }
            Wolf => match other {
                Rabbit | Deer => true,
                Rock | Burrow | Tree | Grass | Clover | Wolf => false,
            }
        }
    }
    pub fn can_pass(&self, other: &Self) -> bool {
        use EntityType::*;

        match self {
            Rock | Burrow | Grass | Clover | Tree => false,
            Rabbit | Deer | Wolf => match other {
                Rock | Burrow => false,
                Grass  | Clover | Tree => true,
                Rabbit | Deer   | Wolf => true,
            }
            Rabbit  => match other {
                Rock => false,
                Grass  | Clover | Tree | Burrow => true,
                Rabbit | Deer   | Wolf => true,
            }
        }
    }
    pub fn typical_physical_state(&self)  -> Option<PhysicalState> {
        use EntityType::*;
        match self {
            Rabbit=> Some(
                PhysicalState{
                    health: Health(50.0),
                    meat: Meat(40.0),
                    attack: None,
                    satiation: Satiation(10.0)
                }),
            Deer => Some(
                PhysicalState{
                    health: Health(300.0),
                    meat: Meat(100.0),
                    attack: None,
                    satiation: Satiation(10.0)
                }),
            Wolf => Some(
                PhysicalState{
                    health: Health(300.0),
                    meat: Meat(100.0),
                    attack: Some(Attack(60.0)),
                    satiation: Satiation(10.0)
                }),
            Rock | Grass | Clover | Tree | Burrow  => None
        }
    }
    pub fn rate(&self) -> Rarity {
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
}
impl Default for EntityType {
    fn default() -> Self {
        Self::Rock
    }
}

pub type Rarity = usize;

