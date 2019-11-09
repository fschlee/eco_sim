use super::world::{PhysicalState, Health, Meat, Satiation, Attack};

#[derive(Ord, PartialOrd, Eq, PartialEq, Copy, Clone, Debug)]
pub enum EntityType {
    Rock,
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
        match (self, other) {
            (Rabbit, Grass) => true,
            (Rabbit, Clover) => true,
            (Deer, Grass) => true,
            (Deer, Clover) => true,
            (Deer, Tree) => true,
            (Wolf, Rabbit) => true,
            (Wolf, Deer) => true,
            _ => false,
        }
    }
    pub fn can_pass(&self, other: &Self) -> bool {
        use EntityType::*;
        match (self, other) {
            (Rabbit, Grass) => true,
            (Rabbit, Clover) => true,
            (Rabbit, Tree) => true,
            (Rabbit, Rabbit) => true,
            (Deer, Grass) => true,
            (Deer, Clover) => true,
            (Deer, Tree) => true,
            (Deer, Deer) => true,
            (Wolf, Grass) => true,
            (Wolf, Clover) => true,
            (Wolf, Tree) => true,
            (Wolf, Rabbit) => true,
            (Wolf, Deer) => true,
            (Wolf, Wolf) => true,
            _ => false,
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
            _ => None
        }
    }
}
impl Default for EntityType {
    fn default() -> Self {
        Self::Rock
    }
}
pub const ENTITY_TYPES : [EntityType; ENTITY_TYPE_COUNT] = [
    EntityType::Rock,
    EntityType::Tree,
    EntityType::Grass,
    EntityType::Clover,
    EntityType::Rabbit,
    EntityType::Deer,
    EntityType::Wolf,
];

pub fn et_idx(et : EntityType) -> usize {
    match et {
        EntityType::Rock => 0,
        EntityType::Tree => 1,
        EntityType::Grass => 2,
        EntityType::Clover => 3,
        EntityType::Rabbit => 4,
        EntityType::Deer => 5,
        EntityType::Wolf => 6,
    }
}

pub const ENTITY_TYPE_COUNT: usize = 7;