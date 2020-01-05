use crate::entity_type::{Count, EntityType};
use crate::util::clip;
use std::borrow::Borrow;

#[derive(PartialOrd, PartialEq, Copy, Clone, Debug, Default)]
#[repr(transparent)]
pub struct Wrapper(pub f32);

const SIZE: usize = EntityType::COUNT + 4;

type WrapperArray = [Wrapper; SIZE];
#[derive(Clone, Debug)]
pub struct EmotionalState {
    arr: WrapperArray,
}
impl EmotionalState {
    const HUNGER: usize = EntityType::COUNT + 0;
    const FEAR: usize = EntityType::COUNT + 1;
    const TIREDNESS: usize = EntityType::COUNT + 2;
    const AGGRESSION: usize = EntityType::COUNT + 3;
    pub fn pref(&self, et: EntityType) -> Preference {
        self.arr[et.idx()]
    }

    pub fn new(food_preferences: Vec<(EntityType, f32)>) -> Self {
        let mut es = Self {
            arr: Default::default(),
        };
        for (et, r) in food_preferences {
            es.arr[et.idx()] += r;
        }
        es += Hunger::default();
        es
    }
    pub fn preferences(&self) -> &[Preference] {
        &self.arr[0..EntityType::COUNT]
    }
    pub fn set_preference(&mut self, et: EntityType, pref: f32) {
        let p = &mut self.arr[et.idx()];
        *p -= 1.0;
        *p += pref;
    }
    pub fn hunger(&self) -> Hunger {
        Hunger(self.arr[Self::HUNGER].0)
    }
    pub fn set_hunger(&mut self, hunger: Hunger) {
        *self -= Hunger(1.0);
        *self += hunger;
    }
    pub fn tiredness(&self) -> Tiredness {
        Tiredness(self.arr[Self::TIREDNESS].0)
    }
    pub fn set_tiredness(&mut self, tiredness: Tiredness) {
        *self -= Tiredness(1.0);
        *self += tiredness;
    }
    pub fn fear(&self) -> Fear {
        Fear(self.arr[Self::FEAR].0)
    }
    pub fn set_fear(&mut self, fear: Fear) {
        *self -= Fear(1.0);
        *self += fear;
    }
    pub fn aggression(&self) -> Aggression {
        Aggression(self.arr[Self::AGGRESSION].0)
    }
    pub fn set_aggression(&mut self, aggression: Aggression) {
        *self -= Aggression(1.0);
        *self += aggression;
    }
    pub fn average<B>(iter: impl Iterator<Item = B>) -> Self
    where
        B: Borrow<Self>,
    {
        let c = 1;
        let mut start = Self {
            arr: Default::default(),
        };
        for b in iter {
            let em = b.borrow();
            for i in 0..start.arr.len() {
                start.arr[i].0 += em.arr[i].0
            }
        }
        let inv = 1.0 / c as f32;
        for e in &mut start.arr {
            (*e).0 *= inv;
        }
        start
    }
    pub(crate) fn encode(&self) -> &[f32; SIZE] {
        // Safe because wrapper has the same representation as f32
        unsafe { std::mem::transmute(&self.arr) }
    }
    pub(crate) const SIZE: usize = SIZE;
}

type Reward = f32;
pub type Preference = Wrapper;

impl std::ops::AddAssign<&EmotionalState> for EmotionalState {
    fn add_assign(&mut self, rhs: &Self) {
        for i in 0..self.arr.len() {
            self.arr[i].0 = 0.5 * (self.arr[i].0 + rhs.arr[i].0)
        }
    }
}

impl std::ops::AddAssign<f32> for Wrapper {
    fn add_assign(&mut self, rhs: f32) {
        self.0 = clip(self.0 + rhs, 0.0, 1.0);
    }
}

impl std::ops::SubAssign<f32> for Wrapper {
    fn sub_assign(&mut self, rhs: f32) {
        self.0 = clip(self.0 - rhs, 0.0, 1.0);
    }
}

#[derive(PartialOrd, PartialEq, Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Hunger(pub f32);

impl Default for Hunger {
    fn default() -> Self {
        Self(0.39)
    }
}
impl std::ops::AddAssign<Hunger> for EmotionalState {
    fn add_assign(&mut self, rhs: Hunger) {
        self.arr[Self::HUNGER] += rhs.0;
    }
}
impl std::ops::SubAssign<Hunger> for EmotionalState {
    fn sub_assign(&mut self, rhs: Hunger) {
        self.arr[Self::HUNGER] -= rhs.0;
    }
}

#[derive(PartialOrd, PartialEq, Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Tiredness(pub f32);

impl Default for Tiredness {
    fn default() -> Self {
        Self(0.0)
    }
}
impl std::ops::AddAssign<Tiredness> for EmotionalState {
    fn add_assign(&mut self, rhs: Tiredness) {
        self.arr[Self::TIREDNESS] += rhs.0;
    }
}
impl std::ops::SubAssign<Tiredness> for EmotionalState {
    fn sub_assign(&mut self, rhs: Tiredness) {
        self.arr[Self::TIREDNESS] -= rhs.0;
    }
}

#[derive(PartialOrd, PartialEq, Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Fear(pub f32);

impl Default for Fear {
    fn default() -> Self {
        Self(0.0)
    }
}
impl std::ops::AddAssign<Fear> for EmotionalState {
    fn add_assign(&mut self, rhs: Fear) {
        self.arr[Self::FEAR] += rhs.0;
    }
}
impl std::ops::SubAssign<Fear> for EmotionalState {
    fn sub_assign(&mut self, rhs: Fear) {
        self.arr[Self::FEAR] -= rhs.0;
    }
}

#[derive(PartialOrd, PartialEq, Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Aggression(pub f32);

impl Default for Aggression {
    fn default() -> Self {
        Self(0.0)
    }
}
impl std::ops::AddAssign<Aggression> for EmotionalState {
    fn add_assign(&mut self, rhs: Aggression) {
        self.arr[Self::AGGRESSION] += rhs.0;
    }
}
impl std::ops::SubAssign<Aggression> for EmotionalState {
    fn sub_assign(&mut self, rhs: Aggression) {
        self.arr[Self::AGGRESSION] -= rhs.0;
    }
}
