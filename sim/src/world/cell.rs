use smallvec::SmallVec;
use std::mem::{size_of, MaybeUninit};

use super::{MAP_HEIGHT, MAP_WIDTH};
use crate::entity::WorldEntity;
use crate::Position;

pub trait Cell: std::ops::Deref<Target = [WorldEntity]> + Sized + Clone + Sync {
    type Locator;
    fn empty_init() -> [[Self; MAP_WIDTH]; MAP_HEIGHT];
    fn retain<F: FnMut(&WorldEntity) -> bool>(&mut self, f: F);
    fn push(&mut self, we: WorldEntity);
    fn unknown() -> Self;
    fn is_empty(&self) -> bool;
}

pub type DefCell = SmallVec<[WorldEntity; 3]>;

impl Cell for DefCell {
    type Locator = Position;
    fn empty_init() -> [[Self; MAP_WIDTH]; MAP_HEIGHT] {
        empty_initialize()
    }

    fn retain<F: FnMut(&WorldEntity) -> bool>(&mut self, mut f: F) {
        self.retain(|we| f(&*we))
    }

    fn push(&mut self, we: WorldEntity) {
        self.push(we)
    }

    fn unknown() -> Self {
        SmallVec::new()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }
}

#[derive(Clone, Debug)]
pub enum Occupancy {
    Empty,
    Filled(Vec<WorldEntity>),
    ExpectedFilled(Vec<WorldEntity>, Vec<f32>),
    Unknown,
}
impl Occupancy {
    const EMPTY_THRESHOLD: f32 = 0.2;
    fn initialize(none: Occupancy) -> [[Occupancy; MAP_WIDTH]; MAP_HEIGHT] {
        unsafe {
            let bytes =
                std::slice::from_raw_parts(&none as *const _ as *const u8, size_of::<Occupancy>());
            if bytes.iter().all(|b| *b == 0u8) {
                return MaybeUninit::zeroed().assume_init();
            }
        }
        let mut a: [[MaybeUninit<Occupancy>; MAP_WIDTH]; MAP_HEIGHT] =
            unsafe { MaybeUninit::uninit().assume_init() };
        for row in &mut a[..] {
            for elem in &mut row[..] {
                *elem = MaybeUninit::new(none.clone());
            }
        }
        unsafe { std::mem::transmute(a) }
    }
}

impl std::ops::Deref for Occupancy {
    type Target = [WorldEntity];

    fn deref(&self) -> &Self::Target {
        use Occupancy::*;
        match self {
            Empty | Unknown => &[],
            Filled(v) => &*v,
            ExpectedFilled(wes, probs) => &*wes,
        }
    }
}
impl Cell for Occupancy {
    type Locator = Position;
    fn empty_init() -> [[Self; MAP_WIDTH]; MAP_HEIGHT] {
        Occupancy::initialize(Occupancy::Unknown)
    }

    fn retain<F: FnMut(&WorldEntity) -> bool>(&mut self, mut f: F) {
        use Occupancy::*;
        match self {
            Empty | Unknown => (),
            ExpectedFilled(wes, probs) => {
                let mut i = 0;
                wes.retain(|we: &WorldEntity| {
                    i += 1;
                    if f(we) {
                        probs.remove(i);
                        true
                    } else {
                        false
                    }
                })
            }
            Filled(v) => v.retain(f),
        }
    }

    fn push(&mut self, we: WorldEntity) {
        use Occupancy::*;
        match self {
            Empty | Unknown | ExpectedFilled(_, _) => {
                *self = Filled(vec![we]);
            }
            Filled(v) => v.push(we),
        }
    }

    fn unknown() -> Self {
        Occupancy::Unknown
    }

    fn is_empty(&self) -> bool {
        use Occupancy::*;
        match self {
            Empty => true,
            ExpectedFilled(_, probs) => probs.iter().sum::<f32>() < Self::EMPTY_THRESHOLD,
            _ => false,
        }
    }
}

fn none_initialize<T>() -> [[Option<Vec<T>>; MAP_WIDTH]; MAP_HEIGHT] {
    let none = None::<Vec<T>>;
    unsafe {
        let bytes =
            std::slice::from_raw_parts(&none as *const _ as *const u8, size_of::<Option<Vec<T>>>());
        if bytes.iter().all(|b| *b == 0u8) {
            return MaybeUninit::zeroed().assume_init();
        }
    }
    let mut a: [[MaybeUninit<Option<Vec<T>>>; MAP_WIDTH]; MAP_HEIGHT] =
        unsafe { MaybeUninit::uninit().assume_init() };
    for row in &mut a[..] {
        for elem in &mut row[..] {
            *elem = MaybeUninit::new(None);
        }
    }
    unsafe { std::mem::transmute(a) }
}
fn empty_initialize<A: smallvec::Array>() -> [[SmallVec<A>; MAP_WIDTH]; MAP_HEIGHT] {
    unsafe { MaybeUninit::zeroed().assume_init() }
}
