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

pub type Prob = f32;

#[derive(Clone, Debug)]
pub enum Occupancy {
    Empty,
    Filled(Vec<WorldEntity>),
    ExpectedFilled(Vec<WorldEntity>, Vec<Prob>),
    Unknown,
}
impl Occupancy {
    const EMPTY_THRESHOLD: Prob = 0.2;
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
    pub fn reduce_prob(&mut self, entity: &WorldEntity, prob: Prob) -> Prob {
        assert!(prob > 0.0 && prob < 1.0);
        use Occupancy::*;
        let mut reduced_prob = 0.0;
        match self {
            Empty | Unknown => (),
            Filled(_) => {
                let mut tmp = Unknown;
                std::mem::swap(&mut tmp, self);
                reduced_prob = 1.0 * (1.0 - prob);
                if let Filled(v) = tmp {
                    let ps = v
                        .iter()
                        .map(|e| if e == entity { reduced_prob } else { 1.0 })
                        .collect();
                    *self = ExpectedFilled(v, ps)
                }
            }
            ExpectedFilled(ws, ps) => {
                if let Some(i) = ws.iter().position(|e| e == entity) {
                    reduced_prob = ps[i] * (1.0 - prob);
                    ps[i] = reduced_prob;
                }
            }
        }
        reduced_prob
    }
    pub fn push_prob(&mut self, entity: WorldEntity, prob: Prob) {
        assert!(prob > 0.0 && prob < 1.0);
        use Occupancy::*;
        match self {
            Empty | Unknown => *self = ExpectedFilled(vec![entity], vec![prob]),
            Filled(_) => {
                let mut tmp = Unknown;
                std::mem::swap(&mut tmp, self);
                if let Filled(mut v) = tmp {
                    let mut ps = vec![1.0; v.len()];
                    v.push(entity);
                    ps.push(prob);
                    *self = ExpectedFilled(v, ps)
                }
            }
            ExpectedFilled(ws, ps) => {
                ws.push(entity);
                ps.push(prob);
            }
        }
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
                    if f(we) {
                        i += 1;
                        true
                    } else {
                        probs.remove(i);
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
            Empty => *self = Filled(vec![we]),
            Unknown => {
                *self = ExpectedFilled(vec![we], vec![1.0]);
            }
            ExpectedFilled(ws, ps) => {
                ws.push(we);
                ps.push(1.0);
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
