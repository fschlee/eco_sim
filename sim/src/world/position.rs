use crate::world::{MAP_HEIGHT, MAP_WIDTH};
use std::collections::HashMap;

pub type Coord = i16; // Needs to be signed and to be able to store values of at least 2 * max(MAP_WIDTH, MAP_HEIGHT)

#[derive(Ord, PartialOrd, Eq, PartialEq, Hash, Copy, Clone, Debug)]
pub struct Position {
    pub x: Coord,
    pub y: Coord,
}

const CLOSEST_DIRS: [[Dir; 4]; 4] = [
    [Dir::R, Dir::D, Dir::U, Dir::L],
    [Dir::L, Dir::U, Dir::D, Dir::R],
    [Dir::U, Dir::L, Dir::R, Dir::D],
    [Dir::D, Dir::R, Dir::L, Dir::U],
];
impl Position {
    pub fn is_neighbour(&self, other: &Position) -> bool {
        self != other
            && ((self.x as i64) - (other.x as i64)).abs() <= 1
            && ((self.y as i64) - (other.y as i64)).abs() <= 1
    }
    pub const fn neighbours(&self) -> impl IntoIterator<Item = Position> {
        NeighborIter {
            pos: *self,
            dir: Some(Dir::R),
        }
    }
    pub const fn distance(&self, other: &Position) -> Coord {
        let dist = (self.x - other.x).abs() + (self.y - other.y).abs();
        dist
    }
    pub const fn within_bounds(&self) -> bool {
        self.x < MAP_WIDTH as Coord && self.y < MAP_HEIGHT as Coord
    }
    pub const fn step(&self, dir: Dir) -> Option<Position> {
        use Dir::*;
        match dir {
            R if self.x + 1 < MAP_WIDTH as Coord => Some(Position {
                x: self.x + 1,
                y: self.y,
            }),
            L if self.x > 0 => Some(Position {
                x: self.x - 1,
                y: self.y,
            }),
            D if self.y + 1 < MAP_HEIGHT as Coord => Some(Position {
                x: self.x,
                y: self.y + 1,
            }),
            U if self.y > 0 as Coord => Some(Position {
                x: self.x,
                y: self.y - 1,
            }),
            R | L | D | U => None,
        }
    }
    fn possible_steps(&self, goal: &Position) -> impl Iterator<Item = Position> + '_ {
        (&CLOSEST_DIRS[self.dir(goal) as usize])
            .iter()
            .filter_map(move |dir| self.step(*dir))
    }
    pub const fn dir(&self, other: &Position) -> Dir {
        let x_diff = self.x - other.x;
        let y_diff = self.y - other.y;
        if x_diff.abs() > y_diff.abs() {
            if x_diff > 0 {
                Dir::L
            } else {
                Dir::R
            }
        } else if y_diff > 0 {
            Dir::U
        } else {
            Dir::D
        }
    }
    pub fn iter() -> impl Iterator<Item = Position> {
        (0..MAP_HEIGHT).into_iter().flat_map(|y| {
            (0..MAP_WIDTH).into_iter().map(move |x| Position {
                x: x as Coord,
                y: y as Coord,
            })
        })
    }
    pub const fn idx(&self) -> usize {
        MAP_WIDTH * self.y as usize + self.x as usize
    }
}
#[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq)]
pub enum Dir {
    R = 0,
    L = 1,
    U = 2,
    D = 3,
}
impl Dir {
    const fn next(self) -> Option<Dir> {
        use Dir::*;
        match self {
            R => Some(U),
            U => Some(L),
            L => Some(D),
            D => None,
        }
    }
    pub const fn opposite(self) -> Dir {
        use Dir::*;
        match self {
            R => L,
            U => D,
            L => R,
            D => U,
        }
    }
    pub(crate) fn closest(&self) -> &[Dir] {
        &CLOSEST_DIRS[*self as usize]
    }
}

impl std::fmt::Display for Dir {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        use Dir::*;
        match self {
            R => write!(f, "right"),
            L => write!(f, "left"),
            U => write!(f, "up"),
            D => write!(f, "down"),
        }
    }
}

struct NeighborIter {
    pos: Position,
    dir: Option<Dir>,
}
impl Iterator for NeighborIter {
    type Item = Position;
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(dir) = self.dir {
            self.dir = dir.next();
            if let p @ Some(_) = self.pos.step(dir) {
                return p;
            }
        }
        None
    }
}

pub(crate) enum PositionMap<T: Sized> {
    Vec(Vec<(Position, T)>),
    Map(HashMap<Position, T>),
}
impl<T: Sized> PositionMap<T> {
    pub fn new() -> Self {
        Self::Vec(Vec::new())
    }
    pub fn get(&self, k: &Position) -> Option<&T> {
        match self {
            Self::Vec(vec) => vec
                .iter()
                .find_map(|(p, t)| if *p == *k { Some(t) } else { None }),
            Self::Map(m) => m.get(k),
        }
    }
    pub fn insert(&mut self, k: Position, v: T) -> Option<T> {
        match self {
            Self::Vec(vec) => {
                if MAP_HEIGHT * MAP_WIDTH > 200 && vec.len() > 100 {
                    let mut map = HashMap::new();
                    for (p, t) in vec.drain(..) {
                        map.insert(p, t);
                    }
                    *self = Self::Map(map);
                    return self.insert(k, v);
                }
                match vec.iter_mut().find(|(p, t)| *p == k) {
                    Some((_p, t)) => {
                        let mut r = v;
                        std::mem::swap(t, &mut r);
                        Some(r)
                    }
                    None => {
                        vec.push((k, v));
                        None
                    }
                }
            }
            Self::Map(m) => m.insert(k, v),
        }
    }
}
