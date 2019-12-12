use std::collections::HashMap;
use crate::world::{MAP_WIDTH, MAP_HEIGHT};

#[derive(Ord, PartialOrd, Eq, PartialEq, Hash, Copy, Clone, Debug)]
pub struct Position {
    pub x: u32,
    pub y: u32,
}

const CLOSEST_DIRS : [[Dir; 4]; 4] = [
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
    pub fn neighbours(&self) -> impl IntoIterator<Item=Position> {
        NeighborIter { pos: *self, dir: Some(Dir::R)}
    }
    pub fn distance(&self, other: & Position) -> u32 {
        ((self.x as i32 - other.x as i32).abs() + (self.y as i32 - other.y as i32).abs()) as u32
    }
    pub fn within_bounds(& self) -> bool {
        self.x < MAP_WIDTH as u32 && self.y < MAP_HEIGHT as u32
    }
    pub fn step(&self, dir: Dir)-> Option<Position>{
        use Dir::*;
        match dir {
            R if self.x +1 < MAP_WIDTH as u32  => Some(Position { x: self.x+1, y: self.y }),
            L if self.x > 0  => Some(Position { x: self.x - 1, y: self.y }),
            D if self.y +1 < MAP_HEIGHT as u32 => Some(Position { x: self.x, y: self.y +1 }),
            U if self.y > 0 as u32 => Some(Position { x: self.x, y: self.y  - 1 }),
            R | L | D | U => None,
        }
    }
    fn possible_steps(&self, goal: &Position) -> impl Iterator<Item=Position> + '_ {
        (&CLOSEST_DIRS[self.dir(goal) as usize]).iter().filter_map(move|dir|
            self.step(*dir))
    }
    pub fn dir(&self, other: &Position) -> Dir {
        let x_diff = self.x as i32 - other.x as i32;
        let y_diff = self.y as i32 - other.y as i32;
        if x_diff.abs() > y_diff.abs() {
            if x_diff > 0 {
                Dir::L
            }
            else {
                Dir::R
            }
        }
        else if y_diff > 0 {
            Dir::U
        }
        else {
            Dir::D
        }
    }
    pub fn iter() -> impl Iterator<Item=Position> {
        (0.. MAP_WIDTH).into_iter().zip((0..MAP_HEIGHT).into_iter())
            .map(move |(x, y)| Position{x : x as u32, y: y as u32})
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
    fn next(self) -> Option<Dir>{
        use Dir::*;
        match self {
            R => Some(U),
            U => Some(L),
            L => Some(D),
            D => None,
        }
    }
    pub fn opposite(self) -> Dir {
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
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error>  {
        use Dir::*;
        match self {
            R => write!(f, "right"),
            L => write!(f, "left"),
            U => write!(f, "up"),
            D => write!(f, "down")
        }
    }
}

struct NeighborIter {
    pos : Position,
    dir : Option<Dir>,
}
impl Iterator for NeighborIter{
    type Item = Position;
    fn next(&mut self) -> Option<Self::Item>{
        while let Some(dir) = self.dir {
            self.dir = dir.next();
            if let p@Some(_) = self.pos.step(dir){
                return p
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
            Self::Vec(vec) => vec.iter().find_map(|(p, t)| {
                if *p == *k {
                    Some(t)
                }
                else {
                    None
                }
            }),
            Self::Map(m) => m.get(k)
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
                    return self.insert(k, v)
                }
                match vec.iter_mut().find(|(p, t)| *p == k ) {
                    Some((p, t)) => {
                        let mut r = v;
                        std::mem::swap(t, & mut r);
                        Some(r)
                    },
                    None => {
                        vec.push((k, v));
                        None
                    }
                }
            }
            Self::Map(m) => m.insert(k, v)
        }
    }
}
