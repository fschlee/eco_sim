use crate::world::{MAP_HEIGHT, MAP_WIDTH};
use std::fmt::Display;

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
        self.x >= 0 && self.y >= 0 && self.x < MAP_WIDTH as Coord && self.y < MAP_HEIGHT as Coord
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
    pub fn iter_outwards(&self) -> impl Iterator<Item = Position> {
        PositionWalker::new(*self, (MAP_WIDTH + MAP_HEIGHT) as Coord)
    }
    pub fn iter_ring(
        &self,
        min_radius: Coord,
        max_radius: Coord,
    ) -> impl Iterator<Item = Position> {
        PositionWalker {
            center: *self,
            current_pos: Some(Position {
                x: self.x + (min_radius - 1).min(0),
                y: self.y,
            }),
            radius: min_radius,
            max_radius,
            delta: 0,
        }
    }
}
impl std::fmt::Display for Position {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "({}, {})", self.x, self.y)
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

pub(crate) struct PositionMap<T: Clone + Sized>(Map<Option<T>>);

impl<T: Clone + Sized> PositionMap<T> {
    pub fn new() -> Self {
        Self(Map::new())
    }
    pub fn get(&self, k: &Position) -> Option<&T> {
        self.0[*k].as_ref()
    }
    pub fn insert(&mut self, k: Position, v: T) -> Option<T> {
        let mut ret = Some(v);
        std::mem::swap(&mut ret, &mut self.0[k]);
        ret
    }
}

pub trait ConstDefault {
    const DEFAULT: Self;
}
impl<T> ConstDefault for Option<T> {
    const DEFAULT: Self = None;
}

pub struct Map<T>([[T; MAP_WIDTH]; MAP_HEIGHT]);

impl<T: ConstDefault> Map<T> {
    pub const fn new() -> Self {
        Self([[T::DEFAULT; MAP_WIDTH]; MAP_HEIGHT])
    }
}
impl<T> std::ops::Index<Position> for Map<T> {
    type Output = T;

    fn index(&self, index: Position) -> &Self::Output {
        &self.0[index.y as usize][index.x as usize]
    }
}
impl<T> std::ops::IndexMut<Position> for Map<T> {
    fn index_mut(&mut self, index: Position) -> &mut Self::Output {
        &mut self.0[index.y as usize][index.x as usize]
    }
}

pub struct PositionWalker {
    center: Position,
    current_pos: Option<Position>,
    radius: Coord,
    max_radius: Coord,
    delta: Coord,
}
impl PositionWalker {
    pub fn new(center: Position, max_radius: Coord) -> Self {
        Self {
            center,
            current_pos: Some(center.clone()),
            radius: 0,
            max_radius,
            delta: 0,
        }
    }
    pub fn empty() -> Self {
        Self {
            center: Position { x: 0, y: 0 },
            current_pos: None,
            radius: 1,
            max_radius: 0,
            delta: 0,
        }
    }
    pub fn get_current(&self) -> Option<Position> {
        self.current_pos
    }
}
impl Iterator for PositionWalker {
    type Item = Position;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(current) = self.current_pos {
            if current.x > self.center.x {
                let x = self.center.x + self.delta - self.radius;
                if x >= 0 {
                    self.current_pos = Some(Position { x: x, y: current.y });
                    return Some(current);
                }
            }
            if current.y > self.center.y {
                let x = self.center.x + self.radius - self.delta;
                let y = self.center.y - self.delta;
                if y >= 0 {
                    self.current_pos = Some(Position { x, y });
                    return Some(current);
                }
            }
            if self.delta < self.radius {
                self.delta += 1;
                self.current_pos = Some(Position {
                    x: self.center.x + self.radius - self.delta,
                    y: self.center.y + self.delta,
                });
                return Some(current);
            }
            let v = self.center.x + self.center.y;
            if self.radius < self.max_radius
                && (self.radius < v || self.radius < (MAP_WIDTH + MAP_HEIGHT) as Coord - v)
            {
                self.radius += 1;
                self.delta = 0;
                self.current_pos = Some(Position {
                    x: self.center.x + self.radius,
                    y: self.center.y,
                });
                return Some(current);
            }
            self.current_pos = None;
            return Some(current);
        }
        None
    }
    /*
    fn try_fold<B, F: FnMut(B, Self::Item) -> R, R: std::ops::Try<Ok = B>>(&mut self, init: B, mut f: F) -> R {
        if let Some(current) = & mut self.current_pos {
            let mut tmp = init;
            loop {
                if current.x > self.center.x {
                    let x = self.center.x + self.delta - self.radius;
                    if x >= 0 {
                        current.x = x;
                        tmp = f(tmp, *current)?;
                        continue;
                    }
                }
                if current.y > self.center.y {
                    let x = self.center.x + self.radius - self.delta;
                    let y = self.center.y - self.delta;
                    if x < MAP_WIDTH as Coord && y >= 0 {
                        *current = Position { x, y };
                        tmp = f(tmp, *current)?;
                        continue;
                    }
                }
                if self.delta < self.radius {
                    self.delta += 1;
                    *current = Position {
                        x: self.center.x + self.radius - self.delta,
                        y: self.center.y + self.delta,
                    };
                    tmp = f(tmp, *current)?;
                    continue;
                }
                break;
            }
            let v = self.center.x + self.center.y;
            while self.radius < self.max_radius
                    && (self.radius < v || self.radius < (MAP_WIDTH + MAP_HEIGHT) as Coord - v)
                {
                    self.radius += 1;
                    tmp = f(tmp, *current)?;
                    for delta in 0..self.radius {
                        self.delta = delta;
                        *current = Position {
                            x: self.center.x + self.radius - self.delta,
                            y: self.center.y + self.delta,
                        };
                        tmp = f(tmp, *current)?;
                        let x = self.center.x + self.delta - self.radius;
                        if x >= 0 {
                            current.x = x;
                            tmp = f(tmp, *current)?;
                        }
                        if current.y > self.center.y {
                            let x = self.center.x + self.radius - self.delta;
                            let y = self.center.y - self.delta;
                            if x < MAP_WIDTH as Coord && y >= 0 {
                                *current = Position { x, y };
                                tmp = f(tmp, *current)?;
                            }
                            let x = self.center.x + self.delta - self.radius;
                            if x >= 0 {
                                current.x = x;
                                tmp = f(tmp, *current)?;
                            }
                        }
                    }
                }

            std::ops::Try::from_ok(tmp)
        } else {
            std::ops::Try::from_ok(init)
        }
    }
    */
}
