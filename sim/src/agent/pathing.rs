use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::{Cell, Dir, Observation, Position, PositionMap, RadiusObservation, World, WorldEntity};

struct FindPathIter<C: Copy + Eq + PartialOrd, F, A, O> {
    seeker: WorldEntity,
    cost_acc: A,
    found: F,
    observation: O,
    came_from: PositionMap<Position>,

    costs: PositionMap<C>,
    queue: BinaryHeap<PathNode<C>>,
}
impl<C: Copy + Eq + PartialOrd, F, A, O> FindPathIter<C, F, A, O> {
    pub fn new(
        seeker: WorldEntity,
        start: Position,
        start_cost: C,
        observation: O,
        cost_acc: A,
        found: F,
    ) -> Self {
        let came_from: PositionMap<Position> = PositionMap::new();
        let mut costs = PositionMap::new();
        let mut queue = BinaryHeap::new();
        costs.insert(start, start_cost);
        queue.push(PathNode {
            pos: start,
            exp_cost: start_cost,
        });
        Self {
            seeker,
            cost_acc,
            found,
            observation,
            came_from,
            costs,
            queue,
        }
    }
}
impl<
        C: Copy + Eq + PartialOrd,
        F: Fn(Position, C, &O) -> bool,
        A: Fn(Position, &C, &O) -> C,
        O: Observation,
    > Iterator for FindPathIter<C, F, A, O>
{
    type Item = (Vec<Position>, C);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(PathNode { pos, exp_cost: _ }) = self.queue.pop() {
            let base_cost = *self.costs.get(&pos).unwrap();
            for n in pos.neighbours() {
                if self.observation.known_can_pass(&self.seeker, n) == Some(false) {
                    continue;
                }
                let cost = (self.cost_acc)(n, &base_cost, &self.observation);
                let mut insert = true;
                if let Some(c) = self.costs.get(&n) {
                    if *c <= cost {
                        insert = false;
                    }
                }
                if insert {
                    self.costs.insert(n, cost);
                    self.came_from.insert(n, pos);
                    self.queue.push(PathNode {
                        pos: n,
                        exp_cost: cost,
                    })
                }
                if (self.found)(n, cost, &self.observation) {
                    let mut v = vec![n];
                    let mut current = pos;
                    while let Some(&from) = self.came_from.get(&current) {
                        v.push(current);
                        current = from;
                    }
                    v.reverse();
                    return Some((v, cost));
                }
            }
        }
        None
    }
}

type Cost = i32;
#[derive(Eq, PartialEq, Clone, Debug)]
pub struct PathNode<T: Eq + PartialEq + PartialOrd> {
    pub pos: Position,
    pub exp_cost: T,
}

impl<T: Eq + PartialEq + PartialOrd> Ord for PathNode<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .exp_cost
            .partial_cmp(&self.exp_cost)
            .expect("Non-comparable cost value, probably NaN")
    }
}
impl<T: Eq + PartialEq + PartialOrd> PartialOrd for PathNode<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.exp_cost.partial_cmp(&self.exp_cost)
    }
}

pub trait Pathable: Observation {
    fn path_as(&self, start: Position, goal: Position, entity: &WorldEntity) -> Option<Vec<Dir>> {
        if self.known_can_pass(entity, goal) == Some(false) {
            return None;
        }
        let mut came_from: PositionMap<Dir> = PositionMap::new();
        let mut cost = PositionMap::new();
        let mut queue = BinaryHeap::new();
        cost.insert(start, 0 as Cost);
        queue.push(PathNode {
            pos: start,
            exp_cost: start.distance(&goal) as Cost,
        });
        while let Some(PathNode { pos, exp_cost: _ }) = queue.pop() {
            let base_cost = *cost.get(&pos).unwrap();
            for d in pos.dir(&goal).closest() {
                if let Some(n) = pos.step(*d) {
                    if self.known_can_pass(entity, n) == Some(false) {
                        continue;
                    }
                    if n == goal {
                        let mut v = vec![*d];
                        let mut current = pos;
                        while let Some(from) = came_from.get(&current) {
                            v.push(*from);
                            current = current.step(from.opposite()).unwrap();
                        }
                        v.reverse();
                        return Some(v);
                    } else {
                        let mut insert = true;
                        if let Some(c) = cost.get(&n) {
                            if *c <= base_cost + 1 {
                                insert = false;
                            }
                        }
                        if insert {
                            cost.insert(n, base_cost + 1);
                            came_from.insert(n, *d);
                            queue.push(PathNode {
                                pos: n,
                                exp_cost: base_cost + 1 + n.distance(&goal) as Cost,
                            })
                        }
                    }
                }
            }
        }
        None
    }
    fn find_with_path_as<C: Copy + Eq + PartialOrd>(
        &self,
        start: Position,
        start_cost: C,
        entity: &WorldEntity,
        cost_acc: impl Fn(Position, &C, &Self) -> C,
        found: impl Fn(Position, &Self) -> bool,
    ) -> Option<(Vec<Position>, C)> {
        let mut came_from: PositionMap<Position> = PositionMap::new();
        let mut costs = PositionMap::new();
        let mut queue = BinaryHeap::new();
        costs.insert(start, start_cost);
        queue.push(PathNode {
            pos: start,
            exp_cost: start_cost,
        });
        while let Some(PathNode { pos, exp_cost: _ }) = queue.pop() {
            let base_cost = *costs.get(&pos).unwrap();
            for n in pos.neighbours() {
                if self.known_can_pass(entity, n) == Some(false) {
                    continue;
                }
                let cost = cost_acc(n, &base_cost, &self);
                if found(n, &self) {
                    let mut v = vec![n];
                    let mut current = pos;
                    while let Some(&from) = came_from.get(&current) {
                        v.push(current);
                        current = from;
                    }
                    v.reverse();
                    return Some((v, cost));
                } else {
                    let mut insert = true;
                    if let Some(c) = costs.get(&n) {
                        if *c <= cost {
                            insert = false;
                        }
                    }
                    if insert {
                        costs.insert(n, cost);
                        came_from.insert(n, pos);
                        queue.push(PathNode {
                            pos: n,
                            exp_cost: cost,
                        })
                    }
                }
            }
        }
        None
    }
    fn iter_paths_with_as<
        C: Copy + Eq + PartialOrd,
        A: Fn(Position, &C, &Self) -> C,
        F: Fn(Position, &C, &Self) -> bool,
    >(
        &self,
        start: Position,
        start_cost: C,
        entity: &WorldEntity,
        cost_acc: A,
        found: F,
    ) -> FindPathIter<C, F, A, Self> {
        FindPathIter::new(*entity, start, start_cost, self, cost_acc, found)
    }
}

impl<O: Observation> Pathable for O {}
