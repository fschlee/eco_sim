use rayon::prelude::*;

use crate::entity_type::EntityType;

type GenID = i16;
type ID = u32;

trait Traits: Sized + Ord + Eq + Copy + std::fmt::Debug {}

impl Traits for EntityType {}

impl std::fmt::Display for WorldEntity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{:?} ({})", self.e_type(), self.id())
    }
}
#[derive(Ord, PartialOrd, Eq, PartialEq, Copy, Clone, Debug, Hash)]
pub struct Entity {
    id: ID,
    gen: GenID,
}

#[derive(Ord, PartialOrd, Eq, PartialEq, Copy, Clone, Debug)]
pub struct WorldEntity {
    ent: Entity,
    e_type: EntityType,
}
impl WorldEntity {
    #[inline]
    pub fn id(&self) -> usize {
        self.ent.id as usize
    }
    #[inline]
    pub fn e_type(&self) -> EntityType {
        self.e_type
    }
    pub fn new(ent: Entity, e_type: EntityType) -> Self {
        Self { ent, e_type }
    }
}
impl Into<Entity> for WorldEntity {
    fn into(self) -> Entity {
        self.ent
    }
}
impl<'a> Into<&'a Entity> for &'a WorldEntity {
    fn into(self) -> &'a Entity {
        &self.ent
    }
}
impl Into<Entity> for &WorldEntity {
    fn into(self) -> Entity {
        self.ent
    }
}

#[derive(Clone, Debug, Default)]
pub struct EntityManager {
    generations: Vec<GenID>,
    //  data: Vec<T>,
    valid: Vec<bool>,
    full_to: usize,
    deleted: u32,
}

impl EntityManager {
    pub fn destroy(&mut self, entity: impl Into<Entity>) -> Result<(), &str> {
        let entity = entity.into();
        debug_assert!(entity.id < self.generations.len() as u32);
        let id = entity.id as usize;
        if self.valid[id] && self.generations[id] == entity.gen {
            self.valid[id] = false;
            self.deleted += 1;
            if self.full_to > id {
                self.full_to = id;
            }
            Ok(())
        } else {
            Err("Entity to be destroyed does not exist")
        }
    }
    pub fn fresh(&mut self) -> Entity {
        if self.deleted > 0 {
            for i in (self.full_to + 1)..self.valid.len() {
                if !self.valid[i] {
                    let gen = self.generations[i] + 1;
                    self.generations[i] = gen;
                    self.valid[i] = true;
                    self.deleted -= 1;
                    self.full_to = i;
                    return Entity { id: i as u32, gen };
                }
            }
        }
        debug_assert!(self.deleted == 0);
        let len = self.valid.len();
        let gen = 0;
        self.generations.push(gen);
        self.valid.push(true);
        self.full_to = len;
        Entity {
            id: len as u32,
            gen,
        }
    }
    pub fn put(&mut self, entity: impl Into<Entity>) -> Result<Entity, Entity> {
        let entity = entity.into();
        let Entity { gen, id } = entity;
        let id = id as usize;
        if id >= self.generations.len() {
            self.generations.resize(id + 1, -1);
            self.valid.resize(id + 1, false);
            self.generations[id] = gen;
            self.valid[id] = true;
            return Ok(entity);
        }
        if !self.valid[id] && self.generations[id] < gen {
            self.generations[id] = gen;
            self.valid[id] = true;
            return Ok(entity);
        } else {
            return Err(self.fresh());
        }
    }
}

pub struct EntityIter<'a> {
    em: &'a EntityManager,
    idx: usize,
}

impl<'a> Iterator for EntityIter<'a> {
    type Item = Entity;
    fn next(&mut self) -> Option<Self::Item> {
        while self.idx + 1 < self.em.valid.len() {
            self.idx += 1;
            if self.em.valid[self.idx] {
                return Some(Entity {
                    id: self.idx as u32,
                    gen: self.em.generations[self.idx],
                });
            }
        }
        None
    }
}

impl<'a> IntoIterator for &'a EntityManager {
    type Item = Entity;
    type IntoIter = EntityIter<'a>;
    fn into_iter(self) -> Self::IntoIter {
        EntityIter { em: self, idx: 0 }
    }
}

#[derive(Clone, Debug)]
pub struct Storage<T> {
    content: Vec<Option<T>>,
    generations: Vec<GenID>,
}

impl<T> Storage<T> {
    pub fn get(&self, entity: impl Into<Entity>) -> Option<&T> {
        let Entity { id, gen } = entity.into();
        let id = id as usize;
        if let Some(stored_gen) = self.generations.get(id) {
            if *stored_gen == gen {
                if let Some(opt) = self.content.get(id) {
                    return opt.as_ref();
                }
            }
        }
        None
    }
    pub fn get_mut(&mut self, entity: impl Into<Entity>) -> Option<&mut T> {
        let Entity { id, gen } = entity.into();
        let id = id as usize;
        if let Some(stored_gen) = self.generations.get(id) {
            if *stored_gen == gen {
                if let Some(opt) = self.content.get_mut(id) {
                    return opt.as_mut();
                }
            } else if *stored_gen < gen {
                self.content[id] = None;
            }
        }
        None
    }
    pub fn get_or_insert_with(
        &mut self,
        entity: impl Into<Entity>,
        inserter: impl FnOnce() -> T,
    ) -> &mut T {
        let Entity { id, gen } = &entity.into();
        let id = *id as usize;
        if let Some(stored_gen) = self.generations.get(id) {
            if stored_gen > gen {
                log::error!("invalid entity state");
            } else if stored_gen < gen {
                let val = inserter();
                self.generations[id] = *gen;
                self.content[id] = Some(val);
            }
        } else {
            let val = inserter();
            let end = self.generations.len();
            if id >= end {
                for _ in end..=id {
                    self.generations.push(-1);
                    self.content.push(None);
                }
            }
            self.generations[id] = *gen;
            self.content[id] = Some(val);
        }
        self.content[id].get_or_insert_with(|| unreachable!())
    }
    pub fn insert(&mut self, entity: impl Into<Entity>, val: T) -> Option<(GenID, T)> {
        let Entity { id, gen } = entity.into();
        let id = id as usize;
        let end = self.generations.len();
        if id >= end {
            for _ in end..=id {
                self.generations.push(-1);
                self.content.push(None);
            }
        }
        let old_gen = self.generations[id];
        let mut old_cont = Some(val);
        std::mem::swap(&mut self.content[id], &mut old_cont);
        self.generations[id] = gen;
        if old_gen >= 0 && old_cont.is_some() {
            return Some((old_gen, old_cont.unwrap()));
        }
        None
    }
    pub fn remove(&mut self, entity: impl Into<Entity>) -> Option<T> {
        let Entity { id, gen } = &entity.into();
        let id = *id as usize;
        if let Some(stored_gen) = self.generations.get(id) {
            if stored_gen <= gen {
                let mut val = None;
                std::mem::swap(&mut self.content[id], &mut val);
                return val;
            }
        }
        None
    }
    pub fn new() -> Self {
        Self {
            content: Vec::new(),
            generations: Vec::new(),
        }
    }
    pub fn split_out_mut<'a>(
        &'a mut self,
        entity: impl Into<Entity>,
    ) -> Option<(&'a mut T, StorageSlice<'a, T>)> {
        let entity = entity.into();
        if self.get(entity).is_none() {
            return None;
        }
        let idx = entity.id as usize;
        let Self {
            content,
            generations,
        } = self;
        let (con0, con_) = content.as_mut_slice().split_at_mut(idx);
        let (gen0, gen1) = generations.as_mut_slice().split_at_mut(idx);
        let (e, con1) = con_.split_at_mut(1);
        if let Some(Some(elem)) = e.get_mut(0) {
            return Some((
                elem,
                StorageSlice::<T> {
                    con0,
                    con1,
                    gen0,
                    gen1,
                    idx,
                },
            ));
        }
        None
    }
    pub fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut T> + 'a {
        self.content.iter_mut().filter_map(|t| t.as_mut())
    }
}
impl<T: Send> Storage<T> {
    pub fn par_iter_mut<'a>(&'a mut self) -> impl ParallelIterator<Item = &'a mut T> + 'a {
        self.content.par_iter_mut().filter_map(|t| t.as_mut())
    }
}

impl<T> Default for Storage<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T> IntoIterator for &'a Storage<T> {
    type Item = &'a T;
    type IntoIter = StorageIter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        StorageIter { st: self, idx: 0 }
    }
}

pub struct StorageIter<'a, T> {
    st: &'a Storage<T>,
    idx: usize,
}

impl<'a, T> Iterator for StorageIter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        while self.idx + 1 < self.st.content.len() {
            self.idx += 1;

            if let Some(t) = &self.st.content[self.idx] {
                return Some(t);
            }
        }
        None
    }
}

#[derive(Clone)]
pub struct StorageSlice<'a, T> {
    gen0: &'a [GenID],
    gen1: &'a [GenID],
    con0: &'a [Option<T>],
    con1: &'a [Option<T>],
    idx: usize,
}
impl<'a, T> StorageSlice<'a, T> {}

pub struct StorageSliceIter<'a, T> {
    slice: &'a StorageSlice<'a, T>,
    idx: usize,
}

impl<'a, T> Iterator for StorageSliceIter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<&'a T> {
        while self.idx + 1 < self.slice.idx {
            self.idx += 1;

            if let Some(t) = &self.slice.con0[self.idx] {
                return Some(t);
            }
        }
        // skipping non-accessible element
        if self.idx == self.slice.idx {
            self.idx += 1;
        }
        while self.idx + 1 - self.slice.idx < self.slice.con1.len() {
            self.idx += 1;

            if let Some(t) = &self.slice.con1[self.idx] {
                return Some(t);
            }
        }
        None
    }
}

impl<'a, T> IntoIterator for &'a StorageSlice<'a, T> {
    type IntoIter = StorageSliceIter<'a, T>;
    type Item = &'a T;
    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter {
            slice: self,
            idx: 0,
        }
    }
}

impl<'a, 'b, T> IntoIterator for &'b &'a StorageSlice<'a, T> {
    type IntoIter = StorageSliceIter<'a, T>;
    type Item = &'a T;
    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter {
            slice: *self,
            idx: 0,
        }
    }
}

pub trait Source<'a, T> {
    fn get(&'a self, entity: Entity) -> Option<T>;
    fn iter(&'a self) -> Box<dyn Iterator<Item = T> + 'a>;
}

impl<'a, T: 'a, F: 'static> Source<'a, T> for &'a StorageSlice<'a, F>
where
    &'a F: Into<T>,
{
    fn get(&'a self, entity: Entity) -> Option<T> {
        let idx = entity.id as usize;
        if idx < self.idx {
            if let Some(stored_gen) = self.gen0.get(idx) {
                if *stored_gen == entity.gen {
                    if let Some(opt) = self.con0.get(idx) {
                        return opt.as_ref().map(Into::into);
                    }
                }
            }
        } else if idx > self.idx {
            if let Some(stored_gen) = self.gen1.get(idx) {
                if *stored_gen == entity.gen {
                    if let Some(opt) = self.con1.get(idx) {
                        return opt.as_ref().map(Into::into);
                    }
                }
            }
        }
        None
    }
    fn iter(&'a self) -> Box<dyn Iterator<Item = T> + 'a> {
        Box::new(self.into_iter().map(Into::into))
    }
}
/*
impl<T> Source<T> for Storage<T> {
    fn get(&self, entity: impl Into<Entity>) -> Option<&T> {
        Storage::get(self, entity)
    }
}
*/
/*
impl<'a, T, F> Source<T> for StorageSlice<'a, F> where &F: Into<T> {
    fn get(&self, entity: impl Into<Entity>) -> Option<T> {
        self.get( entity).map(|e | e.into())
    }
}*/
impl<'a, T, I: 'a> Source<'a, T> for Storage<I>
where
    &'a I: Into<T>,
{
    fn get(&'a self, entity: Entity) -> Option<T> {
        self.get(entity).map(|e| e.into())
    }
    fn iter(&'a self) -> Box<dyn Iterator<Item = T> + 'a> {
        Box::new(self.into_iter().map(|e| e.into()))
    }
}

pub struct StorageAdapter<'a, T, U> {
    storage: &'a Storage<U>,
    fun: Box<dyn Fn(&U) -> Option<&T> + Sync + Send>,
}
impl<'a, T, U> StorageAdapter<'a, T, U> {
    pub fn new(storage: &'a Storage<U>, fun: Box<dyn Fn(&U) -> Option<&T> + Sync + Send>) -> Self {
        Self { storage, fun }
    }
}
impl<'a, T, U> Source<'a, &'a T> for StorageAdapter<'a, T, U> {
    fn get(&'a self, entity: Entity) -> Option<&'a T> {
        self.storage.get(entity).and_then(|t| (*self.fun)(t))
    }

    fn iter(&'a self) -> Box<dyn Iterator<Item = &'a T> + 'a> {
        Box::new(
            self.storage
                .iter()
                .flat_map(move |u| (*self.fun)(u).into_iter()),
        )
    }
}
