use crate::entity_type::EntityType;

type GenID = i16;

#[derive(Ord, PartialOrd, Eq, PartialEq, Copy, Clone, Debug)]
pub struct Entity {
    pub id: u32,
    pub gen: GenID,
    pub e_type: EntityType,
}
impl std::fmt::Display for Entity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{:?} ({})", self.e_type, self.id)
    }
}

#[derive(Clone, Debug, Default)]
pub struct EntityManager {
    generations: Vec<GenID>,
    types: Vec<EntityType>,
    valid: Vec<bool>,
    full_to: usize,
    deleted: u32,
}

impl EntityManager {

    pub fn destroy(&mut self, entity: Entity) -> Result<(), &str> {
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
    pub fn fresh(&mut self, e_type: EntityType) -> Entity {
        if self.deleted > 0 {
            for i in (self.full_to + 1)..self.valid.len() {
                if !self.valid[i] {
                    let gen = self.generations[i] + 1;
                    self.generations[i] = gen;
                    self.types[i] = e_type;
                    self.valid[i] = true;
                    self.deleted -= 1;
                    self.full_to = i;
                    return Entity { id: i as u32, gen, e_type };
                }
            }
        }
        debug_assert!(self.deleted == 0);
        let len = self.valid.len();
        let gen = 0;
        self.generations.push(gen);
        self.types.push(e_type);
        self.valid.push(true);
        self.full_to = len;
        Entity {
            id: len as u32,
            gen,
            e_type
        }
    }
    pub fn put(&mut self, entity: Entity) -> Result<Entity, Entity> {
        let Entity{gen, id, e_type} = entity;
        let id = id as usize;
        if id >= self.generations.len() {
            self.generations.resize(id + 1, -1);
            self.valid.resize(id + 1, false);
            self.generations[id] = gen;
            self.types[id] = e_type;
            self.valid[id] = true;
            return Ok(entity);
        } if !self.valid[id] && self.generations[id] < gen {
            self.generations[id] = gen;
            self.types[id] = e_type;
            self.valid[id] = true;
            return Ok(entity)
        } else {
            return Err(self.fresh(e_type))
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
                    e_type: self.em.types[self.idx],
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
    pub fn get(&self, entity: &Entity) -> Option<&T> {
        let Entity { id, gen, e_type : _ } = entity;
        let id = *id as usize;
        if let Some(stored_gen) = self.generations.get(id) {
            if stored_gen == gen {
                if let Some(opt) = self.content.get(id) {
                    return opt.as_ref();
                }
            }
        }
        None
    }
    pub fn get_mut(&mut self, entity: &Entity) -> Option<&mut T> {
        let Entity { id, gen, e_type : _ } = entity;
        let id = *id as usize;
        if let Some(stored_gen) = self.generations.get(id) {
            if stored_gen == gen {
                if let Some(opt) = self.content.get_mut(id) {
                    return opt.as_mut();
                }
            }
            else if stored_gen < gen {
                self.content[id] = None;
            }
        }
        None
    }
    pub fn get_or_insert_with(& mut self, entity: & Entity, inserter: impl FnOnce() -> T) -> &mut T {
        let Entity { id, gen, e_type : _  } = entity;
        let id = *id as usize;
        if let Some(stored_gen) = self.generations.get(id) {
            if stored_gen > gen {
                log::error!("invalid entity state");
            }
            else if stored_gen < gen {
                let val = inserter();
                self.generations[id] = *gen;
                self.content[id] = Some(val);
            }
        }
        else {
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
    pub fn insert(&mut self, entity: &Entity, val: T) -> Option<(GenID, T)> {
        let Entity { id, gen, e_type : _ } = entity;
        let id = *id as usize;
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
        self.generations[id] = *gen;
        if old_gen >= 0 && old_cont.is_some() {
            return Some((old_gen, old_cont.unwrap()));
        }
        None
    }
    pub fn remove(& mut self, entity: &Entity) -> Option<T> {
        let Entity { id, gen, e_type : _ } = entity;
        let id = *id as usize;
        if let Some(stored_gen) = self.generations.get(id) {
            if stored_gen <= gen {
                let mut val = None;
                std::mem::swap(& mut self.content[id], & mut val);
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
    pub fn split_out_mut<'a>(& 'a mut self, entity: &Entity) -> Option<(& 'a mut T, StorageSlice<'a, T>)> {
        if self.get(entity).is_none() {
            return None;
        }
        let idx = entity.id as usize;
        let Self {  content, generations} = self;
        let (con0, con_) = content.as_mut_slice().split_at_mut(idx);
        let (gen0, gen1) = generations.as_mut_slice().split_at_mut(idx);
        let (e, con1) = con_.split_at_mut(1);
        if let Some(Some(elem)) = e.get_mut(0) {
            return Some((elem, StorageSlice::<T>{con0, con1, gen0, gen1, idx }))
        }
        None
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
pub struct StorageSlice<'a,T> {
    gen0: & 'a [GenID],
    gen1: & 'a [GenID],
    con0: & 'a [Option<T>],
    con1: & 'a [Option<T>],
    idx: usize,
}
impl<'a,T>  StorageSlice<'a,T>  {
    pub fn get(&self, entity: &Entity) -> Option<&'a T> {
        let idx = entity.id as usize;
        if idx < self.idx {
            if let Some(stored_gen) = self.gen0.get(idx) {
                if *stored_gen == entity.gen {
                    if let Some(opt) = self.con0.get(idx) {
                        return opt.as_ref();
                    }
                }
            }
        }
        else if idx > self.idx {
            if let Some(stored_gen) = self.gen1.get(idx) {
                if *stored_gen == entity.gen {
                    if let Some(opt) = self.con1.get(idx) {
                        return opt.as_ref();
                    }
                }
            }
        }
        None
    }
}

pub struct StorageSliceIter<'a, T> {
    slice : & 'a StorageSlice<'a, T>,
    idx : usize,
}

impl<'a, T> Iterator for StorageSliceIter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<&'a T>{
        while self.idx + 1 < self.slice.idx {
            self.idx += 1;

            if let Some(t) = &self.slice.con0[self.idx] {
                return Some(t);
            }
        }
        // skipping non-accessible element
        if self.idx == self.slice.idx {
            self.idx +=1;
        }
        while self.idx + 1  - self.slice.idx < self.slice.con1.len()  {
            self.idx += 1;

            if let Some(t) = &self.slice.con1[self.idx] {
                return Some(t);
            }
        }
        None
    }

}
impl<'a, T> IntoIterator for & 'a StorageSlice<'a,T> {
    type IntoIter = StorageSliceIter<'a, T> ;
    type Item = &'a T;
    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter{ slice: self, idx: 0 }
    }
}