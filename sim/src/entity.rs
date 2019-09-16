#[derive(Ord, PartialOrd, Eq, PartialEq, Copy, Clone, Debug)]
pub struct Entity {
    pub id: u32,
    pub gen: i32,
}

#[derive(Clone, Debug, Default)]
pub struct EntityManager {
    generations: Vec<i32>,
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
            if (self.full_to > id) {
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
    pub fn put(&mut self, entity: Entity) -> Result<Entity, Entity> {
        let Entity{gen, id} = entity;
        let id = id as usize;
        if id >= self.generations.len() {
            self.generations.resize(id + 1, -1);
            self.valid.resize(id + 1, false);
            self.generations[id] = gen;
            self.valid[id] = true;
            return Ok(entity);
        } if !self.valid[id] && self.generations[id] < gen {
            self.generations[id] = gen;
            self.valid[id] = true;
            return Ok(entity)
        } else {
            return Err(self.fresh())
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
    generations: Vec<i32>,
}

impl<T> Storage<T> {
    pub fn get(&self, entity: &Entity) -> Option<&T> {
        let Entity { id, gen } = entity;
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
        let Entity { id, gen } = entity;
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
    pub fn insert(&mut self, entity: &Entity, val: T) -> Option<(i32, T)> {
        let Entity { id, gen } = entity;
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
    pub fn remove(& mut self, entity: &Entity) {
        let Entity { id, gen } = entity;
        let id = *id as usize;
        if let Some(stored_gen) = self.generations.get(id) {
            if stored_gen <= gen {
                self.content[id] = None;
            }
        }
    }
    pub fn new() -> Self {
        Self {
            content: Vec::new(),
            generations: Vec::new(),
        }
    }
}
impl<T> Default for Storage<T> {
    fn default() -> Self {
        Self::new()
    }
}
