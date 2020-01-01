use super::*;

pub mod buffer;
pub mod depth;
pub mod descriptors;
pub mod texture;

use std::collections::hash_map::RandomState;
use std::hash::Hasher;

pub use buffer::BufferBundle;

pub use texture::{LoadedTexture, TextureSpec};
pub type Texture<B> = texture::LoadedTexture<B>;

pub enum Tex {}

pub trait BufferType {
    fn usage() -> BufferUsage;
}

pub struct VtxBuff<V> {
    phantom: PhantomData<*const V>,
}

impl<V> BufferType for VtxBuff<V> {
    fn usage() -> BufferUsage {
        BufferUsage::VERTEX
    }
}

pub enum IdxBuff {}
impl BufferType for IdxBuff {
    fn usage() -> BufferUsage {
        BufferUsage::INDEX
    }
}

pub enum TempUniform {}

pub type FontId = usize;
#[derive(Default)]
pub struct Id<T> {
    pub id: u32,
    phantom: PhantomData<*const T>,
}
impl<T> Id<T> {
    pub fn new(id: u32) -> Self {
        Self {
            id,
            phantom: PhantomData,
        }
    }
}
impl<T> Clone for Id<T> {
    fn clone(&self) -> Id<T> {
        Id {
            id: self.id,
            phantom: PhantomData,
        }
    }
}

impl<T> Copy for Id<T> {}
impl<T> PartialEq for Id<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl<T> Eq for Id<T> {}
impl<T> std::fmt::Debug for Id<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Id {{ id: {}, phantom }}", self.id)
    }
}
impl<T> std::hash::Hash for Id<T> {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.id.hash(hasher);
    }
}

pub struct ResourceManager<B: Backend> {
    device: Arc<Dev<B>>,
    adapter: Arc<Adapter<B>>,
    mem_atom: usize,
    command_pool: ManuallyDrop<B::CommandPool>,
    //   command_queue: ManuallyDrop<CommandQueue<back::Backend, Trans>>,
    textures: Vec<Option<Texture<B>>>,
    old_textures: Vec<Texture<B>>,
    old_texture_expirations: Vec<i32>,
    old_pools: Vec<<B as Backend>::DescriptorPool>,
    old_pool_expirations: Vec<i32>,
    pub descriptor_set_layouts: Vec<<B as Backend>::DescriptorSetLayout>,
    descriptor_pool: ManuallyDrop<<B as Backend>::DescriptorPool>,
    max_texture_sets: usize,
    ub_count: usize,
    descriptor_map: std::collections::HashMap<Id<Tex>, <B as Backend>::DescriptorSet, RandomState>,
    unmapped_descriptor_count: usize,
    pub uniform_buffers: Vec<BufferBundle<B>>,
    pub uniform_buffer_descs: Vec<<B as Backend>::DescriptorSet>,
    registered_buffers: Vec<BufferBundle<B>>,
    buffer_usages: Vec<BufferUsage>,
    old_buffers: Vec<BufferBundle<B>>,
    old_buffer_expirations: Vec<i32>,
    old_buffer_descriptors: Vec<Option<<B as Backend>::DescriptorSet>>,
}

impl<B: Backend + BackendExt> ResourceManager<B> {
    pub const MAX_UB_COUNT: usize = 256;
    pub const AUTO_UB_COUNT: usize = 4;
    pub const TEXTURE_AND_SAMPLER: usize = 0;
    pub const UNIFORM_VS: usize = 1;
    const DELETE_DELAY: i32 = 4;
    const TEXTURES_PER_BINDING: usize = 1;
    pub fn new(
        device: Arc<Dev<B>>,
        adapter: Arc<Adapter<B>>,
        command_pool: B::CommandPool,
        //       command_queue: CommandQueue<back::Backend, Trans>,
    ) -> Result<Self, Error> {
        let mem_atom = adapter.physical_device.limits().non_coherent_atom_size;
        let max_texture_sets = 24;
        let ub_count = Self::AUTO_UB_COUNT;
        let mut descriptor_set_layouts: Vec<<B as Backend>::DescriptorSetLayout> = Vec::new();
        let mut descs = vec![
            DescriptorSetLayoutBinding {
                binding: 0,
                ty: gfx_hal::pso::DescriptorType::SampledImage,
                count: 1,
                stage_flags: ShaderStageFlags::FRAGMENT,
                immutable_samplers: false,
            },
            DescriptorSetLayoutBinding {
                binding: 1,
                ty: gfx_hal::pso::DescriptorType::Sampler,
                count: 1,
                stage_flags: ShaderStageFlags::FRAGMENT,
                immutable_samplers: false,
            },
        ];

        unsafe {
            descriptor_set_layouts.push(
                device
                    .create_descriptor_set_layout(&descs, &[])
                    .map_err(|_| "Couldn't make a DescriptorSetLayout")?,
            );
            descriptor_set_layouts.push(
                device
                    .create_descriptor_set_layout(
                        &vec![DescriptorSetLayoutBinding {
                            binding: 0,
                            ty: gfx_hal::pso::DescriptorType::UniformBuffer,
                            count: 1,
                            stage_flags: ShaderStageFlags::VERTEX,
                            immutable_samplers: false,
                        }],
                        &[],
                    )
                    .map_err(|_| "Couldn't make a DescriptorSetLayout")?,
            )
        }
        // 2. you create a descriptor pool, and when making that descriptor pool
        //    you specify how many sets you want to be able to allocate from the
        //    pool, as well as the maximum number of each kind of descriptor you
        //    want to be able to allocate from that pool, total, for all sets.
        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(
                    max_texture_sets + Self::MAX_UB_COUNT, // sets
                    &[
                        gfx_hal::pso::DescriptorRangeDesc {
                            ty: gfx_hal::pso::DescriptorType::SampledImage,
                            count: max_texture_sets * Self::TEXTURES_PER_BINDING,
                        },
                        gfx_hal::pso::DescriptorRangeDesc {
                            ty: gfx_hal::pso::DescriptorType::Sampler,
                            count: max_texture_sets * Self::TEXTURES_PER_BINDING,
                        },
                        gfx_hal::pso::DescriptorRangeDesc {
                            ty: gfx_hal::pso::DescriptorType::UniformBuffer,
                            count: Self::MAX_UB_COUNT,
                        },
                    ],
                    gfx_hal::pso::DescriptorPoolCreateFlags::empty(),
                )
                .map_err(|_| "Couldn't create a descriptor pool!")?
        };
        let mut uniform_buffers = Vec::new();
        let size = 20 * 256;
        if true || !B::can_push_graphics_constants() {
            for i in 0..ub_count {
                uniform_buffers.push(
                    BufferBundle::new(
                        adapter.as_ref(),
                        device.as_ref(),
                        size,
                        BufferUsage::UNIFORM,
                    )
                    .expect("Failed creating uniform buffer"),
                );
            }
        }
        let mut this = Self {
            device,
            adapter,
            mem_atom,
            command_pool: ManuallyDrop::new(command_pool),
            //           command_queue: ManuallyDrop::new(command_queue),
            textures: Vec::new(),
            descriptor_set_layouts,
            descriptor_pool: ManuallyDrop::new(descriptor_pool),
            old_textures: Vec::new(),
            old_texture_expirations: Vec::new(),
            old_pool_expirations: Vec::new(),
            old_pools: Vec::new(),
            descriptor_map: std::collections::HashMap::with_capacity(max_texture_sets),
            unmapped_descriptor_count: 0,
            max_texture_sets,
            ub_count,
            uniform_buffers,
            registered_buffers: Vec::new(),
            buffer_usages: Vec::new(),
            old_buffers: Vec::new(),
            old_buffer_expirations: Vec::new(),
            old_buffer_descriptors: Vec::new(),
            uniform_buffer_descs: Vec::new(),
        };
        if true || !B::can_push_graphics_constants() {
            this.allocate_uniform_descs()?;
        }
        Ok(this)
    }

    fn allocate_uniform_descs(&mut self) -> Result<(), Error> {
        self.uniform_buffer_descs.clear();
        for i in 0..Self::AUTO_UB_COUNT {
            unsafe {
                let new_set = self
                    .descriptor_pool
                    .allocate_set(&self.descriptor_set_layouts[Self::UNIFORM_VS])?;
                self.device
                    .write_descriptor_sets(vec![gfx_hal::pso::DescriptorSetWrite {
                        set: &new_set,
                        binding: 0,
                        array_offset: 0,
                        descriptors: Some(gfx_hal::pso::Descriptor::Buffer(
                            self.uniform_buffers[i].buffer.deref(),
                            None..None,
                        )),
                    }]);
                self.uniform_buffer_descs.push(new_set);
            }
        }
        Ok(())
    }
    pub fn get_texture(&self, id: Id<Tex>) -> Option<&Texture<B>> {
        match self.textures.get(id.id as usize) {
            Some(Some(tex)) => Some(tex),
            _ => None,
        }
    }
    pub fn get_descriptor_set(&self, id: Id<Tex>) -> Option<&<B as Backend>::DescriptorSet> {
        if self.descriptor_map.contains_key(&id) {
            self.descriptor_map.get(&id)
        } else {
            None
        }
    }
    pub fn get_or_write_descriptor_set(
        &mut self,
        id: Id<Tex>,
    ) -> Result<&<B as Backend>::DescriptorSet, Error> {
        if self.get_descriptor_set(id).is_some() {
            return self.get_descriptor_set(id).ok_or("Impossible".into());
        }
        if self.descriptor_map.len() + self.unmapped_descriptor_count >= self.max_texture_sets {
            self.refresh_pool()?;
            self.unmapped_descriptor_count = 0;
            warn!("refreshed descriptor pool");
        }
        match self.textures.get(id.id as usize).and_then(|a| a.as_ref()) {
            None => {
                error!("Texture id {} requested but does not exist", id.id);
                return Err("unknown texture id".into());
            }
            Some(texture) => unsafe {
                match self
                    .descriptor_pool
                    .allocate_set(&self.descriptor_set_layouts[0])
                {
                    Ok(new_set) => {
                        let vec = vec![
                            gfx_hal::pso::DescriptorSetWrite {
                                set: &new_set,
                                binding: 0,
                                array_offset: 0,
                                descriptors: Some(gfx_hal::pso::Descriptor::Image(
                                    texture.image_view.deref(),
                                    Layout::ShaderReadOnlyOptimal,
                                )),
                            },
                            gfx_hal::pso::DescriptorSetWrite {
                                set: &new_set,
                                binding: 1,
                                array_offset: 0,
                                descriptors: Some(gfx_hal::pso::Descriptor::Sampler(
                                    texture.sampler.deref(),
                                )),
                            },
                        ];
                        self.device.write_descriptor_sets(vec);
                        self.descriptor_map.insert(id, new_set);
                        Ok(self.descriptor_map.get(&id).unwrap())
                    }
                    Err(err) => {
                        error!("{} for texture #{}", err, id.id);
                        Err("couldn't allocate descriptor set".into())
                    }
                }
            },
        }
    }
    fn refresh_pool(&mut self) -> Result<(), Error> {
        unsafe {
            let pool = self
                .device
                .create_descriptor_pool(
                    self.max_texture_sets + self.ub_count, // sets
                    &[
                        gfx_hal::pso::DescriptorRangeDesc {
                            ty: gfx_hal::pso::DescriptorType::SampledImage,
                            count: self.max_texture_sets * Self::TEXTURES_PER_BINDING,
                        },
                        gfx_hal::pso::DescriptorRangeDesc {
                            ty: gfx_hal::pso::DescriptorType::Sampler,
                            count: self.max_texture_sets * Self::TEXTURES_PER_BINDING,
                        },
                        gfx_hal::pso::DescriptorRangeDesc {
                            ty: gfx_hal::pso::DescriptorType::UniformBuffer,
                            count: Self::MAX_UB_COUNT,
                        },
                    ],
                    gfx_hal::pso::DescriptorPoolCreateFlags::empty(),
                )
                .map_err(|_| "Couldn't create a descriptor pool!")?;
            self.old_pools
                .push(ManuallyDrop::take(&mut self.descriptor_pool));
            self.old_pool_expirations.push(Self::DELETE_DELAY);
            self.descriptor_pool = ManuallyDrop::new(pool);
        }
        self.allocate_uniform_descs()?;
        self.ub_count = Self::AUTO_UB_COUNT;
        self.descriptor_map.clear();
        Ok(())
    }
    fn padded_size(&self, proper_size: usize) -> usize {
        ((proper_size + self.mem_atom - 1) / self.mem_atom) * self.mem_atom
    }

    pub(crate) fn register_buffer<T: Sized + Copy, U: BufferType>(
        &mut self,
        slice: &[T],
    ) -> Result<Id<U>, Error> {
        let idx = self.registered_buffers.len();
        let usage = U::usage();
        {
            let size = size_of::<T>() * slice.len();
            let pad = self.padded_size(size);
            let buff = BufferBundle::new(self.adapter.deref(), self.device.deref(), pad, usage)?;
            let res = unsafe { buff.write_range(&self.device, 0..(pad as u64), slice) };
            self.registered_buffers.push(buff);
            self.buffer_usages.push(usage);
            res?;
        }
        Ok(Id::new(idx as u32))
    }
    pub fn replace_buffer<T: Sized + Copy, U: BufferType>(
        &mut self,
        id: Id<U>,
        slice: &[T],
    ) -> Result<Id<U>, Error> {
        let idx = id.id as usize;
        let usage = U::usage();
        assert!(self.registered_buffers.len() < idx);
        assert!(self.buffer_usages[idx] == usage);
        {
            let size = size_of::<T>() * slice.len();
            let pad = self.padded_size(size);
            let mut buff =
                BufferBundle::new(self.adapter.deref(), self.device.deref(), pad, usage)?;
            let res = unsafe { buff.write_range(&self.device, 0..(pad as u64), slice) };
            std::mem::swap(&mut self.registered_buffers[idx], &mut buff);
            self.old_buffers.push(buff);
            self.old_buffer_expirations.push(Self::DELETE_DELAY);
            res?;
        }
        Ok(id)
    }
    pub fn get_buffer<U: BufferType>(&self, id: Id<U>) -> Option<&BufferBundle<B>> {
        let idx = id.id as usize;
        if idx >= self.buffer_usages.len() || self.buffer_usages[idx] != U::usage() {
            return None;
        }
        self.registered_buffers.get(idx)
    }
    pub fn write_temp_uniform_buffer_with_descriptor<T: Sized + Copy>(
        &mut self,
        slice: &[T],
    ) -> Result<Id<TempUniform>, Error> {
        let idx = self.old_buffers.len();
        let usage = BufferUsage::UNIFORM;
        let size = size_of::<T>() * slice.len();
        let pad = self.padded_size(size);
        unsafe {
            let mut buff =
                BufferBundle::new(self.adapter.deref(), self.device.deref(), pad, usage)?;
            let res = buff.write_range(&self.device, 0..(pad as u64), slice);
            self.old_buffers.push(buff);
            self.old_buffer_expirations.push(Self::DELETE_DELAY);
            if self.ub_count + 1 >= Self::MAX_UB_COUNT {
                self.refresh_pool()?;
            }
            self.ub_count += 1;
            let new_set = self
                .descriptor_pool
                .allocate_set(&self.descriptor_set_layouts[Self::UNIFORM_VS])?;
            self.device
                .write_descriptor_sets(vec![gfx_hal::pso::DescriptorSetWrite {
                    set: &new_set,
                    binding: 0,
                    array_offset: 0,
                    descriptors: Some(gfx_hal::pso::Descriptor::Buffer(
                        self.old_buffers[idx].buffer.deref(),
                        None..None,
                    )),
                }]);
            let len_diff = idx - self.old_buffer_descriptors.len();
            if len_diff > 0 {
                for _ in 0..len_diff {
                    self.old_buffer_descriptors.push(None);
                }
            }
            debug_assert_eq!(idx, self.old_buffer_descriptors.len());
            self.old_buffer_descriptors.push(Some(new_set));
        }
        Ok(Id::new(idx as u32))
    }
    pub fn get_temp_buffer_descriptor(
        &self,
        id: Id<TempUniform>,
    ) -> Option<&<B as Backend>::DescriptorSet> {
        let idx = id.id as usize;
        if self.old_buffer_descriptors.len() <= 4
            || self.old_buffer_expirations.len() <= idx
            || self.old_buffer_expirations[idx] < 4
        {
            None
        } else {
            self.old_buffer_descriptors[idx].as_ref()
        }
    }
    pub fn remove_texture(&mut self, id: Id<Tex>) -> Result<(), Error> {
        if let Some(mtex) = self.textures.get_mut(id.id as usize) {
            if let Some(tex) = mtex.take() {
                self.old_textures.push(tex);
                self.old_texture_expirations.push(Self::DELETE_DELAY);
                return Ok(());
            }
        }
        Err("No texture present".into())
    }

    pub fn replace_texture<'a>(
        &mut self,
        id: Id<Tex>,
        spec: &'a TextureSpec,
        command_queue: &mut B::CommandQueue,
    ) -> Result<(), Error> {
        if let Some(mtex) = self.textures.get_mut(id.id as usize) {
            if mtex.is_some() {
                let mut other = LoadedTexture::from_texture_spec(
                    self.adapter.deref(),
                    self.device.deref(),
                    &mut self.command_pool,
                    command_queue,
                    spec,
                )
                .ok();
                std::mem::swap(mtex, &mut other);
                if let Some(otex) = other {
                    self.old_textures.push(otex);
                    self.old_texture_expirations.push(Self::DELETE_DELAY);
                }
                self.descriptor_map.remove(&id);
                self.unmapped_descriptor_count += 1;
                return Ok(());
            }
        }
        Err("No texture present".into())
    }
    pub fn add_texture<'a>(
        &mut self,
        spec: &'a TextureSpec,
        command_queue: &mut B::CommandQueue,
    ) -> Result<Id<Tex>, Error> {
        let id = Id::new(self.textures.len() as u32);
        let tex = LoadedTexture::from_texture_spec(
            self.adapter.deref(),
            self.device.deref(),
            &mut self.command_pool,
            command_queue,
            spec,
        )
        .map(Some)?;
        self.textures.push(tex);
        Ok(id)
    }
    pub fn texture_count(&self) -> usize {
        self.textures.len()
    }
    pub fn tick(&mut self) {
        for i in (0..self.old_texture_expirations.len()).rev() {
            if self.old_texture_expirations[i] <= 0 {
                self.old_texture_expirations.remove(i);
                unsafe {
                    self.old_textures
                        .remove(i)
                        .manually_drop(self.device.deref());
                }
            } else {
                self.old_texture_expirations[i] -= 1;
            }
        }
        for i in (0..self.old_pool_expirations.len()).rev() {
            if self.old_pool_expirations[i] <= 0 {
                self.old_pool_expirations.remove(i);
                unsafe {
                    self.device
                        .destroy_descriptor_pool(self.old_pools.remove(i));
                }
            } else {
                self.old_pool_expirations[i] -= 1;
            }
        }
        for i in (0..self.old_buffers.len()).rev() {
            if self.old_buffer_expirations[i] <= 0 {
                unsafe {
                    self.old_buffers
                        .remove(i)
                        .manually_drop(self.device.deref());
                }
                self.old_buffer_expirations.remove(i);
                if self.old_buffer_descriptors.len() > i {
                    self.old_buffer_descriptors.remove(i);
                }
            } else {
                self.old_buffer_expirations[i] -= 1;
            }
        }
    }
    /*
        pub fn update_texture(rect: con_back::cc::rt::Rect<u32>, data: &[u8], id: con_back::Id<Tex>)-> Result<(), & 'static str> {

        }
    */
    pub fn dispose(&mut self) {
        unsafe {
            for descriptor_set_layout in self.descriptor_set_layouts.drain(..) {
                self.device
                    .destroy_descriptor_set_layout(descriptor_set_layout)
            }
            self.device
                .destroy_descriptor_pool(ManuallyDrop::take(&mut self.descriptor_pool));
            for mut tex in self.textures.iter() {
                if let Some(tex) = tex {
                    tex.manually_drop(self.device.deref());
                }
            }
        }
        unsafe {
            self.device
                .destroy_command_pool(ManuallyDrop::take(&mut self.command_pool));
            //          ManuallyDrop::take(&mut self.command_queue)
        }
    }
}
