use super::*;

pub mod texture;
pub mod descriptors;
pub mod buffer;

use std::collections::{hash_map::RandomState};
use std::hash::Hasher;

pub use buffer::BufferBundle;

pub use texture::{TextureSpec, LoadedTexture};
pub type Texture<B> = texture::LoadedTexture<B>;

pub enum Tex {}

pub type FontId = usize;
#[derive(Default)]
pub struct Id<T> {
    pub id : u32,
    phantom : PhantomData<* const T>,
}
impl<T> Id<T>{
    pub fn new(id: u32) -> Self {
        Self{ id, phantom: PhantomData }
    }
}
impl <T> Clone for Id<T> {
    fn clone(&self) -> Id<T> {
        Id {id: self.id, phantom: PhantomData}
    }
}

impl <T> Copy for Id<T> {}
impl <T> PartialEq for Id<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl <T> Eq for Id<T> {

}
impl<T> std::fmt::Debug for Id<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Id {{ id: {}, phantom }}", self.id)
    }
}
impl<T> std::hash::Hash for Id<T> {
    fn hash<H: Hasher>(&self, hasher: &mut H){
        self.id.hash(hasher);
    }
}

pub struct ResourceManager<B: Backend, Trans: Capability + Supports<Transfer>> {
    device: Arc<Dev<B>>,
    adapter: Arc<Adapter<B>>,
    command_pool: ManuallyDrop<CommandPool<B, Trans>>,
    //   command_queue: ManuallyDrop<CommandQueue<back::Backend, Trans>>,
    textures: Vec<Option<Texture<B>>>,
    old_textures: Vec<Texture<B>>,
    old_texture_expirations: Vec<i32>,
    old_pools: Vec<<B as Backend>::DescriptorPool>,
    old_pool_expirations: Vec<i32>,
    pub descriptor_set_layouts : Vec<<B as Backend>::DescriptorSetLayout>,
    descriptor_pool: ManuallyDrop<<B as Backend>::DescriptorPool>,
    pool_size: usize,
    descriptor_map: std::collections::HashMap<Id<Tex>, <B as Backend>::DescriptorSet, RandomState>,
}


impl<B: Backend, Trans: Capability + Supports<Transfer>> ResourceManager<B, Trans> {
    const DELETE_DELAY : i32 = 4;
    const TEXTURES_PER_BINDING : usize = 1;
    pub fn new(
        device: Arc<Dev<B>>,
        adapter: Arc<Adapter<B>>,
        command_pool: CommandPool<B, Trans>,
        //       command_queue: CommandQueue<back::Backend, Trans>,
    )->Result<Self, Error> {
        let pool_size = 32;
        let mut descriptor_set_layouts: Vec<<B as Backend>::DescriptorSetLayout> = Vec::new();

        unsafe {
            descriptor_set_layouts.push(device
                .create_descriptor_set_layout(
                    &[
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
                    ],
                    &[],
                ).map_err(|_| "Couldn't make a DescriptorSetLayout")?)
        }
        // 2. you create a descriptor pool, and when making that descriptor pool
        //    you specify how many sets you want to be able to allocate from the
        //    pool, as well as the maximum number of each kind of descriptor you
        //    want to be able to allocate from that pool, total, for all sets.
        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(
                    pool_size, // sets
                    &[
                        gfx_hal::pso::DescriptorRangeDesc {
                            ty: gfx_hal::pso::DescriptorType::SampledImage,
                            count: pool_size * Self::TEXTURES_PER_BINDING,
                        },
                        gfx_hal::pso::DescriptorRangeDesc {
                            ty: gfx_hal::pso::DescriptorType::Sampler,
                            count: pool_size * Self::TEXTURES_PER_BINDING,
                        },
                    ],
                    gfx_hal::pso::DescriptorPoolCreateFlags::empty(),
                )
                .map_err(|_| "Couldn't create a descriptor pool!")?
        };
        Ok(Self{device,
            adapter,
            command_pool: ManuallyDrop::new(command_pool),
            //           command_queue: ManuallyDrop::new(command_queue),
            textures: Vec::new(),
            descriptor_set_layouts,
            descriptor_pool: ManuallyDrop::new(descriptor_pool),
            old_textures: Vec::new(),
            old_texture_expirations: Vec::new(),
            old_pool_expirations : Vec::new(),
            old_pools: Vec::new(),
            descriptor_map: std::collections::HashMap::with_capacity(pool_size ),
            pool_size
        })
    }
    pub fn get_texture(&self, id: Id<Tex>) -> Option<&Texture<B>> {
        match self.textures.get(id.id as usize) {
            Some(Some(tex)) => Some(tex),
            _ => None
        }
    }
    pub fn get_descriptor_set(& mut self, id: Id<Tex>) -> Result<& <B as Backend>::DescriptorSet, Error> {
        if self.descriptor_map.contains_key(&id) {
            match self.descriptor_map.get(&id) {
                Some(desc) => Ok(&desc),
                None => Err("unknown texture id".into())
            }
        } else {
            self.set_descriptor_set(id)
        }
    }
    fn set_descriptor_set(& mut self, id: Id<Tex>) -> Result<& <B as Backend>::DescriptorSet, Error> {


        if self.descriptor_map.len() >= self.pool_size {
            self.refresh_pool()?;
            warn!("refreshed descriptor pool");

        }
        match self.textures.get(id.id as usize).and_then(|a| a.as_ref()) {
            None => {
                error!("Texture id {} requested but does not exist", id.id);
                return Err("unknown texture id".into());
            }
            Some(texture) => {
                unsafe {
                    match self.descriptor_pool.allocate_set(&self.descriptor_set_layouts[0]) {
                        Ok(new_set) => {
                            self.device.write_descriptor_sets(vec![
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
                                    descriptors: Some(gfx_hal::pso::Descriptor::Sampler(texture.sampler.deref())),
                                },
                            ]);
                            self.descriptor_map.insert(id, new_set);
                            Ok(self.descriptor_map.get(&id).unwrap())
                        },
                        Err(err) => { error!("{} for texture #{}", err, id.id); Err("couldn't allocate descriptor set".into())}
                    }
                }
            }
        }
    }
    fn refresh_pool(&mut self) -> Result<(), Error> {
        unsafe {
            let pool = self.device
                .create_descriptor_pool(
                    self.pool_size, // sets
                    &[
                        gfx_hal::pso::DescriptorRangeDesc {
                            ty: gfx_hal::pso::DescriptorType::SampledImage,
                            count: self.pool_size * Self::TEXTURES_PER_BINDING,
                        },
                        gfx_hal::pso::DescriptorRangeDesc {
                            ty: gfx_hal::pso::DescriptorType::Sampler,
                            count: self.pool_size * Self::TEXTURES_PER_BINDING,
                        },
                    ],
                    gfx_hal::pso::DescriptorPoolCreateFlags::empty(),
                ).map_err(|_| "Couldn't create a descriptor pool!")?;
            self.old_pools.push(ManuallyDrop::take(& mut self.descriptor_pool));
            self.old_texture_expirations.push(Self::DELETE_DELAY);
            self.descriptor_pool = ManuallyDrop::new(pool);
        }
        self.descriptor_map.clear();
        Ok(())
    }
    pub fn remove_texture(& mut self, id: Id<Tex>) -> Result<(), Error> {
        if let Some(mtex) = self.textures.get_mut(id.id as usize) {
            if let Some(tex)=mtex.take() {
                self.old_textures.push(tex);
                self.old_texture_expirations.push(Self::DELETE_DELAY);
                return Ok(());
            }
        }
        Err("No texture present".into())
    }
    pub fn replace_texture<'a>(
        & mut self,
        id: Id<Tex>,
        spec: &'a TextureSpec,
        command_queue: & mut CommandQueue<B, Trans>) -> Result<(), Error> {
        if let Some(mtex) = self.textures.get_mut(id.id as usize) {
            if mtex.is_some() {
                let mut other = LoadedTexture::from_texture_spec(
                    self.adapter.deref(),
                    self.device.deref(),
                    &mut self.command_pool,
                    command_queue, spec)
                    .ok();
                std::mem::swap(mtex, &mut other);
                if let Some(otex) = other {
                    self.old_textures.push(otex);
                    self.old_texture_expirations.push(Self::DELETE_DELAY);
                }
                self.descriptor_map.remove(&id);
                return Ok(());
            }
        }
        Err("No texture present".into())
    }
    pub fn add_texture<'a>(
        & mut self,
        spec: &'a TextureSpec,
        command_queue: & mut CommandQueue<B, Trans>) -> Result<Id<Tex>, Error> {
        let id = Id::new(self.textures.len() as u32);
        let tex = LoadedTexture::from_texture_spec(
            self.adapter.deref(),
            self.device.deref(),
            &mut self.command_pool,
            command_queue, spec)
            .map(Some)?;
        self.textures.push(tex);
        Ok(id)
    }
    pub fn texture_count(&self) -> usize {
        self.textures.len()
    }
    pub fn tick(&mut self){
        for i in (0 ..self.old_texture_expirations.len()).rev() {
            if self.old_texture_expirations[i] <= 0 {
                self.old_texture_expirations.remove(i);
                unsafe {
                    self.old_textures.remove(i).manually_drop(self.device.deref());
                }
            } else {
                self.old_texture_expirations[i] -= 1;
            }
        }
        for i in (0 ..self.old_pool_expirations.len()).rev() {
            if self.old_pool_expirations[i] <= 0 {
                self.old_pool_expirations.remove(i);
                unsafe {
                    self.device.destroy_descriptor_pool(self.old_pools.remove(i));
                }
            } else {
                self.old_pool_expirations[i] -= 1;
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
                self.device.destroy_descriptor_set_layout(descriptor_set_layout)
            }
            self.device.destroy_descriptor_pool(ManuallyDrop::take(&mut self.descriptor_pool));
            for mut tex in self.textures.iter() {
                if let Some(tex) = tex {
                    tex.manually_drop(self.device.deref());
                }
            }
        }
        unsafe {
            self.device.destroy_command_pool(ManuallyDrop::take(& mut self.command_pool).into_raw());
            //          ManuallyDrop::take(&mut self.command_queue)
        }
    }
}

