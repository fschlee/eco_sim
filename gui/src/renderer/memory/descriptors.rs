use super::*;
use crate::renderer::memory::texture::LoadedTexture;

use gfx_hal::pso::DescriptorRangeDesc;

use std::ops::{DerefMut};
use crate::renderer::memory::buffer::BufferBundle;
use itertools::Itertools;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub enum Desc {}

pub struct DescriptorLayoutSpec<B: Backend> {
    pub textures: Vec<(usize, ShaderStageFlags)>,
    pub buffers: Vec<(gfx_hal::pso::DescriptorType, ShaderStageFlags)>,
    phantom_b: PhantomData<B>,
}
impl<B: Backend> DescriptorLayoutSpec<B> {
    pub fn new(textures: Vec<(usize, ShaderStageFlags)>, buffers: Vec<(gfx_hal::pso::DescriptorType, ShaderStageFlags)>) -> Self {
        Self{textures, buffers, phantom_b: PhantomData}
    }
    pub fn build_layout(&self, device: & Dev<B>) -> Result<<B as Backend>::DescriptorSetLayout, & 'static str> {
        let mut acc = Vec::new();
        let mut bind_count = 0;
        for (s,f) in &self.textures {
            acc.push(DescriptorSetLayoutBinding {
                binding: bind_count,
                ty: gfx_hal::pso::DescriptorType::SampledImage,
                count: *s,
                stage_flags: *f,
                immutable_samplers: false,
            });
            bind_count += 1;
            acc.push(DescriptorSetLayoutBinding {
                binding: bind_count,
                ty: gfx_hal::pso::DescriptorType::Sampler,
                count: 1,
                stage_flags: *f,
                immutable_samplers: false,
            });
            bind_count += 1;
        }
        for (dt, f) in &self.buffers {
            acc.push(DescriptorSetLayoutBinding {
                binding: bind_count,
                ty: *dt,
                count: 1,
                stage_flags: *f,
                immutable_samplers: false,
            });
            bind_count += 1;
        }
        unsafe {
            device.create_descriptor_set_layout(&acc, &[])
                .map_err(|e| {error!("{}", e); "out of memory"})
        }
    }
    pub fn write_to_set(
        &self, device: &Dev<B>,
        written_set: & <B as Backend>::DescriptorSet,
        textures: &[LoadedTexture<B>],
        texture_indices:  &[usize],
        buffers: &[BufferBundle<B>],
        buffer_indices: &[usize]) {
        let mut write_vec = Vec::new();
        for &i in texture_indices {
            write_vec.push(
                gfx_hal::pso::DescriptorSetWrite {
                    set: written_set,
                    binding: 2* i as u32,
                    array_offset: 0,
                    descriptors: Some(gfx_hal::pso::Descriptor::Image(
                        textures[i].image_view.deref(),
                        Layout::ShaderReadOnlyOptimal,
                    )),
                });
            write_vec.push(
                gfx_hal::pso::DescriptorSetWrite {
                    set: written_set,
                    binding: 1 + 2* i as u32,
                    array_offset: 0,
                    descriptors: Some(gfx_hal::pso::Descriptor::Sampler(textures[i].sampler.deref())),
                });
        }
        let base = self.textures.len() * 2;
        for &i in buffer_indices {
            let (ds, _) = self.buffers[i];
            write_vec.push(
                gfx_hal::pso::DescriptorSetWrite {
                    set: written_set,
                    binding: (base + i) as u32,
                    array_offset: 0,
                    descriptors: Some(gfx_hal::pso::Descriptor::Buffer(buffers[i].buffer.deref(), None..None)),
                });
        }
        unsafe {
            device.write_descriptor_sets(write_vec )
        }
    }
    pub fn create_pool_range(& self, pool_size: usize) -> Vec<DescriptorRangeDesc> {
        let mut out = Vec::new();
        out.push(DescriptorRangeDesc{
            ty: gfx_hal::pso::DescriptorType::SampledImage,
            count: pool_size * (self.textures.iter().fold(0, |acc, (sz, _)| acc + *sz) )  });
        out.push(DescriptorRangeDesc{
            ty: gfx_hal::pso::DescriptorType::Sampler,
            count: pool_size * (self.textures.len()) });
        let mut c = 0;
        for (dt, v) in self.buffers.clone().into_iter().into_group_map().iter() {
            out.push(DescriptorRangeDesc {
                ty: *dt,
                count: pool_size * v.len(),
            });
        }
        out
    }
}

pub struct DescriptorPoolManager<B: Backend> {
    pub descriptor_set_layouts : Vec<<B as Backend>::DescriptorSetLayout>,
    descriptor_pool: ManuallyDrop<<B as Backend>::DescriptorPool>,
    descriptor_sets: Vec<<B as Backend>::DescriptorSet>,
    pool_size: usize,
    pub spec: DescriptorLayoutSpec<B>
}

impl<B: Backend>  DescriptorPoolManager<B> {
    pub fn new(device: & Dev<B>, spec: DescriptorLayoutSpec<B>, pool_size: usize) -> Result<Self, & 'static str> {
        let layout = spec.build_layout(device)?;
        let pool = unsafe {
            device.create_descriptor_pool(pool_size, spec.create_pool_range(pool_size), gfx_hal::pso::DescriptorPoolCreateFlags::empty())
        }.map_err(|e| { error!("{}", e); "not enough memory for descriptor pool"})?;
        Ok(Self{
            descriptor_set_layouts:  vec![layout],
            descriptor_pool: ManuallyDrop::new(pool),
            pool_size,
            descriptor_sets: Vec::new(),
            spec
        })
    }
    pub fn register_set(
        & mut self,
        device: & Dev<B>,
        textures: &[LoadedTexture<B>],
        texture_indices:  &[usize],
        buffers: &[BufferBundle<B>],
        buffer_indices: &[usize]
    ) -> Result<Id<Desc>, & str> {
        if self.descriptor_sets.len() < self.pool_size {
            let mut new_set = unsafe {
                 self.descriptor_pool.allocate_set(&self.descriptor_set_layouts[0])
            }.map_err(|e| {error!("{}", e); "not enough memory"})?;
            self.spec.write_to_set(device, & mut new_set, textures, texture_indices, buffers, buffer_indices);
            let l = self.descriptor_sets.len();
            self.descriptor_sets.push(new_set);
            return Ok(Id::new(l as u32))
        }
        Err("too many descriptor set allocations")
    }
    pub fn reset(& mut self, device : & Dev<B>) -> Result<<B as Backend>::DescriptorPool, & str> {
        unsafe {
            let mut new = device.create_descriptor_pool(self.pool_size, self.spec.create_pool_range(self.pool_size), gfx_hal::pso::DescriptorPoolCreateFlags::empty())
                .map_err(|e| { error!("{}", e); "not enough memory for descriptor pool"})?;
            std::mem::swap(& mut new, self.descriptor_pool.deref_mut());
            self.descriptor_sets.clear();
            Ok(new)
        }
    }
    pub fn disperse(& mut self, device : & Dev<B>) {
        unsafe {
            for l in self.descriptor_set_layouts.drain(..) {
                device.destroy_descriptor_set_layout(l);
            }
            device.destroy_descriptor_pool(ManuallyDrop::take(& mut self.descriptor_pool));
        }
    }
    pub fn get_desc(& self, id: Id<Desc>) -> Result<& <B as Backend>::DescriptorSet, & str>{
        self.descriptor_sets.get(id.id as usize).ok_or("descriptor set not registered")
    }
    pub fn get_layouts(& self) -> & Vec<<B as Backend>::DescriptorSetLayout> {
        &self.descriptor_set_layouts
    }
}
