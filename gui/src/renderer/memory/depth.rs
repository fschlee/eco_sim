use crate::error::Error;
use gfx_hal::adapter::Adapter;
use gfx_hal::window::Extent2D;
use gfx_hal::{
    adapter::PhysicalDevice,
    device::Device,
    format::Swizzle,
    format::{Aspects, Format},
    image::{SubresourceRange, ViewKind},
    memory::{Properties, Requirements},
    Backend, MemoryTypeId,
};
use std::mem::ManuallyDrop;

pub struct DepthImage<B: Backend> {
    pub image: ManuallyDrop<B::Image>,
    pub requirements: Requirements,
    pub memory: ManuallyDrop<B::Memory>,
    pub image_view: ManuallyDrop<B::ImageView>,
}
impl<B: Backend> DepthImage<B> {
    pub fn new(
        adapter: &Adapter<B>,
        device: &<B as Backend>::Device,
        extent: Extent2D,
    ) -> Result<Self, Error> {
        unsafe {
            let mut image = device.create_image(
                gfx_hal::image::Kind::D2(extent.width, extent.height, 1, 1),
                1,
                Format::D32Sfloat,
                gfx_hal::image::Tiling::Optimal,
                gfx_hal::image::Usage::DEPTH_STENCIL_ATTACHMENT,
                gfx_hal::image::ViewCapabilities::empty(),
            )?;
            let requirements = device.get_image_requirements(&image);
            let (memory_type_idx, _) = adapter
                .physical_device
                .memory_properties()
                .memory_types
                .iter()
                .enumerate()
                .find(|&(idx, mt)| {
                    requirements.type_mask & (1 << idx) != 0
                        && mt.properties.contains(Properties::DEVICE_LOCAL)
                })
                .ok_or("Couldn't find appropriate memory")?;
            let memory =
                device.allocate_memory(MemoryTypeId(memory_type_idx), requirements.size)?;
            device.bind_image_memory(&memory, 0, &mut image)?;
            let image_view = device.create_image_view(
                &image,
                ViewKind::D2,
                Format::D32Sfloat,
                Swizzle::NO,
                SubresourceRange {
                    aspects: Aspects::DEPTH,
                    levels: 0..1,
                    layers: 0..1,
                },
            )?;
            Ok(Self {
                image: ManuallyDrop::new(image),
                requirements,
                memory: ManuallyDrop::new(memory),
                image_view: ManuallyDrop::new(image_view),
            })
        }
    }
    pub unsafe fn dispose(self, device: &<B as Backend>::Device) {
        let Self {
            image,
            requirements,
            memory,
            image_view,
        } = self;
        device.destroy_image(ManuallyDrop::into_inner(image));
        device.free_memory(ManuallyDrop::into_inner(memory));
        device.destroy_image_view(ManuallyDrop::into_inner(image_view));
    }
}
