use super::*;

pub struct LoadedTexture<B: Backend> {
    pub image: ManuallyDrop<B::Image>,
    pub format: Format,
    pub requirements: Requirements,
    pub memory: ManuallyDrop<B::Memory>,
    pub image_view: ManuallyDrop<B::ImageView>,
    pub sampler: ManuallyDrop<B::Sampler>,
}
// type Texture = LoadedImage<B: Backend, D: Device<Backend>>;

impl<B: Backend> LoadedTexture<B> {
    pub fn from_image<C: Capability + Supports<Transfer>>(
        adapter: &Adapter<B>, device: &Dev<B>, command_pool: &mut CommandPool<B, C>,
        command_queue: &mut CommandQueue<B, C>, img: image::RgbaImage,
    ) -> Result<Self, &'static str> {
        Self::from_buffer(adapter, device, command_pool, command_queue, &(*img), img.width(), img.height(), Format::Rgba8Srgb)
    }
    pub fn from_texture_spec<C: Capability + Supports<Transfer>>(
        adapter: &Adapter<B>, device: &Dev<B>, command_pool: &mut CommandPool<B, C>,
        command_queue: &mut CommandQueue<B, C>,
        spec: & TextureSpec,
    ) -> Result<Self, &'static str> {
        Self::from_buffer(adapter, device, command_pool, command_queue, spec.buffer, spec.width, spec.height, spec.format)
    }
    pub fn from_buffer<C: Capability + Supports<Transfer>>(
        adapter: &Adapter<B>, device: &Dev<B>, command_pool: &mut CommandPool<B, C>,
        command_queue: &mut CommandQueue<B, C>,
        buffer: &[u8],
        width: u32,
        height: u32,
        format: Format,
    ) -> Result<Self, &'static str> {
        let pixel_size = (format.surface_desc().bits / 8) as usize; // size_of::<image::Rgba<u8>>();
        let row_size = pixel_size * width as usize;
        let limits = adapter.physical_device.limits();
        let row_alignment_mask = limits.optimal_buffer_copy_pitch_alignment  as u32 - 1;
        let row_pitch = ((row_size as u32 + row_alignment_mask) & !row_alignment_mask) as usize;
        debug_assert!(row_pitch as usize >= row_size);
        // 1. make a staging buffer with enough memory for the image, and a
        //    transfer_src usage
        let required_bytes = row_pitch * height as usize;

        unsafe {
            // 0. First we compute some memory related values.

            let staging_bundle:  BufferBundle<B> =
                BufferBundle::new(&adapter, device, required_bytes, BufferUsage::TRANSFER_SRC)?;

            // 2. use mapping writer to put the image data into that buffer
            let mut writer = device
                .acquire_mapping_writer::<u8>(&staging_bundle.memory, 0..staging_bundle.requirements.size)
                .map_err(|_| "Couldn't acquire a mapping writer to the staging buffer!")?;
            for y in 0..height as usize {
                let row = &buffer[y * row_size..(y + 1) * row_size];
                let dest_base = y * row_pitch;
                writer[dest_base..dest_base + row.len()].copy_from_slice(row);
            }
            device
                .release_mapping_writer(writer)
                .map_err(|_| "Couldn't release the mapping writer to the staging buffer!")?;

            // 3. Make an image with transfer_dst and SAMPLED usage
            let mut the_image = device
                .create_image(
                    gfx_hal::image::Kind::D2(width, height, 1, 1),
                    1,
                    format,
                    gfx_hal::image::Tiling::Optimal,
                    gfx_hal::image::Usage::TRANSFER_DST | gfx_hal::image::Usage::SAMPLED,
                    gfx_hal::image::ViewCapabilities::empty(),
                )
                .map_err(|_| "Couldn't create the image!")?;

            // 4. allocate memory for the image and bind it
            let requirements = device.get_image_requirements(&the_image);
            let memory_type_id = adapter
                .physical_device
                .memory_properties()
                .memory_types
                .iter()
                .enumerate()
                .find(|&(id, memory_type)| {
                    // BIG NOTE: THIS IS DEVICE LOCAL NOT CPU VISIBLE
                    requirements.type_mask & (1 << id) != 0
                        && memory_type.properties.contains(Properties::DEVICE_LOCAL)
                })
                .map(|(id, _)| MemoryTypeId(id))
                .ok_or("Couldn't find a memory type to support the image!")?;
            let memory = device
                .allocate_memory(memory_type_id, requirements.size)
                .map_err(|_| "Couldn't allocate image memory!")?;
            device
                .bind_image_memory(&memory, 0, &mut the_image)
                .map_err(|_| "Couldn't bind the image memory!")?;

            // 5. create image view and sampler
            let image_view = device
                .create_image_view(
                    &the_image,
                    gfx_hal::image::ViewKind::D2,
                    format,
                    gfx_hal::format::Swizzle::NO,
                    SubresourceRange {
                        aspects: Aspects::COLOR,
                        levels: 0..1,
                        layers: 0..1,
                    },
                )
                .map_err(|_| "Couldn't create the image view!")?;
            let sampler = device
                .create_sampler(gfx_hal::image::SamplerInfo::new(
                    gfx_hal::image::Filter::Linear,
                    gfx_hal::image::WrapMode::Clamp,
                ))
                .map_err(|_| "Couldn't create the sampler!")?;


            // 6. create a command buffer
            let mut cmd_buffer = command_pool.acquire_command_buffer::<gfx_hal::command::OneShot>();
            cmd_buffer.begin();

            // 7. Use a pipeline barrier to transition the image from empty/undefined
            //    to TRANSFER_WRITE/TransferDstOptimal
            let image_barrier = gfx_hal::memory::Barrier::Image {
                states: (gfx_hal::image::Access::empty(), Layout::Undefined)
                    ..(
                    gfx_hal::image::Access::TRANSFER_WRITE,
                    Layout::TransferDstOptimal,
                ),
                target: &the_image,
                families: None,
                range: SubresourceRange {
                    aspects: Aspects::COLOR,
                    levels: 0..1,
                    layers: 0..1,
                },
            };
            cmd_buffer.pipeline_barrier(
                PipelineStage::TOP_OF_PIPE..PipelineStage::TRANSFER,
                gfx_hal::memory::Dependencies::empty(),
                &[image_barrier],
            );

            // 8. perform copy from staging buffer to image
            cmd_buffer.copy_buffer_to_image(
                &staging_bundle.buffer,
                &the_image,
                Layout::TransferDstOptimal,
                &[gfx_hal::command::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_width: (row_pitch / pixel_size) as u32,
                    buffer_height: height,
                    image_layers: gfx_hal::image::SubresourceLayers {
                        aspects: Aspects::COLOR,
                        level: 0,
                        layers: 0..1,
                    },
                    image_offset: gfx_hal::image::Offset { x: 0, y: 0, z: 0 },
                    image_extent: gfx_hal::image::Extent {
                        width: width,
                        height: height,
                        depth: 1,
                    },
                }],
            );

            // 9. use pipeline barrier to transition the image to SHADER_READ access/
            //    ShaderReadOnlyOptimal layout
            let image_barrier = gfx_hal::memory::Barrier::Image {
                states: (
                    gfx_hal::image::Access::TRANSFER_WRITE,
                    Layout::TransferDstOptimal,
                )
                    ..(
                    gfx_hal::image::Access::SHADER_READ,
                    Layout::ShaderReadOnlyOptimal,
                ),
                target: &the_image,
                families: None,
                range: SubresourceRange {
                    aspects: Aspects::COLOR,
                    levels: 0..1,
                    layers: 0..1,
                },
            };
            cmd_buffer.pipeline_barrier(
                PipelineStage::TRANSFER..PipelineStage::FRAGMENT_SHADER,
                gfx_hal::memory::Dependencies::empty(),
                &[image_barrier],
            );

            // 10. Submit the cmd buffer to queue and wait for it
            cmd_buffer.finish();
            let upload_fence = device
                .create_fence(false)
                .map_err(|_| "Couldn't create an upload fence!")?;
            command_queue.submit_nosemaphores(Some(&cmd_buffer), Some(&upload_fence));
            device
                .wait_for_fence(&upload_fence, core::u64::MAX)
                .map_err(|_| "Couldn't wait for the fence!")?;
            device.destroy_fence(upload_fence);

            // 11. Destroy the staging bundle and one shot buffer now that we're done
            staging_bundle.manually_drop(device);
            command_pool.free(Some(cmd_buffer));

            Ok(Self {
                format,
                image: ManuallyDrop::new(the_image),
                requirements,
                memory: ManuallyDrop::new(memory),
                image_view: ManuallyDrop::new(image_view),
                sampler: ManuallyDrop::new(sampler),
            })
        }
    }


    pub unsafe fn manually_drop(&self, device: &Dev<B>) {
        use core::ptr::read;
        device.destroy_sampler(ManuallyDrop::into_inner(read(&self.sampler)));
        device.destroy_image_view(ManuallyDrop::into_inner(read(&self.image_view)));
        device.destroy_image(ManuallyDrop::into_inner(read(&self.image)));
        device.free_memory(ManuallyDrop::into_inner(read(&self.memory)));
    }
}

#[derive(Debug, Copy, Clone)]
pub struct TextureSpec<'a> {
    pub format: Format,
    pub width: u32,
    pub height: u32,
    pub buffer: &'a [u8]
}
impl<'a> TextureSpec<'a> {
    pub fn save_as_image(&self, path: & std::path::Path) -> Result<(), String> {
        use image::ColorType::*;
        let color_format = match self.format {
            Format::R8Unorm => Gray(8),
            Format::Rgba8Srgb => RGBA(8),
            _ => Err("texture format with unknown image color mapping".to_string())?
        };
        image::save_buffer(path, self.buffer, self.width, self.height, color_format).map_err(|e| format!("{:?}", e))
    }
}

