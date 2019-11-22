pub mod con_back;
pub mod ui_pipeline;
pub mod pipeline_2d;
pub mod memory;
pub mod init;

use memory::*;
use std::{sync::{Arc, Mutex}, ops::DerefMut};
use nalgebra::{Vector2, Vector3};

use log::{error, info, warn};

use arrayvec::ArrayVec;
use core::{
    marker::PhantomData,
    mem::{size_of, ManuallyDrop},
    ops::Deref,
};
use gfx_hal::{
    adapter::{Adapter, MemoryTypeId, PhysicalDevice},
    buffer::{IndexBufferView, Usage as BufferUsage},
    command::{ClearColor, ClearValue, CommandBuffer, MultiShot, OneShot, Primary, Secondary, Submittable, RenderPassInlineEncoder},
    device::Device,
    format::{Aspects, ChannelType, Format, Swizzle},
    image::{Extent, Layout, SubresourceRange, Usage, ViewKind},
    memory::{Properties, Requirements},
    pass::{Attachment, AttachmentLoadOp, AttachmentOps, AttachmentStoreOp, Subpass, SubpassDesc},
    pool::{CommandPool, CommandPoolCreateFlags},
    pso::{
        AttributeDesc, BakedStates, BasePipeline, BlendDesc, BlendOp, BlendState, ColorBlendDesc,
        ColorMask, DepthStencilDesc, DepthTest, DescriptorSetLayoutBinding, ElemOffset, ElemStride,
        Element, EntryPoint, Face, Factor, FrontFace, GraphicsPipelineDesc, GraphicsShaderSet,
        InputAssemblerDesc, LogicOp, PipelineCreationFlags, PipelineStage, PolygonMode, Rasterizer,
        Rect, ShaderStageFlags, Specialization, StencilTest, VertexBufferDesc, Viewport,
    },
    queue::{
        capability::{Capability, Supports, Transfer},
        family::QueueGroup,
        CommandQueue, Submission,
    },
    window::{Extent2D, PresentMode, Swapchain, SwapchainConfig},
    Backend, DescriptorPool, Gpu, Graphics, IndexType, Instance, Primitive, QueueFamily, Surface,
};

use winit::{
    dpi::LogicalSize, Window
};
use self::memory::{BufferBundle, ResourceManager, TextureSpec, Texture, Id};

use init::InstSurface;
use gfx_hal::command::{CommandBufferInheritanceInfo, SecondaryCommandBuffer, SubpassCommandBuffer};
use gfx_hal::query::Type::PipelineStatistics;
use crate::renderer::memory::descriptors::DescriptorPoolManager;
use crate::error::{Error, LogError};



type Back<IS> = <<IS as InstSurface>::Instance as Instance>::Backend;
type Dev<B> = <B as Backend>::Device;

const CLEAR_COLOR : [ClearValue; 1] =  [ClearValue::Color(ClearColor::Uint([0x2E, 0x34, 0x36, 0]))];

#[cfg(feature = "reload_shaders")]
const UI_SHADERS : [ShaderSpec; 2] = [
        ShaderSpec{ kind: shaderc::ShaderKind::Vertex, source_path:"resources/ui.vert", source: None},
        ShaderSpec{ kind: shaderc::ShaderKind::Fragment, source_path: "resources/ui.frag", source: None},
    ];
#[cfg(not(feature = "reload_shaders"))]
const UI_SHADERS : [ShaderSpec; 2] = [
    ShaderSpec{ kind: shaderc::ShaderKind::Vertex, source_path:"resources/ui.vert", source: Some(include_str!("../../resources/ui.vert")) },
    ShaderSpec{ kind: shaderc::ShaderKind::Fragment, source_path: "resources/ui.frag", source: Some(include_str!("../../resources/ui.frag"))},
];
#[cfg(feature = "reload_shaders")]
const SHADERS_2D : [ShaderSpec; 2] = [
    ShaderSpec{ kind: shaderc::ShaderKind::Vertex, source_path:"resources/2d.vert", source: None},
    ShaderSpec{ kind: shaderc::ShaderKind::Fragment, source_path: "resources/2d.frag", source: None},
];
#[cfg(not(feature = "reload_shaders"))]
const SHADERS_2D : [ShaderSpec; 2] = [
    ShaderSpec{ kind: shaderc::ShaderKind::Vertex, source_path:"resources/2d.vert", source: Some(include_str!("../../resources/2d.vert")) },
    ShaderSpec{ kind: shaderc::ShaderKind::Fragment, source_path: "resources/2d.frag", source: Some(include_str!("../../resources/2d.frag"))},
];

pub struct Renderer<'a, IS : InstSurface> {
//    type D = <IS::Backend as Backend>::Device,
//    type B = IS::Backend,
    pub hal_state: HalState<Back<IS>>,
    pub texture_manager: ResourceManager<Back<IS>, Graphics>,
    pub ui_pipeline: ui_pipeline::UiPipeline<Back<IS>>,
    pub pipeline_2d: pipeline_2d::Pipeline2D<Back<IS>>,
    ui_vbuff: Vec<BufferBundle<Back<IS>>>,
    old_buffers: Vec<BufferBundle<Back<IS>>>,
    old_buffer_expirations: Vec<i32>,
    pub window: &'a Window,
    //  : <back::Backend as Backend>::Surface,
    inst_surface: ManuallyDrop<IS>,
    mem_atom: usize,
    pub adapter: Arc<Adapter<Back<IS>>>,
    pub device: ManuallyDrop<Arc<Dev<Back<IS>>>>,
    pub queue_group: ManuallyDrop<QueueGroup<Back<IS>, Graphics>>,
    snd_command_pools: Vec<CommandPool<Back<IS>, Graphics>>,
   // snd_command_buffers: Vec<CommandBuffer<Back<IS>, Graphics, MultiShot, Secondary>>,
}

impl<'a, IS : InstSurface>  Renderer<'a, IS>
{
    const DELETE_DELAY : i32 = 4;
    pub fn new(window: & 'a Window) -> Self {

        let (mut inst_surface, adapter, mut _device, queue_group)
            = init::init_device::<IS>(window).expect("failed to initiate 3D device");
        //debug_assert!(queue_group.queues.len() == 2);
        let device = Arc::new(_device);
        let adapter = Arc::new(adapter);
        let mem_atom = adapter.physical_device.limits().non_coherent_atom_size;

        let primary_pool = unsafe {
            device
                .create_command_pool_typed(&queue_group, CommandPoolCreateFlags::RESET_INDIVIDUAL)
                .expect("Could not create the raw draw command pool!")
        };

        let hal_state = HalState::init(window, inst_surface.get_mut_surface(),  &adapter,  device.clone(), primary_pool).expect("failed to set up device for rendering");
        let mut snd_command_pools = Vec::new();
        for _ in 0 ..hal_state.frames_in_flight {
            unsafe {
                snd_command_pools.push(device
                    .create_command_pool_typed(&queue_group, CommandPoolCreateFlags::RESET_INDIVIDUAL)
                    .expect("Could not create the raw draw command pool!"))
            };
        }
        let (vert_art, frag_art) = Self::compile_ui_shaders().expect("couldn't compile shader");

        let transfer_pool = unsafe {
            device
                .create_command_pool_typed(&queue_group, CommandPoolCreateFlags::TRANSIENT)
                .expect("Could not create the raw transfer command pool!")
        };
        let texture_manager = ResourceManager::new(device.clone(), adapter.clone(), transfer_pool).expect("failed to create texture manager");
        let ui_pipeline = ui_pipeline::UiPipeline::create(device.clone(), hal_state.render_area, hal_state.render_pass.deref(), vert_art, frag_art, &texture_manager.descriptor_set_layouts).expect("failed to create pipeline");
        let mut art_2d = complile_shaders(&SHADERS_2D).expect("couldn't compile shader");
        let frag_2d = art_2d.remove(1);
        let vert_2d = art_2d.remove(0);
        let pipeline_2d = pipeline_2d::Pipeline2D::create(device.clone(), hal_state.render_area, hal_state.render_pass.deref(), vert_2d, frag_2d, &texture_manager.descriptor_set_layouts).expect("failed to create pipeline");
        let queue_group = ManuallyDrop::new(queue_group);
        Self{ hal_state,
            ui_pipeline,
            pipeline_2d,
            ui_vbuff: Vec::new(),
            old_buffers: Vec::new(),
            old_buffer_expirations: Vec::new(),
            window,
            inst_surface: ManuallyDrop::new(inst_surface),
            adapter,
            device: ManuallyDrop::new(device),
            queue_group,
            texture_manager,
            mem_atom,
            snd_command_pools, // : ManuallyDrop::new(snd_command_pool),
           // snd_command_buffers,
        }
    }
    /*
    fn draw_queue(& self) ->  & mut CommandQueue<back::Backend, Graphics> {
        & mut self.queue_group.queues[0]
    }
    fn transfer_queue(& self) ->  & mut CommandQueue<back::Backend, Graphics> {
        let l = self.queue_group.queues.len();
        & mut self.queue_group.queues[l -1]
    }*/
    pub fn tick(&mut self, cmds: &Vec<con_back::Command>, ui_updates: Vec<crate::ui::UIUpdate>, render_data: &crate::simulation::RenderData) -> Result<(), Error>{
        use crate::ui::UIUpdate::*;
        let mut restart = false;
        let mut refresh = false;
        for update in ui_updates {
            match update {
                Resized{..} => {
                    info!("restarting render");
                    restart = true;

                },
                Refresh => {
                    refresh = true;
                }
                _ => ()
            }
        }
        if restart {
            self.restart()?;
        } else if refresh {
            self.reload_shaders()?;
        }
        self.dec_old();
        self.texture_manager.tick();

        let render_area = self.hal_state.render_area.clone();
        let sim_idx = self.temp_buffer(&render_data.indices, BufferUsage::INDEX)?;
        let sim_vtx = self.temp_buffer(&render_data.vertices, BufferUsage::VERTEX)?;


        let sim_idx_buff = & self.old_buffers[sim_idx];

        let sim_vtx_buff = & self.old_buffers[sim_vtx..=sim_vtx];

        let pipeline = &mut self.ui_pipeline;
        let pipeline_2d = &mut self.pipeline_2d;
        let mm = & mut self.texture_manager;
        let vbuffs = &self.ui_vbuff;
        let  draw_queue = & mut self.queue_group.queues[0];
        self.hal_state.with_inline_encoder( draw_queue, |enc| {

            {
                pipeline_2d.execute(enc, mm, sim_vtx_buff, sim_idx_buff, render_area, &render_data.commands);
            }
            {
                pipeline.execute(enc, mm, vbuffs, render_area, cmds)
            }

        })
    }
    fn compile_ui_shaders() -> Result<(shaderc::CompilationArtifact, shaderc::CompilationArtifact), Error> {

        let mut v = complile_shaders(&UI_SHADERS)?;
        if v.len() == 2 {
            let frag = v.remove(1);
            let vert = v.remove(0);
            Ok((vert, frag))
        } else {
            Err("unexpected number of compilation artifacts".into())
        }
    }

    fn reload_shaders(&mut self) -> Result<(), Error> {
        #[cfg(feature = "reload_shaders")]
        {
            println!("reloading shaders");
            {
                let  draw_queue = & mut self.queue_group.queues[0];
                self.hal_state.draw_clear_frame( draw_queue, [0.8, 0.8, 0.8, 1.0]);
                self.hal_state.draw_clear_frame( draw_queue, [0.8, 0.8, 0.8, 1.0]);
                self.hal_state.draw_clear_frame( draw_queue, [0.8, 0.8, 0.8, 1.0]);
            }

            let (vert_art, frag_art) = Self::compile_ui_shaders()?;
            self.ui_pipeline = ui_pipeline::UiPipeline::create(self.device.deref().clone(), self.hal_state.render_area, self.hal_state.render_pass.deref(), vert_art, frag_art, &self.texture_manager.descriptor_set_layouts)?;
            let mut art_2d = complile_shaders(&SHADERS_2D)?;
            let frag_2d = art_2d.remove(1);
            let vert_2d = art_2d.remove(0);
            self.pipeline_2d = pipeline_2d::Pipeline2D::create(self.device.deref().clone(), self.hal_state.render_area, self.hal_state.render_pass.deref(),vert_2d, frag_2d, &self.texture_manager.descriptor_set_layouts)?;
        }
        #[cfg(not(feature= "reload_shaders"))]
        {
            println!("not using feature reload_shaders");
        }
        Ok(())
    }
    fn restart(& mut self)-> Result<(), Error> {
        let pool = self.hal_state.dispose();
        info!("disposing old");
        self.hal_state = HalState::init(& self.window, self.inst_surface.get_mut_surface(),  &mut self.adapter,  self.device.deref().clone(),   pool)?;
        info!("disposed");
        let (vert_art, frag_art) = Self::compile_ui_shaders()?;
        let mut art_2d = complile_shaders(&SHADERS_2D)?;
        let frag_2d = art_2d.remove(1);
        let vert_2d = art_2d.remove(0);
        self.ui_pipeline = ui_pipeline::UiPipeline::create(self.device.deref().clone(), self.hal_state.render_area, self.hal_state.render_pass.deref(), vert_art, frag_art, &self.texture_manager.descriptor_set_layouts)?;
        self.pipeline_2d = pipeline_2d::Pipeline2D::create(self.device.deref().clone(), self.hal_state.render_area, self.hal_state.render_pass.deref(), vert_2d, frag_2d, &self.texture_manager.descriptor_set_layouts)?;
        Ok(())
    }
    pub fn set_ui_buffer(& mut self, vtx: Vec<con_back::UiVertex>) -> Result<(), Error>{
        let proper_size = (vtx.len() * size_of::<f32>() * 6);
        let padded_size = ((proper_size + self.mem_atom - 1) / self.mem_atom)  * self.mem_atom;
        let size = (vtx.len() * size_of::<f32>() * 6) as u64;
        if self.ui_vbuff.len() < 1 || self.ui_vbuff[0].requirements.size <= padded_size as u64 {
            let device = self.device.deref().deref();
            for b in self.ui_vbuff.drain(..){
                self.old_buffers.push(b);
                self.old_buffer_expirations.push(Self::DELETE_DELAY);
            } // b.manually_drop(device));
            let vb = BufferBundle::new(& self.adapter, self.device.deref().deref(), padded_size, BufferUsage::VERTEX)?;
            self.ui_vbuff.insert(0, vb);
        }
        unsafe {
            let mut vtx_target = self.device.acquire_mapping_writer(& self.ui_vbuff[0].memory, 0..(padded_size as u64))
                .map_err(|_| "Failed to acquire a vertex buffer mapping writer!")?;
            vtx_target[0..(vtx.len())].copy_from_slice(&vtx[0.. vtx.len()]);
            self.device.release_mapping_writer(vtx_target)
                .map_err(|_| "Couldn't release the VB mapping writer!")?;
        }
        Ok(())
    }
    fn dec_old(&mut self){
        for i in (0..self.old_buffers.len()).rev() {
            if self.old_buffer_expirations[i] <= 0 {
                unsafe {
                    self.old_buffers.remove(i).manually_drop(self.device.deref());
                    self.old_buffer_expirations.remove(i);
                }
            } else {
                self.old_buffer_expirations[i] -= 1;
            }
        }
    }
    pub fn replace_texture<'b>(& mut self, id: Id<Tex>, spec: &'b TextureSpec) -> Result<(), Error> {
        let l = self.queue_group.queues.len();

        let  transfer_queue = & mut self.queue_group.queues[l -1];
         self.texture_manager.replace_texture(id, spec, transfer_queue)
    }
    pub fn add_texture<'b>(& mut self, spec: &'b TextureSpec) -> Result<Id<Tex>, Error> {
        let l = self.queue_group.queues.len();
        let  transfer_queue = & mut self.queue_group.queues[l -1];
        self.texture_manager.add_texture(spec, transfer_queue)
    }
    fn padded_size(& self, proper_size: usize) -> usize {
        ((proper_size + self.mem_atom - 1) / self.mem_atom)  * self.mem_atom
    }
    fn temp_buffer<T: Sized + Copy>(& mut self, slice: & [T], usage: BufferUsage) -> Result<usize, Error> {
        let idx = self.old_buffers.len();
        {
            let size = size_of::<T>() * slice.len();
            let pad = self.padded_size(size);
            let buff = BufferBundle::new(self.adapter.deref(), self.device.deref(), pad, usage)?;
            unsafe {
                let mut target = self.device.acquire_mapping_writer(&buff.memory, 0..(pad as u64)).map_err(|e| {
                    error!("{}", e);
                    "can't acquire mapping writer"
                })?;
                target[0..slice.len()].copy_from_slice(slice);
                self.device.release_mapping_writer(target)
                    .map_err(|_| "Couldn't release the mapping writer!")?;
            }
            self.old_buffers.push(buff);
            self.old_buffer_expirations.push(4);
        }
       Ok(idx)
    }
}

impl<'a, IS: InstSurface> Drop for Renderer<'a, IS>{
    fn drop(&mut self) {
        while self.old_buffers.len() > 0 {
            let  draw_queue = & mut self.queue_group.queues[0];
            let _ = self.hal_state.draw_clear_frame( draw_queue, [0.8, 0.8, 0.8, 1.0]);
            self.dec_old();
            self.texture_manager.tick();
        }
        self.texture_manager.dispose();
        let pool = self.hal_state.dispose();
   //     self.queue_group.queues.push(draw_queue);
   //     self.queue_group.queues.push(transfer_queue);
       //  self.snd_command_buffers.drain(..);
        unsafe {
            self.device.destroy_command_pool(pool.into_raw());
            for pool in self.snd_command_pools.drain(..) {
                self.device.destroy_command_pool(pool.into_raw());
            }
       //     self.device.destroy_command_pool(ManuallyDrop::take(& mut self.snd_command_pool).into_raw());
            ManuallyDrop::drop(&mut self.queue_group);
            for b in self.ui_vbuff.drain(..){
                b.manually_drop(self.device.deref());
            }
            {
                let arc = ManuallyDrop::take(&mut self.device);
                match Arc::try_unwrap(arc){
                    Err(_arc) => warn!("device still exists"),
                    Ok(_) => ()
                }
            }

            ManuallyDrop::drop(& mut self.inst_surface);
        }

    }
}


pub struct HalState< B: Backend>{
    device: Arc<Dev<B>>,
    current_frame: usize,
    next_frame: usize,
    frames_in_flight: usize,
    in_flight_fences: Vec<<B as Backend>::Fence>,
    render_finished_semaphores: Vec<<B as Backend>::Semaphore>,
    image_available_semaphores: Vec<<B as Backend>::Semaphore>,
 //   command_queue: ManuallyDrop<CommandQueue<back::Backend, Graphics>>,
    command_buffers: Vec<CommandBuffer<B, Graphics, MultiShot, Primary>>,
    command_pool: ManuallyDrop<CommandPool<B, Graphics>>,
    framebuffers: Vec<<B as Backend>::Framebuffer>,
    image_views: Vec<(<B as Backend>::ImageView)>,
    current_image: usize,
    render_pass: ManuallyDrop<<B as Backend>::RenderPass>,
    render_area: Rect,
    swapchain: ManuallyDrop<<B as Backend>::Swapchain>,
}



impl<B: Backend> HalState<B> {
    pub fn init(
        window: &Window,
        surface:&mut <B as Backend>::Surface,
        adapter: &Adapter<B>,
        device: Arc<Dev<B>>,
        mut command_pool : CommandPool<B, Graphics>,
        // queue_group: Arc<QueueGroup<back::Backend, Graphics>>
    ) -> Result<Self, Error> {
        // Create A Swapchain, this is extra long
        let (swapchain, extent, images, format, frames_in_flight) = {
            let (caps, preferred_formats, present_modes) =
                surface.compatibility(&adapter.physical_device);
            let format = match preferred_formats {
                None => Format::Rgba8Srgb,
                Some(formats) => match formats
                    .iter()
                    .find(|format| format.base_format().1 == ChannelType::Srgb)
                    .cloned()
                    {
                        Some(srgb_format) => srgb_format,
                        None => formats.get(0).cloned().ok_or("Preferred format list was empty!")?,
                    },
            };
            let default_extent = {
                let window_client_area = window
                    .get_inner_size()
                    .ok_or("Window doesn't exist!")?
                    .to_physical(window.get_hidpi_factor());
                Extent2D {
                    width: window_client_area.width as u32,
                    height: window_client_area.height as u32,
                }
            };
            let mut swapchain_config = SwapchainConfig::from_caps(&caps, format, default_extent);
            let image_count = if swapchain_config.present_mode == PresentMode::Mailbox {
                (caps.image_count.end() - 1).min(3)
            } else {
                (caps.image_count.end() - 1).min(2)
            };
            swapchain_config.image_count = image_count;
            let extent = swapchain_config.extent;
            let (swapchain, images) = unsafe {
                device.create_swapchain(surface, swapchain_config, None)
                    .map_err(|_| "Failed to create the swapchain!")?
            };
            (swapchain, extent, images, format, image_count as usize)
        };
        // println!("{}:{}", extent.width, extent.height);
        // Create Our Sync Primitives
        let (image_available_semaphores, render_finished_semaphores, in_flight_fences) = {
            let mut image_available_semaphores: Vec<<B as Backend>::Semaphore> = vec![];
            let mut render_finished_semaphores: Vec<<B as Backend>::Semaphore> = vec![];
            let mut in_flight_fences: Vec<<B as Backend>::Fence> = vec![];
            for _ in 0..frames_in_flight {
                in_flight_fences.push(
                    device
                        .create_fence(true)
                        .map_err(|_| "Could not create a fence!")?,
                );
                image_available_semaphores.push(
                    device
                        .create_semaphore()
                        .map_err(|_| "Could not create a semaphore!")?,
                );
                render_finished_semaphores.push(
                    device
                        .create_semaphore()
                        .map_err(|_| "Could not create a semaphore!")?,
                );
            }
            (
                image_available_semaphores,
                render_finished_semaphores,
                in_flight_fences,
            )
        };

        // Define A RenderPass
        let render_pass = {
            let color_attachment = Attachment {
                format: Some(format),
                samples: 1,
                ops: AttachmentOps {
                    load: AttachmentLoadOp::Clear,
                    store: AttachmentStoreOp::Store,
                },
                stencil_ops: AttachmentOps::DONT_CARE,
                layouts: Layout::Undefined..Layout::Present,
            };
            let subpass = SubpassDesc {
                colors: &[(0, Layout::ColorAttachmentOptimal)],
                depth_stencil: None,
                inputs: &[],
                resolves: &[],
                preserves: &[],
            };
            unsafe {
                device
                    .create_render_pass(&[color_attachment], &[subpass], &[])
                    .map_err(|_| "Couldn't create a render pass!")?
            }
        };

        // Create The ImageViews
        let image_views: Vec<_> = images
                .into_iter()
                .map(|image| unsafe {
                    device
                        .create_image_view(
                            &image,
                            ViewKind::D2,
                            format,
                            Swizzle::NO,
                            SubresourceRange {
                                aspects: Aspects::COLOR,
                                levels: 0..1,
                                layers: 0..1,
                            },
                        )
                        .map_err(|_| "Couldn't create the image_view for the image!".into())
                })
                .collect::<Result<Vec<_>, &str>>()?;

        // Create Our FrameBuffers
        let framebuffers: Vec<<B as Backend>::Framebuffer> = {
            image_views
                .iter()
                .map(|image_view| unsafe {
                    device
                        .create_framebuffer(
                            &render_pass,
                            vec![image_view],
                            Extent {
                                width: extent.width as u32,
                                height: extent.height as u32,
                                depth: 1,
                            },
                        )
                        .map_err(|_| "Failed to create a framebuffer!".into())
                })
                .collect::<Result<Vec<_>, &str>>()?
        };

        // Create Our CommandBuffers
        let command_buffers: Vec<_> = framebuffers
            .iter()
            .map(|_| command_pool.acquire_command_buffer())
            .collect();

        Ok(Self {
            device,
 //           command_queue: ManuallyDrop::new(command_queue),
            swapchain: ManuallyDrop::new(swapchain),
            render_area: extent.to_extent().rect(),
            render_pass: ManuallyDrop::new(render_pass),
            image_views,
            framebuffers,
            command_pool: ManuallyDrop::new(command_pool),
            command_buffers,
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            frames_in_flight,
            current_frame: 0,
            next_frame: 0,
            current_image: 0,
        })
    }


    /// Draw a frame that's just cleared to the color specified.
    pub fn draw_clear_frame(
            &mut self,
            command_queue: & mut CommandQueue<B, Graphics>,
            color: [f32; 4]) -> Result<(), Error> {
        // SETUP FOR THIS FRAME
        let image_available = &self.image_available_semaphores[self.current_frame];
        let render_finished = &self.render_finished_semaphores[self.current_frame];
        // Advance the frame _before_ we start using the `?` operator
        self.current_frame = (self.current_frame + 1) % self.frames_in_flight;

        let (i_u32, i_usize) = unsafe {
            let (image_index, sbopt) = self
                .swapchain
                .acquire_image(core::u64::MAX, Some(image_available), None)
                .map_err(|_| "Couldn't acquire an image from the swapchain!")?;
            (image_index, image_index as usize)
        };

        let flight_fence = &self.in_flight_fences[i_usize];
        unsafe {
            self.device
                .wait_for_fence(flight_fence, core::u64::MAX)
                .map_err(|_| "Failed to wait on the fence!")?;
            self.device
                .reset_fence(flight_fence)
                .map_err(|_| "Couldn't reset the fence!")?;
        }

        // RECORD COMMANDS
        unsafe {
            let buffer = &mut self.command_buffers[i_usize];
            let clear_values = [ClearValue::Color(ClearColor::Sfloat(color))];
            buffer.begin(false);
            buffer.begin_render_pass_inline(
                &self.render_pass,
                &self.framebuffers[i_usize],
                self.render_area,
                clear_values.iter(),
            );

            buffer.finish();
        }

        // SUBMISSION AND PRESENT
        let command_buffers = &self.command_buffers[i_usize..=i_usize];
        let wait_semaphores: ArrayVec<[_; 1]> =
            [(image_available, PipelineStage::COLOR_ATTACHMENT_OUTPUT)].into();
        let signal_semaphores: ArrayVec<[_; 1]> = [render_finished].into();
        // yes, you have to write it twice like this. yes, it's silly.
        let present_wait_semaphores: ArrayVec<[_; 1]> = [render_finished].into();
        let submission = Submission {
            command_buffers,
            wait_semaphores,
            signal_semaphores,
        };

        unsafe {
            command_queue.submit(submission, Some(flight_fence));
            self
                .swapchain
                .present(command_queue, i_u32, present_wait_semaphores)
                .map_err(|_| "Failed to present into the swapchain!".into()).map(|_| ())
        }
    }

    pub fn with_inline_encoder<F>(
            &mut self,
            command_queue: & mut CommandQueue<B, Graphics>,
            draw: F) -> Result<(), Error>
        where F : FnOnce(&mut RenderPassInlineEncoder<B>) {
        // SETUP FOR THIS FRAME
        let image_available = &self.image_available_semaphores[self.current_frame];
        let render_finished = &self.render_finished_semaphores[self.current_frame];
        // Advance the frame _before_ we start using the `?` operator
        self.current_frame = (self.current_frame + 1) % self.frames_in_flight;

        let (i_u32, i_usize) = unsafe {
            let (image_index, sbopt) = self
                .swapchain
                .acquire_image(core::u64::MAX, Some(image_available), None)
                .map_err(|_| "Couldn't acquire an image from the swapchain!")?;
            (image_index, image_index as usize)
        };

        let flight_fence = &self.in_flight_fences[i_usize];
        unsafe {
            self.device
                .wait_for_fence(flight_fence, core::u64::MAX)
                .map_err(|_| "Failed to wait on the fence!")?;
            self.device
                .reset_fence(flight_fence)
                .map_err(|_| "Couldn't reset the fence!")?;
        }


        // RECORD COMMANDS
        unsafe {
            let buffer = &mut self.command_buffers[i_usize];
            buffer.begin(false);
            {
                let mut encoder = buffer.begin_render_pass_inline(
                    &self.render_pass,
                    &self.framebuffers[i_usize],
                    self.render_area,
                    CLEAR_COLOR.iter(),
                );

                {
                    draw(& mut encoder);
                }

                // self.pipeline.execute(&mut encoder, memory_manager, &vertex_buffers, index_buffer_view, self.render_area, time_f32, cmds);
            }
            buffer.finish();
        }


        // SUBMISSION AND PRESENT
        let command_buffers = &self.command_buffers[i_usize..=i_usize];
        let wait_semaphores: ArrayVec<[_; 1]> =
            [(image_available, PipelineStage::COLOR_ATTACHMENT_OUTPUT)].into();
        let signal_semaphores: ArrayVec<[_; 1]> = [render_finished].into();
        // yes, you have to write it twice like this. yes, it's silly.
        let present_wait_semaphores: ArrayVec<[_; 1]> = [render_finished].into();
        let submission = Submission {
            command_buffers,
            wait_semaphores,
            signal_semaphores,
        };
        unsafe {
            // let the_command_queue = queues.get_unchecked_mut(0);
            command_queue.submit(submission, Some(flight_fence));
            self
                .swapchain
                .present(command_queue, i_u32, present_wait_semaphores)
                .map_err(|_| "Failed to present into the swapchain!")?;
        };
        Ok(())
    }
    pub fn prepare_command_buffers(& mut self)-> Result<(& mut CommandBuffer<B, Graphics, MultiShot, Primary>, gfx_hal::pass::Subpass<B>, & <B as Backend>::Framebuffer, usize) , Error> {
        self.next_frame = (self.current_frame + 1) % self.frames_in_flight;
        let image_available = &self.image_available_semaphores[self.current_frame];
        let (i_u32, i_usize) = unsafe {
            let (image_index, sbopt) = self
                .swapchain
                .acquire_image(core::u64::MAX, Some(image_available), None)
                .map_err(|_| "Couldn't acquire an image from the swapchain!")?;
            (image_index, image_index as usize)
        };
        self.current_image = i_usize;
        let flight_fence = &self.in_flight_fences[i_usize];
        unsafe {
            self.device
                .wait_for_fence(flight_fence, core::u64::MAX)
                .map_err(|_| "Failed to wait on the fence!")?;
            self.device
                .reset_fence(flight_fence)
                .map_err(|_| "Couldn't reset the fence!")?;
        }
        let sub_pass = gfx_hal::pass::Subpass{index: 0, main_pass : self.render_pass.deref()};

        Ok((& mut self.command_buffers[i_usize], sub_pass, &self.framebuffers[i_usize], i_usize))
    }
    pub fn submit<'a>(
        &mut self,
        command_queue: & mut CommandQueue<B, Graphics>,
    ) -> Result<(), Error>
    {
        if self.current_frame == self.next_frame {
            return Err("frame not set up, nothing to submit to".into())
        }
        let image_available = &self.image_available_semaphores[self.current_frame];
        let render_finished = &self.render_finished_semaphores[self.current_frame];
        self.current_frame = self.next_frame;
        let command_buffers = &self.command_buffers[self.current_image..=self.current_image];
        let wait_semaphores: ArrayVec<[_; 1]> =
            [(image_available, PipelineStage::COLOR_ATTACHMENT_OUTPUT)].into();
        let signal_semaphores: ArrayVec<[_; 1]> = [render_finished].into();
        // yes, you have to write it twice like this. yes, it's silly.
        let present_wait_semaphores: ArrayVec<[_; 1]> = [render_finished].into();
        let submission = Submission {
            command_buffers,
            wait_semaphores,
            signal_semaphores,
        };
        unsafe {
            // let the_command_queue = queues.get_unchecked_mut(0);
            command_queue.submit(submission, Some(&self.in_flight_fences[self.current_image]));
            self
                .swapchain
                .present(command_queue, self.current_image as u32, present_wait_semaphores)
                .map_err(|_| "Failed to present into the swapchain!")?;
        }
        Ok(())
    }


    /// We have to clean up "leaf" elements before "root" elements. Basically, we
    /// clean up in reverse of the order that we created things.
    fn dispose(&mut self)-> (CommandPool<B, Graphics>)  {
        let _ = self.device.wait_idle();
        unsafe {
            for fence in self.in_flight_fences.drain(..) {
                self.device.destroy_fence(fence)
            }
            for semaphore in self.render_finished_semaphores.drain(..) {
                self.device.destroy_semaphore(semaphore)
            }
            for semaphore in self.image_available_semaphores.drain(..) {
                self.device.destroy_semaphore(semaphore)
            }
            for framebuffer in self.framebuffers.drain(..) {
                self.device.destroy_framebuffer(framebuffer);
            }
            for image_view in self.image_views.drain(..) {
                self.device.destroy_image_view(image_view);
            }
            // self.device.destroy_command_pool(ManuallyDrop::take(&mut self.command_pool).into_raw());
            self.device
                .destroy_render_pass(ManuallyDrop::take(&mut self.render_pass));
            self.device
                .destroy_swapchain(ManuallyDrop::take(&mut self.swapchain));
            ManuallyDrop::take(&mut self.command_pool)
        }
    }
}
/*
impl Drop for HalState{
    fn drop(&mut self) {

        let _ = self.device.wait_idle();
        unsafe {

            for fence in self.in_flight_fences.drain(..) {
                self.device.destroy_fence(fence)
            }
            for semaphore in self.render_finished_semaphores.drain(..) {
                self.device.destroy_semaphore(semaphore)
            }
            for semaphore in self.image_available_semaphores.drain(..) {
                self.device.destroy_semaphore(semaphore)
            }
            for framebuffer in self.framebuffers.drain(..) {
                self.device.destroy_framebuffer(framebuffer);
            }
            for image_view in self.image_views.drain(..) {
                self.device.destroy_image_view(image_view);
            }

            self.device.destroy_command_pool(ManuallyDrop::take(&mut self.command_pool).into_raw());
        //
            self.device.destroy_swapchain(ManuallyDrop::take(&mut self.swapchain));
            self.device.destroy_render_pass(ManuallyDrop::take(&mut self.render_pass));

        }
    }
}
*/


struct ShaderSpec {
    pub kind: shaderc::ShaderKind,
    pub source_path: & 'static str,
    pub source: Option<& 'static str>,
}
fn complile_shaders(shaders: &[ShaderSpec]) -> Result<Vec<shaderc::CompilationArtifact>, Error>{
    let mut compiler = shaderc::Compiler::new().unwrap();
    let mut res = Vec::with_capacity(shaders.len());
    for ShaderSpec{ kind, source_path, source } in shaders {
        let file = std::path::Path::new(source_path).file_name().unwrap().to_str().unwrap();
        let artifact = match source {
            None => {
                let source = std::fs::read_to_string(source_path).map_err(|_| "shader source not found")?;
                compiler.compile_into_spirv(
                    &source,
                    *kind,
                    file,
                    "main",
                    None,
                ).map_err(|_| "couldn't compile shader")?
            }
            Some(source) => {
                compiler.compile_into_spirv(
                    *source,
                    *kind,
                    file,
                    "main",
                    None,
                ).map_err(|_| "couldn't compile shader")?
            }
        };
        res.push(artifact);
    }
    Ok(res)
}
