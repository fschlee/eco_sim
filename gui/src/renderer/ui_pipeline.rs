use super::con_back::Command;
use super::*;
use conrod_core as cc;
use gfx_hal::{
    command::CommandBuffer,
    pso::{Primitive, VertexInputRate},
};

pub struct UiPipeline<B: Backend> {
    device: Arc<Dev<B>>,
    pub layouts: ManuallyDrop<<B as Backend>::PipelineLayout>,
    pub gfx_pipeline: ManuallyDrop<<B as Backend>::GraphicsPipeline>,
}

impl<B: Backend + BackendExt> UiPipeline<B> {
    pub fn create(
        device: Arc<Dev<B>>,
        render_area: Rect,
        render_pass: &<B as Backend>::RenderPass,
        vertex_compile_artifact: shaderc::CompilationArtifact,
        fragment_compile_artifact: shaderc::CompilationArtifact,
        descriptor_set_layouts: &Vec<<B as Backend>::DescriptorSetLayout>,
    ) -> Result<UiPipeline<B>, Error> {
        let vertex_shader_module = unsafe {
            device
                .create_shader_module(vertex_compile_artifact.as_binary())
                .map_err(|_| "Couldn't make the vertex module")?
        };
        let fragment_shader_module = unsafe {
            device
                .create_shader_module(fragment_compile_artifact.as_binary())
                .map_err(|_| "Couldn't make the fragment module")?
        };
        let (vs_entry, fs_entry) = (
            EntryPoint {
                entry: "main",
                module: &vertex_shader_module,
                specialization: Specialization::EMPTY,
            },
            EntryPoint {
                entry: "main",
                module: &fragment_shader_module,
                specialization: Specialization::EMPTY,
            },
        );
        let shaders = GraphicsShaderSet {
            vertex: vs_entry,
            hull: None,
            domain: None,
            geometry: None,
            fragment: Some(fs_entry),
        };

        let input_assembler = InputAssemblerDesc::new(Primitive::TriangleList);

        let vertex_buffers: Vec<VertexBufferDesc> = vec![VertexBufferDesc {
            binding: 0,
            stride: (size_of::<f32>() * (2 + 2) + 2 * size_of::<u32>()) as ElemStride,
            rate: VertexInputRate::Vertex,
        }];
        let position_attribute = AttributeDesc {
            location: 0,
            binding: 0,
            element: Element {
                format: Format::Rg32Sfloat,
                offset: 0,
            },
        };
        let uv_attribute = AttributeDesc {
            location: 1,
            binding: 0,
            element: Element {
                format: Format::Rg32Sfloat,
                offset: (size_of::<f32>() * 2) as ElemOffset,
            },
        };
        let mode_attribute = AttributeDesc {
            location: 2,
            binding: 0,
            element: Element {
                format: Format::R32Uint,
                offset: (size_of::<f32>() * 4) as ElemOffset,
            },
        };
        let color_attribute = AttributeDesc {
            location: 3,
            binding: 0,
            element: Element {
                format: Format::Rgba8Unorm,
                offset: (size_of::<f32>() * 4 + size_of::<u32>()) as ElemOffset,
            },
        };

        let attributes: Vec<AttributeDesc> = vec![
            position_attribute,
            color_attribute,
            mode_attribute,
            uv_attribute,
        ];

        let rasterizer = Rasterizer {
            depth_clamping: false,
            polygon_mode: PolygonMode::Fill,
            cull_face: Face::NONE, // Face::BACK,
            front_face: FrontFace::Clockwise,
            depth_bias: None,
            conservative: false,
        };

        let depth_stencil = DepthStencilDesc {
            depth: None,
            depth_bounds: false,
            stencil: None,
        };

        let blender = BlendDesc {
            logic_op: None,
            targets: vec![ColorBlendDesc {
                mask: ColorMask::ALL,
                blend: Some(BlendState::ALPHA),
            }],
        };

        let baked_states = BakedStates {
            viewport: Some(Viewport {
                rect: render_area,
                depth: (0.0..1.0),
            }),
            scissor: None,
            blend_color: None,
            depth_bounds: None,
        };
        let push_constants = vec![(ShaderStageFlags::VERTEX, 0..8)]; //needs to be divisible by 4 for dx12?
        let layout = unsafe {
            device
                .create_pipeline_layout(descriptor_set_layouts, push_constants)
                .map_err(|_| "Couldn't create a pipeline layout")?
        };
        let gfx_pipeline = {
            let desc = GraphicsPipelineDesc {
                shaders,
                rasterizer,
                vertex_buffers,
                attributes,
                input_assembler,
                blender,
                depth_stencil,
                multisampling: None,
                baked_states,
                layout: &layout,
                subpass: Subpass {
                    index: 0,
                    main_pass: render_pass,
                },
                flags: PipelineCreationFlags::empty(),
                parent: BasePipeline::None,
            };

            unsafe {
                device.create_graphics_pipeline(&desc, None).map_err(|e| {
                    error!("{}", e);
                    "Couldn't create a graphics pipeline!"
                })?
            }
        };
        unsafe {
            device.destroy_shader_module(vertex_shader_module);
            device.destroy_shader_module(fragment_shader_module);
        }

        Ok(UiPipeline {
            device,
            layouts: ManuallyDrop::new(layout),
            gfx_pipeline: ManuallyDrop::new(gfx_pipeline),
        })
    }
    pub fn execute<'a>(
        &mut self,
        encoder: &mut impl CommandBuffer<B>,
        resource_manager: &mut ResourceManager<B>,
        vertex_buffers: &[BufferBundle<B>],
        render_area: Rect,
        cmds: &[Command],
    ) {
        unsafe {
            let vert = vertex_buffers.iter().map(|b| (b.buffer.deref(), 0));
            encoder.bind_graphics_pipeline(&self.gfx_pipeline);
            encoder.bind_vertex_buffers(0, vert);
            // encoder.bind_index_buffer(index_buffer);

            // encoder.bind_graphics_descriptor_sets(&self.layout, 0, Some(&self.descriptor_sets[next_descriptor]), &[], );
            resource_manager.uniform_buffers[3]
                .write(
                    &self.device,
                    0,
                    &[
                        (render_area.w as f32).to_bits(),
                        (render_area.h as f32).to_bits(),
                    ],
                )
                .expect("couldn't write to uniform buffer");
            let mut last_tex = None;
            for cmd in cmds.iter() {
                if last_tex.is_none() || (cmd.texture_id.is_some() && cmd.texture_id != last_tex) {
                    last_tex = cmd.texture_id;
                    let id = cmd.texture_id.unwrap_or(Id::new(0));
                    let desc = if let Some(desc) = resource_manager.get_descriptor_set(id) {
                        desc
                    } else {
                        resource_manager.get_or_write_descriptor_set(id).log();
                        resource_manager.get_descriptor_set(id).unwrap()
                    };
                    encoder.bind_graphics_descriptor_sets(
                        &self.layouts,
                        0,
                        vec![desc, &resource_manager.uniform_buffer_descs[3]],
                        &[],
                    );
                }
                let sr = cmd.clip_rect;
                let scissor = calc_scissor(sr, render_area);
                encoder.set_scissors(0, Some(scissor));
                // encoder.draw_indexed(0..6, 0, 0..1);
                encoder.draw(cmd.vtx_offset..cmd.vtx_offset + cmd.elem_count, 0..1);
            }
        }
    }
}

impl<B: Backend> Drop for UiPipeline<B> {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_pipeline_layout(ManuallyDrop::take(&mut self.layouts));
            self.device
                .destroy_graphics_pipeline(ManuallyDrop::take(&mut self.gfx_pipeline));
        }
    }
}

fn calc_scissor(sr: cc::Rect, extent: Rect) -> Rect {
    // return extent;
    Rect {
        x: (sr.x.start.round() as i16).max(extent.x),
        y: (-sr.y.end.round() as i16).max(extent.y),
        w: ((sr.x.end - sr.x.start).max(0.0).round() as i16).min(extent.w),
        h: ((sr.y.end - sr.y.start).max(0.0).round() as i16).min(extent.h),
    }
}
