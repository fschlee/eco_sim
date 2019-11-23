#![feature(manually_drop_take)]

use winit::{event_loop::EventLoop, window::{WindowBuilder, Window}};
use log::{debug, error, info, trace, warn};
use winit::dpi::LogicalSize;
use std::time::Instant;
use strum_macros::{Display, EnumIter};

mod renderer;
pub mod ui;
pub mod conrod_winit;
pub mod simulation;
pub mod error;

use renderer::init::{init_device};

const MAX_RENDER_FAILS : u32 = 100;

const MAX_FPS : f32 = 60.0;
const MIN_FRAME_DELAY: f32 = 1.0/ MAX_FPS;

#[derive(Debug, Copy, Clone, PartialEq, Eq, EnumIter, Display)]
enum BackendSelection {
#[cfg(all(feature = "vulkan", not(macos)))]
    Vulcan,
#[cfg(all(feature = "dx12", not(macos)))]
    Dx12,
#[cfg(all(feature = "dx11", not(macos)))]
    Dx11,
#[cfg(all(macos, feature="metal"))]
    Metal,
#[cfg(feature = "gl")]
    OpenGl,
}

fn main() {
    env_logger::init();
    let mut event_loop = EventLoop::new();
    let window_builder = WindowBuilder::new()
        .with_inner_size(LogicalSize { width: 1640.0, height: 1024.0 })
        .with_title("Eco sim");

#[cfg(feature = "gl")]
    let (di, window) = renderer::init::init_gl(window_builder, &event_loop).expect("could not initialize device");
#[cfg(not(feature = "gl"))]
    let window = window_builder.build(&event_loop).expect("could not create window");
#[cfg(all(macos, feature="metal"))]
    let di = init_device::<(gfx_backend_metal::Instance, <gfx_backend_metal::Backend as gfx_hal::Backend>::Surface)>(&window);
#[cfg(all(feature = "vulkan", not(macos)))]
    let di = init_device::<(gfx_backend_vulkan::Instance, <gfx_backend_vulkan::Backend as gfx_hal::Backend>::Surface)>(&window);
#[cfg(all(feature = "dx12", not(macos)))]
    let di = init_device::<(gfx_backend_dx12::Instance, <gfx_backend_dx12::Backend as gfx_hal::Backend>::Surface)>(&window);
#[cfg(all(feature = "dx11", not(macos)))]
    let di = init_device::<(gfx_backend_dx11::Instance, <gfx_backend_dx11::Backend as gfx_hal::Backend>::Surface)>(&window);
#[cfg(not(feature = "gl"))]
    let di = di.expect("could not initialize device");
    let window_client_area = window
        .inner_size()
        .to_physical(window.hidpi_factor());
    let mut renderer = renderer::Renderer::new(window_client_area, di);
    let mut ui_state = ui::UIState::new(window);
    let mut game_state = simulation::GameState::new();
    let mut fail_counter = 0;
    let mut instant = Instant::now();
    let glyph_cache = conrod_core::text::GlyphCache::builder().dimensions(1024,1024).build();
    let mut ui_processor = renderer::con_back::UiProcessor::from_glyph_cache_with_filled_queue(glyph_cache);
    let mut ui_cmds = Vec::new();
    event_loop.run( move|event, window_target, cntr_flow| {
        let close = ui_state.process(event, &game_state);
        if close || fail_counter > MAX_RENDER_FAILS {
            *cntr_flow = winit::event_loop::ControlFlow::Exit;
        }
        let now = Instant::now();
        let delta = now.duration_since(instant).as_secs_f32();
        if delta < MIN_FRAME_DELAY {
            return;
        }
        instant = now;
        ui_state.update(&game_state);
        if let Some(mut prims) = ui_state.conrod.draw_if_changed(){
            let dpi_factor = ui_state.window.hidpi_factor() as f32;
            let winit::dpi::LogicalSize{width, height, ..} = ui_state.window.inner_size();
            let (cmds, vtx) = ui_processor.process_primitives(& mut prims, dpi_factor, width as f32, height as f32)
                .or_else(|_| {
                    ui_processor.update_gyph_cache(renderer::con_back::GlyphWalker::new(ui_state.conrod.draw(), dpi_factor ));
                    ui_processor.process_primitives(& mut ui_state.conrod.draw(), dpi_factor, width as f32, height as f32)
                }).unwrap();
            renderer.set_ui_buffer(vtx).unwrap();
            ui_cmds = cmds;
            if ui_processor.tex_updated {
                let id0 = renderer::memory::Id::new(0);
                let spec = ui_processor.get_texture_spec();
                if renderer.texture_manager.texture_count() < 1 {
                    let id = renderer.add_texture(&spec);
                    assert_eq!(id, Ok(id0));
                } else {
                    if let Err(_) = renderer.replace_texture(id0, &spec) {
                        let id = renderer.add_texture(&spec);
                        assert_eq!(id, Ok(id0));
                    }
                }
            }
        }
        let now = Instant::now();
        let delta = now.duration_since(instant).as_secs_f32();
        instant = now;
        game_state.update(ui_state.actions.drain(..), delta);
        let sim_updates = game_state.get_render_data();
        match renderer.tick(&ui_cmds, ui_state.ui_updates.drain(..), &sim_updates) {
            Ok(_) => fail_counter = 0,
            Err(err) => {
                error!("{}", err);
                fail_counter += 1;
            }
        }
    });
}
