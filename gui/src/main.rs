#![feature(manually_drop_take)]
#![feature(label_break_value)]
#![feature(type_ascription)]

use winit::{EventsLoop, WindowBuilder, Window};
use log::{debug, error, info, trace, warn};
use winit::dpi::LogicalSize;
use std::time::Instant;


mod renderer;
pub mod ui;
pub mod conrod_winit;
pub mod simulation;

const MAX_RENDER_FAILS : u32 = 100;

fn main() {
    env_logger::init();
    let mut event_loop = EventsLoop::new();
    let window = WindowBuilder::new()
        .with_dimensions(LogicalSize { width: 1280.0, height: 1024.0 })
        .with_title("Eco sim")
        .build(&event_loop)
        .unwrap();
    /*
    println!("vulkan devices:");
    renderer::init::list_adapters::<gfx_backend_vulkan::Instance>();
    println!("dx11 devices:");
    renderer::init::list_adapters::<gfx_backend_dx11::Instance>();
    //* */
println!("dx12 devices:");
renderer::init::list_adapters::<gfx_backend_dx12::Instance>();

println!("openGL devices:");
renderer::init::list_adapters::<gfx_backend_gl::Instance>();*/
    game_loop(window, event_loop);
}
fn game_loop(window: Window, mut event_loop: EventsLoop){
    let mut ui_state = ui::UIState::new(&window);
    let mut game_state = simulation::GameState::new();
#[cfg(macos)]
    let mut renderer = renderer::Renderer::<(gfx_backend_metal::Instance, <<gfx_backend_metal::Instance as gfx_hal::Instance>::Backend as gfx_hal::Backend>::Surface)>::new(&window);
#[cfg(all(feature = "gl", not(macos)))]
    let mut renderer = renderer::Renderer::<(gfx_backend_gl::Surface)>::new(&window);
#[cfg(all(feature = "vulkan", not(macos)))]
    let mut renderer = renderer::Renderer::<(gfx_backend_vulkan::Instance, <gfx_backend_vulkan::Backend as gfx_hal::Backend>::Surface)>::new(&window);
#[cfg(all(feature = "dx12", not(macos)))]
    let mut renderer = renderer::Renderer::<(gfx_backend_dx12::Instance, <<gfx_backend_dx12::Instance as gfx_hal::Instance>::Backend as gfx_hal::Backend>::Surface)>::new(&window);
#[cfg(all(feature = "dx11", not(macos)))]
    let mut renderer = renderer::Renderer::<(gfx_backend_dx11::Instance, gfx_backend_dx11::Surface)>::new(&window);
    let mut quit = false;
    let mut fail_counter = 0;
    let mut instant = Instant::now();
    let glyph_cache = conrod_core::text::GlyphCache::builder().dimensions(1024,1024).build();
    let mut ui_processor = renderer::con_back::UiProcessor::from_glyph_cache_with_filled_queue(glyph_cache);
    let mut ui_cmds = Vec::new();
    while !quit && fail_counter < MAX_RENDER_FAILS {
        let (close, ui_updates, actions) = ui_state.process(& mut event_loop, &game_state);
        quit = close;
        if let Some(mut prims) = ui_state.conrod.draw_if_changed(){
            let dpi_factor = window.get_hidpi_factor() as f32;
            if let Some(winit::dpi::LogicalSize{width, height, ..}) = window.get_inner_size(){
                let (cmds, vtx) = ui_processor.process_primitives(& mut prims, dpi_factor, width as f32, height as f32)
                    .or_else(|_| {
                    ui_processor.update_gyph_cache(renderer::con_back::GlyphWalker::new(ui_state.conrod.draw(), dpi_factor ));
                    ui_processor.process_primitives(& mut ui_state.conrod.draw(), dpi_factor, width as f32, height as f32)
                }).unwrap();
                // println!("vtx {:?}", &vtx);
                renderer.set_ui_buffer(vtx).unwrap();
                ui_cmds = cmds;
                // println!("cmds {:?}", ui_cmds);

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
        }
        let now = Instant::now();
        let delta = now.duration_since(instant).as_secs_f32();
        instant = now;
        game_state.update(actions, delta);
        let sim_updates = game_state.get_render_data();
        match renderer.tick(&ui_cmds, ui_updates, &sim_updates) {
            Ok(_) => fail_counter = 0,
            Err(err) => {
                error!("{}", err);
                fail_counter += 1;
            }
        }
    }
}
