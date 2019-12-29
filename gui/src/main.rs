#![feature(manually_drop_take)]

use enum_macros::EnumIter;
use log::{error, info};
use std::str::FromStr;
use std::time::Instant;
use strum_macros::{Display, EnumString};
use winit::dpi::LogicalSize;
use winit::{
    event::Event::EventsCleared,
    event_loop::EventLoop,
    platform::desktop::EventLoopExtDesktop,
    window::{Window, WindowBuilder},
};

pub mod conrod_winit;
pub mod error;
mod renderer;
pub mod simulation;
pub mod ui;

use crate::renderer::Renderer;
use error::LogError;
use renderer::init::{init_device, DeviceInit, InstSurface};
use ui::AppState;

const MAX_RENDER_FAILS: u32 = 100;

const MAX_FPS: f32 = 60.0;
const MIN_FRAME_DELAY: f32 = 1.0 / MAX_FPS;

#[derive(Debug, Copy, Clone, PartialEq, Eq, EnumIter, Display, EnumString)]
pub enum BackendSelection {
    #[cfg(all(feature = "vulkan", not(macos)))]
    #[strum(serialize = "Vulkan", serialize = "vulkan")]
    Vulkan,
    #[cfg(all(feature = "dx12", not(macos)))]
    #[strum(
        serialize = "DirectX12",
        serialize = "direct_x12",
        serialize = "Dx12",
        serialize = "dx12",
        serialize = "DX12",
        serialize = "dx_12"
    )]
    Dx12,
    #[cfg(all(feature = "dx11", not(macos)))]
    #[strum(
        serialize = "DirectX11",
        serialize = "direct_x11",
        serialize = "Dx11",
        serialize = "dx11",
        serialize = "DX11",
        serialize = "dx_11"
    )]
    Dx11,
    #[cfg(all(macos, feature = "metal"))]
    #[strum(serialize = "metal", serialize = "Metal")]
    Metal,
    #[cfg(feature = "gl")]
    #[strum(
        serialize = "OpenGL",
        serialize = "Open_GL",
        serialize = "open_gl",
        serialize = "Open-GL",
        serialize = "GL",
        serialize = "gl"
    )]
    OpenGl,
}

fn main() {
    env_logger::init();
    let mut event_loop = EventLoop::new();
    let window_builder = WindowBuilder::new()
        .with_inner_size(LogicalSize {
            width: 1640.0,
            height: 1024.0,
        })
        .with_title("Eco sim");
    let args: Vec<String> = std::env::args().collect();
    let back = match args.get(1).map(|s| BackendSelection::from_str(s)) {
        Some(Ok(b)) => {
            println!("{:?}", b);
            b
        }
        _ => BackendSelection::iter()
            .next()
            .expect("No backend available"),
    };
    let adapter_selection = args.get(2).and_then(|s| usize::from_str(s).ok());
    use BackendSelection::*;
    match back {
        #[cfg(feature = "gl")]
        OpenGl => {
            let (di, window) =
                renderer::init::init_gl(window_builder, &event_loop, adapter_selection)
                    .expect("could not initialize device");
            init_all(event_loop, window, di)
        }
        back => {
            let window = window_builder
                .build(&event_loop)
                .expect("could not create window");
            match back {
                #[cfg(all(feature = "vulkan", not(macos)))]
                Vulkan => {
                    let di = init_device::<(
                        gfx_backend_vulkan::Instance,
                        <gfx_backend_vulkan::Backend as gfx_hal::Backend>::Surface,
                    )>(&window, adapter_selection)
                    .expect("could not initialize device");
                    init_all(event_loop, window, di)
                }
                #[cfg(all(feature = "dx12", not(macos)))]
                Dx12 => {
                    let di = init_device::<(
                        gfx_backend_dx12::Instance,
                        <gfx_backend_dx12::Backend as gfx_hal::Backend>::Surface,
                    )>(&window, adapter_selection)
                    .expect("could not initialize device");
                    init_all(event_loop, window, di)
                }
                #[cfg(all(feature = "dx11", not(macos)))]
                Dx11 => {
                    println!("dx11");
                    let di = init_device::<(
                        gfx_backend_dx11::Instance,
                        <gfx_backend_dx11::Backend as gfx_hal::Backend>::Surface,
                    )>(&window, adapter_selection)
                    .expect("could not initialize device");
                    init_all(event_loop, window, di)
                }
                #[cfg(all(macos, feature = "metal"))]
                Metal => {
                    let di = init_device::<(
                        gfx_backend_metal::Instance,
                        <gfx_backend_metal::Backend as gfx_hal::Backend>::Surface,
                    )>(&window, adapter_selection)
                    .expect("could not initialize device");
                    init_all(event_loop, window, di)
                }
                #[cfg(feature = "gl")]
                OpenGl => unreachable!(),
            }
        }
    };
}

fn init_all<IS: InstSurface + 'static>(
    mut event_loop: EventLoop<()>,
    window: Window,
    device_init: DeviceInit<IS>,
) {
    let window_client_area = window.inner_size().to_physical(window.hidpi_factor());
    let mut renderer = renderer::Renderer::new(window_client_area, device_init);

    let mut ui_state = ui::UIState::new(window, renderer::init::adapter_list());
    let mut game_state = simulation::GameState::new();

    let glyph_cache = conrod_core::text::GlyphCache::builder()
        .dimensions(1024, 1024)
        .build();
    let mut ui_processor =
        renderer::con_back::UiProcessor::from_glyph_cache_with_filled_queue(glyph_cache);
    game_loop(renderer, ui_processor, ui_state, game_state, event_loop);
}
fn game_loop<IS: InstSurface + 'static>(
    mut renderer: renderer::Renderer<IS>,
    mut ui_processor: renderer::con_back::UiProcessor,
    mut ui_state: ui::UIState,
    mut game_state: simulation::GameState,
    mut event_loop: EventLoop<()>,
) -> AppState {
    let spec = ui_processor.get_texture_spec();
    renderer.add_texture(&spec);
    let mut fail_counter = 0;
    let mut instant = Instant::now();
    let mut ui_cmds = Vec::new();
    let mut app_state = AppState::Continue;
    event_loop.run_return(|event, _window_target, cntr_flow| {
        if event != EventsCleared {
            match ui_state.process(event, &game_state) {
                ap @ AppState::Exit | ap @ AppState::ChangeAdapter(_, _) => {
                    app_state = ap;
                    *cntr_flow = winit::event_loop::ControlFlow::Exit;
                }
                AppState::Continue if fail_counter > MAX_RENDER_FAILS => {
                    *cntr_flow = winit::event_loop::ControlFlow::Exit
                }
                AppState::Continue => (),
            }
        } else {
            ui_state.update(&game_state);
            if let Some(mut prims) = ui_state.conrod.draw_if_changed() {
                let dpi_factor = ui_state.window.hidpi_factor() as f32;
                let winit::dpi::LogicalSize { width, height, .. } = ui_state.window.inner_size();
                let (cmds, vtx) = ui_processor
                    .process_primitives(&mut prims, dpi_factor, width as f32, height as f32)
                    .or_else(|e| {
                        info!("{}", e);
                        ui_processor.update_gyph_cache(renderer::con_back::GlyphWalker::new(
                            ui_state.conrod.draw(),
                            dpi_factor,
                        ));
                        ui_processor.process_primitives(
                            &mut ui_state.conrod.draw(),
                            dpi_factor,
                            width as f32,
                            height as f32,
                        )
                    })
                    .unwrap();
                renderer.set_ui_buffer(vtx).unwrap();
                ui_cmds = cmds;
                if ui_processor.tex_updated {
                    let id0 = renderer::memory::Id::new(0);
                    let spec = ui_processor.get_texture_spec();
                    if renderer.texture_manager.texture_count() < 1 {
                        let id = renderer.add_texture(&spec);
                        assert_eq!(id, Ok(id0));
                    } else {
                        if let Err(e) = renderer.replace_texture(id0, &spec) {
                            info!("{}", e);
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
        }
    });
    if let AppState::ChangeAdapter(back, idx) = app_state {
        std::mem::drop(renderer);
        set_adapter(ui_processor, ui_state, game_state, event_loop, back, idx)
    } else {
        app_state
    }
}

fn set_adapter(
    mut ui_processor: renderer::con_back::UiProcessor,
    mut ui_state: ui::UIState,
    mut game_state: simulation::GameState,
    mut event_loop: EventLoop<()>,
    back: BackendSelection,
    idx: usize,
) -> AppState {
    let adapter_selection = Some(idx);
    let wca = ui_state
        .window
        .inner_size()
        .to_physical(ui_state.window.hidpi_factor());

    match back {
        #[cfg(all(feature = "vulkan", not(macos)))]
        Vulkan => {
            let di = init_device::<(
                gfx_backend_vulkan::Instance,
                <gfx_backend_vulkan::Backend as gfx_hal::Backend>::Surface,
            )>(&ui_state.window, adapter_selection)
            .expect("could not initialize device");
            game_loop(
                Renderer::new(wca, di),
                ui_processor,
                ui_state,
                game_state,
                event_loop,
            )
        }
        #[cfg(all(feature = "dx12", not(macos)))]
        Dx12 => {
            let di = init_device::<(
                gfx_backend_dx12::Instance,
                <gfx_backend_dx12::Backend as gfx_hal::Backend>::Surface,
            )>(&ui_state.window, adapter_selection)
            .expect("could not initialize device");
            game_loop(
                Renderer::new(wca, di),
                ui_processor,
                ui_state,
                game_state,
                event_loop,
            )
        }
        #[cfg(all(feature = "dx11", not(macos)))]
        Dx11 => {
            println!("dx11");
            let di = init_device::<(
                gfx_backend_dx11::Instance,
                <gfx_backend_dx11::Backend as gfx_hal::Backend>::Surface,
            )>(&ui_state.window, adapter_selection)
            .expect("could not initialize device");
            game_loop(
                Renderer::new(wca, di),
                ui_processor,
                ui_state,
                game_state,
                event_loop,
            )
        }
        #[cfg(all(macos, feature = "metal"))]
        Metal => {
            let di = init_device::<(
                gfx_backend_metal::Instance,
                <gfx_backend_metal::Backend as gfx_hal::Backend>::Surface,
            )>(&ui_state.window, adapter_selection)
            .expect("could not initialize device");
            game_loop(
                Renderer::new(wca, di),
                ui_processor,
                ui_state,
                game_state,
                event_loop,
            )
        }
        #[cfg(feature = "gl")]
        OpenGl => unimplemented!("explicitly selecting an adapter with open Gl is not supported"),
    }
}
