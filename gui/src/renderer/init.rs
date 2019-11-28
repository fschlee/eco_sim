use winit::{event_loop::EventLoop, window::{WindowBuilder, Window}};
use gfx_hal::{Backend, Instance, adapter::{Adapter, PhysicalDevice},
              queue::{QueueGroup, QueueFamily}, window::Surface, Features};
use log::{error};
use winit::dpi::LogicalSize;

use crate::error::Error;

pub const WINDOW_NAME: &str = "Textures";

pub trait InstSurface {
    type Back : Backend;
    type SurfaceInfo : Surface<Self::Back>;
    fn get_surface<'a>(&'a self) -> & 'a <<Self as InstSurface>::Back as Backend>::Surface;
    fn get_surface_info<'a>(& 'a self) -> & 'a Self::SurfaceInfo;
    fn get_mut_surface<'a>(& 'a mut self) -> & 'a mut <<Self as InstSurface>::Back as Backend>::Surface;
    fn get_instance<'a>(&'a self) -> & 'a <<Self as InstSurface>::Back as Backend>::Instance;
    fn create(name: & str, version: u32, window: &Window) -> Self;
}
#[cfg(all(feature = "vulkan", not(macos)))]
impl InstSurface for (gfx_backend_vulkan::Instance, <gfx_backend_vulkan::Backend as Backend>::Surface)  {
    type Back = gfx_backend_vulkan::Backend;
    type SurfaceInfo = <Self::Back as Backend>::Surface;
    fn get_surface<'a>(&'a self) -> &'a <<Self as InstSurface>::Back as Backend>::Surface {
        &self.1
    }
    fn get_surface_info<'a>(&'a self) -> & 'a Self::SurfaceInfo {
        &self.1
    }
    fn get_mut_surface<'a>(&'a mut self) -> &'a mut <<Self as InstSurface>::Back as Backend>::Surface {
        &mut self.1
    }
    fn get_instance<'a>(&'a self) -> &'a <<Self as InstSurface>::Back as Backend>::Instance {
        &self.0
    }
    fn create(name: &str, version: u32, window: &Window) -> Self {
        let inst = gfx_backend_vulkan::Instance::create(name, version).expect("unsupported backend");
        let surf = unsafe {
            inst.create_surface(window).expect("couldn't initialize surface")
        };
        (inst, surf)
    }
}

#[cfg(feature = "dx11")]
impl InstSurface for (gfx_backend_dx11::Instance, <gfx_backend_dx11::Backend as Backend>::Surface)  {
    type Back = gfx_backend_dx11::Backend;
    type SurfaceInfo = <Self::Back as Backend>::Surface;
    fn get_surface<'a>(&'a self) -> &'a  <<Self as InstSurface>::Back as Backend>::Surface {
        &self.1
    }
    fn get_surface_info<'a>(&'a self) -> & 'a Self::SurfaceInfo {
        &self.1
    }
    fn get_mut_surface<'a>(&'a mut self) -> &'a mut  <<Self as InstSurface>::Back as Backend>::Surface {
        & mut self.1
    }

    fn get_instance<'a>(&'a self) -> &'a <<Self as InstSurface>::Back as Backend>::Instance {
        &self.0
    }
    fn create(name: &str, version: u32, window: &Window) -> Self {
        let inst = gfx_backend_dx11::Instance::create(name, version).expect("unsupported backend");
        let surf = unsafe {
            inst.create_surface(window).expect("couldn't create surface")
        };
        (inst, surf)
    }
}
#[cfg(feature = "dx12")]
impl InstSurface for (gfx_backend_dx12::Instance, <gfx_backend_dx12::Backend as Backend>::Surface)  {
    type Back = gfx_backend_dx12::Backend;
    type SurfaceInfo = <Self::Back as Backend>::Surface;
    fn get_surface<'a>(&'a self) -> &'a  <<Self as InstSurface>::Back as Backend>::Surface {
        &self.1
    }
    fn get_surface_info<'a>(&'a self) -> & 'a Self::SurfaceInfo {
        &self.1
    }
    fn get_mut_surface<'a>(&'a mut self) -> &'a mut  <<Self as InstSurface>::Back as Backend>::Surface {
        & mut self.1
    }

    fn get_instance<'a>(&'a self) -> &'a <<Self as InstSurface>::Back as Backend>::Instance {
        &self.0
    }
    fn create(name: &str, version: u32, window: &Window) -> Self {
        let inst = gfx_backend_dx12::Instance::create(name, version).expect("unsupported backend");
        let surf = unsafe {
            inst.create_surface(window).expect("couldn't create surface")
        };
        (inst, surf)
    }
}
#[cfg(all(feature="metal", macos))]
impl InstSurface for (gfx_backend_metal::Instance, <gfx_backend_metal::Backend as Backend>::Surface)  {
    type Back = gfx_backend_metal::Backend;
    type SurfaceInfo = <Self::Back as Backend>::Surface;
    fn get_surface<'a>(&'a self) -> &'a <<Self as InstSurface>::Back as Backend>::Surface {
        &self.1
    }
    fn get_surface_info<'a>(&'a self) -> & 'a Self::SurfaceInfo {
        &self.1
    }
    fn get_mut_surface<'a>(&'a mut self) -> &'a mut <<Self as InstSurface>::Back as Backend>::Surface {
        & mut self.1
    }

    fn get_instance<'a>(&'a self) -> &'a <<Self as InstSurface>::Back as Backend>::Instance {
        &self.0
    }
    fn create(name: &str, version: u32, window: &Window) -> Self {
        let inst = gfx_backend_metal::Instance::create(name, version).expect("unsupported backend");
        let surf = unsafe {
            inst.create_surface(window).expect("couldn't create surface")
        };
        (inst, surf)
    }
}

#[cfg(feature = "gl")]
impl InstSurface for gfx_backend_gl::Instance   {
    type Back = gfx_backend_gl::Backend;
    type SurfaceInfo = gfx_backend_gl::Surface;
    fn get_surface<'a>(&'a self) -> &'a <<Self as InstSurface>::Back as Backend>::Surface {
        match self {
            gfx_backend_gl::Instance::Surface(surface) => & surface,
            gfx_backend_gl::Instance::Headless(ctx) => unimplemented!()
        }
    }
    fn get_surface_info<'a>(&'a self) -> & 'a Self::SurfaceInfo {
        match self {
            gfx_backend_gl::Instance::Surface(surface) => & surface,
            gfx_backend_gl::Instance::Headless(ctx) => unimplemented!()
        }
    }
    fn get_mut_surface<'a>(&'a mut self) -> &'a mut <<Self as InstSurface>::Back as Backend>::Surface {
        match self {
            gfx_backend_gl::Instance::Surface(surface) => surface,
            gfx_backend_gl::Instance::Headless(ctx) => unimplemented!()
        }
    }

    fn get_instance<'a>(&'a self) -> &'a <<Self as InstSurface>::Back as Backend>::Instance {
        self
    }
    fn create(name: &str, version: u32, window: &Window) -> Self {
        unimplemented!()
        /*
        let mut ev = gfx_backend_gl::glutin::event_loop::EventLoop::new();
        let ctx = gfx_backend_gl::glutin::ContextBuilder::new()
            .build_headless(&ev,
                            gfx_backend_gl::glutin::dpi::PhysicalSize::from_logical(
                                window.get_inner_size().unwrap(),
                                window.get_hidpi_factor())).expect("couldn't create context");
        let surf = gfx_backend_gl::Surface::from_context(gfx_backend_gl::glutin::ContextWrapper::from(ctx));
        let inst = <Self::Back as Backend>::Instance::create(name, version).expect("unsupported backend");
        (inst, surf) */
    }
}


pub struct DeviceInit<IS: InstSurface>(pub IS, pub Adapter<IS::Back>, pub <IS::Back as Backend>::Device, pub QueueGroup<IS::Back>);

pub fn init_device<IS: InstSurface>(window: &Window, adapter_selection: Option<usize>) -> Result<DeviceInit<IS>, Error> {
    init_helper(IS::create(WINDOW_NAME, 1, window), adapter_selection)
}
fn init_helper<IS: InstSurface>(inst_surface: IS, adapter_selection: Option<usize>) -> Result<DeviceInit<IS>, Error> {
    let instance = inst_surface.get_instance();
    let surface = inst_surface.get_surface();
    let mut adapters : Vec<_> = instance.enumerate_adapters().into_iter()
        .filter(|a| {
            println!("{}", a.info.name);
            a.queue_families.iter()
                .any(|qf| qf.queue_type().supports_graphics() && surface.supports_queue_family(qf))
        }).collect();
    if adapters.len() < 1 {
        return Err("couldn't find suitable graphics adapter".into())
    }
    let adapter_i = match adapter_selection {
        Some(i) if i < adapters.len() => i,
        _ => (0..adapters.len()).max_by_key(|i| {
            let a = &adapters[*i];
            let mut score = 0;
        //  println!("{}: {:?}", a.info.name, a.physical_device.limits());
            if a.info.device_type == gfx_hal::adapter::DeviceType::DiscreteGpu {
                score += 4000;
            }
            score += name_score(&a.info.name);
            score
        }).unwrap_or(0)
    };
    let adapter = adapters.remove(adapter_i);
    println!("selected {}", adapter.info.name);
    let family = adapter.queue_families.iter().find(|family| {
            surface.supports_queue_family(family) && family.queue_type().supports_graphics()
        }).unwrap(); // Already seen at least one suitable qf earlier
    let mut gpu = unsafe {
        adapter
            .physical_device
            .open(&[(family, &[1.0])], Features::empty())
            .map_err(|e| {error!("{}", e); "Couldn't open the PhysicalDevice!"})?
    };
    let device = gpu.device;
    let queue_group = gpu.queue_groups.pop().ok_or("Can't get queue group")?;
    Ok(DeviceInit(inst_surface, adapter, device, queue_group))
}


#[cfg(feature = "gl")]
pub fn init_gl(window_builder: WindowBuilder, ev: &EventLoop<()>, adapter_selection: Option<usize>) -> Result<(DeviceInit<gfx_backend_gl::Instance>, Window), Error> {
    let ctx = gfx_backend_gl::glutin::ContextBuilder::new()
        .build_windowed(window_builder, ev).expect("couldn't create context");
    let (glutin_context, window) = unsafe {
        ctx.make_current().expect("Failed to make the context current").split()
    };
    let surface = gfx_backend_gl::Surface::from_context(glutin_context);
    let di = init_helper(gfx_backend_gl::Instance::Surface(surface), adapter_selection)?;
    Ok((di, window))
}

fn name_score(name: &str) -> i32 {
    let mut score = 0;
    let lower = name.to_lowercase();
    for (n, s) in &[("rtx", 1000), ("rx", 500), ("vega", 500), ("gtx", 500), ("nvidia", 1000), ("geforce", 500), ("radeon", 500), ("amd", 500), ("intel", -50), ("mircosoft", -200), ("basic", -200)]{
        if lower.matches(n).next().is_some() {
            score += s;
        }
    }
    score
}