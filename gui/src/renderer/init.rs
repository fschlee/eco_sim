use winit::{Window, EventsLoop, WindowBuilder};
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
#[cfg(feature = "vulkan")]
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
#[cfg(feature = "gl")]
impl InstSurface for gfx_backend_gl::Surface  {
    type Instance = gfx_backend_gl::Surface;
    fn get_surface<'a>(&'a self) -> &'a <<Self::Instance as Instance>::Backend as Backend>::Surface {
        self
    }

    fn get_mut_surface<'a>(&'a mut self) -> &'a mut <<Self::Instance as Instance>::Backend as Backend>::Surface {
        self
    }

    fn get_instance<'a>(&'a self) -> &'a Self::Instance {
        self
    }
    fn create(name: &str, version: u32, window: &Window) -> Self {
        let wb = WindowBuilder::new()
            .with_dimensions(LogicalSize{ width: 400.0, height: 200.0 })
            .with_title("quad".to_string());;
        let ev = EventsLoop::new();
        let cb = gfx_backend_gl::config_context(gfx_backend_gl::glutin::ContextBuilder::new(), gfx_hal::format::Format::Rgba8Srgb, None).with_vsync(true);;
        let win = gfx_backend_gl::glutin::GlWindow::new(wb, cb, &ev).expect("no gl window");
        let surf = gfx_backend_gl::Surface::from_window(win);
        surf
    }
}
#[cfg(feature = "dx11")]
impl InstSurface for (gfx_backend_dx11::Instance, gfx_backend_dx11::Surface)  {
    type Instance = gfx_backend_dx11::Instance;
    fn get_surface<'a>(&'a self) -> &'a <<Self::Instance as Instance>::Backend as Backend>::Surface {
        &self.1
    }

    fn get_mut_surface<'a>(&'a mut self) -> &'a mut <<Self::Instance as Instance>::Backend as Backend>::Surface {
        & mut self.1
    }

    fn get_instance<'a>(&'a self) -> &'a Self::Instance {
        &self.0
    }
    fn create(name: &str, version: u32, window: &Window) -> Self {
        let inst = gfx_backend_dx11::Instance::create(name, version);
        let surf = inst.create_surface(window);
        (inst, surf)
    }
}
#[cfg(feature = "dx12")]
impl InstSurface for (gfx_backend_dx12::Instance, <<gfx_backend_dx12::Instance as Instance>::Backend as Backend>::Surface)  {
    type Instance = gfx_backend_dx12::Instance;
    fn get_surface<'a>(&'a self) -> &'a <<Self::Instance as Instance>::Backend as Backend>::Surface {
        &self.1
    }

    fn get_mut_surface<'a>(&'a mut self) -> &'a mut <<Self::Instance as Instance>::Backend as Backend>::Surface {
        & mut self.1
    }

    fn get_instance<'a>(&'a self) -> &'a Self::Instance {
        &self.0
    }
    fn create(name: &str, version: u32, window: &Window) -> Self {
        let inst = gfx_backend_dx12::Instance::create(name, version);
        let surf = inst.create_surface_from_hwnd(winit::WindowExtWindows::hwnd(window));
        (inst, surf)
    }
}
#[cfg(macos)]
impl InstSurface for (gfx_backend_metal::Instance, <<gfx_backend_metal::Instance as Instance>::Backend as Backend>::Surface)  {
    type Instance = gfx_backend_metal::Instance;
    fn get_surface<'a>(&'a self) -> &'a <<Self::Instance as Instance>::Backend as Backend>::Surface {
        &self.1
    }

    fn get_mut_surface<'a>(&'a mut self) -> &'a mut <<Self::Instance as Instance>::Backend as Backend>::Surface {
        & mut self.1
    }

    fn get_instance<'a>(&'a self) -> &'a Self::Instance {
        &self.0
    }
    fn create(name: &str, version: u32, window: &Window) -> Self {
        let inst = gfx_backend_metal::Instance::create(name, version);
        let surf = inst.create_surface(window);
        (inst, surf)
    }
}

pub fn init_device<IS: InstSurface>(window: &Window) -> Result<(IS, Adapter<IS::Back>, <IS::Back as Backend>::Device, QueueGroup<IS::Back>), Error> {
    let inst_surface = IS::create(WINDOW_NAME, 1, window);
    let instance = inst_surface.get_instance();
    let surface = inst_surface.get_surface();
    let mut adapters  = instance.enumerate_adapters().into_iter()
        .filter(|a| {

            a.queue_families.iter()
                .any(|qf| qf.queue_type().supports_graphics() && surface.supports_queue_family(qf))
        });
    let first = adapters.next();
    let adapter = adapters.find(|a| a.info.device_type == gfx_hal::adapter::DeviceType::DiscreteGpu).or(first)
        .ok_or("Couldn't find a graphical Adapter!")?;
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
    Ok((inst_surface, adapter, device, queue_group))
}
/*
pub fn list_adapters<IS: Initialize>() {
    let instance = I::create(WINDOW_NAME, 1);
    for a in &instance.enumerate_adapters(){
        println!("{}: {}", a.info.name, a.info.vendor);
    }
}*/
/*
pub trait ComQueueCont<B : Backend, C: Capability> : Sized + DerefMut<Target = CommandQueue<B, C>> {
    type Container;
    fn split(queue_group: QueueGroup<B, C>) -> (Self, Self, Self::Container);
    fn merge(first: Self, second: Self, container: Self::Container) ->  QueueGroup<B, C>;
    fn
}

pub struct SingletonQueue<B : Backend, C: Capability> {
    inner : Rc<RefCell<CommandQueue<B, C>>>,
}
impl<B : Backend, C: Capability> Deref for SingletonQueue<B, C> {
    type Target = CommandQueue<B, C>;
    fn deref<'a>(&'a self) -> &'a Self::Target {
        self.inner.deref()
    }
}
impl<B : Backend, C: Capability> DerefMut for SingletonQueue<B, C> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.deref_mut()
    }
}

impl<B  : Backend, C: Capability> SingletonQueue<B, C> {
    fn new(inner: Rc<RefCell<CommandQueue<B, C>>>) -> Self {
        Self{inner}
    }
}

impl<B: Backend, C: Capability> ComQueueCont<B, C> for SingletonQueue<B, C> {
    type Container = QueueGroup<B, C>;
    fn split(mut queue_group: QueueGroup<B, C>) -> (Self, Self, Self::Container) {
        let cq = queue_group.queues.remove(0);
        let rc = Rc::new(RefCell::new(cq));
        (SingletonQueue::new(rc), SingletonQueue::new(rc.clone()), queue_group)
    }
    fn merge(first: Self, second: Self, mut container: Self::Container) -> QueueGroup<B, C> {
        {
            let f = first;
        }
        match Rc::try_unwrap(second.inner){
            Ok(mut  m) =>
                container.queues.insert(0, m.into_inner()),
            Err(_) => error!("ownership violation")
        }
       ;
        container
    }
}

*/
