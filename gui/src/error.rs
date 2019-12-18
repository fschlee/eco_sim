use log::{error};
use crate::renderer::con_back::ConBackError;

pub trait LogError {
    fn log(&self);
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Error {
    Static(&'static str),
    Owned(String),
    ConBack(ConBackError),
    // IO(std::io::Error),
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        use Error::*;
        match self {
            Static(m) => write!(f, "{}", m),
            Owned((m))=> write!(f, "{}", m),
            ConBack(e) => write!(f, "{}", e),
        }
    }
}
impl<T> LogError for Result<T, Error> {
    fn log(&self){
        match self {
            Ok(_) => (),
            Err(err) => error!("{}", err),
        }
    }
}
impl From<ConBackError> for Error {
    fn from(e: ConBackError) -> Self {
        Error::ConBack(e)
    }
}
impl From<& 'static str> for Error {
    fn from(m: &'static str) -> Self {
        Error::Static(m)
    }
}

impl From<String> for Error {
    fn from(m: String) -> Self {
        Error::Owned(m)
    }
}
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
       // Error::IO(e)
        Error::Owned(format!("{}", e))
    }
}
impl From<gfx_hal::device::MapError> for Error {
    fn from(e: gfx_hal::device::MapError) -> Self {
        Error::Owned(format!("{}", e))
    }
}
impl From<gfx_hal::device::OutOfMemory> for Error {
    fn from(e: gfx_hal::device::OutOfMemory) -> Self {
        Error::Owned(format!("{}", e))
    }
}
impl From<gfx_hal::pso::AllocationError> for Error {
    fn from(e: gfx_hal::pso::AllocationError) -> Self {
        Error::Owned(format!("{}", e))
    }
}
impl From<gfx_hal::device::OomOrDeviceLost> for Error {
    fn from(e: gfx_hal::device::OomOrDeviceLost) -> Self {
        Error::Static("Out of memory, device lost")
    }
}
impl From<gfx_hal::window::PresentError> for Error {
    fn from(e: gfx_hal::window::PresentError) -> Self {
        Error::Owned(format!("Could not present into swapchain: {}", e))
    }
}