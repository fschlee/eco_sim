use log::{error};

pub trait LogError {
    fn log(&self);
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Error {
    Static(&'static str),
    Owned(String),
    // IO(std::io::Error),
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{:?}", self)
    }
}
impl<T> LogError for Result<T, Error> {
    fn log(&self){
        match self {
            Ok(_) => (),
            Err(Error::Static(m)) => error!("{}", m),
            Err(Error::Owned(m)) => error!("{}", m),
         //   Err(Error::IO(e)) => error!("{}", e),
        }
    }
}

impl From<& 'static str> for Error {
    fn from(m: &'static str) -> Self {
        Error::Static(m)
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