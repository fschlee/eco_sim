pub trait BackendExt {
    fn can_push_graphics_constants() -> bool {
        false
    }
}

#[cfg(all(feature = "vulkan", not(macos)))]
impl BackendExt for gfx_backend_vulkan::Backend {
    fn can_push_graphics_constants() -> bool {
        true
    }
}

#[cfg(all(feature = "metal", macos))]
impl BackendExt for gfx_backend_metal::Backend {
    fn can_push_graphics_constants() -> bool {
        true
    }
}

#[cfg(feature = "dx12")]
impl BackendExt for gfx_backend_dx12::Backend {
    fn can_push_graphics_constants() -> bool {
        true
    }
}

#[cfg(feature = "dx11")]
impl BackendExt for gfx_backend_dx11::Backend {
    fn can_push_graphics_constants() -> bool {
        false
    }
}
#[cfg(feature = "gl")]
impl BackendExt for gfx_backend_gl::Backend {
    fn can_push_graphics_constants() -> bool {
        false
    }
}
