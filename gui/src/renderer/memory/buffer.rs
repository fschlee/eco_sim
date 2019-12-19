use super::*;
use gfx_hal::MemoryTypeId;

pub struct BufferBundle<B: Backend> {
    pub buffer: ManuallyDrop<B::Buffer>,
    pub requirements: Requirements,
    pub memory: ManuallyDrop<B::Memory>,
}

impl<B: Backend> BufferBundle<B> {
    pub fn new(
        adapter: &Adapter<B>,
        device: &Dev<B>,
        size: usize,
        usage: BufferUsage,
    ) -> Result<Self, Error> {
        unsafe {
            let mut buffer = device
                .create_buffer(size as u64, usage)
                .map_err(|_| "Couldn't create a buffer!")?;
            let requirements = device.get_buffer_requirements(&buffer);
            let memory_type_id = adapter
                .physical_device
                .memory_properties()
                .memory_types
                .iter()
                .enumerate()
                .find(|&(id, memory_type)| {
                    requirements.type_mask & (1 << id) != 0
                        && memory_type.properties.contains(Properties::CPU_VISIBLE)
                })
                .map(|(id, _)| MemoryTypeId(id))
                .ok_or("Couldn't find a memory type to support the buffer!")?;
            let memory = device
                .allocate_memory(memory_type_id, requirements.size)
                .map_err(|e| {
                    format!(
                        "Couldn't allocate buffer memory({} needed): {:?}",
                        requirements.size, e
                    )
                })?;
            device
                .bind_buffer_memory(&memory, 0, &mut buffer)
                .map_err(|e| format!("Couldn't bind buffer memory: {:?}", e))?;
            Ok(Self {
                buffer: ManuallyDrop::new(buffer),
                requirements,
                memory: ManuallyDrop::new(memory),
            })
        }
    }

    pub unsafe fn write_range<T: Copy>(
        &self,
        device: &B::Device,
        range: std::ops::Range<u64>,
        source: &[T],
    ) -> Result<(), Error> {
        let memory = &*(self.memory);
        let mut target = device.map_memory(memory, range.clone())?;
        std::slice::from_raw_parts_mut(target as *mut T, source.len()).copy_from_slice(source);
        let res = device.flush_mapped_memory_ranges(Some((memory, range)));
        device.unmap_memory(memory);
        res?;
        Ok(())
    }
    pub fn write<T: Sized + Copy>(
        &self,
        device: &B::Device,
        offset: usize,
        source: &[T],
    ) -> Result<(), Error> {
        let range_start = (offset * size_of::<T>()) as u64;
        assert!(range_start + (source.len() * size_of::<T>()) as u64 <= self.requirements.size);
        let range = range_start..self.requirements.size;
        let memory = &*(self.memory);

        unsafe {
            let mut target = device.map_memory(memory, range.clone())?;
            std::slice::from_raw_parts_mut(target as *mut T, source.len()).copy_from_slice(source);
            let res = device.flush_mapped_memory_ranges(Some((memory, range)));
            device.unmap_memory(memory);
            res?;
        }

        Ok(())
    }
    pub fn write_slice<T: Sized + Copy, F: FnMut(&mut [T])>(
        &self,
        device: &B::Device,
        mut writer: F,
    ) -> Result<(), Error> {
        let range = 0..self.requirements.size;
        let memory = &*(self.memory);
        let len = self.requirements.size as usize / size_of::<T>();

        unsafe {
            let mut target = device.map_memory(memory, range.clone())?;
            writer(std::slice::from_raw_parts_mut(target as *mut T, len));
            let res = device.flush_mapped_memory_ranges(Some((memory, range)));
            device.unmap_memory(memory);
            res?;
        }

        Ok(())
    }
    pub unsafe fn manually_drop(&self, device: &Dev<B>) {
        use core::ptr::read;
        device.destroy_buffer(ManuallyDrop::into_inner(read(&self.buffer)));
        device.free_memory(ManuallyDrop::into_inner(read(&self.memory)));
    }
}
