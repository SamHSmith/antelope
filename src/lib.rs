pub mod window;

#[macro_use]
extern crate log;

#[cfg(test)]
mod tests {
    use glfw::{Context, Key, Action, Monitor, ffi};
    use glfw::ffi::glfwGetPrimaryMonitor;
    use std::time::Duration;

    #[test]
    fn clear_and_export_blue_image() {
        use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice};
        use vulkano::device::{Device, Features, DeviceExtensions};
        use vulkano::format::{Format, ClearValue};
        use vulkano::image::{Dimensions, StorageImage};
        use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBuffer};
        use vulkano::sync::GpuFuture;
        use vulkano::buffer::{CpuAccessibleBuffer, BufferUsage};

        let inst = Instance::new(None, &InstanceExtensions::none(), None).expect("Failed to create vulkan instance");

        let physical = PhysicalDevice::enumerate(&inst).next().expect("Oops, there is no device");

        let queue_family = physical.queue_families().find(|q| q.supports_graphics()).expect("There is no queue family that support graphics");

        let (device, mut queues) =
            Device::new(physical,
                        &Features::none(),
                        &DeviceExtensions::none(),
                        [(queue_family, 0.5)].iter().cloned()).expect("Could not create device.");

        let queue = queues.next().unwrap();

        let image = StorageImage::new(device.clone(), Dimensions::Dim2d { width: 1024, height: 1024 },
                                      Format::R8G8B8A8Unorm, Some(queue.family())).unwrap();

        let buf = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(),
                                                 (0..1024 * 1024 * 4).map(|_| 0u8)).expect("failed to create buffer");

        let command_buffer = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap()
            .clear_color_image(image.clone(), ClearValue::Float([0.0, 0.0, 1.0, 1.0])).unwrap()
            .copy_image_to_buffer(image.clone(), buf.clone()).unwrap()
            .build().unwrap();

        let finished = command_buffer.execute(queue.clone()).unwrap();

        finished.then_signal_fence_and_flush().unwrap().wait(None).unwrap();

        use image::{ImageBuffer, Rgba};

        let buffer_content = buf.read().unwrap();
        let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();

        std::fs::create_dir_all("test_dump").unwrap();
        image.save("test_dump/blue.png").unwrap();
    }

    #[test]
    fn create_window() {
        use crate::window::VulkanWindow;

        let glfw = crate::antelope::init_glfw();

        let monitor_count=crate::antelope::get_monitor_count(&glfw);

        let mut windows = Vec::<VulkanWindow>::new();

        for i in 0..monitor_count {
            let m = crate::antelope::get_monitor(&glfw, i);

            windows.push(VulkanWindow::new_windowed_fullscreen(&glfw, "Fullscreen",&m));
        }

        std::thread::sleep(Duration::new(5, 0));
    }
}

pub mod antelope {
    use glfw::{Glfw, Monitor, ffi};

    pub fn init_glfw() -> Glfw {
        glfw::init(glfw::LOG_ERRORS).unwrap()
    }

    pub fn get_monitor(glfw: &Glfw, monitor: i32) -> Monitor {
        unsafe {
            let mut count: i32 = 0;
            let monitors = glfw::ffi::glfwGetMonitors(&mut count as *mut i32);
            assert!(monitor < count);

            let ptr = monitors.offset(monitor as isize).read();

            let final_monitor: Monitor = std::mem::transmute(FakeMonitor { ptr });

            return final_monitor;
        }
    }

    pub fn get_monitor_count(glfw: &Glfw) -> i32{
        unsafe {
            let mut count: i32 = 0;
            glfw::ffi::glfwGetMonitors(&mut count as *mut i32);
            return count;
        }
    }

    struct FakeMonitor {
        pub ptr: *mut ffi::GLFWmonitor
    }
}
