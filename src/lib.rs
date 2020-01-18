#[derive(Default, Debug, Clone)]
pub struct Vertex {
    pub position: [f32; 2],
}

#[cfg(test)]
mod tests {
    use vulkano::command_buffer::CommandBuffer;
    use vulkano::device::Features;
    use vulkano::format::{ClearValue, Format};
    use vulkano::image::{Dimensions, StorageImage};
    use vulkano::instance::InstanceExtensions;

    #[test]
    fn clear_and_export_blue_image() {
        println!("HELLO");
        let inst = Instance::new(None, &InstanceExtensions::none(), None)
            .expect("Failed to create vulkan instance");

        let physical = PhysicalDevice::enumerate(&inst)
            .next()
            .expect("Oops, there is no device");

        let queue_family = physical
            .queue_families()
            .find(|q| q.supports_graphics())
            .expect("There is no queue family that support graphics");

        let (device, mut queues) = Device::new(
            physical,
            &Features::none(),
            &DeviceExtensions::none(),
            [(queue_family, 0.5)].iter().cloned(),
        )
        .expect("Could not create device.");

        let queue = queues.next().unwrap();

        let image = StorageImage::new(
            device.clone(),
            Dimensions::Dim2d {
                width: 1024,
                height: 1024,
            },
            Format::R8G8B8A8Unorm,
            Some(queue.family()),
        )
        .unwrap();

        let buf = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            (0..1024 * 1024 * 4).map(|_| 0u8),
        )
        .expect("failed to create buffer");

        let command_buffer = AutoCommandBufferBuilder::new(device.clone(), queue.family())
            .unwrap()
            .clear_color_image(image.clone(), ClearValue::Float([0.0, 0.0, 1.0, 1.0]))
            .unwrap()
            .copy_image_to_buffer(image.clone(), buf.clone())
            .unwrap()
            .build()
            .unwrap();

        let finished = command_buffer.execute(queue.clone()).unwrap();

        finished
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        use image::{ImageBuffer, Rgba};

        let buffer_content = buf.read().unwrap();
        let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();

        std::fs::create_dir_all("test_dump").unwrap();
        image.save("test_dump/blue.png").unwrap();
    }

    use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
    use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
    use vulkano::device::{Device, DeviceExtensions};
    use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass};
    use vulkano::image::SwapchainImage;
    use vulkano::instance::{Instance, PhysicalDevice};
    use vulkano::pipeline::viewport::Viewport;
    use vulkano::pipeline::GraphicsPipeline;
    use vulkano::swapchain;
    use vulkano::swapchain::{
        AcquireError, ColorSpace, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError,
    };
    use vulkano::sync;
    use vulkano::sync::{FlushError, GpuFuture};

    use vulkano_win::VkSurfaceBuild;

    use winit::{Event, EventsLoop, Window, WindowBuilder, WindowEvent};

    use std::sync::Arc;

    #[test]
    fn triangle() {
        crate::window::main_loop();
    }
}

pub mod window;
