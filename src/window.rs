use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{
    AutoCommandBuffer, AutoCommandBufferBuilder, CommandBufferExecFuture, DynamicState,
};
use vulkano::device::{Device, DeviceExtensions, Queue};
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass};
use vulkano::image::{AttachmentImage, SwapchainImage};
use vulkano::instance::{Instance, PhysicalDevice, QueueFamily};
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::swapchain;
use vulkano::swapchain::{
    AcquireError, PresentMode, SurfaceTransform, Swapchain, SwapchainAcquireFuture,
    SwapchainCreationError,
};
use vulkano::sync;
use vulkano::sync::{FlushError, GpuFuture, JoinFuture};

use vulkano_win::VkSurfaceBuild;

use winit::{Event, EventsLoop, WindowBuilder, WindowEvent};

use std::borrow::BorrowMut;
use std::clone::Clone;
use std::sync::Arc;
use std::time::Instant;

#[derive(Default, Debug, Clone)]
struct TestVertex {
    pub position: [f32; 2],
}

pub fn main_loop<Window, F>()
where
    F: Frame,
    Window: crate::window::Window<F>,
{
    // The first step of any Vulkan program is to create an instance.
    let instance = {
        // When we create an instance, we have to pass a list of extensions that we want to enable.
        //
        // All the window-drawing functionalities are part of non-core extensions that we need
        // to enable manually. To do so, we ask the `vulkano_win` crate for the list of extensions
        // required to draw to a window.
        let extensions = vulkano_win::required_extensions();

        // Now creating the instance.
        Instance::new(None, &extensions, None).unwrap()
    };

    // We then choose which physical device to use.
    //
    // In a real application, there are three things to take into consideration:
    //
    // - Some devices may not support some of the optional features that may be required by your
    //   application. You should filter out the devices that don't support your app.
    //
    // - Not all devices can draw to a certain surface. Once you create your window, you have to
    //   choose a device that is capable of drawing to it.
    //
    // - You probably want to leave the choice between the remaining devices to the user.
    //
    // For the sake of the example we are just going to use the first device, which should work
    // most of the time.
    //TODO Add proper device selection
    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
    // Some little debug infos.
    println!(
        "Using device: {} (type: {:?})",
        physical.name(),
        physical.ty()
    );

    // The objective of this example is to draw a triangle on a window. To do so, we first need to
    // create the window.
    //
    // This is done by creating a `WindowBuilder` from the `winit` crate, then calling the
    // `build_vk_surface` method provided by the `VkSurfaceBuild` trait from `vulkano_win`. If you
    // ever get an error about `build_vk_surface` being undefined in one of your projects, this
    // probably means that you forgot to import this trait.
    //
    // This returns a `vulkano::swapchain::Surface` object that contains both a cross-platform winit
    // window and a cross-platform Vulkan surface that represents the surface of the window.
    let mut events_loop = EventsLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&events_loop, instance.clone())
        .unwrap();
    let window = surface.window();

    // The next step is to choose which GPU queue will execute our draw commands.
    //
    // Devices can provide multiple queues to run commands in parallel (for example a draw queue
    // and a compute queue), similar to CPU threads. This is something you have to have to manage
    // manually in Vulkan.
    //
    // In a real-life application, we would probably use at least a graphics queue and a transfers
    // queue to handle data transfers in parallel. In this example we only use one queue.
    //
    // We have to choose which queues to use early on, because we will need this info very soon.
    let queue_family = physical
        .queue_families()
        .find(|&q| {
            // We take the first queue that supports drawing to our window.
            q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
        })
        .unwrap();

    // Now initializing the device. This is probably the most important object of Vulkan.
    //
    // We have to pass five parameters when creating a device:
    //
    // - Which physical device to connect to.
    //
    // - A list of optional features and extensions that our program needs to work correctly.
    //   Some parts of the Vulkan specs are optional and must be enabled manually at device
    //   creation. In this example the only thing we are going to need is the `khr_swapchain`
    //   extension that allows us to draw to a window.
    //
    // - A list of layers to enable. This is very niche, and you will usually pass `None`.
    //
    // - The list of queues that we are going to use. The exact parameter is an iterator whose
    //   items are `(Queue, f32)` where the floating-point represents the priority of the queue
    //   between 0.0 and 1.0. The priority of the queue is a hint to the implementation about how
    //   much it should prioritize queues between one another.
    //
    // The list of created queues is returned by the function alongside with the device.
    let mut device_ext = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };

    Window::get_device_extensions(&mut device_ext);

    let (device, mut queues) = Device::new(
        physical,
        physical.supported_features(),
        &device_ext,
        [(queue_family, 0.5)].iter().cloned(), //TODO add support for user selected queue families
    )
    .unwrap();

    // Since we can request multiple queues, the `queues` variable is in fact an iterator. In this
    // example we use only one queue, so we just retrieve the first and only element of the
    // iterator and throw it away.
    let queue = queues.next().unwrap(); //TODO add support for multiple queues

    // Before we can draw on the surface, we have to create what is called a swapchain. Creating
    // a swapchain allocates the color buffers that will contain the image that will ultimately
    // be visible on the screen. These images are returned alongside with the swapchain.
    let (mut swapchain, images) = {
        // Querying the capabilities of the surface. When we create the swapchain we can only
        // pass values that are allowed by the capabilities.
        let caps = surface.capabilities(physical).unwrap();

        let usage = caps.supported_usage_flags;

        // The alpha mode indicates how the alpha value of the final image will behave. For example
        // you can choose whether the window will be opaque or transparent.
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();

        // Choosing the internal format that the images will have.
        let format = caps.supported_formats[0].0;

        // The dimensions of the window, only used to initially setup the swapchain.
        // NOTE:
        // On some drivers the swapchain dimensions are specified by `caps.current_extent` and the
        // swapchain size must use these dimensions.
        // These dimensions are always the same as the window dimensions
        //
        // However other drivers dont specify a value i.e. `caps.current_extent` is `None`
        // These drivers will allow anything but the only sensible value is the window dimensions.
        //
        // Because for both of these cases, the swapchain needs to be the window dimensions, we just use that.
        let initial_dimensions = if let Some(dimensions) = window.get_inner_size() {
            // convert to physical pixels
            let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
            [dimensions.0, dimensions.1]
        } else {
            // The window no longer exists so exit the application.
            return;
        };

        // Please take a look at the docs for the meaning of the parameters we didn't mention.
        Swapchain::new(
            device.clone(),
            surface.clone(),
            caps.min_image_count,
            format,
            initial_dimensions,
            1,
            usage,
            &queue,
            SurfaceTransform::Identity,
            alpha,
            PresentMode::Immediate, //TODO add custom present modes
            true,
            None,
        )
        .unwrap()
    };

    // The render pass we created above only describes the layout of our framebuffers. Before we
    // can draw we also need to create the actual framebuffers.
    //
    // Since we need to draw to multiple images, we are going to create a different framebuffer for
    // each image.

    // Initialization is finally finished!

    // In some situations, the swapchain will become invalid by itself. This includes for example
    // when the window is resized (as the images of the] swapchain will no longer match the
    // window's) or, on Android, when the application went to the background and goes back to the
    // foreground.
    //
    // In this situation, acquiring a swapchain image or presenting it will return an error.
    // Rendering to an image of that swapchain will not produce any error, but may or may not work.
    // To continue rendering, we need to recreate the swapchain by creating a new swapchain.
    // Here, we remember that we need to do this for the next loop iteration.
    let mut recreate_swapchain = false;

    // In the loop below we are going to submit commands to the GPU. Submitting a command produces
    // an object that implements the `GpuFuture` trait, which holds the resources for as long as
    // they are in use by the GPU.
    //
    // Destroying the `GpuFuture` blocks until the GPU is finished executing it. In order to avoid
    // that, we store the submission of the previous frame here.
    let mut previous_frame_end = Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>;

    let mut win = Window::setup(&device, swapchain.format(), queue_family, &queue);

    let mut framebuffers = win.create_framebuffers(&device, &images);

    let mut totalmillies = 0.0;
    let mut totalcounts = 0;
    let mut last_printout = Instant::now();

    loop {
        //perf
        let framestart = Instant::now();

        // It is important to call this function from time to time, otherwise resources will keep
        // accumulating and you will eventually reach an out of memory error.
        // Calling this function polls various fences in order to determine what the GPU has
        // already processed, and frees the resources that are no longer needed.
        previous_frame_end.cleanup_finished();

        // Whenever the window resizes we need to recreate everything dependent on the window size.
        // In this example that includes the swapchain, the framebuffers and the dynamic state viewport.
        if recreate_swapchain {
            // Get the new dimensions of the window.
            let dimensions = if let Some(dimensions) = window.get_inner_size() {
                let dimensions: (u32, u32) =
                    dimensions.to_physical(window.get_hidpi_factor()).into();
                [dimensions.0, dimensions.1]
            } else {
                return;
            };

            let (new_swapchain, new_images) = match swapchain.recreate_with_dimension(dimensions) {
                Ok(r) => r,
                // This error tends to happen when the user is manually resizing the window.
                // Simply restarting the loop is the easiest way to fix this issue.
                Err(SwapchainCreationError::UnsupportedDimensions) => continue,
                Err(err) => panic!("{:?}", err),
            };

            swapchain = new_swapchain;
            // Because framebuffers contains an Arc on the old swapchain, we need to
            // recreate framebuffers as well.
            framebuffers = win.create_framebuffers(&device, &new_images);

            recreate_swapchain = false;
        }

        // Before we can draw on the output, we have to *acquire* an image from the swapchain. If
        // no image is available (which happens if you submit draw commands too quickly), then the
        // function will block.
        // This operation returns the index of the image that we are allowed to draw upon.
        //
        // This function can block if no image is available. The parameter is an optional timeout
        // after which the function call will return an error.
        let (image_num, acquire_future) =
            match swapchain::acquire_next_image(swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    recreate_swapchain = true;
                    continue;
                }
                Err(err) => panic!("{:?}", err),
            };

        let future = previous_frame_end
            .join(acquire_future)
            .then_execute(
                queue.clone(),
                win.render(&device, queue_family, &framebuffers[image_num]),
            )
            .unwrap()
            // The color output is now expected to contain our triangle. But in order to show it on
            // the screen, we have to *present* the image by calling `present`.
            //
            // This function does not actually present the image immediately. Instead it submits a
            // present command at the end of the queue. This means that it will only be presented once
            // the GPU has finished executing the command buffer that draws the triangle.
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                // This wait is required when using NVIDIA or running on macOS. See https://github.com/vulkano-rs/vulkano/issues/1247
                future.wait(None).unwrap();
                previous_frame_end = Box::new(future) as Box<_>;
            }
            Err(FlushError::OutOfDate) => {
                recreate_swapchain = true;
                previous_frame_end = Box::new(sync::now(device.clone())) as Box<_>;
            }
            Err(e) => {
                println!("{:?}", e);
                previous_frame_end = Box::new(sync::now(device.clone())) as Box<_>;
            }
        }

        // Note that in more complex programs it is likely that one of `acquire_next_image`,
        // `command_buffer::submit`, or `present` will block for some time. This happens when the
        // GPU's queue is full and the driver has to wait until the GPU finished some work.
        //
        // Unfortunately the Vulkan API doesn't provide any way to not wait or to detect when a
        // wait would happen. Blocking may be the desired behavior, but if you don't want to
        // block you should spawn a separate thread dedicated to submissions.

        // Handling the window events in order to close the program when the user wants to close
        // it.
        let mut done = false;
        events_loop.poll_events(|ev| match ev {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => done = true,
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => recreate_swapchain = true,
            _ => (),
        });
        if done {
            return;
        }

        totalcounts += 1;
        totalmillies += framestart.elapsed().as_secs_f64() * 1000.0;

        if last_printout.elapsed().as_secs() >= 8 {
            println!("Avg Frametime: {}", totalmillies / f64::from(totalcounts));
            totalcounts = 0;
            totalmillies = 0.0;
            last_printout = Instant::now();
        }
    }
}

pub trait Frame {
    fn get_framebuffer(&self) -> &Arc<dyn FramebufferAbstract + Send + Sync>;
}

pub trait Window<F>
where
    F: Frame,
{
    /*
    Used to request for device extensions.
    TODO Add fail case.
    */
    fn get_device_extensions(extensions: &mut DeviceExtensions) {}

    fn setup(
        device: &Arc<Device>,
        swapchain_format: vulkano::format::Format,
        graphics_family: QueueFamily,
        graphics_queue: &Arc<Queue>,
    ) -> Self;

    fn get_dynamic_state_ref(&mut self) -> &mut DynamicState;
    fn get_render_pass(&mut self) -> &Arc<dyn RenderPassAbstract + Sync + Send>;

    fn create_framebuffers(
        &mut self,
        device: &Arc<Device>,
        images: &[Arc<SwapchainImage<winit::Window>>],
    ) -> Vec<F>;

    fn render(
        &mut self,
        device: &Arc<Device>,
        queue_family: QueueFamily,
        framebuffer: &F,
    ) -> AutoCommandBuffer;
}

pub struct DemoTriangleRenderer {
    vertex_buffer: Arc<CpuAccessibleBuffer<[TestVertex]>>,
    render_pass: Arc<dyn RenderPassAbstract + Sync + Send>,
    pipeline: Arc<dyn GraphicsPipelineAbstract + Sync + Send>,
    dynamic_state: DynamicState,
}

pub struct TriangleFrame {
    framebuffer: Arc<dyn FramebufferAbstract + Send + Sync>,
}

impl Frame for TriangleFrame {
    fn get_framebuffer(&self) -> &Arc<dyn FramebufferAbstract + Send + Sync> {
        &self.framebuffer
    }
}

impl Window<TriangleFrame> for DemoTriangleRenderer {
    fn setup(
        device: &Arc<Device>,
        swapchain_format: vulkano::format::Format,
        _graphics_family: QueueFamily,
        _graphics_queue: &Arc<Queue>,
    ) -> Self {
        let vertex_buffer = {
            CpuAccessibleBuffer::<[TestVertex]>::from_iter(
                device.clone(),
                BufferUsage::all(),
                [
                    TestVertex {
                        position: [-0.5, -0.25],
                    },
                    TestVertex {
                        position: [0.0, 0.5],
                    },
                    TestVertex {
                        position: [0.25, -0.1],
                    },
                ]
                .iter()
                .cloned(),
            )
            .unwrap()
        };

        let render_pass = Arc::new(
            vulkano::single_pass_renderpass!(
                device.clone(),
                attachments: {
                    // `color` is a custom name we give to the first and only attachment.
                    color: {
                        // `load: Clear` means that we ask the GPU to clear the content of this
                        // attachment at the start of the drawing.
                        load: Clear,
                        // `store: Store` means that we ask the GPU to store the output of the draw
                        // in the actual image. We could also ask it to discard the result.
                        store: Store,
                        // `format: <ty>` indicates the type of the format of the image. This has to
                        // be one of the types of the `vulkano::format` module (or alternatively one
                        // of your structs that implements the `FormatDesc` trait). Here we use the
                        // same format as the swapchain.
                        format: swapchain_format,
                        samples: 1,
                    },
                    depth: {
                        load: Clear,
                        store: DontCare,
                        format: vulkano::format::Format::D32Sfloat,
                        samples: 1,
                    }
                },
                pass: {
                    // We use the attachment named `color` as the one and only color attachment.
                    color: [color],
                    // No depth-stencil attachment is indicated with empty brackets.
                    depth_stencil: {depth}
                }
            )
            .unwrap(),
        );

        vulkano::impl_vertex!(TestVertex, position);
        mod vs {
            vulkano_shaders::shader! {
                ty: "vertex",
                src: "
#version 450
layout(location = 0) in vec2 position;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}"
            }
        }

        mod fs {
            vulkano_shaders::shader! {
                ty: "fragment",
                src: "
#version 450
layout(location = 0) out vec4 f_color;
void main() {
    f_color = vec4(1.0, 0.0, 0.0, 1.0);
}
"
            }
        }

        let vs = vs::Shader::load(device.clone()).unwrap();
        let fs = fs::Shader::load(device.clone()).unwrap();

        let pipeline = Arc::new(
            GraphicsPipeline::start()
                // We need to indicate the layout of the vertices.
                // The type `SingleBufferDefinition` actually contains a template parameter corresponding
                // to the type of each vertex. But in this code it is automatically inferred.
                .vertex_input_single_buffer::<TestVertex>()
                // A Vulkan shader can in theory contain multiple entry points, so we have to specify
                // which one. The `main` word of `main_entry_point` actually corresponds to the name of
                // the entry point.
                .vertex_shader(vs.main_entry_point(), ())
                // The content of the vertex buffer describes a list of triangles.
                .triangle_list()
                // Use a resizable viewport set to draw over the entire window
                .viewports_dynamic_scissors_irrelevant(1)
                // See `vertex_shader`.
                .fragment_shader(fs.main_entry_point(), ())
                // We have to indicate which subpass of which render pass this pipeline is going to be used
                // in. The pipeline will only be usable from this particular subpass.
                .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                // Now that our builder is filled, we call `build()` to obtain an actual pipeline.
                .build(device.clone())
                .unwrap(),
        );

        let dynamic_state = DynamicState {
            line_width: None,
            viewports: None,
            scissors: None,
            compare_mask: None,
            write_mask: None,
            reference: None,
        };

        DemoTriangleRenderer {
            vertex_buffer,
            render_pass,
            pipeline,
            dynamic_state,
        }
    }
    fn get_dynamic_state_ref(&mut self) -> &mut DynamicState {
        self.dynamic_state.borrow_mut()
    }

    fn get_render_pass(&mut self) -> &Arc<dyn RenderPassAbstract + Sync + Send> {
        self.render_pass.borrow_mut()
    }

    fn create_framebuffers(
        &mut self,
        device: &Arc<Device>,
        images: &[Arc<SwapchainImage<winit::Window>>],
    ) -> Vec<TriangleFrame> {
        let dimensions = images[0].dimensions();

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [dimensions[0] as f32, dimensions[1] as f32],
            depth_range: 0.0..1.0,
        };
        self.get_dynamic_state_ref().viewports = Some(vec![viewport]);

        let depth_buffer = AttachmentImage::transient(
            device.clone(),
            dimensions,
            vulkano::format::Format::D32Sfloat,
        )
        .unwrap();

        images
            .iter()
            .map(|image| TriangleFrame {
                framebuffer: Arc::new(
                    Framebuffer::start(self.get_render_pass().clone())
                        .add(image.clone())
                        .unwrap()
                        .add(depth_buffer.clone())
                        .unwrap()
                        .build()
                        .unwrap(),
                ),
            })
            .collect::<Vec<_>>()
    }

    fn render(
        &mut self,
        device: &Arc<Device>,
        queue_family: QueueFamily,
        framebuffer: &TriangleFrame,
    ) -> AutoCommandBuffer {
        // We now create a buffer that will store the shape of our triangle.

        // Specify the color to clear the framebuffer with i.e. blue
        let clear_values = vec![[0.0, 0.0, 0.2, 1.0].into(), 1f32.into()];

        // In order to draw, we have to build a *command buffer*. The command buffer object holds
        // the list of commands that are going to be executed.
        //
        // Building a command buffer is an expensive operation (usually a few hundred
        // microseconds), but it is known to be a hot path in the driver and is expected to be
        // optimized.
        //
        // Note that we have to pass a queue family when we create the command buffer. The command
        // buffer will only be executable on that given queue family.
        let command_buffer =
            AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue_family)
                .unwrap()
                // Before we can draw, we have to *enter a render pass*. There are two methods to do
                // this: `draw_inline` and `draw_secondary`. The latter is a bit more advanced and is
                // not covered here.
                //
                // The third parameter builds the list of values to clear the attachments with. The API
                // is similar to the list of attachments when building the framebuffers, except that
                // only the attachments that use `load: Clear` appear in the list.
                .begin_render_pass(framebuffer.framebuffer.clone(), false, clear_values)
                .unwrap()
                // We are now inside the first subpass of the render pass. We add a draw command.
                //
                // The last two parameters contain the list of resources to pass to the shaders.
                // Since we used an `EmptyPipeline` object, the objects have to be `()`.
                .draw(
                    self.pipeline.clone(),
                    &self.dynamic_state,
                    vec![self.vertex_buffer.clone()],
                    (),
                    (),
                )
                .unwrap()
                // We leave the render pass by calling `draw_end`. Note that if we had multiple
                // subpasses we could have called `next_inline` (or `next_secondary`) to jump to the
                // next subpass.
                .end_render_pass()
                .unwrap()
                // Finish building the command buffer by calling `build`.
                .build()
                .unwrap();

        command_buffer
    }
}
