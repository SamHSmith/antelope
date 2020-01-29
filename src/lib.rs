#[cfg(test)]
mod tests {
    use vulkano::command_buffer::{AutoCommandBuffer, CommandBuffer, CommandBufferExecFuture};
    use vulkano::device::{Features, Queue};
    use vulkano::format::{ClearValue, Format};
    use vulkano::image::{Dimensions, StorageImage};
    use vulkano::instance::{InstanceExtensions, QueueFamily};

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

    use crate::camera::RenderCamera;
    use crate::mesh::{Mesh, MeshCreateInfo, Vertex};
    use crate::window::{DemoTriangleRenderer, Window};
    use cgmath::{Deg, Euler, Matrix4, Quaternion, Vector3};
    use gltf::mesh::Reader;
    use gltf::Gltf;
    use std::borrow::BorrowMut;
    use std::sync::Arc;
    use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, DeviceLocalBuffer};
    use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
    use vulkano::device::{Device, DeviceExtensions};
    use vulkano::framebuffer::{FramebufferAbstract, RenderPassAbstract, Subpass};
    use vulkano::half::f16;
    use vulkano::instance::{Instance, PhysicalDevice};
    use vulkano::pipeline::vertex::SingleBufferDefinition;
    use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
    use vulkano::swapchain::SwapchainAcquireFuture;
    use vulkano::sync::{GpuFuture, JoinFuture};

    #[test]
    fn triangle() {
        crate::window::main_loop::<DemoMeshRenderer>();
    }

    pub struct DemoMeshRenderer {
        vertex_buffer: Arc<DeviceLocalBuffer<[Vertex]>>,
        index_buffer: Arc<DeviceLocalBuffer<[u32]>>,
        render_pass: Arc<dyn RenderPassAbstract + Sync + Send>,
        pipeline: Arc<dyn GraphicsPipelineAbstract + Sync + Send>,
        dynamic_state: DynamicState,
    }

    impl Window for DemoMeshRenderer {
        type RenderReturn = CommandBufferExecFuture<
            JoinFuture<Box<dyn GpuFuture>, SwapchainAcquireFuture<winit::Window>>,
            AutoCommandBuffer,
        >;

        fn setup(
            device: &Arc<Device>,
            swapchain_format: vulkano::format::Format,
            graphics_family: QueueFamily,
            graphics_queue: &Arc<Queue>,
        ) -> Self {
            let verts = vec![
                Vertex {
                    position: [-0.5, -0.9, -0.5],
                    colour: [1.0, 0.0, 0.0],
                    normal: [1.0, 0.0, 0.0],
                    tangent: [1.0, 0.0, 0.0, 1.0],
                    texcoord: [0.0, 0.0],
                },
                Vertex {
                    position: [0.0, 0.5, -0.5],
                    colour: [0.0, 1.0, 0.0],
                    normal: [1.0, 0.0, 0.0],
                    tangent: [1.0, 0.0, 0.0, 1.0],
                    texcoord: [0.0, 0.0],
                },
                Vertex {
                    position: [0.25, -0.1, -0.5],
                    colour: [0.0, 0.0, 1.0],
                    normal: [1.0, 0.0, 0.0],
                    tangent: [1.0, 0.0, 0.0, 1.0],
                    texcoord: [0.0, 0.0],
                },
            ];

            let (mesh, create_buff) = Mesh::create(
                MeshCreateInfo {
                    verticies: verts,
                    indicies: vec![0, 1, 2],
                },
                &device,
                graphics_family,
            );

            create_buff
                .execute(graphics_queue.clone())
                .unwrap()
                .then_signal_fence_and_flush()
                .unwrap()
                .wait(None)
                .unwrap();

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
                        }
                    },
                    pass: {
                        // We use the attachment named `color` as the one and only color attachment.
                        color: [color],
                        // No depth-stencil attachment is indicated with empty brackets.
                        depth_stencil: {}
                    }
                )
                .unwrap(),
            );

            vulkano::impl_vertex!(Vertex, position, colour, normal, tangent, texcoord);
            mod vs {
                vulkano_shaders::shader! {
                    ty: "vertex",
                    src: "
#version 450
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 colour;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec4 tangent;
layout(location = 4) in vec2 texcoord;

layout(location = 0) out vec3 frag_colour;
layout(location = 1) out vec3 frag_normal;
layout(location = 2) out vec4 frag_tangent;
layout(location = 3) out vec2 frag_texcoord;

layout(push_constant) uniform PushConstants {
    dmat4 viewproj;
} push_constants;

void main() {
    gl_Position = vec4(push_constants.viewproj * vec4(position, 1.0));
    
    frag_colour=colour;
    frag_normal=normal;
    frag_tangent=tangent;
    frag_texcoord=texcoord;
}"
                }
            }

            mod fs {
                vulkano_shaders::shader! {
                    ty: "fragment",
                    src: "
#version 450
layout(location = 0) in vec3 colour;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 tangent;
layout(location = 3) in vec2 texcoord;

layout(location = 0) out vec4 f_colour;
void main() {
    f_colour = vec4(colour, 1.0);
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
                    .vertex_input_single_buffer::<Vertex>()
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

            DemoMeshRenderer {
                vertex_buffer: mesh.vertbuff,
                index_buffer: mesh.indbuff,
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

        fn render(
            &mut self,
            device: &Arc<Device>,
            queue_family: QueueFamily,
            framebuffer: &Arc<dyn FramebufferAbstract + Send + Sync>,
        ) -> AutoCommandBuffer {
            // We now create a buffer that will store the shape of our triangle.

            // Specify the color to clear the framebuffer with i.e. blue
            let clear_values = vec![[0.0, 0.0, 0.2, 1.0].into()];

            let cam = RenderCamera {
                position: Vector3 {
                    x: 0.0,
                    y: 0.9,
                    z: 5.5,
                },
                rotation: Quaternion::from(Euler::new(Deg(0.0), Deg(0.0), Deg(0.0))),
                aspect: 1.0,
                fov: 90.0,
                far: 10000.0,
                near: 0.1,
            };

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
                    .begin_render_pass(framebuffer.clone(), false, clear_values)
                    .unwrap()
                    // We are now inside the first subpass of the render pass. We add a draw command.
                    //
                    // The last two parameters contain the list of resources to pass to the shaders.
                    // Since we used an `EmptyPipeline` object, the objects have to be `()`.
                    .draw_indexed(
                        self.pipeline.clone(),
                        &self.dynamic_state,
                        vec![self.vertex_buffer.clone()],
                        self.index_buffer.clone(),
                        (),
                        (cam.to_matrix()),
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
}

pub mod camera;
pub mod mesh;
pub mod window;
