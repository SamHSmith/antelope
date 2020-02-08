#[cfg(test)]
mod tests {
    use vulkano::command_buffer::{AutoCommandBuffer, CommandBuffer, CommandBufferExecFuture};
    use vulkano::device::{Features, Queue};
    use vulkano::format::{ClearValue, Format};
    use vulkano::image::{
        AttachmentImage, Dimensions, ImageLayout, ImageUsage, StorageImage, SwapchainImage,
    };
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
    use crate::mesh::{Mesh, MeshCreateInfo, PostVertex, RenderInfo, Vertex};
    use crate::window::{DemoTriangleRenderer, Frame, TriangleFrame, Window};
    use cgmath::{Deg, Euler, Matrix4, Quaternion, SquareMatrix, Vector3};
    use gltf::mesh::Reader;
    use gltf::Gltf;
    use std::borrow::{Borrow, BorrowMut};
    use std::sync::{Arc, Mutex};
    use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, DeviceLocalBuffer};

    use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
    use vulkano::device::{Device, DeviceExtensions};
    use vulkano::framebuffer::{FramebufferAbstract, RenderPassAbstract, Subpass};

    use vulkano::instance::{Instance, PhysicalDevice};

    use vulkano::pipeline::viewport::Viewport;
    use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};

    use std::time::Duration;
    use vulkano::descriptor::descriptor::DescriptorDesc;
    use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
    use vulkano::descriptor::PipelineLayoutAbstract;
    use vulkano::sync::GpuFuture;
    use winit::os::unix::x11::util::PointerState;

    #[test]
    fn triangle() {
        let (thread, win) = crate::window::main_loop::<DemoTriangleRenderer, TriangleFrame>();
        std::thread::sleep(Duration::new(2, 0));
        win.stop();
    }

    #[test]
    fn mesh() {
        let (thread, win) = crate::window::main_loop::<DemoMeshRenderer, MeshFrame>();
        thread.join().unwrap_err();
    }

    pub struct DemoMeshRenderer {
        render_info: RenderInfo,
        render_pass: Mutex<Arc<dyn RenderPassAbstract + Sync + Send>>,
        pipeline: [Arc<dyn GraphicsPipelineAbstract + Sync + Send>; 2],
        pipeline_layout: [Arc<dyn PipelineLayoutAbstract + Send + Sync>; 2],
        dynamic_state: Mutex<DynamicState>,
        should_stop: Mutex<bool>,
    }

    struct MeshFrame {
        framebuffer: Arc<dyn FramebufferAbstract + Send + Sync>,
        albedobuffer: Arc<AttachmentImage>,
        normalbuffer: Arc<AttachmentImage>,
        depthbuffer: Arc<AttachmentImage>,
    }

    impl Frame for MeshFrame {
        fn get_framebuffer(&self) -> &Arc<dyn FramebufferAbstract + Send + Sync> {
            &self.framebuffer
        }
    }

    impl Window<MeshFrame> for DemoMeshRenderer {
        fn get_device_extensions(extensions: &mut DeviceExtensions) {
            extensions.khr_storage_buffer_storage_class = true;
        }

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
                Vertex {
                    position: [-1.5, -0.9, -0.9],
                    colour: [1.0, 1.0, 0.0],
                    normal: [1.0, 1.0, 0.0],
                    tangent: [1.0, 0.0, 0.0, 1.0],
                    texcoord: [0.0, 0.0],
                },
                Vertex {
                    position: [-1.0, 0.5, -0.9],
                    colour: [1.0, 1.0, 0.0],
                    normal: [1.0, 1.0, 0.0],
                    tangent: [1.0, 0.0, 0.0, 1.0],
                    texcoord: [0.0, 0.0],
                },
                Vertex {
                    position: [0.2, -0.1, -0.9],
                    colour: [1.0, 1.0, 0.0],
                    normal: [1.0, 0.0, 0.0],
                    tangent: [1.0, 0.0, 0.0, 1.0],
                    texcoord: [0.0, 0.0],
                },
            ];

            let (mesh, create_buff) = Mesh::create(
                MeshCreateInfo {
                    verticies: verts,
                    indicies: vec![0, 1, 2, 3, 4, 5],
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
                vulkano::ordered_passes_renderpass!(
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
                            store: Store,
                            format: Format::D32Sfloat,
                            samples: 1,
                        },
                        albedo: {
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
                            format: Format::R32G32B32A32Sfloat,
                            samples: 1,
                        },
                        normals: {
                            load: Clear,
                            store: Store,
                            format: Format::R16G16B16A16Sfloat,
                            samples: 1,
                        }
                    },
                    passes: [
                    {
                        // We use the attachment named `color` as the one and only color attachment.
                        color: [albedo,normals],
                        // No depth-stencil attachment is indicated with empty brackets.
                        depth_stencil: {depth},
                        input: []
                    },
                    {
                        // We use the attachment named `color` as the one and only color attachment.
                        color: [color],
                        // No depth-stencil attachment is indicated with empty brackets.
                        depth_stencil: {},
                        input: [albedo,normals,depth]
                    }
                    ]
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
    uint index;
} push_constants;

layout(binding = 0) uniform CameraBlock
{
    dmat4 viewproj;
} camera;

layout(binding = 1) buffer TransformBlock
{
    dmat4 mat[];
} transform;

void main() {
    gl_Position = vec4(camera.viewproj * transform.mat[push_constants.index] * vec4(position, 1.0));
    
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
layout(location = 1) out vec4 f_normal;

void main() {
    f_colour = vec4(colour, 1.0);
    f_normal = vec4(normal, 0.0);
}
"
                }
            }

            let vs = vs::Shader::load(device.clone()).unwrap();
            let fs = fs::Shader::load(device.clone()).unwrap();

            let pipeline1 = Arc::new(
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
                    .depth_stencil_simple_depth()
                    .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                    // Now that our builder is filled, we call `build()` to obtain an actual pipeline.
                    .build(device.clone())
                    .unwrap(),
            );

            vulkano::impl_vertex!(PostVertex, pos);

            mod vs2 {
                vulkano_shaders::shader! {
                    ty: "vertex",
                    src: "
#version 450
layout(location = 0) in vec2 pos;

void main() {
    gl_Position = vec4(pos,0.0,1.0);
}"
                }
            }

            mod fs2 {
                vulkano_shaders::shader! {
                    ty: "fragment",
                    src: "
#version 450

layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput u_diffuse;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput u_normal;
layout(input_attachment_index = 2, set = 0, binding = 2) uniform subpassInput u_depth;

layout(location = 0) out vec4 f_colour;

void main() {
    if(subpassLoad(u_depth).x==1)
        discard;

    f_colour = vec4(subpassLoad(u_normal).xyz, 1.0);
}
"
                }
            }

            let vs2 = vs2::Shader::load(device.clone()).unwrap();
            let fs2 = fs2::Shader::load(device.clone()).unwrap();

            let pipeline2 = Arc::new(
                GraphicsPipeline::start()
                    // We need to indicate the layout of the vertices.
                    // The type `SingleBufferDefinition` actually contains a template parameter corresponding
                    // to the type of each vertex. But in this code it is automatically inferred.
                    .vertex_input_single_buffer::<PostVertex>()
                    // A Vulkan shader can in theory contain multiple entry points, so we have to specify
                    // which one. The `main` word of `main_entry_point` actually corresponds to the name of
                    // the entry point.
                    .vertex_shader(vs2.main_entry_point(), ())
                    // The content of the vertex buffer describes a list of triangles.
                    .triangle_list()
                    // Use a resizable viewport set to draw over the entire window
                    .viewports_dynamic_scissors_irrelevant(1)
                    // See `vertex_shader`.
                    .fragment_shader(fs2.main_entry_point(), ())
                    // We have to indicate which subpass of which render pass this pipeline is going to be used
                    // in. The pipeline will only be usable from this particular subpass.
                    //.depth_stencil_simple_depth()
                    .render_pass(Subpass::from(render_pass.clone(), 1).unwrap())
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
                render_info: RenderInfo {
                    meshes: vec![mesh.clone(), mesh.clone()],
                    mats: vec![
                        Matrix4::from_translation(Vector3 {
                            x: 0.0,
                            y: 2.0,
                            z: 0.0,
                        }),
                        Matrix4::from_translation(Vector3 {
                            x: 3.0,
                            y: 2.0,
                            z: 0.0,
                        }),
                    ],
                },
                render_pass: Mutex::new(render_pass),
                pipeline: [pipeline1.clone(), pipeline2.clone()],
                pipeline_layout: [pipeline1.clone(), pipeline2.clone()],
                dynamic_state: Mutex::new(dynamic_state),
                should_stop: Mutex::new(false),
            }
        }

        fn get_dynamic_state_ref(&self) -> &Mutex<DynamicState> {
            self.dynamic_state.borrow()
        }

        fn get_render_pass(&self) -> &Mutex<Arc<dyn RenderPassAbstract + Sync + Send>> {
            self.render_pass.borrow()
        }

        fn stop(&self) {
            let mut data = self.should_stop.lock().unwrap();
            *data = true;
        }

        fn should_stop(&self) -> bool {
            *self.should_stop.lock().unwrap()
        }

        fn create_framebuffers(
            &self,
            device: &Arc<Device>,
            images: &[Arc<SwapchainImage<winit::Window>>],
        ) -> Vec<MeshFrame> {
            let dimensions = images[0].dimensions();

            let viewport = Viewport {
                origin: [0.0, 0.0],
                dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                depth_range: 0.0..1.0,
            };
            self.get_dynamic_state_ref().lock().unwrap().viewports = Some(vec![viewport]);

            let mut usage = ImageUsage::none();
            usage.input_attachment = true;
            usage.transient_attachment = true;

            let albedo_buffer = AttachmentImage::with_usage(
                device.clone(),
                dimensions,
                vulkano::format::Format::R32G32B32A32Sfloat,
                usage,
            )
            .unwrap();

            let normal_buffer = AttachmentImage::with_usage(
                device.clone(),
                dimensions,
                vulkano::format::Format::R16G16B16A16Sfloat,
                usage,
            )
            .unwrap();

            let depth_buffer = AttachmentImage::with_usage(
                device.clone(),
                dimensions,
                vulkano::format::Format::D32Sfloat,
                usage,
            )
            .unwrap();

            images
                .iter()
                .map(|image| MeshFrame {
                    framebuffer: Arc::new(
                        vulkano::framebuffer::Framebuffer::start(
                            self.get_render_pass().lock().unwrap().clone(),
                        )
                        .add(image.clone())
                        .unwrap()
                        .add(depth_buffer.clone())
                        .unwrap()
                        .add(albedo_buffer.clone())
                        .unwrap()
                        .add(normal_buffer.clone())
                        .unwrap()
                        .build()
                        .unwrap(),
                    ),
                    albedobuffer: albedo_buffer.clone(),
                    normalbuffer: normal_buffer.clone(),
                    depthbuffer: depth_buffer.clone(),
                })
                .collect::<Vec<_>>()
        }

        fn render(
            &self,
            device: &Arc<Device>,
            queue_family: QueueFamily,
            framebuffer: &MeshFrame,
        ) -> AutoCommandBuffer {
            // We now create a buffer that will store the shape of our triangle.

            // Specify the color to clear the framebuffer with i.e. blue
            let clear_values = vec![
                [0.0, 0.0, 0.2, 1.0].into(),
                1f32.into(),
                [0.0, 0.0, 0.0, 0.0].into(),
                [0.0, 0.0, 0.0, 0.0].into(),
            ];

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

            let post_area = CpuAccessibleBuffer::from_iter(
                device.clone(),
                BufferUsage::all(),
                [
                    PostVertex { pos: [-1.0, -1.0] },
                    PostVertex { pos: [1.0, -1.0] },
                    PostVertex { pos: [-1.0, 1.0] },
                    PostVertex { pos: [-1.0, 1.0] },
                    PostVertex { pos: [1.0, -1.0] },
                    PostVertex { pos: [1.0, 1.0] },
                ]
                .iter()
                .cloned(),
            )
            .unwrap();

            let descriptor_set2 =
                PersistentDescriptorSet::start(self.pipeline_layout[1].clone(), 0)
                    .add_image(framebuffer.albedobuffer.clone())
                    .unwrap()
                    .add_image(framebuffer.normalbuffer.clone())
                    .unwrap()
                    .add_image(framebuffer.depthbuffer.clone())
                    .unwrap()
                    .build()
                    .unwrap();

            let mat: Matrix4<f64> = Matrix4::from_translation(Vector3 {
                x: 0.0,
                y: 4.0,
                z: 0.0,
            });

            let camera_uniform = CpuAccessibleBuffer::from_iter(
                device.clone(),
                BufferUsage::all(),
                [cam.to_matrix()].iter().cloned(),
            )
            .unwrap();

            debug_assert_eq!(self.render_info.mats.len(), self.render_info.meshes.len()); //The code will still render if the assert fails. But this
            let transforms = CpuAccessibleBuffer::from_iter(
                //a good indication of a bug somewhere else in the code.
                device.clone(),
                BufferUsage::all(),
                self.render_info.mats.clone().iter().cloned(),
            )
            .unwrap();

            let descriptor_set1 = Arc::new(
                PersistentDescriptorSet::start(self.pipeline_layout[0].clone(), 0)
                    .add_buffer(camera_uniform)
                    .unwrap()
                    .add_buffer(transforms)
                    .unwrap()
                    .build()
                    .unwrap(),
            );

            let mut command_buffer =
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
                    .unwrap();

            for i in 0..self.render_info.meshes.len() {
                command_buffer = command_buffer
                    .draw_indexed(
                        self.pipeline[0].clone(),
                        &self.dynamic_state.lock().unwrap(),
                        vec![self.render_info.meshes[i].vertbuff.clone()],
                        self.render_info.meshes[i].indbuff.clone(),
                        descriptor_set1.clone(),
                        i,
                    )
                    .unwrap();
            }

            command_buffer = command_buffer
                .next_subpass(false)
                .unwrap()
                .draw(
                    self.pipeline[1].clone(),
                    &self.dynamic_state.lock().unwrap(),
                    vec![post_area.clone()],
                    descriptor_set2,
                    (),
                )
                .unwrap()
                .end_render_pass()
                .unwrap();

            // Finish building the command buffer by calling `build`.
            command_buffer.build().unwrap()
        }
    }
}

pub mod camera;
pub mod mesh;
pub mod window;
