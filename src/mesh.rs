use cgmath::Matrix4;
use std::ops::BitOr;
use std::sync::Arc;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, DeviceLocalBuffer};
use vulkano::command_buffer::{AutoCommandBuffer, AutoCommandBufferBuilder};
use vulkano::device::Device;
use vulkano::instance::QueueFamily;

#[derive(Default, Debug, Clone)]
pub struct Vertex {
    pub position: [f32; 3],
    pub colour: [f32; 3],
    pub normal: [f32; 3],
    pub tangent: [f32; 4],
    pub texcoord: [f32; 2],
}
#[derive(Default, Debug, Clone)]
pub struct PostVertex {
    pub pos: [f32; 2],
}

pub struct MeshCreateInfo {
    pub indicies: Vec<u32>,
    pub verticies: Vec<Vertex>,
}

pub struct Mesh {
    pub indbuff: Arc<DeviceLocalBuffer<[u32]>>,
    pub vertbuff: Arc<DeviceLocalBuffer<[Vertex]>>,
}

pub struct RenderInfo {
    pub meshes: Vec<Arc<Mesh>>,
    pub mats: Vec<Matrix4<f64>>,
}

impl Mesh {
    pub fn create(
        info: MeshCreateInfo,
        device: &Arc<Device>,
        queue_family: QueueFamily,
    ) -> (Arc<Mesh>, AutoCommandBuffer) {
        let i = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::transfer_source(),
            info.indicies.iter().cloned(),
        )
        .unwrap();
        let v = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::transfer_source(),
            info.verticies.iter().cloned(),
        )
        .unwrap();

        let di = DeviceLocalBuffer::<[u32]>::array(
            device.clone(),
            info.indicies.len(),
            BufferUsage::transfer_destination().bitor(BufferUsage::index_buffer()),
            vec![queue_family],
        )
        .unwrap();

        let dv = DeviceLocalBuffer::<[Vertex]>::array(
            device.clone(),
            info.verticies.len(),
            BufferUsage::transfer_destination().bitor(BufferUsage::vertex_buffer()),
            vec![queue_family],
        )
        .unwrap();

        let mut cmd = AutoCommandBufferBuilder::new(device.clone(), queue_family.clone()).unwrap();

        cmd = cmd
            .copy_buffer(i, di.clone())
            .unwrap()
            .copy_buffer(v, dv.clone())
            .unwrap();

        (
            Arc::new(Mesh {
                indbuff: di,
                vertbuff: dv,
            }),
            cmd.build().unwrap(),
        )
    }
}
