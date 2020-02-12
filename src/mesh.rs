use cgmath::{Deg, Euler, Matrix4, Quaternion, Vector3};

use std::ops::BitOr;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};

use crate::camera::RenderCamera;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, DeviceLocalBuffer};
use vulkano::command_buffer::{AutoCommandBuffer, AutoCommandBufferBuilder};
use vulkano::device::Device;
use vulkano::instance::QueueFamily;

#[derive(Default, Debug, Clone)]
pub struct Vertex {
    pub position: [f32; 3],
    pub colour: [f32; 4],
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

pub struct RawMesh {
    pub indbuff: Arc<DeviceLocalBuffer<[u32]>>,
    pub vertbuff: Arc<DeviceLocalBuffer<[Vertex]>>,
}

pub struct Mesh {
    raw: Mutex<Option<RawMesh>>,
}
impl Mesh {
    #[inline]
    pub fn get_raw(&self) -> Option<RawMesh> {
        let lock = self.raw.lock().unwrap();

        return if lock.is_none() {
            None
        } else {
            Some(lock.as_ref().unwrap().clone())
        };
    }
}

pub struct RenderInfo {
    pub meshes: Vec<Arc<Mesh>>,
    pub mats: Vec<Matrix4<f64>>,
    pub camera: RenderCamera,
}

impl RenderInfo {
    pub fn empty() -> Self {
        RenderInfo {
            meshes: Vec::new(),
            mats: Vec::new(),
            camera: RenderCamera {
                position: Vector3 {
                    x: 0.0,
                    y: 0.0,
                    z: 5.0,
                },
                rotation: Quaternion::from(Euler::new(Deg(0.0), Deg(0.0), Deg(0.0))),
                aspect: 1.0,
                fov: 90.0,
                far: 10000.0,
                near: 0.1,
            },
        }
    }
}

impl RawMesh {
    pub fn create(
        info: MeshCreateInfo,
        device: &Arc<Device>,
        queue_family: QueueFamily,
    ) -> (RawMesh, AutoCommandBuffer) {
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
            RawMesh {
                indbuff: di,
                vertbuff: dv,
            },
            cmd.build().unwrap(),
        )
    }
}

impl Clone for RawMesh {
    fn clone(&self) -> Self {
        RawMesh {
            vertbuff: self.vertbuff.clone(),
            indbuff: self.indbuff.clone(),
        }
    }
}

pub struct MeshFactory {
    sender: Mutex<Sender<(Arc<Mesh>, MeshCreateInfo)>>,
    receiver: Mutex<Receiver<(Arc<Mesh>, MeshCreateInfo)>>,
}

impl MeshFactory {
    pub fn new() -> Self {
        let (tx, rx) = channel::<(Arc<Mesh>, MeshCreateInfo)>();

        MeshFactory {
            sender: Mutex::new(tx),
            receiver: Mutex::new(rx),
        }
    }

    pub fn create_mesh(&self, info: MeshCreateInfo) -> Arc<Mesh> {
        let mesh = Arc::new(Mesh {
            raw: Mutex::new(Option::None),
        });
        self.sender
            .lock()
            .unwrap()
            .send((mesh.clone(), info))
            .unwrap();
        mesh
    }

    pub fn perform_mesh_creation(
        &self,
        device: &Arc<Device>,
        queue_family: QueueFamily,
    ) -> AutoCommandBuffer {
        let mut buffers = Vec::<AutoCommandBuffer>::new();
        let receiver = self.receiver.lock().unwrap();

        loop {
            let recv = receiver.try_recv();
            if recv.is_err() {
                //TODO timeout maybe???
                break;
            }
            let (mesh, info) = recv.unwrap();

            let (raw, create_buff) = RawMesh::create(info, &device, queue_family);

            buffers.push(create_buff);
            let mut reff = mesh.raw.lock().unwrap();
            *reff = Some(raw);
        }
        let mut builder =
            AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue_family)
                .unwrap();
        unsafe {
            for buff in buffers {
                builder = builder.execute_commands(buff).unwrap();
            }
        }
        builder.build().unwrap()
    }
}
