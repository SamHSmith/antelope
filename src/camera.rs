use cgmath::{Deg, Euler, Matrix4, PerspectiveFov, Quaternion, SquareMatrix, Vector3};

#[derive(Debug, Clone)]
pub struct RenderCamera {
    pub position: Vector3<f64>,
    pub rotation: Quaternion<f64>,
    pub fov: f64,
    pub aspect: f64,
    pub near: f64,
    pub far: f64,
}

impl RenderCamera {
    pub fn to_matrix(&self) -> Matrix4<f64> {
        let mut mat = Matrix4::<f64>::identity();

        mat = mat
            * Matrix4::from(PerspectiveFov {
                aspect: self.aspect,
                near: self.near,
                far: self.far,
                fovy: Deg(self.fov).into(),
            });

        mat = mat * Matrix4::from(-self.rotation);
        mat = mat * Matrix4::from_translation(-self.position);

        mat
    }

    pub fn new() -> RenderCamera {
        RenderCamera {
            position: Vector3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            rotation: Quaternion::from(Euler::new(Deg(0.0), Deg(0.0), Deg(0.0))),
            aspect: 1.0,
            fov: 90.0,
            far: 100000.0,
            near: 0.1,
        }
    }
}
