use cgmath::{Deg, Matrix4, PerspectiveFov, Quaternion, SquareMatrix, Vector3};

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
}
