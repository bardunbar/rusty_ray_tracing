use std::ops::{Add, Mul, AddAssign, Div, Sub, Neg, DivAssign};
use quad_rand::gen_range;

#[repr(C)]
#[derive(Default, Copy, Clone)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z:f32) -> Self {
        Vec3 { x, y, z }
    }

    #[inline]
    pub fn length(&self) -> f32 {
        f32::sqrt(self.length_squared())
    }

    #[inline]
    pub fn length_squared(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn make_unit_vector(&mut self) {
        let k = 1.0 / self.length_squared();
        self.x *= k;
        self.y *= k;
        self.z *= k;
    }

    pub fn unit_vector(v: &Vec3) -> Vec3 {
        *v / v.length()
    }

    pub fn dot(a: &Vec3, b: &Vec3) -> f32 {
        a.x * b.x + a.y * b.y + a.z * b.z
    }

    pub fn cross(a: &Vec3, b: &Vec3) -> Vec3 {
        let x = a.y * b.z - a.z * b.y;
        let y = a.z * b.x - a.x * b.z;
        let z = a.x * b.y - a.y * b.x;
        Vec3::new(x, y, z)
    }

    pub fn random() -> Vec3 {
        let mut result = Vec3::default();
        while {
            result.x = gen_range::<f32>(-1.0, 1.0);
            result.y = gen_range::<f32>(-1.0, 1.0);
            result.z = gen_range::<f32>(-1.0, 1.0);

            result.length_squared() > 1.0
        } {}
        result
    }

    pub fn random_in_disk() -> Vec3 {
        let mut p = Vec3::default();
        while {
            p.x = gen_range(0.0, 1.0);
            p.y = gen_range(0.0, 1.0);
            p = p - Vec3::new(1.0, 1.0, 0.0);
            Vec3::dot(&p, &p) >= 1.0
        } {}
        p
    }

    pub fn reflect(incoming: &Vec3, normal: &Vec3) -> Vec3 {
        *incoming - 2.0 * Vec3::dot(incoming, normal) * normal
    }

    pub fn refract(incoming: &Vec3, normal: &Vec3, ni_over_nt: f32, out_refracted: &mut Vec3) -> bool {
        let normalized = Vec3::unit_vector(incoming);
        let dt = Vec3::dot(&normalized, normal);
        let discriminant = 1.0 - ni_over_nt * ni_over_nt * (1.0 - dt * dt);
        if discriminant > 0.0 {
            *out_refracted = ni_over_nt * (normalized - *normal * dt) - *normal * f32::sqrt(discriminant);
            true
        } else {
            false
        }
    }
}

impl Add<Vec3> for Vec3 {
    type Output = Vec3;
    fn add(self, rhs: Self) -> Self {
        Vec3 {x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z}
    }
}

impl AddAssign<Vec3> for Vec3 {
    fn add_assign(&mut self, rhs: Vec3) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Self::Output {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}

impl Sub<Vec3> for Vec3 {
    type Output = Vec3;
    fn sub(self, rhs: Vec3) -> Self::Output {
        Vec3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Sub<&Vec3> for Vec3 {
    type Output = Vec3;
    fn sub(self, rhs: &Vec3) -> Self::Output {
        Vec3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Sub<&Vec3> for &Vec3 {
    type Output = Vec3;
    fn sub(self, rhs: &Vec3) -> Self::Output {
        *self - *rhs
    }
}

impl Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, rhs: f32) -> Self::Output {
        Vec3 { x: self.x * rhs, y: self.y * rhs, z: self.z * rhs }
    }
}

impl Mul<Vec3> for f32 {
    type Output = Vec3;
    fn mul(self, rhs: Vec3) -> Self::Output {
        rhs * self
    }
}

impl Mul<&Vec3> for f32 {
    type Output = Vec3;
    fn mul(self, rhs: &Vec3) -> Self::Output {
        *rhs * self
    }
}

impl Mul<Vec3> for Vec3 {
    type Output = Vec3;
    fn mul(self, rhs: Vec3) -> Self::Output {
        Vec3::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }
}

impl Div<f32> for Vec3 {
    type Output = Vec3;
    fn div(self, rhs: f32) -> Self::Output {
        Vec3 { x: self.x / rhs, y: self.y / rhs, z: self.z / rhs}
    }
}

impl DivAssign<f32> for Vec3 {
    fn div_assign(&mut self, rhs: f32) {
        self.x /= rhs;
        self.y /= rhs;
        self.z /= rhs;
    }
}



#[derive(Default)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Ray { origin, direction }
    }

    pub fn origin(&self) -> Vec3 {
        self.origin
    }

    pub fn direction(&self) -> Vec3 {
        self.direction
    }

    pub fn point_at_parameter(&self, t: f32) -> Vec3 {
        self.origin + (self.direction * t)
    }
}

pub fn random() -> f32 {
    gen_range(0.0, 1.0)
}