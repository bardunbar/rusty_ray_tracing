use crate::math::*;
use std::f32::consts::PI as PI_F32;

pub struct Camera {
    origin: Vec3,
    lower_left_corner: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
    view_u: Vec3,
    view_v: Vec3,
    lens_radius: f32,
}

impl Camera {
    pub fn new(look_from: &Vec3, look_at: &Vec3, up: &Vec3, v_fov: f32, aspect: f32, aperture: f32, focal_distance: f32) -> Self {

        let theta = v_fov * PI_F32 / 180.0;
        let half_height = f32::tan(theta / 2.0);
        let half_width = aspect * half_height;

        let w = Vec3::unit_vector(&(look_from - look_at));
        let u = Vec3::unit_vector(&Vec3::cross(up, &w));
        let v = Vec3::cross(&w, &u);

        let origin = *look_from;
        let lower_left_corner = origin - (half_width * focal_distance * u) - (half_height * focal_distance * v) - w * focal_distance;
        let horizontal = 2.0 * half_width * focal_distance * u;
        let vertical = 2.0 * half_height * focal_distance * v;

        Camera { 
            // This simple camera needs to change if the aspect ratio changes!!
            origin, 
            lower_left_corner,
            horizontal,
            vertical,
            view_u: u,
            view_v: v,
            lens_radius: aperture / 2.0,
        }
    }

    pub fn get_ray(&self, u: f32, v: f32) -> Ray {
        let rd = self.lens_radius * Vec3::random_in_disk();
        let offset = self.view_u * rd.x + self.view_v * rd.y;
        Ray::new(self.origin + offset, self.lower_left_corner + u * self.horizontal + v * self.vertical - self.origin - offset)
    }
}
