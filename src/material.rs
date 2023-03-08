use crate::{math::{Ray, Vec3}, HitRecord};
use quad_rand::gen_range;

// pub trait Material {
//     fn scatter(incoming_ray: &Ray, hit_record: &HitRecord, attenuation: &mut Vec3, scattered: &mut Ray) -> bool;
// }

#[derive(Default)]
pub struct MaterialParams {
    pub albedo: Vec3,
    pub fuzz: f32,
    pub metallicity: f32,
    pub dielectric: f32,
}

#[derive(Default, Clone, Copy)]
pub struct Material {
    albedo: Vec3,
    fuzz: f32,
    metallicity: f32,    
    dielectric: f32,
    // use_dielectric: bool
}

impl Material {
    pub fn new(params: MaterialParams) -> Self {

        let metallicity = params.metallicity.clamp(0.0, 1.0);
        let fuzz = params.fuzz.clamp(0.0, 1.0);
        
        Material { 
            albedo: params.albedo,
            fuzz,
            metallicity,
            dielectric: params.dielectric,
         }
    }

    pub fn scatter(&self, incoming_ray: &Ray, hit_record: &HitRecord, attenuation: &mut Vec3, scattered: &mut Ray) -> bool {

        let reflect = Vec3::reflect(&Vec3::unit_vector(&incoming_ray.direction), &hit_record.normal);
        
        if self.dielectric != 0.0 {
            *attenuation = Vec3::new(1.0, 1.0, 1.0);
            let mut refracted = Vec3::default();
            
            let outward_normal;
            let ni_over_nt;
            let cosine;
            if Vec3::dot(&incoming_ray.direction(), &hit_record.normal) > 0.0 {
                outward_normal = -hit_record.normal;
                ni_over_nt = self.dielectric;
                cosine = self.dielectric * Vec3::dot(&incoming_ray.direction(), &hit_record.normal);
            } else {
                outward_normal = hit_record.normal;
                ni_over_nt = 1.0 / self.dielectric;
                cosine = -Vec3::dot(&incoming_ray.direction(), &hit_record.normal) / incoming_ray.direction().length();
            }
    
            let reflect_prob;
            if Vec3::refract(&incoming_ray.direction(), &outward_normal, ni_over_nt, &mut refracted) {
                reflect_prob = schlick(cosine, self.dielectric);
            } else {
                reflect_prob = 1.0;
            }
            
            if gen_range(0.0, 1.0) < reflect_prob
            {
                *scattered = Ray::new(hit_record.p, reflect);
            } else {
                *scattered = Ray::new(hit_record.p, refracted);
            }

            return true;

        }

        let fuzz_vector = Vec3::random();
        let target = hit_record.p + hit_record.normal + fuzz_vector;
        let fuzzed_reflect = reflect + self.fuzz * fuzz_vector;
        let reflection = (1.0 - self.metallicity) * (target - hit_record.p) + self.metallicity * fuzzed_reflect;
        *scattered = Ray::new(hit_record.p, reflection);
        *attenuation = self.albedo;

        true
    }
}

fn schlick(cosine: f32, ref_index: f32) -> f32 {
    let mut r0 = (1.0 - ref_index) / (1.0 + ref_index);
    r0 = r0 * r0;
    r0 + (1.0 - r0) * f32::powf(1.0 - cosine, 5.0)
}