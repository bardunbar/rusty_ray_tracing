use camera::Camera;
use miniquad::*;

mod math;
mod camera;
mod material;

use math::Vec3;
use math::Ray;
use math::random;
use chrono::Local;
use material::Material;
use material::MaterialParams;
use std::thread::available_parallelism;
use image::ImageBuffer;

#[repr(C)]
struct Vec2 {
    x: f32,
    y: f32,
}

#[repr(C)]
struct Vertex {
    pos: Vec2,
    uv: Vec2,
}

#[derive(Default, Copy, Clone)]
pub struct HitRecord {
    t: f32,
    p: Vec3,
    normal: Vec3,
    material: Option<Material>
}

pub trait Hitable {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32, hit_record: &mut HitRecord) -> bool;
    fn get_material(&self) -> Option<Material>;
}

pub struct Sphere {
    center: Vec3,
    radius: f32,
    material: Material,
}

impl Sphere {
    pub fn new(center: Vec3, radius: f32, material: Material) -> Self {
        Sphere { center, radius, material }
    }
}

impl Hitable for Sphere {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32, hit_record: &mut HitRecord) -> bool {
        let oc = ray.origin() - self.center;
        let a = Vec3::dot(&ray.direction(), &ray.direction());
        let b = Vec3::dot(&oc, &ray.direction());
        let c = Vec3::dot(&oc, &oc) - self.radius * self.radius;
        let discriminant = (b * b) - (a * c);

        if discriminant > 0.0 {
            let temp = (-b - f32::sqrt(b * b - a * c)) / a;
            if temp < t_max && temp > t_min {
                hit_record.t = temp;
                hit_record.p = ray.point_at_parameter(hit_record.t);
                hit_record.normal = (hit_record.p - self.center) / self.radius;
                hit_record.material = self.get_material();
                return true
            }

            let temp = (-b + f32::sqrt(b * b - a * c)) / a;
            if temp < t_max && temp > t_min {
                hit_record.t = temp;
                hit_record.p = ray.point_at_parameter(hit_record.t);
                hit_record.normal = (hit_record.p - self.center) / self.radius;
                hit_record.material = self.get_material();
               return true
            }
            false
        } else {
            false
        }
    }

    fn get_material(&self) -> Option<Material> {
        Some(self.material)
    }
}

pub struct World {
    objects: Vec<Box<dyn Hitable>>
}

impl World {
    pub fn new() -> Self {
        World { objects: Vec::new() }
    }

    pub fn add(&mut self, object: impl Hitable + 'static) {
        self.objects.push(Box::new(object));
    }
}

impl Hitable for World {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32, hit_record: &mut HitRecord) -> bool {
        let mut record = HitRecord::default();
        let mut hit_any = false;
        let mut closest_so_far = t_max;

        for object in self.objects.iter() {
            if object.hit(ray, t_min, closest_so_far, &mut record) {
                hit_any = true;
                closest_so_far = record.t;
                *hit_record = record;
            }
        }

        hit_any
    }

    fn get_material(&self) -> Option<Material> {
        None
    }
}


fn color(ray: &Ray, world: &impl Hitable, depth: u32) -> Vec3 {
    let mut record = HitRecord::default();
    if world.hit(ray, 0.0001, f32::MAX, &mut record) {
        let mut scattered = Ray::default();
        let mut attenuation = Vec3::default();
        if let Some(material) = record.material {
            if depth < 50 && material.scatter(ray, &record, &mut attenuation, &mut scattered) {
                attenuation * color(&scattered, world, depth + 1)
            }
            else {
                Vec3::default()
            }
        }
        else
        {
            Vec3::default()
        }
    } else {
        // Process the color from the skybox
        let normalized_direction = Vec3::unit_vector(&ray.direction());
        let t = 0.5 * (normalized_direction.y + 1.0);
        (1.0 - t) * Vec3::new(1.0, 1.0, 1.0) + t * Vec3::new(0.5, 0.7, 1.0)
    }
}

fn build_random_scene() -> World {
    let mut world = World::new();

    // Add the ground
    world.add( Sphere::new(Vec3::new(0.0, -1000.0, 0.0), 1000.0, 
        Material::new(MaterialParams {
            albedo: Vec3::new(0.5, 0.5, 0.5), 
            ..Default::default()
        })
    ));

    for a in -11..11 {
        for b in -11..11 {
            let choose_mat = random();
            let center = Vec3::new(a as f32 + 0.9 * random(), 0.2, b as f32 + 0.9 * random());
            if (center - Vec3::new(4.0, 0.2, 0.0)).length() > 0.9 {
                let mut params = MaterialParams::default();
                if choose_mat > 0.95 {
                    params.dielectric = 1.5;
                }
                else if choose_mat > 0.8 {
                    params.metallicity = 1.0;
                    params.fuzz = 0.5 * random()
                }

                params.albedo = Vec3::new(random()*random(), random() * random(), random() * random());
                world.add(Sphere::new(center, 0.2, Material::new(params)));
            }
        }
    }

    world.add(Sphere::new(Vec3::new(0.0, 1.0, 0.0), 1.0,
        Material::new(MaterialParams {
            dielectric: 1.5,
            ..Default::default()
        })));
    world.add(Sphere::new(Vec3::new(-4.0, 1.0, 0.0), 1.0,
        Material::new(MaterialParams {
            albedo: Vec3::new(0.4, 0.2, 0.1),
            ..Default::default()
        })));
    world.add(Sphere::new(Vec3::new(4.0, 1.0, 0.0), 1.0,
        Material::new(MaterialParams {
            metallicity: 1.0,
            fuzz: 0.0,
            albedo: Vec3::new(0.7, 0.6, 0.5),
            ..Default::default()
        })));

    world
}

const SCREEN_WIDTH: usize = 1024;
const SCREEN_HEIGHT: usize = SCREEN_WIDTH / 2;

const RENDER_WIDTH: usize = 1024;
const RENDER_HEIGHT: usize = RENDER_WIDTH / 2;
const LEN: usize = RENDER_WIDTH * RENDER_HEIGHT * 4;
const SAMPLES: usize = 10;

struct Application {
    pipeline: Pipeline,
    bindings: Bindings,
    world: World,
    camera: Camera,
    pixel_buffer: Vec<u8>,
    cur_x: usize,
    cur_y: usize,
    rendering: bool,
}

impl Application {
    fn new(ctx: &mut GraphicsContext) -> Self {

        println!("Available thread count: {}", available_parallelism().unwrap().get());

        let vertices: [Vertex; 4] = [
            Vertex { pos : Vec2 { x: -1.0, y: -1.0 }, uv: Vec2 { x: 0., y: 0. } },
            Vertex { pos : Vec2 { x:  1.0, y: -1.0 }, uv: Vec2 { x: 1., y: 0. } },
            Vertex { pos : Vec2 { x:  1.0, y:  1.0 }, uv: Vec2 { x: 1., y: 1. } },
            Vertex { pos : Vec2 { x: -1.0, y:  1.0 }, uv: Vec2 { x: 0., y: 1. } },
        ];

        let vertex_buffer = Buffer::immutable(ctx, BufferType::VertexBuffer, &vertices);

        let indices: [u16; 6] = [0, 1, 2, 0, 2, 3];
        let index_buffer = Buffer::immutable(ctx, BufferType::IndexBuffer, &indices);

        let world = build_random_scene();

        let pixels = vec![0; LEN];
        let texture = Texture::from_rgba8(ctx, RENDER_WIDTH as u16, RENDER_HEIGHT as u16, &pixels);

        let bindings = Bindings {
            vertex_buffers: vec![vertex_buffer],
            index_buffer: index_buffer,
            images: vec![texture],
        };

        let shader = Shader::new(ctx, shader::VERTEX, shader::FRAGMENT, shader::meta()).unwrap();

        let pipeline = Pipeline::new(
            ctx,
            &[BufferLayout::default()],
            &[
                VertexAttribute::new("pos", VertexFormat::Float2),
                VertexAttribute::new("uv", VertexFormat::Float2),
            ],
            shader,
        );

        let look_from = Vec3::new(12.0, 2.0, 3.0);
        let look_at = Vec3::new(0.0, 0.5, 0.0);
        let dist = (look_from - look_at).length();

        Application {
            pipeline,
            bindings,
            world,
            camera: Camera::new(
                &look_from, 
                &look_at, 
                &Vec3::new(0.0, 1.0, 0.0), 
            20.0, 
                SCREEN_WIDTH as f32 / SCREEN_HEIGHT as f32,
                0.25, 
                dist
            ),
            pixel_buffer: pixels,
            cur_x: 0,
            cur_y: 0,
            rendering: true
        }
    }
}

impl EventHandler for Application {
    
    fn update(&mut self, ctx: &mut Context) {
        use std::time::Instant;

        if self.rendering {
            let start_time = Instant::now();
            let mut stop_flag = false;
            for y in self.cur_y..RENDER_HEIGHT {
                for x in self.cur_x..RENDER_WIDTH {
                    let mut pixel_color = Vec3::default();
                    for _ in 0..SAMPLES {
    
                        let u = ((x as f32) + random()) / (RENDER_WIDTH as f32);
                        let v = ((y as f32) + random()) / (RENDER_HEIGHT as f32); 
        
                        let ray = self.camera.get_ray(u, v);
                        pixel_color += color(&ray, &self.world, 0);
                    }
    
                    pixel_color /= SAMPLES as f32;
                    pixel_color = Vec3::new(f32::sqrt(pixel_color.x), f32::sqrt(pixel_color.y), f32::sqrt(pixel_color.z));
    
                    let r = pixel_color.x;
                    let g = pixel_color.y;
                    let b = pixel_color.z;
    
                    self.pixel_buffer[(RENDER_WIDTH * 4) * y + (x * 4) + 0] = (256.0 * r) as u8;
                    self.pixel_buffer[(RENDER_WIDTH * 4) * y + (x * 4) + 1] = (256.0 * g) as u8;
                    self.pixel_buffer[(RENDER_WIDTH * 4) * y + (x * 4) + 2] = (256.0 * b) as u8;
                    self.pixel_buffer[(RENDER_WIDTH * 4) * y + (x * 4) + 3] = 0xFF;

                    self.cur_x = x + 1;

                    let current_time = Instant::now();
                    let duration_ms = current_time.duration_since(start_time).as_millis();

                    stop_flag = duration_ms > 200;

                    if stop_flag {
                        break;
                    }
                }
                
                if stop_flag {
                    break;
                }

                self.cur_x = 0;
                self.cur_y = y + 1;
            }
    
            if self.rendering && !stop_flag {
                self.rendering = false;
 
                

                let mut buff = ImageBuffer::new(RENDER_WIDTH as u32, RENDER_HEIGHT as u32);
                for (x, y, pixel) in buff.enumerate_pixels_mut() {
                    let inv_y = RENDER_HEIGHT - 1 - y as usize;
                    let r = self.pixel_buffer[inv_y as usize * (RENDER_WIDTH * 4) + (x as usize * 4) + 0];
                    let g = self.pixel_buffer[inv_y as usize * (RENDER_WIDTH * 4) + (x as usize * 4) + 1];
                    let b = self.pixel_buffer[inv_y as usize * (RENDER_WIDTH * 4) + (x as usize * 4) + 2];
                    *pixel = image::Rgb([r, g, b]);
                }

                let dt = Local::now();
                let file_path = format!("output_{}.png", dt.format("%Y%m%d%H%M%S"));

                buff.save_with_format(file_path, image::ImageFormat::Png).unwrap()
                
            }

            let texture = &self.bindings.images[0]; 
            texture.update(ctx, &self.pixel_buffer);
        }
    }

    fn draw(&mut self, ctx: &mut Context) {
        ctx.begin_default_pass(Default::default());

        ctx.apply_pipeline(&self.pipeline);
        ctx.apply_bindings(&self.bindings);

        ctx.draw(0, 6, 1);

        ctx.end_render_pass();

        ctx.commit_frame();
    }

    fn key_down_event(
            &mut self,
            ctx: &mut Context,
            keycode: KeyCode,
            _keymods: KeyMods,
            _repeat: bool,
        ) {
        if keycode == KeyCode::Escape {
            ctx.request_quit();
        }
    }
}

fn main() {
    miniquad::start(
        conf::Conf {
            window_title: "Rust Tracer".to_string(),
            window_width: SCREEN_WIDTH as i32,
            window_height: SCREEN_HEIGHT as i32,
            fullscreen: false,
            ..Default::default()
        }, |ctx| Box::new(Application::new(ctx)));
}


mod shader {
    use miniquad::*;

    pub const VERTEX: &str = r#"#version 100
    attribute vec2 pos;
    attribute vec2 uv;

    varying lowp vec2 texcoord;

    void main() {
        gl_Position = vec4(pos, 0, 1);
        texcoord = uv;
    }"#;

    pub const FRAGMENT: &str = r#"#version 100
    varying lowp vec2 texcoord;

    uniform sampler2D tex;

    void main() {
        gl_FragColor = texture2D(tex, texcoord);
    }"#;

    pub fn meta() -> ShaderMeta {
        ShaderMeta {
            images: vec!["tex".to_string()],
            uniforms: UniformBlockLayout {
                uniforms: vec![],
            },
        }
    }
}