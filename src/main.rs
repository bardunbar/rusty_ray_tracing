
mod math;
mod camera;
mod material;

use std::thread;
use std::{
    time::Instant,
    thread::{
        JoinHandle,
    },
    sync::{
        Arc,
        Mutex,
        RwLock
    }
};

use chrono::{ Duration, Local };
use image::Rgb;
use miniquad::*;
use quad_rand::ChooseRandom;

use math::{Vec3, Rect, Ray, random};
use camera::Camera;
use material::{ MaterialParams, Material };
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

unsafe impl Send for World {}
unsafe impl Sync for World {}

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

#[derive(Clone)]
pub struct RenderStats {
    x: usize,
    y: usize,
    target_x: usize,
    target_y: usize,
    buffer_stride: usize,
    done: bool,
}

fn worker(queue: Arc<Mutex<Vec<RenderJob>>>)
{
    let mut job;
    while {
        let mut job_queue = queue.lock().unwrap();
        job = job_queue.pop();
        job.is_some()
    } {
        if let Some(j) = job {
            render_tile(j.buffer, j.stats, j.camera, j.world);
        }
    }
} 

fn render_tile(pixels: Arc<Mutex<Vec<u8>>>, render_stats: RenderStats, camera: Arc<RwLock<Camera>>, world: Arc<RwLock<World>>) {
    
    let (sx, sy) = (render_stats.x, render_stats.y);
    let (fx, fy) = (render_stats.target_x, render_stats.target_y);
    let buffer_stride = render_stats.buffer_stride;

    let camera_resource = camera.read().unwrap();
    let world_resource = world.read().unwrap();

    for y in sy..fy {
        for x in sx..fx {
            let mut pixel_color = Vec3::default();
            for _ in 0..SAMPLES {

                let u = ((x as f32) + random()) / (RENDER_WIDTH as f32);
                let v = ((y as f32) + random()) / (RENDER_HEIGHT as f32); 

                let ray = camera_resource.get_ray(u, v);
                pixel_color += color(&ray, &*world_resource, 0);
            }

            pixel_color /= SAMPLES as f32;
            pixel_color = Vec3::new(f32::sqrt(pixel_color.x), f32::sqrt(pixel_color.y), f32::sqrt(pixel_color.z));

            let r = pixel_color.x;
            let g = pixel_color.y;
            let b = pixel_color.z;

            let iy = y - sy;
            let ix = x - sx;

            {
                let mut data = pixels.lock().unwrap();

                data[buffer_stride * iy + (ix * 4) + 0] = (256.0 * r) as u8;
                data[buffer_stride * iy + (ix * 4) + 1] = (256.0 * g) as u8;
                data[buffer_stride * iy + (ix * 4) + 2] = (256.0 * b) as u8;
                data[buffer_stride * iy + (ix * 4) + 3] = 0xFF;
            }
        }
    }
}

pub struct RenderJob {
    pub buffer: Arc<Mutex<Vec<u8>>>,
    pub stats: RenderStats, 
    pub camera: Arc<RwLock<Camera>>,
    pub world: Arc<RwLock<World>>,
}

pub struct Tile {
    pixel_buffer: Arc<Mutex<Vec<u8>>>,
    bindings: Bindings,
    progress: RenderStats,
}

impl Tile {
    fn new(ctx: &mut GraphicsContext, tile_rect: Rect) -> Tile {
        let (vx1, vx2, vy1, vy2) = (
            (tile_rect.x / RENDER_WIDTH as f32) * 2.0 - 1.0,
            ((tile_rect.x + tile_rect.w) / RENDER_WIDTH as f32) * 2.0 - 1.0,
            (tile_rect.y / RENDER_HEIGHT as f32) * 2.0 - 1.0,
            ((tile_rect.y + tile_rect.h) / RENDER_HEIGHT as f32) * 2.0 - 1.0);

        let vertices: [Vertex; 4] = [
            Vertex { pos : Vec2 { x: vx1, y: vy1 }, uv: Vec2 { x: 0., y: 0. } },
            Vertex { pos : Vec2 { x:  vx2, y: vy1 }, uv: Vec2 { x: 1., y: 0. } },
            Vertex { pos : Vec2 { x:  vx2, y:  vy2 }, uv: Vec2 { x: 1., y: 1. } },
            Vertex { pos : Vec2 { x: vx1, y:  vy2 }, uv: Vec2 { x: 0., y: 1. } },
        ];

        let vertex_buffer = Buffer::immutable(ctx, BufferType::VertexBuffer, &vertices);

        let indices: [u16; 6] = [0, 1, 2, 0, 2, 3];
        let index_buffer = Buffer::immutable(ctx, BufferType::IndexBuffer, &indices);

        let pixels = vec![0; tile_rect.w as usize * tile_rect.h as usize * 4];
        let texture = Texture::from_rgba8(ctx, tile_rect.w as u16, tile_rect.h as u16, &pixels);

        let bindings = Bindings {
            vertex_buffers: vec![vertex_buffer],
            index_buffer: index_buffer,
            images: vec![texture],
        };

        let mutex = Mutex::new(pixels);
        let arc = Arc::new(mutex);

        Tile {
            progress: RenderStats { 
                x: tile_rect.x as usize, 
                y: tile_rect.y as usize, 
                target_x: tile_rect.x as usize + tile_rect.w as usize, 
                target_y: tile_rect.y as usize + tile_rect.h as usize, 
                buffer_stride: tile_rect.w as usize * 4, 
                done: false },
            pixel_buffer: arc,
            bindings,
        }
    }

    pub fn render(&self, ctx: &mut GraphicsContext) {
        ctx.apply_bindings(&self.bindings);
        ctx.draw(0, 6, 1);
    }
}

const TILE_DIM: usize = 64;
const TILE_WIDTH: usize = 16;
const TILE_HEIGHT: usize = TILE_WIDTH / 2;

const SCREEN_WIDTH: usize = 1024;
const SCREEN_HEIGHT: usize = SCREEN_WIDTH / 2;

const RENDER_WIDTH: usize = TILE_WIDTH * TILE_DIM;
const RENDER_HEIGHT: usize = TILE_HEIGHT * TILE_DIM;
const SAMPLES: usize = 1000;
const THREAD_COUNT: usize = 15;

struct Application {
    pipeline: Pipeline,
    tiles: Vec<Tile>,
    workers: Vec<JoinHandle<()>>,
    rendering: bool,
    start_time: Instant,
}

impl Application {
    fn new(ctx: &mut GraphicsContext) -> Self {

        // println!("Available thread count: {}", thread::available_parallelism().unwrap().get());

        let world = build_random_scene();
        let world_lock = RwLock::new(world);
        let world_resource = Arc::new(world_lock);


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

        let camera = Camera::new(
            &look_from, 
            &look_at, 
            &Vec3::new(0.0, 1.0, 0.0), 
        20.0, 
            SCREEN_WIDTH as f32 / SCREEN_HEIGHT as f32,
            0.25, 
            dist
        );
        let camera_lock = RwLock::new(camera);
        let camera_resource = Arc::new(camera_lock);

        // Create all the tiles to render
        let mut tiles = Vec::new();
        for x in 0..TILE_WIDTH {
            for y in 0..TILE_HEIGHT {
                let tile = Tile::new(ctx, Rect {
                    x: (x * TILE_DIM) as f32,
                    y: (y * TILE_DIM) as f32,
                    w: TILE_DIM as f32,
                    h: TILE_DIM as f32,
                });

                tiles.push(tile);
            }
        }

        // Shuffle the tiles to make the render look more interesting!
        tiles.shuffle();

        // Create a render job for each tile
        let mut jobs = Vec::new();
        for tile in tiles.iter() {
            let job = RenderJob {
                buffer: tile.pixel_buffer.clone(),
                stats: tile.progress.clone(),
                camera: camera_resource.clone(),
                world: world_resource.clone(),
            };

            jobs.push(job);
        }

        let jobs_lock = Mutex::new(jobs);
        let jobs_resource = Arc::new(jobs_lock);

        // Attempt to use as many threads as are available, leaving one for the main thread.
        let thread_count = match thread::available_parallelism() {
            Ok(system_threads) => system_threads.get() - 1,
            Err(_) => THREAD_COUNT,
        };

        println!("Begin render using {} core{}.", thread_count, if thread_count == 1 { "" } else {"s"} );

        let mut workers = Vec::new();
        for _ in 0..thread_count {
            let job_clone = jobs_resource.clone();

            let t = thread::spawn(move ||{
                worker(job_clone);
            });
            workers.push(t);
        }


        Application {
            pipeline,
            workers,
            tiles,
            rendering: true,
            start_time: Instant::now(),
        }
    }
}

impl EventHandler for Application {
    
    fn update(&mut self, ctx: &mut Context) {

        if self.rendering {

            thread::sleep(std::time::Duration::from_millis(100));

            for tile in self.tiles.iter_mut() {
                if !tile.progress.done {

                    {
                        let pixels = tile.pixel_buffer.lock().unwrap();
                        let texture = &tile.bindings.images[0]; 
                        texture.update(ctx, &pixels);
                    }
                }
            }
            let mut all_done = true;
            for worker in self.workers.iter() {
                all_done = all_done && worker.is_finished();
            }
            
            if self.rendering && all_done {
                self.rendering = false;

                let finish_time = Instant::now();

                if let Ok(render_time) = Duration::from_std(finish_time.duration_since(self.start_time)) {
                    let hours = render_time.num_hours();
                    let minutes = render_time.num_minutes() % 60;
                    let seconds = render_time.num_seconds() % 60;
                    let milliseconds = render_time.num_milliseconds() % 1000;
                    println!("Render complete in {:02}:{:02}:{:02}.{:03}", hours, minutes, seconds, milliseconds);
                }
 
                let mut buff = ImageBuffer::new(RENDER_WIDTH as u32, RENDER_HEIGHT as u32);
                
                for tile in self.tiles.iter() {
                    let pixels = tile.pixel_buffer.lock().unwrap();
                    let stats = &tile.progress;
                    
                    for y in 0..TILE_DIM {
                        for x in 0..TILE_DIM {
                            let r = pixels[stats.buffer_stride * y + x * 4 + 0];
                            let g = pixels[stats.buffer_stride * y + x * 4 + 1];
                            let b = pixels[stats.buffer_stride * y + x * 4 + 2];

                            let pixel_x = (x + stats.x) as u32;
                            let pixel_y = ((RENDER_HEIGHT - 1) - (y + stats.y)) as u32; // Invert the Y to accommodate the image crate's format

                            buff.put_pixel(pixel_x, pixel_y, Rgb([r,g,b]));
                        }
                    }
                }

                let dt = Local::now();
                let file_path = format!("output_{}.png", dt.format("%Y%m%d%H%M%S"));
                buff.save_with_format(file_path, image::ImageFormat::Png).unwrap()
                
            }

        }
    }

    fn draw(&mut self, ctx: &mut Context) {
        ctx.begin_default_pass(Default::default());

        ctx.apply_pipeline(&self.pipeline);

        for tile in self.tiles.iter() {
            tile.render(ctx);
        }

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