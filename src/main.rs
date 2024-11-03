use std::{f32::consts::PI, ffi::{c_float, CString}, mem, os::raw::c_char, ptr::NonNull};
use rand::random;
use objc::rc::autoreleasepool;
use objc2::rc::Retained;
use objc2_app_kit::{NSAnyEventMask, NSApplication, NSApplicationActivationPolicy, NSBackingStoreType, NSColor, NSEventType, NSScreen, NSWindow, NSWindowStyleMask};
use objc2_foundation::{CGPoint, MainThreadMarker, NSComparisonResult, NSDate, NSDefaultRunLoopMode, NSRect, NSSize, NSString};

use metal::*;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Float4(c_float, c_float, c_float, c_float);
impl Float4 {
    pub fn new(v1 : Float2, v2 : Float2) -> Self {
        Self(v1.0, v1.1, v2.0, v2.1)
    }
}
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Float3(c_float, c_float, c_float);
impl Float3 {
    pub fn new(v : Float2, f : f32) -> Self {
        Self(v.0, v.1, f)
    }
}
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Float2(c_float, c_float);

pub struct Float2x2 {
    row1 : Float2,
    row2 : Float2
}

#[inline]
fn float2_add(v1 : Float2, v2 : Float2) -> Float2 {
    Float2(v1.0 + v2.0, v1.1 + v2.1)
}

#[inline]
fn dot(v1 : Float2, v2 : Float2) -> f32 {
    v1.0 * v2.0 + v1.1 * v2.1
}

#[inline]
fn matrix_mul(v : Float2, m : Float2x2) -> Float2 {
    Float2(dot(v, m.row1), dot(v, m.row2))
}

#[inline]
fn rotation_matrix(theta : f32) -> Float2x2 {
    let cos_theta = f32::cos(theta);
    let sin_theta = f32::sin(theta);
    Float2x2 {
        row1 : Float2(cos_theta, -sin_theta),
        row2 : Float2(sin_theta, cos_theta)
    }
}

#[inline]
fn apply_rotation_float2(target : Float2, theta : f32) -> Float2 {
    matrix_mul(target, rotation_matrix(theta))
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Color {
    r : f32,
    b : f32,
    g : f32,
    a : f32
}
#[repr(C)]
#[derive(Clone, Copy)]
struct Rect {
    w : f32,
    h : f32,
}

pub trait Collider {
    fn get_collider(&self, positions : &Vec<Float3>) -> Float4;
    //fn check_collision(&self, collider : impl Collider) -> bool;
}

#[derive(Clone, Copy)]
struct Hero {
    rect : Rect,
    color: Color,
    position : Float2 
}

impl Hero {
    fn update_position(&mut self, current_x : f32, current_y: f32) {
        self.position = Float2(current_x, current_y);
    }

    // fn update_color(&mut self, new_color : Float4) {
    //     self.color.r = new_color.0;
    //     self.color.g = new_color.1;
    //     self.color.b = new_color.2;
    //     self.color.a = new_color.3;
    // }
}

impl Collider for Hero {
    fn get_collider(&self, _positions : &Vec<Float3>) -> Float4 {
        Float4(
            (-self.rect.w/ 2.0) + self.position.0, //left
            (self.rect.h / 2.0) + self.position.1, //top
            (self.rect.w / 2.0) + self.position.0, //right
            (-self.rect.h / 2.0) + self.position.1 //bottom
        )
    }
}
pub trait LineCollider {
    fn get_collider(&self, _positions : &Vec<Float3>) -> Float4;
}

struct Arrow {
    index: usize,
    width: f32,
    height: f32,
}

impl Collider for Arrow {
    fn get_collider(&self, positions : &Vec<Float3>) -> Float4 {
        let pos = Float2(positions[self.index].0, positions[self.index].1);
        let theta = positions[self.index].2;
        //let cos_theta = f32::cos(positions[self.index][2]); //* self.width - f32::sin(positions[self.index][2] * self.height);
        //let sin_theta = f32::sin(positions[self.index][2]); //* self.width - f32::cos(positions[self.index][2] * self.height);
        Float4::new(
            float2_add(apply_rotation_float2(Float2(0.0, self.height / 2.0), theta), pos), 
            float2_add(apply_rotation_float2(Float2(0.0, -self.height / 2.0), theta), pos)
        )
    }
}

#[inline]
fn build_arrow_vertices (width : f32, length : f32) -> [f32; 18]{
    [
    width / 6.0, -length / 2.0,
    -width / 6.0, -length / 2.0,
    width / 6.0, length / 4.0,
    width / 6.0, length / 4.0,
    -width / 6.0, length / 4.0,
    -width / 6.0, -length / 2.0,
    0.0, length / 2.0,
    width / 2.0, length / 4.0,
    -width / 2.0, length / 4.0
    ]
}

fn gen_random_arrows(arrows : &mut Vec<Arrow>, n : u16, view_width : f32, view_height : f32) -> (Vec<[f32; 18]>, Vec<[f32; 3]>) {
    let width = 30.0 / view_width;
    let height = 100.0 / view_height;
    
    let mut vertices : Vec<[f32; 18]> = Vec::new();
    let mut positions : Vec<[f32; 3]> = Vec::new();
    for i in 0..n {
        arrows.push(Arrow {index: i as usize, width, height});
        vertices.push(build_arrow_vertices(width, height));
        positions.push([random::<f32>() * 4.0 - 5.0, random::<f32>() * 2.0 - 1.0, PI / 2.0]);
    };

    (vertices, positions)
}

fn gen_random_paths (paths : &mut Vec<Box<dyn Path>>, n : u16) {
    for _ in 0..n {
        let x = (random::<f32>() * 3.0).floor() as u8;
        match x {
            0 => {
                paths.push(Box::new(StraightPath {
                    speed: random::<f32>() / 5.0 + 0.4,
                    rotation: (random::<f32>() * PI * 0.5) + PI * 0.75,
                    origin: Float2(0.0, 1.0),
                }))
            },
            1 => {
                paths.push(Box::new(WavyPath {
                    speed: random::<f32>() / 5.0 + 0.4,
                    amplitude: (random::<f32>() - 0.5) * 2.0,
                    rotation: (random::<f32>() * PI * 0.5) + PI * 0.25,
                    origin: Float2(0.0, 1.0)
                }));
            },
            2 => {
                paths.push(Box::new(CirclePath {
                    speed: random::<f32>() / 5.0 + 1.0,
                    radius: random::<f32>() * 1.5 + 0.5,
                    origin: Float2(0.0, 1.0)
                }));
            },
            _ => paths.push(Box::new(StraightPath {
                speed: random::<f32>() / 20.0,
                rotation: (random::<f32>() + 0.5f32) * PI,
                origin: Float2(0.0, 1.0),
            }))
        }
    }
}

fn line_line_collision(l1 : Float4, l2 : Float4) -> bool {
    let a = Float2(l1.0, l1.1);
    let b = Float2 (l1.2, l1.3);
    let c = Float2(l2.0, l2.1);
    let d = Float2 (l2.2, l2.3);
    
    let denominator : f32 = ((b.0 - a.0) * (d.1 - c.1)) - ((b.1 - a.1) * (d.0 - c.0));
    let numerator1 : f32 = ((a.1 - c.1) * (d.0 - c.0)) - ((a.0 - c.0) * (d.1 - c.1));
    let numerator2 : f32 = ((a.1 - c.1) * (b.0 - a.0)) - ((a.0 - c.0) * (b.1 - a.1));

    // Detect coincident lines (has a problem, read below)
    if denominator == 0.0 {
    return numerator1 == 0.0 && numerator2 == 0.0;
    };
    
    let r : f32 = numerator1 / denominator;
    let s : f32 = numerator2 / denominator;

    return (r >= 0.0 && r <= 1.0) && (s >= 0.0 && s <= 1.0);
}

fn line_rect_collision(line : Float4, rect : Float4)-> bool {
    if line_line_collision(line, Float4(rect.0, rect.1, rect.0, rect.3)) {
        return true;
    }
    if line_line_collision(line, Float4(rect.0, rect.1, rect.2, rect.1)) {
        return true;
    }
    if line_line_collision(line, Float4(rect.2, rect.1, rect.2, rect.3)) {
        return true;
    }
    if line_line_collision(line, Float4(rect.0, rect.3, rect.2, rect.3)) {
        return true;
    }
    return false;
}

fn check_arrow_collisions(arrows : &mut Vec<Arrow>, hero : &Hero, positions: &Vec<Float3>) -> Option<usize>{
    let hero_collider = hero.get_collider(positions);
    for arrow in arrows {
        let arrow_collider = arrow.get_collider(positions);
        if line_rect_collision(arrow_collider, hero_collider) {
            return Some(arrow.index);
        }
        // if hero_collider.0 <= arrow_collider.2 && hero_collider.2 >= arrow_collider.0 {
        //     if hero_collider.1 <= arrow_collider.3 && hero_collider.3 >= arrow_collider.1 {
        //         return Some(arrow.index);
        //     }
        // }
    }
    return None;
}

pub trait Path {
    fn get_position(&self, t : f32) -> Float2;
    fn get_position_and_rotation(&self, t : f32) -> Float3;
}

struct StraightPath {
    speed : f32,
    rotation : f32,
    origin : Float2,
}

impl Path for StraightPath {
    fn get_position(&self, t : f32) -> Float2 {
        float2_add(apply_rotation_float2(Float2(0.0, t * self.speed), -self.rotation), self.origin)
    }
    fn get_position_and_rotation(&self, t : f32) -> Float3 {
        Float3::new(self.get_position(t), self.rotation)
    }
}

struct CirclePath {
    speed : f32,
    radius : f32,
    origin : Float2,
}

impl Path for CirclePath {
    fn get_position(&self, t : f32) -> Float2 {
        float2_add(Float2((t*self.speed).cos() * self.radius, (t*self.speed).sin() * self.radius), self.origin)
    }
    fn get_position_and_rotation(&self, t : f32) -> Float3 {
        Float3::new(self.get_position(t), -(t * self.speed))
    }
}

struct WavyPath {
    speed : f32,
    amplitude : f32,
    rotation : f32,
    origin : Float2,
}

impl Path for WavyPath {
    fn get_position(&self, t : f32) -> Float2 {
        float2_add(apply_rotation_float2(Float2(t * self.speed, t.sin() * self.amplitude), -(self.amplitude / self.amplitude.abs()) * self.rotation), self.origin)
    }
    fn get_position_and_rotation(&self, t : f32) -> Float3 {
        Float3::new(self.get_position(t), self.rotation + (t.cos() / (t.cos() * t.cos() + 1.0).sqrt()).acos())
    }
}

#[inline]
fn initialize_app(thread : MainThreadMarker) -> Retained<NSApplication> {
    let app = NSApplication::sharedApplication(thread);
    app.setActivationPolicy(NSApplicationActivationPolicy::Regular);
    
    return app;
}

#[inline]
fn initialize_window(thread: MainThreadMarker) -> Retained<NSWindow> {
    //Set 
    const GLOBAL_RENDERING_WIDTH : f64 = 1024.0;
    const GLOBAL_RENDERING_HEIGHT : f64 = 768.0;

    let screen_rect : NSRect = NSScreen::mainScreen(thread).unwrap().frame();

    let size  = NSSize{
        width: GLOBAL_RENDERING_WIDTH,
        height: GLOBAL_RENDERING_HEIGHT
    };
    let origin= CGPoint{
        x: (screen_rect.size.width - GLOBAL_RENDERING_WIDTH) * 0.5,
        y: (screen_rect.size.height - GLOBAL_RENDERING_HEIGHT) * 0.5,
    };
    let window_rect : NSRect = NSRect::new(
        origin,
        size
    );


    let window_color = unsafe {NSColor::colorWithSRGBRed_green_blue_alpha(0.0, 0.0, 0.0, 1.0)};
    let ctitle = CString::new("Dodge").expect("CString::new failed!");
    let ctitleptr : NonNull<c_char> = NonNull::new(ctitle.as_ptr() as *mut i8).expect("NonNull::new failed!");

    let window_title = unsafe {
        NSString::initWithCString_encoding(
            thread.alloc::<NSString>(), 
            ctitleptr, 
            NSString::defaultCStringEncoding()
        ).expect("String init failed!")
    };

    let style_mask = 
        NSWindowStyleMask::Titled.union(
        NSWindowStyleMask::Closable.union(
        NSWindowStyleMask::Resizable.union(
        NSWindowStyleMask::Miniaturizable
    )));

    let window  = unsafe { 
        NSWindow::initWithContentRect_styleMask_backing_defer(
        thread.alloc::<NSWindow>(), 
        window_rect, 
        style_mask, 
        NSBackingStoreType::NSBackingStoreBuffered, 
        false)
    };

    window.setBackgroundColor(Some(&window_color));
    window.setTitle(&window_title);
    window.contentView().unwrap().setWantsLayer(true);
    return window;
}

#[inline]
fn set_window_layer(window : &Retained<NSWindow>, layer: &MetalLayer) {
    unsafe {
        window.contentView().unwrap().setLayer(mem::transmute(layer.as_ref()));
    }
}

#[inline]
fn get_next_frame (fps : &f64) -> Retained<NSDate> {
    unsafe {
        NSDate::dateWithTimeIntervalSinceNow(1.0 / fps)
    }
}

#[inline]
fn new_metal_layer(device: &DeviceRef) -> MetalLayer {
    let layer = MetalLayer::new();

    layer.set_device(device);
    layer.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
    layer.set_presents_with_transaction(false);

    return layer;
}

fn prepare_pipeline_state (
    device : &DeviceRef, 
    vertex_shader : &str, 
    fragment_shader : &str
) -> RenderPipelineState {
    let library_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/shaders.metallib");
    let shaderlib = device.new_library_with_file(library_path).unwrap();

    let vert = shaderlib.get_function(vertex_shader, None).unwrap();
    let frag = shaderlib.get_function(fragment_shader, None).unwrap();

    let pipeline_state_descriptor = RenderPipelineDescriptor::new();
    pipeline_state_descriptor.set_vertex_function(Some(&vert));
    pipeline_state_descriptor.set_fragment_function(Some(&frag));

    let pipeline_attachment = pipeline_state_descriptor
        .color_attachments()
        .object_at(0)
        .unwrap();

    pipeline_attachment.set_pixel_format(MTLPixelFormat::RGBA8Unorm);

    device
        .new_render_pipeline_state(&pipeline_state_descriptor)
        .unwrap()
}

fn prepare_render_pass_descriptor( render_pass_descriptor : &RenderPassDescriptorRef, texture : &TextureRef) {

    let render_pass_attachment = render_pass_descriptor.color_attachments().object_at(0).unwrap();
    render_pass_attachment.set_texture(Some(&texture));
    render_pass_attachment.set_load_action(MTLLoadAction::Clear);
    render_pass_attachment.set_clear_color(MTLClearColor::new(0.0, 0.0, 0.0, 1.0));
    render_pass_attachment.set_store_action(MTLStoreAction::Store);
}

fn main() {
    let mtm = MainThreadMarker::new().expect("Current thread isn't main");
    let app = initialize_app(mtm);
    let device = Device::system_default().expect("No device found");
    // let layer = MetalLayer::new();

    // layer.set_device(&device);
    // layer.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
    // layer.set_presents_with_transaction(false);

    let window = initialize_window(mtm);
    let launch_screen = new_metal_layer(&device);
    set_window_layer(&window, &launch_screen);
    let view_width = window.contentView().unwrap().frame().size.width as f32;
    let view_height = window.contentView().unwrap().frame().size.height as f32;

    let hero_pipeline_state = prepare_pipeline_state(
        &device, 
        "rectangle_vertex", 
        "rectangle_shader"
    );
    let arrow_pipeline_state = prepare_pipeline_state(
        &device, 
        "arrow_vertex", 
        "fragment_shader"
    );

    let command_queue = device.new_command_queue();

    unsafe {
        app.finishLaunching();
        app.activateIgnoringOtherApps(true);
        window.makeKeyAndOrderFront(None);
    };

    let center_x = view_width / 2.0;
    let center_y = view_height / 2.0;
    let mut current_x = center_x;
    let mut current_y = center_y;
    
    let start_position = [(current_x - center_x) / view_width, (current_y - center_y) / view_height];

    let mut hero = Hero {
        rect : Rect {
            w : 50.0 / view_width,
            h : 50.0 / view_height,
        },
        color : Color {
            r: 0.5, 
            b: 0.2, 
            g: 0.8, 
            a: 1.0 
        },
        position : Float2((current_x - center_x) / view_width, (current_y - center_y) / view_height)
    };

    let hero_rect = vec![hero.rect];
    let hero_rect_buffer = device.new_buffer_with_data(
        hero_rect.as_ptr() as *const _, 
        mem::size_of::<Rect>() as u64,
        MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeManaged
    );

    let pbuf = device.new_buffer_with_data(
        start_position.as_ptr() as *const _,
        (mem::size_of::<f32>() * start_position.len()) as u64,
        MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeManaged
    );

    //let hero_color = hero.color;
    let hero_color_data = vec![hero.color.r, hero.color.g, hero.color.b, hero.color.a];
    let hit_color_data = vec![1.0f32, 0.0, 0.0, 1.0];

    let cbuf = device.new_buffer_with_data(
        hero_color_data.as_ptr() as *const _, 
        (size_of::<f32>() * hero_color_data.len()) as u64, 
        MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeManaged
    );

    let hit_color_buf = device.new_buffer_with_data(
        hit_color_data.as_ptr() as *const _, 
        (size_of::<f32>() * hit_color_data.len()) as u64, 
        MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeManaged
    );

    // let arrow1_pos = [-150.0f32 / view_width, 100.0 / view_height, PI / 2.0];
    // let arrow2_pos = [-150.0f32 / view_width, 1.0 / view_height, PI / 2.0];

    let num_arrows = 32;
    let mut arrows = Vec::<Arrow>::new();
    arrows.reserve(num_arrows as usize);

    let (arrow_vertices, _) = gen_random_arrows(&mut arrows, num_arrows, view_width, view_height);
    let arrow_positions : Vec<Float3> = vec![Float3(0.0, 1.0, PI); num_arrows as usize];


    let arrow_pbuf = device.new_buffer_with_data(
        arrow_positions.as_ptr() as *const _,
        (mem::size_of::<Float3>() * arrow_positions.len()) as u64,
        MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeManaged
    );

    let dummy_vertex_data = build_arrow_vertices(30.0 / view_width, 150.0 / view_height);
    let arrow_vbuf = device.new_buffer_with_data(
        arrow_vertices.as_ptr() as *const _, 
        (arrow_vertices.len() * dummy_vertex_data.len() * size_of::<f32>()) as u64, 
        MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeManaged
    );

    let mut arrow_paths = Vec::<Box<dyn Path>>::new();
    arrow_paths.reserve(num_arrows as usize);

    gen_random_paths(&mut arrow_paths, num_arrows);

    

    let fps = 60.0;
    let mut key_pressed : u16 = 112;
    let mut frame_time = get_next_frame(&fps);

    let mut t = 0.0;

    loop {
        autoreleasepool(|| {
                if unsafe { frame_time.compare(&NSDate::now()) } == NSComparisonResult::Ascending {
                    frame_time = get_next_frame(&fps);
                    t += 0.01;

                    match key_pressed {
                        0 => current_x -= 10.0,
                        1 => current_y -= 10.0,
                        2 => current_x += 10.0,
                        13 => current_y += 10.0,
                        _ => ()
                    }
                    hero.update_position((current_x - center_x) / view_width, (current_y - center_y) / view_height);
                    let p = pbuf.contents();
                    let position_data = [(current_x - center_x) / view_width, (current_y - center_y) / view_height];
                    
                    unsafe {
                        std::ptr::copy(
                            position_data.as_ptr(),
                            p as *mut f32,
                            position_data.len() as usize
                        );
                    }
                    pbuf.did_modify_range(NSRange::new(
                        0 as u64, 
                        (position_data.len() * size_of::<Float2>()) as u64
                    ));
                    let apbuf : Vec<Float3> = arrow_paths.iter().map(|path| path.get_position_and_rotation(t)).collect();
                    let ap = arrow_pbuf.contents();
                    unsafe {
                        std::ptr::copy(
                            apbuf.as_ptr(),
                            ap as *mut Float3,
                            apbuf.len() as usize
                        );
                    }
                    arrow_pbuf.did_modify_range(NSRange::new(
                        0 as u64,
                        (apbuf.len() * size_of::<Float3>()) as u64
                    ));
                    let render_pass_descriptor = RenderPassDescriptor::new();
                    let drawable = match launch_screen.next_drawable() {
                        Some(drawable) => drawable,
                        None => return,
                    };
                    prepare_render_pass_descriptor(
                        render_pass_descriptor,
                        drawable.texture()
                    );
                    let command_buffer = command_queue.new_command_buffer();
                    let encoder = command_buffer.new_render_command_encoder(&render_pass_descriptor);
                    encoder.set_render_pipeline_state(&hero_pipeline_state);
                    encoder.set_vertex_buffer(0, Some(&hero_rect_buffer), 0);
                    encoder.set_vertex_buffer(1, Some(&pbuf), 0);
                    if let Some(_) = check_arrow_collisions(&mut arrows, &hero, &apbuf) {
                        encoder.set_fragment_buffer(0, Some(&hit_color_buf), 0);
                    } else {
                        encoder.set_fragment_buffer(0, Some(&cbuf), 0);
                    }
                    encoder.draw_primitives(MTLPrimitiveType::TriangleStrip, 0, 4);
                    
                    encoder.set_render_pipeline_state(&arrow_pipeline_state);
                    encoder.set_vertex_buffer(0, Some(&arrow_vbuf), 0);
                    encoder.set_vertex_buffer(1, Some(&arrow_pbuf), 0);
                    encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, 9 * num_arrows as u64);
                    
                    encoder.end_encoding();
                    command_buffer.present_drawable(&drawable);
                    command_buffer.commit();

            }
            //do_projectile_calcs();

            
            loop {
                let event = unsafe {
                    app.nextEventMatchingMask_untilDate_inMode_dequeue(
                        NSAnyEventMask, 
                        Some(&frame_time), 
                        NSDefaultRunLoopMode, 
                        true
                    )
                };
                match event {
                    Some(ref e) => {
                        unsafe {
                            match e.r#type() {
                                NSEventType::KeyDown => {
                                    key_pressed = e.keyCode();
                                    app.sendEvent(&e);
                                },
                                NSEventType::KeyUp => {
                                    if key_pressed == e.keyCode() {
                                        key_pressed = 112;
                                    }
                                    app.sendEvent(&e);
                                }
                                _ => app.sendEvent(&e),
                            }
                        }
                    },
                    None => break
                }
            }

        })
    }

}
