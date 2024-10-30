use std::{f32::consts::PI, ffi::{c_float, CString}, mem, os::raw::c_char, ptr::NonNull};
use rand::random;
use objc::rc::autoreleasepool;
use objc2::rc::Retained;
use objc2_app_kit::{NSAnyEventMask, NSApplication, NSApplicationActivationPolicy, NSBackingStoreType, NSColor, NSEventType, NSScreen, NSWindow, NSWindowStyleMask};
use objc2_foundation::{CGPoint, MainThreadMarker, NSComparisonResult, NSDate, NSDefaultRunLoopMode, NSRect, NSSize, NSString};

use metal::*;

#[repr(C)]
#[derive(Debug)]
pub struct float4(c_float, c_float, c_float, c_float);
#[repr(C)]
#[derive(Debug)]
pub struct float3(c_float, c_float, c_float);
#[repr(C)]
#[derive(Debug)]
pub struct float2(c_float, c_float);

#[repr(C)]
struct Color {
    r : f32,
    b : f32,
    g : f32,
    a : f32
}
#[repr(C)]
struct Rect {
    w : f32,
    h : f32,
}
#[repr(C)]
struct ColorRect {
    rect : Rect,
    color : Color,
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

fn gen_random_arrows(n : u8, view_width : f32, view_height : f32) -> (Vec<[f32; 18]>, Vec<[f32; 3]>) {
    let mut arrows : Vec<[f32; 18]> = Vec::new();
    let mut positions : Vec<[f32; 3]> = Vec::new();
    for _ in 0..n {
        arrows.push(build_arrow_vertices(30.0 / view_width, 100.0 / view_height));
        positions.push([random::<f32>() * 2.0 - 3.0, random::<f32>() * 2.0 - 1.0, PI / 2.0]);
    };

    (arrows, positions)
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
        "fragment_shader"
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

    let hero_rect = vec![ColorRect {
        rect : Rect {
            w : 50.0 / view_width,
            h : 50.0 / view_height,
        },
        color : Color {
            r: 0.5, 
            b: 0.2, 
            g: 0.8, 
            a: 1.0 
        }
    }];

    let hero_rect_buffer = device.new_buffer_with_data(
        hero_rect.as_ptr() as *const _, 
        mem::size_of::<ColorRect>() as u64,
        MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeManaged
    );

    let center_x = view_width / 2.0;
    let center_y = view_height / 2.0;
    let mut current_x = center_x;
    let mut current_y = center_y;
    
    let start_position = [(current_x - center_x) / view_width, (current_y - center_y) / view_height];

    let pbuf = device.new_buffer_with_data(
        start_position.as_ptr() as *const _,
        (mem::size_of::<f32>() * start_position.len()) as u64,
        MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeManaged
    );

    // let arrow1_pos = [-150.0f32 / view_width, 100.0 / view_height, PI / 2.0];
    // let arrow2_pos = [-150.0f32 / view_width, 1.0 / view_height, PI / 2.0];

    // let arrow_positions = vec![arrow1_pos, arrow2_pos];
    let num_arrows = 16;
    let (arrow_vertices, mut arrow_positions) = gen_random_arrows(num_arrows, view_width, view_height);
    let arrow_pbuf = device.new_buffer_with_data(
        arrow_positions.as_ptr() as *const _, 
        (mem::size_of::<float3>() * arrow_positions.len()) as u64,
        MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeManaged
    );
    
    let arrow1_vertex_data = build_arrow_vertices(30.0 / view_width, 150.0 / view_height);
    // let arrow2_vertex_data = build_arrow_vertices(30.0 / view_width, 100.0 / view_height);

    // let arrow_vertices = vec![arrow1_vertex_data, arrow2_vertex_data];
    let arrow_vbuf = device.new_buffer_with_data(
        arrow_vertices.as_ptr() as *const _, 
        (arrow_vertices.len() * arrow1_vertex_data.len() * size_of::<f32>()) as u64, 
        MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeManaged
    );

    let fps = 60.0;
    let mut key_pressed : u16 = 112;
    let mut frame_time = get_next_frame(&fps);
    // let mut i = 0;
    // let mut j = 0;
    loop {
        autoreleasepool(|| {
            //do_game_tick_check();
            unsafe { 
                // i += 1;
                // println!("i: {}", i);
                if frame_time.compare(&NSDate::now()) == NSComparisonResult::Ascending {
                    // j += 1;
                    // println!("j: {}", j);
                    frame_time = get_next_frame(&fps);
                    for arrow in &mut *arrow_positions {
                        arrow[0] += 0.01;
                        if arrow[0] > 1.15 {
                            arrow[0] -= 2.0 + random::<f32>() * 2.0;
                        }
                    }
                    match key_pressed {
                        0 => current_x -= 5.0,
                        1 => current_y -= 5.0,
                        2 => current_x += 5.0,
                        13 => current_y += 5.0,
                        _ => ()
                    }
                    let p = pbuf.contents();
                    let position_data = [(current_x - center_x) / view_width, (current_y - center_y) / view_height];
                    
                    std::ptr::copy(
                        position_data.as_ptr(),
                        p as *mut f32,
                        (position_data.len() * mem::size_of::<float2>()) as usize
                    );
                    pbuf.did_modify_range(NSRange::new(
                        0 as u64, 
                        (position_data.len() * mem::size_of::<float2>()) as u64
                    ));
                    let ap = arrow_pbuf.contents();
                    std::ptr::copy(
                        arrow_positions.as_ptr(),
                        ap as *mut [f32; 3],
                        (arrow_positions.len() * mem::size_of::<[f32; 3]>()) as usize
                    );
                    arrow_pbuf.did_modify_range(NSRange::new(
                        0 as u64,
                        (arrow_positions.len() * mem::size_of::<[f32; 3]>()) as u64
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
                    encoder.draw_primitives(MTLPrimitiveType::TriangleStrip, 0, 4);
                    
                    encoder.set_render_pipeline_state(&arrow_pipeline_state);
                    encoder.set_vertex_buffer(0, Some(&arrow_vbuf), 0);
                    encoder.set_vertex_buffer(1, Some(&arrow_pbuf), 0);
                    encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, 9 * num_arrows as u64);
                    
                    encoder.end_encoding();
                    command_buffer.present_drawable(&drawable);
                    command_buffer.commit();

                }
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
