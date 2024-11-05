#include <metal_stdlib>
using namespace metal;

// struct transform {
//     packed_float2 position;
//     float rotation;
// };

struct rect {
    float w;
    float h;
};

struct ColorInOut {
    float4 position [[position]];
    float4 color;
};

struct PosInOut {
    float4 position [[position]];
};

struct PelInOut {
    float4 position [[position]];
    float2 center;
    float2 dimensions;
};


//check if works for multiple draws
vertex PosInOut rectangle_vertex (
    const device rect *rect [[ buffer(0) ]],
    const device float2 *position [[ buffer(1) ]],
    unsigned int vid [[ vertex_id ]]
) {
    PosInOut out;
    auto device const &pos = position[0];
    
    int vid_bit1 = vid % 2;
    int vid_bit2 = vid / 2;
    float x = pos.x + (rect->w / 2) * (2 * vid_bit1 - 1);
    float y = pos.y - (rect->h / 2) * (2 * vid_bit2 - 1);

    float4 out_pos = float4(x, y, 0, 1);
    out.position = out_pos;

    return out;
}

fragment float4 rectangle_shader (
    PosInOut in [[stage_in]],
    const device float4 *color [[ buffer(0) ]]
) {
    auto device const &col = color[0];
    return col;
}

vertex ColorInOut rect_vertex_instanced (
    const device rect *rect [[ buffer(0) ]],
    const device float2 *position [[ buffer(1) ]],
    unsigned int vid [[ vertex_id ]],
    unsigned int id [[ instance_id ]]
) {
    ColorInOut out;
    auto device const &pos = position[id];
    
    int vid_bit1 = vid % 2;
    int vid_bit2 = vid / 2;
    float x = pos.x + (rect->w / 2) * (2 * vid_bit1 - 1);
    float y = pos.y - (rect->h / 2) * (2 * vid_bit2 - 1);

    float4 out_pos = float4(x, y, 0, 1);
    out.position = out_pos;
    out.color = float4(0.15, 0.45, 0.8, 1.0);
    // out.center = float2(pos.x, pos.y);
    // out.dimensions = float2(rect->w, rect->h);

    return out;
}

fragment float4 pellet_shader (
    PelInOut in [[stage_in]]
) {
    //constant float* fragment_args [[ buffer(0) ]] //put in args
    //float screen_width = fragment_args[0];
    //float screen_height = fragment_args[1];

    //float2 center = float2((in.center.x + 1.0) * screen_width / 2.0, abs(in.center.y - 1.0) * screen_height / 2.0);
    //float circle_gradient = 1.0 - sqrt(pow(in.position.x - center.x, 2) + pow(in.position.y - center.y, 2));
    float4 color = float4(0.15, 0.45, 0.8, 1.0);

    return float4(color.xyz, 1.0);
}

vertex ColorInOut arrow_vertex (
    const device float2 *vertex_array [[ buffer(0) ]],
    const device packed_float4 *pos_life [[ buffer(1) ]],
    unsigned int vid [[ vertex_id ]]
) {
    ColorInOut out;

    auto device const &v = vertex_array[vid];
    auto device const &tran = pos_life[vid / 9];

    float theta = tran.z;
    float2x2 rot_matrix = float2x2(float2(cos(theta), sin(theta)), float2(-sin(theta), cos(theta)));
    float2 rot_and_trans_pos = float2(v.x, v.y) * rot_matrix + float2(tran.x, tran.y);

    float opacity = mix(0.15, 1.0, tran.w / 3.5);
    out.position = float4(rot_and_trans_pos, 0.0, 1.0);
    out.color = float4(1.0, 0.0, 0.0, opacity);

    return out;
}

fragment float4 fragment_shader(
    ColorInOut in [[stage_in]],
    const device float *rot [[ buffer(0) ]]
) {
    auto device const &vision_cone_rotation = rot[0];

    float2 hero_pos_lock = float2(1024.0 / 2.0, 768.0 / 1.625);
    float2 rel_pos = in.position.xy - hero_pos_lock;

    float theta = atan2(rel_pos.x, -rel_pos.y);
    float fov = M_PI_4_F;
    // float vision_cone_rotation = M_PI_2_F;

    //float opacity = mix(1.0, 0.2, abs(vision_cone_rotation - theta) / fov);
    float opacity = (fov - abs(vision_cone_rotation - theta));

    return float4(in.color.rgb , in.color.a * opacity);
}
