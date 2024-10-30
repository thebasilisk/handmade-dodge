#include <metal_stdlib>
using namespace metal;

struct transform {
    packed_float2 position;
    float rotation;
};

struct rect {
    float w;
    float h;
};

struct ColorRect {
    rect rect;
    float4 color;
};

struct ColorInOut {
    float4 position [[position]];
    float4 color;
};


vertex ColorInOut rectangle_vertex (
    const device ColorRect *color_rect [[ buffer(0) ]],
    const device float2 *position [[ buffer(1) ]],
    unsigned int vid [[ vertex_id ]]
) {
    ColorInOut out;
    auto device const &pos = position[0];
    
    int vid_bit1 = vid % 2;
    int vid_bit2 = vid / 2;
    float x = pos.x + (color_rect->rect.w / 2) * (2 * vid_bit1 - 1);
    float y = pos.y - (color_rect->rect.h / 2) * (2 * vid_bit2 - 1);

    float4 out_pos = float4(x, y, 0, 1);
    out.position = out_pos;
    out.color = color_rect->color;

    return out;
}

vertex ColorInOut arrow_vertex (
    const device float2 *vertex_array [[ buffer(0) ]],
    const device transform *transform [[ buffer(1) ]],
    unsigned int vid [[ vertex_id ]]
) {
    ColorInOut out;

    auto device const &v = vertex_array[vid];
    auto device const &tran = transform[vid / 9];

    float theta = tran.rotation;
    float2x2 rot_matrix = float2x2(float2(cos(theta), sin(theta)), float2(-sin(theta), cos(theta)));
    float2 rot_and_trans_pos = float2(v.x, v.y) * rot_matrix + float2(tran.position.x, tran.position.y);

    out.position = float4(rot_and_trans_pos, 0.0, 1.0);
    out.color = float4(1.0, 0.0, 0.0, 1.0);

    return out;
}

fragment float4 fragment_shader(ColorInOut in [[stage_in]]) {
    return in.color;
}