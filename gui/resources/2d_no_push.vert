#version 450
layout (binding = 0) uniform PushConsts {
    layout(offset = 0) float width;
    layout(offset = 4) float height;
    layout(offset = 8) float x_offset;
    layout(offset = 12) float y_offset;
    layout(offset = 16) uint highlighted;
} push;
layout (location = 0) in vec2 position;
layout (location = 1) in vec2 vert_uv;
layout (location = 2) in uint mode;
layout (location = 3) in vec4 color;

layout (location = 0) out gl_PerVertex {
    vec4 gl_Position;
};
layout (location = 1) out vec4 frag_color;
layout (location = 2) out vec2 frag_uv;
layout (location = 3) flat out uint frag_mode;

float high(float old){
    return max(sqrt(old), 0.8 * old + 0.2);
}
void main()
{
    float width = push.width; //1024.0;
    float height = push.height;
    float margin = 80.0;
    gl_Position = vec4(
        (margin + push.x_offset + 80.0 * position.x - 0.5 * width) * height,
        (margin + 80.0 * position.y + push.y_offset - 0.5 * height) * width,
        0.0,
        0.5 * height * width);
    uint low = push.highlighted & 255;
    uint highlight = (push.highlighted & 256) >> 8;
    uint threat = (push.highlighted & 512) >> 9;

    if (threat == 1) {
        float red = max(0.0, (low -2.0) /256.0);
        float green = 1.0 - red;
        float mix = 0.5 * max(red * red, green * green);
        frag_color = (1.0 - mix) * color + vec4(red, green, 0.0, mix);
    }
    else {
        frag_color = color;
    }
    if (highlight == 1)
    {
        frag_color = vec4(high(frag_color.r) ,high(frag_color.g), high(frag_color.b), frag_color.a);
    }

    frag_uv = vert_uv;
    frag_mode = mode;
}