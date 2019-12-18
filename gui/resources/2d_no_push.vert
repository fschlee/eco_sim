#version 460
#ifdef VULKAN
#extension GL_EXT_nonuniform_qualifier : require
#else
#extension GL_EXT_nonuniform_qualifier : require
#endif
struct PushConsts {
    float x_offset;
    float y_offset;
    uint highlighted;
    float z;
};
#ifdef VULKAN
layout (set = 1, binding = 0) uniform PC {
    float width;
    float height;
    uint pad0;
    uint pad1;
    PushConsts inner[256];
} fush;
#else
layout (binding = 0) uniform PC {
    float width;
    float height;
    uint pad0;
    uint pad1;
    PushConsts inner[256];
} fush;
#endif
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

    float width = fush.width; //1024.0;
    float height = fush.height;
    #ifdef VULKAN
    int idx = gl_InstanceIndex;
    PushConsts push = fush.inner[gl_InstanceIndex];
    #else
    PushConsts push = fush.inner[gl_BaseInstance + gl_InstanceID];
    int idx = gl_BaseInstance + gl_InstanceID;
    #endif
    float margin = 80.0;
    gl_Position = vec4(
    (margin + push.x_offset + 80.0 * position.x - 0.5 * width) * height,
    (margin + 80.0 * position.y + push.y_offset - 0.5 * height) * width,
    1.0 - push.z,
    0.5 * height * width);
    frag_color = color;
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
/*
struct PushConsts {
    float width;
    float height;
    float x_offset;
    float y_offset;
    uint highlighted;
};

layout (binding = 2) uniform PC {
    PushConsts inner;
} array[];

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
    PushConsts push = array[gl_InstanceIndex].inner;
    float width = push.width; //1024.0;
    float height = push.height;
    float margin = 80.0;
    gl_Position = vec4(
        (margin + push.x_offset + 80.0 * position.x - 0.5 * width) * height,
        (margin + 80.0 * position.y + push.y_offset - 0.5 * height) * width,
        0.0,
        0.5 * height * width);
    frag_color = color;
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
}*/