#version 450
layout (push_constant) uniform PushConsts {
    layout(offset = 0) float width;
    layout(offset = 4) float height;
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
void main()
{
    float width = push.width; //1024.0;
    float height = push.height;
    gl_Position = vec4(position.x * height, - position.y * width, 0.0, 0.5 * width * height);
        //

    // vec4(position.x * 2.0/1024.0 -1.f, position.y * 2.0/768.0 - 1.f, 0.0, 1.0);
    /*
    if (mode == 2) {
        frag_color = vec4(1.0, 0.0, 0.0, 0.5); //color;
    }
    else if (mode == 1){
        frag_color = vec4(0.0, 0.0, 1.0, 0.5);
    }
    else if (mode == 0){
        frag_color = vec4(0.0, 1.0, 0.0, 0.5);
    }
    else {
        frag_color = vec4(1.0, 1.0, 0.0, 0.5);
    }*/
    frag_color = color;
    // frag_color = vec4((data & 0x0000FF00u) >> 8, (data & 0xFF000000u)>> 24, (data & 0x00FF0000u) >> 16, 0.5);
    // frag_color = vec4((data & 0xFF000000u)>> 24, (data & 0x00FF0000u) >> 16, (data & 0x0000FF00u) >> 8, (data & 0x000000FFu) >> 0);
    // frag_color = vec4(1.0 - ((data & 0xFF000000u)>> 24), 1.0 - ((data & 0x00FF0000u) >> 16), 1.0-((data & 0x0000FF00u) >> 8), 1.0 - ((data & 0x000000FFu) >> 0));
    frag_uv = vert_uv;
    frag_mode = mode;
}