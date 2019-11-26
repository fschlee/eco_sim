#version 450
layout(set = 0, binding = 0) uniform texture2D tex;
layout(set = 0, binding = 1) uniform sampler samp;

in vec4 gl_FragCoord;
layout (location = 1) in vec4 frag_color;
layout (location = 2) in vec2 frag_uv;
layout (location = 3) flat in uint frag_mode;
layout (location = 0) out vec4 color;
void main()
{
    if (frag_mode == 1) {
        color = frag_color * texture(sampler2D(tex, samp), vec2(frag_uv.s, frag_uv.t));
    }
    else {
        color = frag_color;
    }
}
