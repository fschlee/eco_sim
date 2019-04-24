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
    //color = frag_color;

    if (frag_mode == 0) {
        vec4 pos = gl_FragCoord;
        vec4 tex_col = texture(sampler2D(tex, samp), vec2(frag_uv.s, frag_uv.t));
        color = vec4(frag_color.xyz, tex_col.r);
    }
    else if (frag_mode == 1) {
        color =  frag_color * texture(sampler2D(tex, samp), vec2(frag_uv.s, frag_uv.t));
    }
    else if (frag_mode == 2){
        // vec4 pos = gl_FragCoord;
         // color = frag_color;// vec4(0.0, 1.0, 0.0, 1.0);
        // color = vec4(1.0, 0.0, 0.0,texture(sampler2D(tex, samp), vec2(frag_uv.s, frag_uv.t)).a);
        vec4 tex_col = texture(sampler2D(tex, samp), vec2(frag_uv.s, frag_uv.t));
        color = frag_color; // vec4(frag_color.xyz, 0.1); //texture(sampler2D(tex, samp), pos.pq); // texture(sampler2D(tex, samp), vec2(frag_uv.x, frag_uv.y));
    }
    color = vec4(pow(color.x, 2.2), pow(color.y, 2.2), pow(color.z, 2.2), color.a);
}
