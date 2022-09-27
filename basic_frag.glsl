#version 450

layout (binding = 1) uniform sampler2D tex_sampler;

layout (location = 0) in vec3 frag_color;
layout (location = 1) in vec2 frag_tex_coord;

layout (location = 0) out vec4 color_out;

void main() {
  color_out = vec4(frag_color * texture(tex_sampler, frag_tex_coord).rgb, 1.0);
}
