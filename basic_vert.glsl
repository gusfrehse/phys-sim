#version 450

layout (binding = 0) uniform uniform_buffer_object {
    mat4 view;
    mat4 proj;
} ubo;

layout (binding = 2) uniform model_uniform {
    mat4 model;
} model;

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_color;
layout (location = 2) in vec2 in_tex_coord;

layout (location = 0) out vec3 frag_color;
layout (location = 1) out vec2 frag_tex_coord;

void main() {
    gl_Position = ubo.proj * ubo.view * model.model * vec4(in_position, 1.0);
    frag_color = in_color;
    frag_tex_coord = in_tex_coord;
}
