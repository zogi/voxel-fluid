#version 430

layout(std140) uniform CommonUniforms {
    mat4 mvp;
    mat3 eye_orientation;
    vec3 eye_pos;
    float cam_nearz;
    vec3 eye_dir;
    float time;
    vec2 view_size;
};
