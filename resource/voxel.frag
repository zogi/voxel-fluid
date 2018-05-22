uniform sampler2D depth;
uniform sampler3D voxels;

layout(std140, binding=2) uniform GridData {
    vec3 origin;
    vec3 size;
    ivec3 grid_dim;
    vec3 voxel_extinction;
    uint grid_flags;
};

#define GRID_FLAGS_DENSITY_QUANTIZATION_MASK 0xff
#define GRID_FLAGS_DITHER_ENABLED_MASK 0x100

bool ditheringEnabled()
{
    return (grid_flags & GRID_FLAGS_DITHER_ENABLED_MASK) != 0;
}

int densityQuantization()
{
    return int(grid_flags & GRID_FLAGS_DENSITY_QUANTIZATION_MASK) + 1;
}

// PRNG functions. Source: https://thebookofshaders.com/10.
float rand(float n) { return fract(sin(n) * 43758.5453123); }
float rand2(vec2 v, float ofs) { return rand(ofs + dot(v, vec2(12.9898,78.233))); }

// Triangular PDF dither.
// Optimal Dither and Noise Shaping in Image Processing, 2008, Cameron Nicklaus Christou.
float dither(vec2 uv, float time)
{
    return (rand2(uv, time) + rand2(12.3456*uv, 76.5432*time) - 0.5) / 255.0;
}

struct Box {
    vec3 origin;
    vec3 size;
};

struct Ray {
    vec3 origin;
    vec3 direction;
};

vec3 getRayPos(Ray ray, float t)
{
    return ray.origin + t * ray.direction;
}

// Returns the parameters when the ray enters and exits the box in
// the first and second component of a vec2 respectively.
// If the enter parameter is greater than the exit parameter, the ray
// misses the box.
vec2 intersectRayBox(Ray ray, Box box)
{
    vec3 t0 = (box.origin - ray.origin) / ray.direction;
    vec3 t1 = (box.origin + box.size - ray.origin) / ray.direction;
    vec3 tm = min(t0, t1);
    vec3 tM = max(t0, t1);
    float t_enter = max(tm.x, max(tm.y, tm.z));
    float t_exit = min(tM.x, min(tM.y, tM.z));
    return vec2(t_enter, t_exit);
}

vec3 getVoxelExtinction(ivec3 index)
{
    int density_quantization = densityQuantization();
    float density = texture(voxels, (vec3(index) + vec3(0.5, 0.5, 0.5)) / grid_dim).x;
    density = float(int(density * density_quantization)) / (density_quantization - 1);
    density = clamp(density, 0, 1);
    return voxel_extinction * density;
}

in vec2 uv;
void main() {
    Ray ray;
    ray.origin = eye_pos;
    vec3 ray_dir_view = vec3(view_size * (uv - 0.5), -1.0);
    ray.direction = normalize(eye_orientation * ray_dir_view);

    // Find the entry and exit points of the grid domain.
    Box bounds;
    bounds.origin = origin;
    bounds.size = size;

    // Compute entry and exit point parameters.
    vec2 params = intersectRayBox(ray, bounds);
    float t_in = params.x;
    float t_out = params.y;

    // Discard if ray misses the volume bounds.
    if (t_in >= t_out || t_out < 0) {
        discard;
    }

    // Handle camera inside grid bounds.
    float t_near = cam_nearz / dot(eye_dir, ray.direction);
    if (t_in < 0 && t_out > 0) {
        t_in = t_near;
    }

    // Terminate rays using the depth buffer.
    float depth = texture(depth, uv).x;
    float t_thresh = t_near / depth;
    t_in = min(t_in, t_thresh);
    t_out = min(t_out, t_thresh);

    // Perform 3D DDA traversal.
    float eps = 1e-5;
    ivec3 index_min = ivec3(0, 0, 0);
    ivec3 index_entry = ivec3(floor((getRayPos(ray, t_in + eps) - bounds.origin) / bounds.size * grid_dim));
    ivec3 index = index_entry;
    Box voxel;
    voxel.size = bounds.size / grid_dim;
    voxel.origin = bounds.origin + index_entry * voxel.size;
    vec3 transmittance = vec3(1, 1, 1);
    float t = t_in;
    ivec3 step = ivec3(sign(ray.direction));
    for (int i = 0; i < 100; ++i) {
        vec3 extinction = getVoxelExtinction(index);
        vec3 t0 = (voxel.origin - ray.origin) / ray.direction;
        vec3 t1 = (voxel.origin + voxel.size - ray.origin) / ray.direction;
        vec3 tM = max(t0, t1);
        float t_exit = min(tM.x, min(tM.y, tM.z));
        t_exit = min(t_exit, t_out);
        transmittance *= exp(-extinction * (t_exit - t));
        t = t_exit;
        if (t >= t_out)
            break;
        if (t_exit == tM.x) {
            index.x += step.x;
            voxel.origin.x += step.x * voxel.size.x;
        }
        if (t_exit == tM.y) {
            index.y += step.y;
            voxel.origin.y += step.y * voxel.size.y;
        }
        if (t_exit == tM.z) {
            index.z += step.z;
            voxel.origin.z += step.z * voxel.size.z;
        }
        if (any(lessThan(index, index_min)) || any(greaterThanEqual(index, grid_dim))) {
            break;
        }
    }

    // Dither.
    if (ditheringEnabled()) {
        float t = fract(time);
        transmittance += vec3(
            dither(uv, t),
            dither(uv, t + 100.0),
            dither(uv, t + 200.0));
    }

    gl_FragColor = vec4(transmittance, 1.0);
}
