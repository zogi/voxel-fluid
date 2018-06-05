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

bvec3 minMask(vec3 v)
{
    return bvec3(
        uint(v.x <= v.y) & uint(v.x <= v.z),
        uint(v.y < v.x) & uint(v.y <= v.z),
        uint(v.z < v.x) & uint(v.z < v.y));
}

bvec3 maxMask(vec3 v)
{
    return minMask(-v);
}

float min3(vec3 v)
{
    return min(v.x, min(v.y, v.z));
}

float max3(vec3 v)
{
    return max(v.x, max(v.y, v.z));
}

// Return ray parameterns (t values) which makes the ray reach the a given
// coordinate of p. Return the three such t values in a vector.
vec3 paramToPoint(Ray ray, vec3 p)
{
    return (p - ray.origin) / ray.direction;
}

struct BoxIntersection
{
    vec3 t3_in;
    vec3 t3_out;
};
BoxIntersection intersectRayBox(Ray ray, Box box)
{
    vec3 t0 = paramToPoint(ray, box.origin);
    vec3 t1 = paramToPoint(ray, box.origin + box.size);
    BoxIntersection isect;
    isect.t3_in = min(t0, t1);
    isect.t3_out = max(t0, t1);
    return isect;
}

ivec3 gridIndexFromWorldPos(vec3 p)
{
    return ivec3(floor((p - origin) / size * grid_dim));
}

vec3 getVoxelOrigin(ivec3 i)
{
    return origin + i * size / grid_dim;
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
    // Initialize ray.
    Ray ray;
    ray.origin = eye_pos;
    vec3 ray_dir_view = vec3(view_size * (uv - 0.5), -1.0);
    ray.direction = normalize(eye_orientation * ray_dir_view);

    // Compute entry and exit point parameters.
    Box bounds;
    bounds.origin = origin;
    bounds.size = size;
    BoxIntersection isect = intersectRayBox(ray, bounds);
    float t_in = max3(isect.t3_in);
    float t_out = min3(isect.t3_out);

    // Discard if ray misses the volume bounds.
    if (t_in >= t_out || t_out < 0) {
        discard;
    }

    // Handle camera inside grid bounds.
    bool inside_volume = (t_in < 0 && t_out > 0);
    float t_near = cam_nearz / dot(eye_dir, ray.direction);
    if (inside_volume) {
        t_in = t_near;
    }

    // Terminate ray if at current depth.
    float depth = texture(depth, uv).x;
    float t_thresh = t_near / depth;
    if (t_thresh <= t_in)
        discard;
    t_out = min(t_out, t_thresh);

    vec3 radiance = vec3(0, 0, 0);
    vec3 sign_direction = sign(ray.direction);
    vec3 normal = -sign_direction * vec3(maxMask(isect.t3_in));

    // Optical depth since last interface.
    vec3 tau = vec3(0, 0, 0);

    // Perform DDA traversal.
    vec3 voxel_size = size / grid_dim;
    ivec3 i_step = ivec3(sign_direction);
    vec3 t_step = abs(voxel_size / ray.direction);
    vec3 p_entry = getRayPos(ray, t_in + 1e-5);
    ivec3 index = gridIndexFromWorldPos(p_entry);
    vec3 t3 = paramToPoint(ray, getVoxelOrigin(index + clamp(i_step, 0, 1)));
    float t = t_in;
    bool inside_fluid = false;
    vec3 opacity = vec3(1, 1, 1)*0.01;
    for (int i = 0; i < 100; ++i) {
        vec3 extinction = getVoxelExtinction(index);
        bool inside_fluid_next = dot(extinction, vec3(1, 1, 1) / 3.0f) > 0.1;
        if (inside_fluid_next != inside_fluid) {
            vec3 Li = vec3(1, 1, 1);
            vec3 wo = -ray.direction;
            vec3 wi = -reflect(wo, normal);
            vec3 albedo = vec3(0, 0, 0);
            BRDFResult surface = BRDF(wi, wo, normal, 0.0, 0.2, albedo);
            radiance += exp(-tau) * opacity * surface.brdf * Li;
            tau += -log((1 - opacity) + opacity * (1 - surface.F));
        }
        inside_fluid = inside_fluid_next;

        float t_next = min(t_out, min3(t3));
        bvec3 mask = minMask(t3);
        index += i_step * ivec3(mask);
        t3 += t_step * ivec3(mask);
        tau += extinction * (t_next - t);
        t = t_next;
        if (t >= t_out || any(lessThan(index, ivec3(0, 0, 0))) || any(greaterThanEqual(index, grid_dim))) {
            break;
        }
        normal = -sign_direction * vec3(mask);
    }
    vec3 transmittance = exp(-tau);
    vec3 background = vec3(1, 1, 1);
    radiance += transmittance * background;

    // Dither.
    if (ditheringEnabled()) {
        float t = fract(time);
        radiance += vec3(
            dither(uv, t),
            dither(uv, t + 100.0),
            dither(uv, t + 200.0));
    }

    gl_FragColor = vec4(radiance, 1.0);
}
