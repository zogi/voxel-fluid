uniform sampler2D depth;
uniform sampler3D voxels;

layout(std140, binding=2) uniform GridData {
    vec3 origin;
    vec3 size;
    ivec3 grid_dim;
    vec3 voxel_size; // size / grid_dim
    vec3 voxel_extinction;
    uint grid_flags;
    vec3 surface_color;
    float surface_opacity;
    float surface_roughness;
    float surface_level;
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
// Low Complexity, High Fidelity: The Rendering of INSIDE, GDC 2016
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

// Compute ray parameters (t values) to reach p along the three axis.
// Return the three such t values in a vector.
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

bool isOutOfGrid(ivec3 index)
{
    return any(lessThan(index, ivec3(0, 0, 0))) || any(greaterThanEqual(index, grid_dim));
}

float getVoxelDensity(ivec3 index)
{
    int density_quantization = densityQuantization();
    float density = texture(voxels, (vec3(index) + vec3(0.5, 0.5, 0.5)) / grid_dim).x;
    density = float(int(density * density_quantization)) / (density_quantization - 1);
    return clamp(density, 0, 1);
}

struct DDAData
{
    ivec3 i_step;
    vec3 t_step;
    ivec3 index;
    vec3 t3;
};

void DDAInit(Ray ray, ivec3 start_index, inout DDAData data)
{
    data.i_step = ivec3(sign(ray.direction));
    data.t_step = abs(voxel_size / ray.direction);
    data.index = start_index;
    data.t3 = paramToPoint(ray, getVoxelOrigin(start_index + clamp(data.i_step, 0, 1)));
}

void DDAStep(inout DDAData data)
{
    bvec3 mask = minMask(data.t3);
    data.index += data.i_step * ivec3(mask);
    data.t3 += data.t_step * ivec3(mask);
}

vec3 traceTransmittance(Ray ray, ivec3 start_index, int niter)
{
    DDAData data;
    DDAInit(ray, start_index, data);
    vec3 tau = vec3(0, 0, 0);
    float t = 0;
    for (int i = 0; i < niter; ++i) {
        if (isOutOfGrid(data.index)) {
            break;
        }
        float density = getVoxelDensity(data.index);
        float t_next = min3(data.t3);
        tau += (t_next - t) * density * voxel_extinction;
        t = t_next;
        DDAStep(data);
    }
    return exp(-tau);
}

// Optical thickness threshold for ray termination.
// exp(-8) < 0.001
uniform vec3 tau_threshold = vec3(8, 8, 8);

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

    // Just constant white background for now.
    vec3 background = vec3(1, 1, 1);

    // Perform DDA traversal.
    ivec3 i_step = ivec3(sign_direction);
    vec3 t_step = abs(voxel_size / ray.direction);
    vec3 p_entry = getRayPos(ray, t_in + 1e-5);
    ivec3 index = gridIndexFromWorldPos(p_entry);
    vec3 t3 = paramToPoint(ray, getVoxelOrigin(index + clamp(i_step, 0, 1)));
    float t = t_in;
    bool inside_fluid = false;
    ivec3 prev_index = index;
    for (int i = 0; i < 80; ++i) {
        float density = getVoxelDensity(index);

        // Reflect light on fluid-air interface.
        bool inside_fluid_next = density > surface_level;
        if (inside_fluid_next != inside_fluid) {
            vec3 wo = -ray.direction;
            vec3 wi = -reflect(wo, normal);
            Ray reflected;
            reflected.origin = getRayPos(ray, t);
            reflected.direction = wi;

            // Attenuate background radiance by attenuation along reflected direction.
            vec3 Li = background * traceTransmittance(reflected, prev_index, 20);

            // Add reflected radiance contribution.
            BRDFResult surface = GGX_specular(wi, wo, normal, 0.0, surface_roughness);
            radiance += exp(-tau) * surface_opacity * surface_color * surface.brdf * Li;

            // Attenuate light coming from behind the surface by increasing the optical thickness.
            tau += -log(1 - surface_opacity * surface.F + 1e-6);
        }
        inside_fluid = inside_fluid_next;
        prev_index = index;

        // Step to the next cell.
        float t_next = min(t_out, min3(t3));
        bvec3 mask = minMask(t3);
        index += i_step * ivec3(mask);
        t3 += t_step * ivec3(mask);
        tau += (t_next - t) * density * voxel_extinction;
        t = t_next;
        if (t >= t_out || isOutOfGrid(index) || all(greaterThan(tau, tau_threshold))) {
            break;
        }
        normal = -sign_direction * vec3(mask);
    }
    vec3 transmittance = exp(-tau);
    radiance += transmittance * background;

    // Gamma correction.
    radiance = pow(radiance, vec3(1, 1, 1) * 0.4545);

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
