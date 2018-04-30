#include "common.h"
#define FLS_IMPLEMENTATION
#define FLS_USE_REMOTERY
#include "fluid_sim.h"
#include <remotery.h>

std::shared_ptr<spdlog::logger> g_logger;

// === OpenGL helpers ===

#define GL_CHECK() checkLastGLError(__FILE__, __LINE__)
void checkLastGLError(const char *file, int line)
{
    GLenum status = glGetError();
    if (status != GL_NO_ERROR) {
        g_logger->error("{} ({}): GL error {}: {}", file, line, status, gluErrorString(status));
    }
}

template <typename Traits>
class GLObject {
public:
    typedef typename Traits::value_type value_type;

    GLObject() : m_obj(s_null_value) {}
    ~GLObject() { release(); }

    template <typename... Args>
    static GLObject create(Args &&... args)
    {
        GLObject<Traits> res;
        res.m_obj = Traits::create(std::forward<Args>(args)...);
        return res;
    }

    void release()
    {
        if (m_obj != s_null_value) {
            Traits::destroy(m_obj);
            m_obj = s_null_value;
        }
    }

    GLObject(const GLObject &rhs) = delete;
    GLObject &operator=(const GLObject &rhs) = delete;
    GLObject(GLObject &&rhs) : m_obj(rhs.m_obj) { rhs.m_obj = value_type(); }
    GLObject &operator=(GLObject &&rhs)
    {
        std::swap(m_obj, rhs.m_obj);
        rhs.release();
        return *this;
    }

    operator value_type() const { return m_obj; }

private:
    value_type m_obj;
    static const value_type s_null_value;
};
template <typename Traits>
const typename GLObject<Traits>::value_type
    GLObject<Traits>::s_null_value = typename GLObject<Traits>::value_type();

struct GLFBOTraits {
    typedef GLuint value_type;
    static value_type create()
    {
        value_type res;
        glGenFramebuffers(1, &res);
        return res;
    }
    static void destroy(value_type fb) { glDeleteFramebuffers(1, &fb); }
};
typedef GLObject<GLFBOTraits> GLFBO;

struct GLUBOTraits {
    typedef GLuint value_type;
    static value_type create()
    {
        value_type ubo;
        glGenBuffers(1, &ubo);
        return ubo;
    }
    static void destroy(value_type ubo) { glDeleteBuffers(1, &ubo); }
};
typedef GLObject<GLUBOTraits> GLUBO;

struct GLVAOTraits {
    typedef GLuint value_type;
    static value_type create()
    {
        value_type res;
        glGenVertexArrays(1, &res);
        return res;
    }
    static void destroy(value_type vao) { glDeleteVertexArrays(1, &vao); }
};
typedef GLObject<GLVAOTraits> GLVAO;

struct GLTextureTraits {
    typedef GLuint value_type;
    static value_type create()
    {
        value_type res;
        glGenTextures(1, &res);
        return res;
    }
    static void destroy(value_type tex) { glDeleteTextures(1, &tex); }
};
typedef GLObject<GLTextureTraits> GLTexture;

struct GLShaderTraits {
    typedef GLuint value_type;
    static value_type create(GLenum shader_type) { return glCreateShader(shader_type); }
    static void destroy(value_type shader) { glDeleteShader(shader); }
};
typedef GLObject<GLShaderTraits> GLShader;

struct GLProgramTraits {
    typedef GLuint value_type;
    static value_type create() { return glCreateProgram(); }
    static void destroy(value_type program) { glDeleteProgram(program); }
};
typedef GLObject<GLProgramTraits> GLProgram;


GLShader createAndCompileShader(GLenum shader_type, const char *source, const char *defines = nullptr)
{
    GL_CHECK();
    auto shader = GLShader::create(shader_type);
    GL_CHECK();
    if (defines) {
        const char *sources[] = { defines, source };
        glShaderSource(shader, 2, sources, NULL);
        GL_CHECK();
    } else {
        glShaderSource(shader, 1, &source, NULL);
        GL_CHECK();
    }
    glCompileShader(shader);
    GL_CHECK();

    GLint is_compiled = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &is_compiled);
    if (is_compiled == GL_FALSE) {
        int info_log_length;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &info_log_length);
        std::vector<char> error_message(std::max(info_log_length, int(1)));
        glGetShaderInfoLog(shader, info_log_length, nullptr, error_message.data());
        if (!error_message.empty() && error_message[0]) {
            g_logger->error(error_message.data());
        }
        return {};
    }
    GL_CHECK();
    return shader;
}

GLProgram createAndLinkProgram(const std::vector<GLShader> &shaders)
{
    auto program = GLProgram::create();
    for (const auto &shader : shaders) {
        glAttachShader(program, shader);
    }
    glLinkProgram(program);
    GL_CHECK();

    GLint is_linked = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &is_linked);
    if (is_linked == GL_FALSE) {
        int info_log_length;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &info_log_length);
        std::vector<char> error_message(std::max(info_log_length, int(1)));
        glGetProgramInfoLog(program, info_log_length, nullptr, error_message.data());
        if (!error_message.empty() && error_message[0]) {
            g_logger->error(error_message.data());
        }
        return {};
    }
    GL_CHECK();
    return program;
}

// === Misc utility ===

template <typename Func>
class Finally {
public:
    Finally(Func func) : m_func(func) {}
    ~Finally() { m_func(); }

private:
    Func m_func;
};

template <typename Func>
Finally<Func> finally(const Func &func)
{
    return { func };
}

// === Shaders' codes used by the application ===

#pragma pack(push, 1)
struct CommonUniforms {
    glm::mat4 mvp;
    glm::mat3x4 eye_orientation; // Such that eye_orientation * (0, 0, -1) = eye_dir
    glm::vec3 eye_pos;
    float _pad1;
    glm::vec3 eye_dir;
    float _pad2;
    glm::vec2 view_size; // 2 * vec2(tan(fov_x/2), tan(fov_y/2))
    float cam_nearz;
    float time;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct GridData {
    glm::vec3 origin;
    float _pad1;
    glm::vec3 size;
    float _pad2;
    glm::ivec3 grid_dim;
    int _pad3;
};
#pragma pack(pop)

static std::string common_shader_code = R"glsl(
    #version 430
    layout(std140) uniform CommonUniforms {
        mat4 mvp;
        mat3 eye_orientation;
        vec3 eye_pos;
        vec3 eye_dir;
        vec2 view_size;
        float cam_nearz;
        float time;
    };
)glsl";
static std::string octree_renderer_vs_code = common_shader_code + R"glsl(
    in vec2 pos;
    out vec2 uv;
    void main() {
        gl_Position = vec4(pos, 0.0, 1.0);
        uv = 0.5f * pos + 0.5f;
    }
    )glsl";
static std::string octree_renderer_fs_code = common_shader_code + R"glsl(
    uniform sampler2D depth;
    uniform sampler3D voxels;

    layout(std140) uniform GridData {
        vec3 origin;
        vec3 size;
        ivec3 grid_dim;
    };

    uniform bool dithering_enabled = true;

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
    uniform vec3 voxel_extinction;
#if 0
    vec3 getVoxelExtinction(ivec3 index)
    {
        if (length(vec3(index) - vec3(1.5, 1.5, 1.5)) < 2)
            return voxel_extinction;
        else
            return vec3(0, 0, 0);
    }
#else
    vec3 getVoxelExtinction(ivec3 index)
    {
        float density = texture(voxels, (vec3(index) + vec3(0.5, 0.5, 0.5)) / grid_dim).x;
        return voxel_extinction * density;
    }
#endif

    in vec2 uv;
    void main() {
        Ray ray;
        ray.origin = eye_pos;
        vec3 ray_dir_view = vec3(view_size * (uv - 0.5), -1.0);
        ray.direction = normalize(eye_orientation * ray_dir_view);

        // Find the entry and exit points of the octree domain.
        Box bounds;
        bounds.origin = origin;
        bounds.size = size;

        // Compute entry and exit point parameters.
        vec2 params = intersectRayBox(ray, bounds);
        float t_in = params.x;
        float t_out = params.y;

        // Discard if ray misses the octee bounds.
        if (t_in >= t_out || t_out < 0) {
            discard;
        }

        // Handle camera inside octree bounds.
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
        voxel.origin = index_entry * voxel.size;
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
        if (dithering_enabled) {
            float t = fract(time);
            transmittance += vec3(
                dither(uv, t),
                dither(uv, t + 100.0),
                dither(uv, t + 200.0));
        }

        gl_FragColor = vec4(transmittance, 1.0);
    }
    )glsl";

static std::string color_cube_vs_code = common_shader_code + R"glsl(
    in vec3 pos;
    out vec3 color;
    void main() {
        gl_Position = mvp * vec4(pos, 1.0);
        color = pos + 0.5;
    }
    )glsl";
static std::string color_cube_fs_code = common_shader_code + R"glsl(
    in vec3 color;
    void main() {
        gl_FragColor = vec4(color, 1.0);
    }
    )glsl";

static std::string arrow_vs_code = R"glsl(
    #version 410
    uniform mat4 mvp;
    in vec3 pos;
    void main() {
        gl_Position = mvp * vec4(pos, 1.0);
    }
    )glsl";
static std::string arrow_fs_code = R"glsl(
    #version 410
    uniform vec3 color;
    void main() {
        gl_FragColor = vec4(color, 1.0);
    }
    )glsl";

// === Cube VAO ===

static GLVAO createCubeVAO()
{
    const uint16_t cube_indices[] = { 4, 2, 0, 6, 2, 4, 3, 5, 1, 7, 5, 3, 2, 1, 0, 3, 1, 2,
                                      5, 6, 4, 7, 6, 5, 1, 4, 0, 5, 4, 1, 6, 3, 2, 7, 3, 6 };
    GLVAO cube_vao = GLVAO::create();
    glBindVertexArray(cube_vao);

    GLuint index_buffer;
    glGenBuffers(1, &index_buffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cube_indices), cube_indices, GL_STATIC_DRAW);
    GL_CHECK();

    GLuint vertex_buffer;
    glGenBuffers(1, &vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    std::vector<float> quad_vertices;
    quad_vertices.reserve(3 * 8);
    for (int i = 0; i < 8; ++i) {
        quad_vertices.push_back(float((i & 1) >> 0) - 0.5f);
        quad_vertices.push_back(float((i & 2) >> 1) - 0.5f);
        quad_vertices.push_back(float((i & 4) >> 2) - 0.5f);
    }
    glBufferData(GL_ARRAY_BUFFER, quad_vertices.size() * sizeof(float), quad_vertices.data(), GL_STATIC_DRAW);
    GL_CHECK();
    glBindVertexArray(0);
    return cube_vao;
}

// === Framebuffer ===

struct Framebuffer {
    int width, height;
    GLFBO fbo;
    GLTexture color_texture, depth_texture;
    Framebuffer() : width(0), height(0) {}
    void init(int width, int height);
};

void Framebuffer::init(int width, int height)
{
    this->width = width;
    this->height = height;

    // Set up fbo with floating-point depth.

    color_texture = GLTexture::create();
    glBindTexture(GL_TEXTURE_2D, color_texture);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_SRGB8_ALPHA8, width, height);
    glBindTexture(GL_TEXTURE_2D, 0);

    depth_texture = GLTexture::create();
    glBindTexture(GL_TEXTURE_2D, depth_texture);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH_COMPONENT32F, width, height);
    glBindTexture(GL_TEXTURE_2D, 0);

    fbo = GLFBO::create();
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_texture, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_texture, 0);
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        g_logger->error("glCheckFramebufferStatus: {}", status);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

// === Camera ===

struct Camera {
    glm::quat orientation;
    glm::vec3 eye_pos;
    float pivot_distance;

    Camera() : pivot_distance(0.01f) {}
    glm::vec3 getForwardVector() const;
};

glm::vec3 Camera::getForwardVector() const { return glm::rotate(orientation, glm::vec3(0, 0, -1)); }

// === Global State ===

struct GUIState {
    bool test_window_open = false;
    bool console_open = false;
    bool overlay_open = false;
    bool settings_open = true;
};

struct RenderSettings {
    bool dither_voxels = true;
    glm::vec3 voxel_transmit_color = glm::vec3(60 / 255.0f, 195 / 255.0f, 222 / 255.0f);
    float voxel_extinction_intensity = 70.0f;
    bool visualize_velocity_field = true;
};

struct SimulationSettings {
    bool step_by_step = true;
    bool do_advection_step = false;
    bool do_pressure_step = false;
};

static Framebuffer g_framebuffer;
static Camera g_camera;
static float g_time_delta = 0;
static GUIState g_gui_state;
static RenderSettings g_render_settings;
static SimulationSettings g_simulation_settings;

// === Camera controller ===

// Modified version of cinder's CameraUI class (https://github.com/cinder/Cinder).

class CameraUI {
public:
    CameraUI(GLFWwindow *window = nullptr)
        : mWindow(window)
        , mCamera(nullptr)
        , mInitialPivotDistance(0.01f)
        , mMouseWheelMultiplier(-1.1f)
        , mMinimumPivotDistance(0.01f)
        , mLastAction(ACTION_NONE)
        , mEnabled(false)
    {
    }
    void setWindow(GLFWwindow *window) { mWindow = window; }
    void setCamera(Camera *camera) { mCamera = camera; }
    void setEnabled(bool enable) { mEnabled = enable; }

    void tick(); // Called once each frame.

    void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
    void cursorPositionCallback(GLFWwindow *window, double xpos, double ypos);
    void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods);
    void scrollCallback(GLFWwindow *window, double xoffset, double yoffset);

private:
    enum { ACTION_NONE, ACTION_ZOOM, ACTION_PAN, ACTION_TUMBLE };

    glm::ivec2 getWindowSize() const
    {
        if (!mWindow)
            return {};
        int w, h;
        glfwGetWindowSize(mWindow, &w, &h);
        return { w, h };
    }

    glm::vec2 mInitialMousePos;
    Camera mInitialCam;
    Camera *mCamera;
    float mInitialPivotDistance;
    float mMouseWheelMultiplier, mMinimumPivotDistance;
    int mLastAction;
    std::array<bool, 6> mMotionKeyState;

    GLFWwindow *mWindow;
    bool mEnabled;
};

void CameraUI::tick()
{
    if (!mEnabled || !mCamera || !mWindow)
        return;

    constexpr float CAMERA_SPEED = 2.0f; // units per second
    const auto forward_vector = mCamera->getForwardVector();
    const auto right_vector = glm::rotate(mCamera->orientation, glm::vec3(1, 0, 0));
    const auto up_vector = glm::rotate(mCamera->orientation, glm::vec3(0, 1, 0));

    glm::vec3 pos_delta = glm::vec3(0);

    if (mMotionKeyState[0])
        pos_delta += forward_vector;
    if (mMotionKeyState[1])
        pos_delta -= forward_vector;
    if (mMotionKeyState[2])
        pos_delta -= right_vector;
    if (mMotionKeyState[3])
        pos_delta += right_vector;
    if (mMotionKeyState[4])
        pos_delta += up_vector;
    if (mMotionKeyState[5])
        pos_delta -= up_vector;

    pos_delta *= g_time_delta * CAMERA_SPEED;
    mCamera->eye_pos += pos_delta;
    mInitialCam.eye_pos += pos_delta;
}

void CameraUI::keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    const bool pressed = action != GLFW_RELEASE;
    if (key == GLFW_KEY_W) {
        mMotionKeyState[0] = pressed;
    } else if (key == GLFW_KEY_S) {
        mMotionKeyState[1] = pressed;
    } else if (key == GLFW_KEY_A) {
        mMotionKeyState[2] = pressed;
    } else if (key == GLFW_KEY_D) {
        mMotionKeyState[3] = pressed;
    } else if (key == GLFW_KEY_Q) {
        mMotionKeyState[4] = pressed;
    } else if (key == GLFW_KEY_E) {
        mMotionKeyState[5] = pressed;
    }
}

void CameraUI::cursorPositionCallback(GLFWwindow *window, double xpos, double ypos)
{
    if (!mCamera || !mEnabled)
        return;

    const auto mousePos = glm::vec2(xpos, ypos);

    const bool leftDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    const bool middleDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
    const bool rightDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    const bool altPressed = glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS;
    const bool ctrlPressed = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS;

    int action;
    if (rightDown || (leftDown && middleDown) || (leftDown && ctrlPressed))
        action = ACTION_ZOOM;
    else if (middleDown || (leftDown && altPressed))
        action = ACTION_PAN;
    else if (leftDown)
        action = ACTION_TUMBLE;
    else
        return;

    if (action != mLastAction) {
        mInitialCam = *mCamera;
        mInitialPivotDistance = mCamera->pivot_distance;
        mInitialMousePos = mousePos;
    }

    mLastAction = action;

    const auto initial_forward = mInitialCam.getForwardVector();
    const auto world_up = glm::vec3(0, 1, 0);
    const auto window_size = getWindowSize();

    if (action == ACTION_ZOOM) { // zooming
        auto mouseDelta = (mousePos.x - mInitialMousePos.x) + (mousePos.y - mInitialMousePos.y);

        float newPivotDistance =
            powf(2.71828183f, 2 * -mouseDelta / glm::length(glm::vec2(window_size))) *
            mInitialPivotDistance;
        glm::vec3 oldTarget = mInitialCam.eye_pos + initial_forward * mInitialPivotDistance;
        glm::vec3 newEye = oldTarget - initial_forward * newPivotDistance;
        mCamera->eye_pos = newEye;
        mCamera->pivot_distance = std::max<float>(newPivotDistance, mMinimumPivotDistance);

    } else if (action == ACTION_PAN) { // panning
        float deltaX = (mousePos.x - mInitialMousePos.x) / float(window_size.x) * mInitialPivotDistance;
        float deltaY = (mousePos.y - mInitialMousePos.y) / float(window_size.y) * mInitialPivotDistance;
        const auto right = glm::cross(initial_forward, world_up);
        mCamera->eye_pos = mInitialCam.eye_pos - right * deltaX + world_up * deltaY;

    } else { // tumbling
        float deltaX = (mousePos.x - mInitialMousePos.x) / -100.0f;
        float deltaY = (mousePos.y - mInitialMousePos.y) / 100.0f;
        glm::vec3 mW = normalize(initial_forward);

        glm::vec3 mU = normalize(cross(world_up, mW));

        const bool invertMotion = (mInitialCam.orientation * world_up).y < 0.0f;
        if (invertMotion) {
            deltaX = -deltaX;
            deltaY = -deltaY;
        }

        glm::vec3 rotatedVec = glm::angleAxis(deltaY, mU) * (-initial_forward * mInitialPivotDistance);
        rotatedVec = glm::angleAxis(deltaX, world_up) * rotatedVec;

        mCamera->eye_pos = mInitialCam.eye_pos + initial_forward * mInitialPivotDistance + rotatedVec;
        mCamera->orientation =
            glm::angleAxis(deltaX, world_up) * glm::angleAxis(deltaY, mU) * mInitialCam.orientation;
    }
}

void CameraUI::mouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
{
    if (!mCamera || !mEnabled)
        return;

    if (action == GLFW_PRESS) {
        double x, y;
        glfwGetCursorPos(window, &x, &y);
        mInitialMousePos = glm::vec2(x, y);
        mInitialCam = *mCamera;
        mInitialPivotDistance = mCamera->pivot_distance;
        mLastAction = ACTION_NONE;

    } else if (action == GLFW_RELEASE) {
        mLastAction = ACTION_NONE;
    }
}

void CameraUI::scrollCallback(GLFWwindow *window, double xoffset, double yoffset)
{
    if (!mCamera || !mEnabled)
        return;

    // some mice issue mouseWheel events during middle-clicks; filter that out
    if (mLastAction != ACTION_NONE)
        return;

    const auto increment = float(yoffset);

    float multiplier;
    if (mMouseWheelMultiplier > 0)
        multiplier = powf(mMouseWheelMultiplier, increment);
    else
        multiplier = powf(-mMouseWheelMultiplier, -increment);
    const auto eye_dir = mCamera->getForwardVector();
    glm::vec3 newEye = mCamera->eye_pos + eye_dir * (mCamera->pivot_distance * (1 - multiplier));
    mCamera->eye_pos = newEye;
    mCamera->pivot_distance =
        std::max<float>(mCamera->pivot_distance * multiplier, mMinimumPivotDistance);
}

// === Console ===

// From imgui_demo.cpp (https://github.com/ocornut/imgui)

struct Console {
    char InputBuf[256];
    ImVector<char *> Items;
    bool ScrollToBottom;
    ImVector<char *> History;
    int HistoryPos; // -1: new line, 0..History.Size-1 browsing history.
    ImVector<const char *> Commands;

    Console()
    {
        ClearLog();
        memset(InputBuf, 0, sizeof(InputBuf));
        HistoryPos = -1;
        Commands.push_back("help");
        Commands.push_back("history");
        Commands.push_back("clear");
        Commands.push_back("camera");
        Commands.push_back("demo");
    }
    ~Console()
    {
        ClearLog();
        for (int i = 0; i < History.Size; i++)
            free(History[i]);
    }

    // Portable helpers
    static int Stricmp(const char *str1, const char *str2)
    {
        int d;
        while ((d = toupper(*str2) - toupper(*str1)) == 0 && *str1) {
            str1++;
            str2++;
        }
        return d;
    }
    static int Strnicmp(const char *str1, const char *str2, int n)
    {
        int d = 0;
        while (n > 0 && (d = toupper(*str2) - toupper(*str1)) == 0 && *str1) {
            str1++;
            str2++;
            n--;
        }
        return d;
    }
    static char *Strdup(const char *str)
    {
        size_t len = strlen(str) + 1;
        void *buff = malloc(len);
        return (char *)memcpy(buff, (const void *)str, len);
    }

    void ClearLog()
    {
        for (int i = 0; i < Items.Size; i++)
            free(Items[i]);
        Items.clear();
        ScrollToBottom = true;
    }

    void AddLog(const char *fmt, ...) IM_FMTARGS(2)
    {
        // FIXME-OPT
        char buf[1024];
        va_list args;
        va_start(args, fmt);
        vsnprintf(buf, IM_ARRAYSIZE(buf), fmt, args);
        buf[IM_ARRAYSIZE(buf) - 1] = 0;
        va_end(args);
        Items.push_back(Strdup(buf));
        ScrollToBottom = true;
    }

    void Draw(const char *title, bool *p_open)
    {
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(520, 300), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin(title, p_open)) {
            ImGui::End();
            return;
        }

        // As a specific feature guaranteed by the library, after calling Begin() the last Item
        // represent the title bar. So e.g. IsItemHovered() will return true when hovering the title
        // bar. Here we create a context menu only available from the title bar.
        if (ImGui::BeginPopupContextItem()) {
            if (ImGui::MenuItem("Close"))
                *p_open = false;
            ImGui::EndPopup();
        }

        if (ImGui::SmallButton("Clear")) {
            ClearLog();
        }
        ImGui::SameLine();
        bool copy_to_clipboard = ImGui::SmallButton("Copy");
        ImGui::SameLine();
        if (ImGui::SmallButton("Scroll to bottom"))
            ScrollToBottom = true;

        ImGui::Separator();

        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
        static ImGuiTextFilter filter;
        filter.Draw("Filter (\"incl,-excl\") (\"error\")", 180);
        ImGui::PopStyleVar();
        ImGui::Separator();

        const float footer_height_to_reserve =
            ImGui::GetStyle().ItemSpacing.y +
            ImGui::GetFrameHeightWithSpacing(); // 1 separator, 1 input text
        ImGui::BeginChild(
            "ScrollingRegion", ImVec2(0, -footer_height_to_reserve), false,
            ImGuiWindowFlags_HorizontalScrollbar); // Leave room for 1 separator + 1 InputText
        if (ImGui::BeginPopupContextWindow()) {
            if (ImGui::Selectable("Clear"))
                ClearLog();
            ImGui::EndPopup();
        }

        // Display every line as a separate entry so we can change their color or add custom
        // widgets. If you only want raw text you can use ImGui::TextUnformatted(log.begin(),
        // log.end()); NB- if you have thousands of entries this approach may be too inefficient and
        // may require user-side clipping to only process visible items. You can seek and display
        // only the lines that are visible using the ImGuiListClipper helper, if your elements are
        // evenly spaced and you have cheap random access to the elements. To use the clipper we
        // could replace the 'for (int i = 0; i < Items.Size; i++)' loop with:
        //     ImGuiListClipper clipper(Items.Size);
        //     while (clipper.Step())
        //         for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; i++)
        // However take note that you can not use this code as is if a filter is active because it
        // breaks the 'cheap random-access' property. We would need random-access on the
        // post-filtered list. A typical application wanting coarse clipping and filtering may want
        // to pre-compute an array of indices that passed the filtering test, recomputing this array
        // when user changes the filter, and appending newly elements as they are inserted. This is
        // left as a task to the user until we can manage to improve this example code! If your
        // items are of variable size you may want to implement code similar to what
        // ImGuiListClipper does. Or split your data into fixed height items to allow random-seeking
        // into your list.
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 1)); // Tighten spacing
        if (copy_to_clipboard)
            ImGui::LogToClipboard();
        for (int i = 0; i < Items.Size; i++) {
            const char *item = Items[i];
            if (!filter.PassFilter(item))
                continue;
            ImVec4 col =
                ImVec4(1.0f, 1.0f, 1.0f, 1.0f); // A better implementation may store a type per-item.
                                                // For the sample let's just parse the text.
            if (strstr(item, "[error]"))
                col = ImColor(1.0f, 0.4f, 0.4f, 1.0f);
            else if (strncmp(item, "# ", 2) == 0)
                col = ImColor(1.0f, 0.78f, 0.58f, 1.0f);
            ImGui::PushStyleColor(ImGuiCol_Text, col);
            ImGui::TextUnformatted(item);
            ImGui::PopStyleColor();
        }
        if (copy_to_clipboard)
            ImGui::LogFinish();
        if (ScrollToBottom)
            ImGui::SetScrollHere();
        ScrollToBottom = false;
        ImGui::PopStyleVar();
        ImGui::EndChild();
        ImGui::Separator();

        // Command-line
        if (ImGui::InputText(
                "Input", InputBuf, IM_ARRAYSIZE(InputBuf),
                ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CallbackCompletion |
                    ImGuiInputTextFlags_CallbackHistory,
                &TextEditCallbackStub, (void *)this)) {
            char *input_end = InputBuf + strlen(InputBuf);
            while (input_end > InputBuf && input_end[-1] == ' ') {
                input_end--;
            }
            *input_end = 0;
            if (InputBuf[0])
                ExecCommand(InputBuf);
            strcpy(InputBuf, "");
        }

        // Demonstrate keeping auto focus on the input box
        if (ImGui::IsItemHovered() || (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) &&
                                       !ImGui::IsAnyItemActive() && !ImGui::IsMouseClicked(0)))
            ImGui::SetKeyboardFocusHere(-1); // Auto focus previous widget

        ImGui::End();
    }

    void ExecCommand(const char *command_line)
    {
        AddLog("# %s\n", command_line);

        // Insert into history. First find match and delete it so it can be pushed to the back. This
        // isn't trying to be smart or optimal.
        HistoryPos = -1;
        for (int i = History.Size - 1; i >= 0; i--)
            if (Stricmp(History[i], command_line) == 0) {
                free(History[i]);
                History.erase(History.begin() + i);
                break;
            }
        History.push_back(Strdup(command_line));

        const auto is_command = [](const char *command_line, const char *command) {
            const auto cmd_len = strnlen(command, 255);
            const auto cmdline_len = strnlen(command_line, 255);
            if (cmdline_len < cmd_len)
                return false;
            const auto is_word_end = [](char ch) { return std::isspace(ch) || ch == 0; };
            return Strnicmp(command_line, command, int(cmd_len)) == 0 &&
                   is_word_end(command_line[cmd_len]);
        };

        // Process command
        if (Stricmp(command_line, "clear") == 0) {
            ClearLog();

        } else if (Stricmp(command_line, "help") == 0) {
            AddLog("Commands:");
            for (int i = 0; i < Commands.Size; i++)
                AddLog("- %s", Commands[i]);

        } else if (Stricmp(command_line, "history") == 0) {
            int first = History.Size - 10;
            for (int i = first > 0 ? first : 0; i < History.Size; i++)
                AddLog("%3d: %s\n", i, History[i]);

        } else if (is_command(command_line, "camera")) {
            ExecCameraCommand(command_line + 6);

        } else if (Stricmp(command_line, "demo") == 0) {
            g_gui_state.test_window_open = true;
            AddLog("Opened imgui's demo window.");

        } else {
            AddLog("Unknown command: '%s'\n", command_line);
        }
    }

    void ExecCameraCommand(const char *args)
    {
        const auto usage_str = "[error] camera: invalid arguments\nusage: camera [pos|dir] x y z";

        char what[4];
        glm::vec3 v;

        if (sscanf(args, " %4s %f %f %f", what, &v.x, &v.y, &v.z) != 4) {
            AddLog(usage_str);
            return;
        }

        if (Stricmp(what, "pos") == 0) {
            g_camera.eye_pos = v;
            AddLog("Camera position set to (%.2f,%.2f,%.2f)", v.x, v.y, v.z);
        } else if (Stricmp(what, "dir") == 0) {
            g_camera.orientation = glm::quatLookAt(v, glm::vec3(0, 1, 0));
            AddLog("Camera direction set to (%.2f,%.2f,%.2f)", v.x, v.y, v.z);
        } else {
            AddLog(usage_str);
        }
    }

    static int
    TextEditCallbackStub(ImGuiTextEditCallbackData *data) // In C++11 you are better off using lambdas
                                                          // for this sort of forwarding callbacks
    {
        Console *console = (Console *)data->UserData;
        return console->TextEditCallback(data);
    }

    int TextEditCallback(ImGuiTextEditCallbackData *data)
    {
        // AddLog("cursor: %d, selection: %d-%d", data->CursorPos, data->SelectionStart, data->SelectionEnd);
        switch (data->EventFlag) {
        case ImGuiInputTextFlags_CallbackCompletion: {
            // Example of TEXT COMPLETION

            // Locate beginning of current word
            const char *word_end = data->Buf + data->CursorPos;
            const char *word_start = word_end;
            while (word_start > data->Buf) {
                const char c = word_start[-1];
                if (c == ' ' || c == '\t' || c == ',' || c == ';')
                    break;
                word_start--;
            }

            // Build a list of candidates
            ImVector<const char *> candidates;
            for (int i = 0; i < Commands.Size; i++)
                if (Strnicmp(Commands[i], word_start, (int)(word_end - word_start)) == 0)
                    candidates.push_back(Commands[i]);

            if (candidates.Size == 0) {
                // No match
                AddLog("No match for \"%.*s\"!\n", (int)(word_end - word_start), word_start);
            } else if (candidates.Size == 1) {
                // Single match. Delete the beginning of the word and replace it entirely so we've got nice casing
                data->DeleteChars((int)(word_start - data->Buf), (int)(word_end - word_start));
                data->InsertChars(data->CursorPos, candidates[0]);
                data->InsertChars(data->CursorPos, " ");
            } else {
                // Multiple matches. Complete as much as we can, so inputing "C" will complete to
                // "CL" and display "CLEAR" and "CLASSIFY"
                int match_len = (int)(word_end - word_start);
                for (;;) {
                    int c = 0;
                    bool all_candidates_matches = true;
                    for (int i = 0; i < candidates.Size && all_candidates_matches; i++)
                        if (i == 0)
                            c = toupper(candidates[i][match_len]);
                        else if (c == 0 || c != toupper(candidates[i][match_len]))
                            all_candidates_matches = false;
                    if (!all_candidates_matches)
                        break;
                    match_len++;
                }

                if (match_len > 0) {
                    data->DeleteChars((int)(word_start - data->Buf), (int)(word_end - word_start));
                    data->InsertChars(data->CursorPos, candidates[0], candidates[0] + match_len);
                }

                // List matches
                AddLog("Possible matches:\n");
                for (int i = 0; i < candidates.Size; i++)
                    AddLog("- %s\n", candidates[i]);
            }

            break;
        }
        case ImGuiInputTextFlags_CallbackHistory: {
            // Example of HISTORY
            const int prev_history_pos = HistoryPos;
            if (data->EventKey == ImGuiKey_UpArrow) {
                if (HistoryPos == -1)
                    HistoryPos = History.Size - 1;
                else if (HistoryPos > 0)
                    HistoryPos--;
            } else if (data->EventKey == ImGuiKey_DownArrow) {
                if (HistoryPos != -1)
                    if (++HistoryPos >= History.Size)
                        HistoryPos = -1;
            }

            // A better implementation would preserve the data on the current input line along with cursor position.
            if (prev_history_pos != HistoryPos) {
                data->CursorPos = data->SelectionStart = data->SelectionEnd = data->BufTextLen =
                    (int)snprintf(
                        data->Buf, (size_t)data->BufSize, "%s",
                        (HistoryPos >= 0) ? History[HistoryPos] : "");
                data->BufDirty = true;
            }
        }
        }
        return 0;
    }
};

// === Stat overlay ===

// From imgui_demo.cpp (https://github.com/ocornut/imgui)

static void ShowOverlay(bool *p_open)
{
    const float DISTANCE = 10.0f;
    static int corner = 0;
    ImVec2 window_pos = ImVec2(
        (corner & 1) ? ImGui::GetIO().DisplaySize.x - DISTANCE : DISTANCE,
        (corner & 2) ? ImGui::GetIO().DisplaySize.y - DISTANCE : DISTANCE);
    ImVec2 window_pos_pivot = ImVec2((corner & 1) ? 1.0f : 0.0f, (corner & 2) ? 1.0f : 0.0f);
    ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0f, 0.0f, 0.0f, 0.3f)); // Transparent background
    if (ImGui::Begin(
            "Stats", p_open,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize |
                ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings)) {
        const auto &p = g_camera.eye_pos;
        const auto &d = g_camera.getForwardVector();
        ImGui::Text("Camera position: (%.2f,%.2f,%.2f)", p.x, p.y, p.z);
        ImGui::Text("Camera direction: (%.2f,%.2f,%.2f)", d.x, d.y, d.z);
        // ImGui::Separator();
        if (ImGui::BeginPopupContextWindow()) {
            if (ImGui::MenuItem("Top-left", NULL, corner == 0))
                corner = 0;
            if (ImGui::MenuItem("Top-right", NULL, corner == 1))
                corner = 1;
            if (ImGui::MenuItem("Bottom-left", NULL, corner == 2))
                corner = 2;
            if (ImGui::MenuItem("Bottom-right", NULL, corner == 3))
                corner = 3;
            ImGui::EndPopup();
        }
        ImGui::End();
    }
    ImGui::PopStyleColor();
}

// === Settings window ===

// From imgui_demo.cpp (https://github.com/ocornut/imgui)

static void ShowSettings(bool *p_open)
{
    ImGui::SetNextWindowSize(ImVec2(430, 450), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Settings", p_open)) {
        ImGui::End();
        return;
    }

    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2, 2));
    ImGui::Columns(2);
    ImGui::Separator();

    // Rendering settings.
    {
        ImGui::PushID(0);
        ImGui::AlignTextToFramePadding();
        ImGui::SetNextTreeNodeOpen(true, ImGuiCond_FirstUseEver);

        const bool node_open = ImGui::TreeNode("Object", "Rendering");
        ImGui::NextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::NextColumn();
        if (node_open) {

            // sRGB.
            const bool srgb_enabled = glIsEnabled(GL_FRAMEBUFFER_SRGB) == GL_TRUE;
            ImGui::PushID(0);
            ImGui::AlignTextToFramePadding();
            ImGui::Text("enable sRGB");
            ImGui::NextColumn();
            bool f = srgb_enabled;
            ImGui::Checkbox("", &f);
            if (f && !srgb_enabled) {
                glEnable(GL_FRAMEBUFFER_SRGB);
            } else if (!f && srgb_enabled) {
                glDisable(GL_FRAMEBUFFER_SRGB);
            }
            ImGui::NextColumn();
            ImGui::PopID();

            // Dithering.
            ImGui::PushID(1);
            ImGui::AlignTextToFramePadding();
            ImGui::Text("enable dithering");
            ImGui::NextColumn();
            ImGui::Checkbox("", &g_render_settings.dither_voxels);
            ImGui::NextColumn();
            ImGui::PopID();

            // Voxel extinction color.
            ImGui::PushID(2);
            ImGui::AlignTextToFramePadding();
            ImGui::Text("transmit color");
            ImGui::NextColumn();
            ImGui::ColorPicker3("", &g_render_settings.voxel_transmit_color[0]);
            ImGui::NextColumn();
            ImGui::PopID();

            // Voxel extinction intensity.
            ImGui::PushID(3);
            ImGui::AlignTextToFramePadding();
            ImGui::Text("extinction intensity");
            ImGui::NextColumn();
            ImGui::SliderFloat("", &g_render_settings.voxel_extinction_intensity, 0, 10);
            ImGui::NextColumn();
            ImGui::PopID();

            ImGui::TreePop();
        }
        ImGui::PopID();
    }

    // Simulation settings.
    {
        ImGui::PushID(1);
        ImGui::AlignTextToFramePadding();
        ImGui::SetNextTreeNodeOpen(true, ImGuiCond_FirstUseEver);

        const bool node_open = ImGui::TreeNode("Object", "Simulation");
        ImGui::NextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::NextColumn();
        if (node_open) {

            // Simulate step-by-step.
            ImGui::PushID(0);
            ImGui::AlignTextToFramePadding();
            ImGui::Text("step-by-step-animation");
            ImGui::NextColumn();
            ImGui::Checkbox("", &g_simulation_settings.step_by_step);
            ImGui::NextColumn();
            ImGui::PopID();

            // Advance simulation by one step.
            ImGui::PushID(1);
            ImGui::AlignTextToFramePadding();
            g_simulation_settings.do_advection_step = ImGui::Button("Advect");
            ImGui::NextColumn();
            g_simulation_settings.do_pressure_step = ImGui::Button("Pressure Solve");
            ImGui::NextColumn();
            ImGui::PopID();

            // Visualize velocity field.
            ImGui::PushID(2);
            ImGui::AlignTextToFramePadding();
            ImGui::Text("visualize velocity field");
            ImGui::NextColumn();
            ImGui::Checkbox("", &g_render_settings.visualize_velocity_field);
            ImGui::NextColumn();
            ImGui::PopID();

            ImGui::TreePop();
        }
        ImGui::PopID();
    }

    ImGui::Columns(1);
    ImGui::Separator();
    ImGui::PopStyleVar();
    ImGui::End();
}


int main()
{
    // Set up logging.
    g_logger = spdlog::stderr_logger_mt("fluid");
    g_logger->set_level(spdlog::level::debug);
    glfwSetErrorCallback([](int error, const char *description) {
        g_logger->error("GLFW Error {}: {}", error, description);
    });

    // Initialize GLFW.
    if (!glfwInit()) {
        g_logger->critical("Failed to initialize GLFT.");
        abort();
    }
    const auto terminate_glfw = finally([]() { glfwTerminate(); });

    // Create window.
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_DEPTH_BITS, 0);
    glfwWindowHint(GLFW_STENCIL_BITS, 0);
    GLFWwindow *window = glfwCreateWindow(640, 480, "", nullptr, nullptr);
    if (!window) {
        g_logger->critical("Failed to create window.");
        abort();
    }
    const auto destroy_window = finally([&window]() { glfwDestroyWindow(window); });

    // Set up OpenGL context.
    {
        glfwMakeContextCurrent(window);
        if (!glfwExtensionSupported("GL_ARB_clip_control")) {
            g_logger->critical(
                "The 'GL_ARB_clip_control' extension is required by the application but "
                "not supported by the GL context.");
            abort();
        }

        const auto err = glewInit();
        if (GLEW_OK != err) {
            g_logger->critical("GLEW Error: {}", glewGetErrorString(err));
            return 1;
        }
        g_logger->info("Using GLEW {}", glewGetString(GLEW_VERSION));
    }

    // Initialize Remotery.
    Remotery *remotery;
    rmt_CreateGlobalInstance(&remotery);
    rmt_BindOpenGL();
    const auto terminate_remotery = finally([remotery]() {
        // Unfortunately rmt_UnbindOpenGL hangs if there is no remotry client connected to our
        // process. See: https://github.com/Celtoys/Remotery/issues/112.
        // rmt_UnbindOpenGL();
        // rmt_DestroyGlobalInstance(remotery);
    });

    // Enable vsync.
    glfwSwapInterval(1);
    GL_CHECK();

    // Set up reverse-Z.
    glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);
    GL_CHECK();
    glClearDepth(0.0);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_GEQUAL);
    const auto perspectiveInvZ = [](float fov_x, float aspect_ratio, float z_near) {
        float f = 1.0f / tan(fov_x / 2.0f);
        return glm::mat4(
            f, 0.0f, 0.0f, 0.0f, 0.0f, f * aspect_ratio, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f,
            0.0f, z_near, 0.0f);
    };
    const auto cam_fov_x = glm::radians(100.0f);
    const auto cam_near = 0.01f;

    // Set up backface culling.
    glEnable(GL_CULL_FACE);
    glFrontFace(GL_CCW);
    glCullFace(GL_BACK);

    // Other GL state.
    glClearColor(1, 1, 1, 1);
    glEnable(GL_FRAMEBUFFER_SRGB);

    // Set up a buffer for common uniforms.
    GLUBO common_ubo = GLUBO::create();
    constexpr GLuint common_ubo_bind_point = 1;
    glBindBufferBase(GL_UNIFORM_BUFFER, common_ubo_bind_point, common_ubo);
    // Allocate storage.
    glBufferData(GL_UNIFORM_BUFFER, sizeof(CommonUniforms), nullptr, GL_STATIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    GL_CHECK();

    // Set up a buffer for grid data.
    GLUBO grid_data_ubo = GLUBO::create();
    constexpr GLuint grid_data_ubo_bind_point = 2;
    glBindBufferBase(GL_UNIFORM_BUFFER, grid_data_ubo_bind_point, grid_data_ubo);
    // Allocate storage.
    glBufferData(GL_UNIFORM_BUFFER, sizeof(GridData), nullptr, GL_STATIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    GL_CHECK();

    // Set up the voxel shader program.
    GLProgram otr_program;
    const GLuint otr_depth_texture_unit = 1;
    GLuint otr_depth_uniform_loc = UINT_MAX; // will be set below.
    {
        std::vector<GLShader> shaders;
        shaders.push_back(createAndCompileShader(GL_VERTEX_SHADER, octree_renderer_vs_code.c_str()));
        shaders.push_back(createAndCompileShader(GL_FRAGMENT_SHADER, octree_renderer_fs_code.c_str()));
        otr_program = createAndLinkProgram(shaders);

        // Common uniforms.
        const auto common_ubo_index = glGetUniformBlockIndex(otr_program, "CommonUniforms");
        if (common_ubo_index != GL_INVALID_INDEX)
            glUniformBlockBinding(otr_program, common_ubo_index, common_ubo_bind_point);
        GL_CHECK();

        // Grid data.
        const auto grid_data_ubo_index = glGetUniformBlockIndex(otr_program, "GridData");
        if (grid_data_ubo_index != GL_INVALID_INDEX)
            glUniformBlockBinding(otr_program, grid_data_ubo_index, grid_data_ubo_bind_point);
        GL_CHECK();

        // Get depth sampler uniform locaction.
        otr_depth_uniform_loc = glGetUniformLocation(otr_program, "depth");
        GL_CHECK();
    }

    // Set up the program to draw the colorful cube.
    GLProgram color_cube_program;
    {
        std::vector<GLShader> shaders;
        shaders.push_back(createAndCompileShader(GL_VERTEX_SHADER, color_cube_vs_code.c_str()));
        shaders.push_back(createAndCompileShader(GL_FRAGMENT_SHADER, color_cube_fs_code.c_str()));
        color_cube_program = createAndLinkProgram(shaders);
        GL_CHECK();

        // Common uniforms.
        auto common_ubo_index = glGetUniformBlockIndex(color_cube_program, "CommonUniforms");
        if (common_ubo_index != GL_INVALID_INDEX)
            glUniformBlockBinding(color_cube_program, common_ubo_index, common_ubo_bind_point);
        GL_CHECK();
    }

    // Set up quad vertex data for the octree renderer program.
    auto quad_vao = GLVAO::create();
    {
        const glm::vec2 quad_vertices[6] = { { -1.0f, -1.0f }, { 1.0f, 1.0f },   { -1.0f, 1.0f },
                                             { 1.0f, 1.0f },   { -1.0f, -1.0f }, { 1.0f, -1.0f } };
        glBindVertexArray(quad_vao);
        GLuint quad_vertex_buffer;
        glGenBuffers(1, &quad_vertex_buffer);
        glBindBuffer(GL_ARRAY_BUFFER, quad_vertex_buffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quad_vertices), quad_vertices, GL_STATIC_DRAW);
        const auto vpos_location = glGetAttribLocation(otr_program, "pos");
        glEnableVertexAttribArray(vpos_location);
        glVertexAttribPointer(vpos_location, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);
        glBindVertexArray(0);
        GL_CHECK();
    }

    // Set up cube vertex data for the simple program.
    auto cube_vao = createCubeVAO();
    {
        glBindVertexArray(cube_vao);
        const auto vpos_location = glGetAttribLocation(color_cube_program, "pos");
        glEnableVertexAttribArray(vpos_location);
        glVertexAttribPointer(vpos_location, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);
        glBindVertexArray(0);
        GL_CHECK();
    }

    // Init framebuffer.
    {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);
        g_framebuffer.init(width, height);
    }

    // Init camera.
    g_camera.eye_pos = glm::vec3(0, 0, 2);
    g_camera.orientation = glm::quat_identity<float, glm::highp>();
    static CameraUI camera_ui;
    camera_ui.setWindow(window);
    camera_ui.setCamera(&g_camera);
    camera_ui.setEnabled(true);

    // Init imgui.
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfwGL3_Init(window, false);
    const auto terminate_imgui = finally([]() {
        ImGui_ImplGlfwGL3_Shutdown();
        ImGui::DestroyContext();
    });
    const auto &imgui_io = ImGui::GetIO();
    ImGui::StyleColorsClassic();
    Console console;

    // Init fluid simulator and voxel texture.
    const auto grid_dim = glm::ivec3(8, 8, 8);
    GridData grid_data;
    grid_data.grid_dim = grid_dim;
    grid_data.origin = glm::vec3(0, 0, 0);
    grid_data.size = glm::vec3(4, 2, 2);
    {
        void *ubo_ptr = glMapNamedBuffer(grid_data_ubo, GL_WRITE_ONLY);
        memcpy(ubo_ptr, &grid_data, sizeof(grid_data));
        glUnmapNamedBuffer(grid_data_ubo);
        GL_CHECK();
    }
    const int fluid_fps = 30;

    const auto center = grid_data.grid_dim / 2 - 1;
    const auto wall_pos = glm::vec3(0, center.y, center.z);
    const auto source_pos = glm::vec3(1, center.y, center.z);
    sim::FluidSim fluid_sim(grid_dim, 1, 1.0f / fluid_fps, 1);
    fluid_sim.solidCells().emplace_back(wall_pos, glm::vec3(10.0f, 0.0f, 0.0f));
    {
        auto &grid = fluid_sim.grid();

        for (int i = 0; i < grid_dim.x; ++i)
            for (int j = 0; j < grid_dim.y; ++j)
                for (int k = 0; k < grid_dim.z; ++k)
                    grid.cell(i, j, k).concentration = 0;

        for (int i = 0; i <= grid_dim.x; ++i)
            for (int j = 0; j < grid_dim.y; ++j)
                for (int k = 0; k < grid_dim.z; ++k)
                    grid.u(i, j, k) = 0; // 10;
    }
    auto voxels = GLTexture::create();
    glBindTexture(GL_TEXTURE_3D, voxels);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexStorage3D(GL_TEXTURE_3D, 1, GL_R8, grid_dim.x, grid_dim.y, grid_dim.z);
    GL_CHECK();
    glBindTexture(GL_TEXTURE_3D, 0);
    std::vector<uint8_t> voxel_storage(grid_dim.x * grid_dim.y * grid_dim.z);
    const GLuint otr_voxels_texture_unit = 2;
    const GLuint otr_voxels_uniform_loc = glGetUniformLocation(otr_program, "voxels");

    // Init graphics resources related to fluid sim visualization.

    // Set up a simple program with uniform flat color.
    GLProgram arrow_program;
    {
        std::vector<GLShader> shaders;
        shaders.push_back(createAndCompileShader(GL_VERTEX_SHADER, arrow_vs_code.c_str()));
        shaders.push_back(createAndCompileShader(GL_FRAGMENT_SHADER, arrow_fs_code.c_str()));
        arrow_program = createAndLinkProgram(shaders);
        GL_CHECK();
    }
    const GLuint arrow_program_mvp_loc = glGetUniformLocation(arrow_program, "mvp");
    const GLuint arrow_program_color_loc = glGetUniformLocation(arrow_program, "color");

    auto arrow_vao = GLVAO::create();
    {
        glBindVertexArray(arrow_vao);
        GLuint arrow_vertex_buffer;
        glGenBuffers(1, &arrow_vertex_buffer);
        glBindBuffer(GL_ARRAY_BUFFER, arrow_vertex_buffer);
        const glm::vec3 arrow_vertices[] = { { 0, 0, 0 },     { 1, 0, 0 }, { 1, 0, 0 },
                                             { 0.8, 0.1, 0 }, { 1, 0, 0 }, { 0.8, -0.1, 0 } };
        glBufferData(GL_ARRAY_BUFFER, sizeof(arrow_vertices), arrow_vertices, GL_STATIC_DRAW);
        const auto vpos_location = glGetAttribLocation(arrow_program, "pos");
        glEnableVertexAttribArray(vpos_location);
        glVertexAttribPointer(vpos_location, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);
        glBindVertexArray(0);
        GL_CHECK();
    }
    const auto drawArrow = [&arrow_vao, &arrow_program, arrow_program_mvp_loc, arrow_program_color_loc](
                               const glm::vec3 &from, const glm::vec3 &to, const glm::vec3 &color,
                               const glm::mat4 &view_proj) {
        glUseProgram(arrow_program);
        const float scale_factor = glm::length(to - from);
        const auto rotate = glm::mat4(glm::rotation({ 1, 0, 0 }, (to - from) / scale_factor));
        const auto scale = glm::scale(glm::vec3(scale_factor, scale_factor, scale_factor));
        const auto translate = glm::translate(from);
        const auto model = translate * rotate * scale;
        const auto mvp = view_proj * model;
        glUniformMatrix4fv(arrow_program_mvp_loc, 1, GL_FALSE, &mvp[0][0]);
        glUniform3f(arrow_program_color_loc, color.r, color.g, color.b);
        glBindVertexArray(arrow_vao);
        glDrawArrays(GL_LINES, 0, 6);
        GL_CHECK();
    };

    // Event callbacks.

    glfwSetFramebufferSizeCallback(window, [](GLFWwindow *, int width, int height) {
        glViewport(0, 0, width, height);
        g_framebuffer.init(width, height);
    });

    glfwSetKeyCallback(window, [](GLFWwindow *window, int key, int scancode, int action, int mods) {
        // Imgui.
        ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
        const auto &io = ImGui::GetIO();
        if (io.WantCaptureKeyboard)
            return;

        // Close window on ESC.
        if (key == GLFW_KEY_ESCAPE)
            glfwSetWindowShouldClose(window, GLFW_TRUE);

        // Open console when the F2 key is pressed.
        if (key == GLFW_KEY_F2 && action == GLFW_PRESS)
            g_gui_state.console_open = !g_gui_state.console_open;

        // Show overlay when the F3 key is pressed (yes, I like Minecraft).
        if (key == GLFW_KEY_F3 && action == GLFW_PRESS)
            g_gui_state.overlay_open = !g_gui_state.overlay_open;

        // Show settings window when the F4 key is pressed.
        if (key == GLFW_KEY_F4 && action == GLFW_PRESS)
            g_gui_state.settings_open = !g_gui_state.settings_open;

        // Camera.
        camera_ui.keyCallback(window, key, scancode, action, mods);
    });

    glfwSetCharCallback(window, [](GLFWwindow *window, unsigned int codepoint) {
        // Imgui.
        ImGui_ImplGlfw_CharCallback(window, codepoint);
    });

    glfwSetCursorPosCallback(window, [](GLFWwindow *window, double xpos, double ypos) {
        // Imgui.
        const auto &io = ImGui::GetIO();
        if (io.WantCaptureMouse)
            return;
        // Camera.
        camera_ui.cursorPositionCallback(window, xpos, ypos);
    });

    glfwSetMouseButtonCallback(window, [](GLFWwindow *window, int button, int action, int mods) {
        // Imgui.
        ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
        const auto &io = ImGui::GetIO();
        if (io.WantCaptureMouse)
            return;
        // Camera.
        camera_ui.mouseButtonCallback(window, button, action, mods);
    });

    glfwSetScrollCallback(window, [](GLFWwindow *window, double xoffset, double yoffset) {
        // Imgui.
        ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
        const auto &io = ImGui::GetIO();
        if (io.WantCaptureMouse)
            return;
        // Camera.
        camera_ui.scrollCallback(window, xoffset, yoffset);
    });

    // Main loop.
    glm::mat4 model = glm::mat4(1);
    while (!glfwWindowShouldClose(window)) {

        // Process messages.
        glfwPollEvents();

        // Start a new imgui frame.
        ImGui_ImplGlfwGL3_NewFrame();
        g_time_delta = imgui_io.DeltaTime;

        // Build GUI.
        {
            if (g_gui_state.console_open)
                console.Draw("console", &g_gui_state.console_open);
            if (g_gui_state.overlay_open)
                ShowOverlay(&g_gui_state.overlay_open);
            if (g_gui_state.settings_open)
                ShowSettings(&g_gui_state.settings_open);
            if (g_gui_state.test_window_open)
                ImGui::ShowDemoWindow(&g_gui_state.test_window_open);
        }

        // Update objects.
        {
            camera_ui.tick();
        }

        // Update fluid.
        {
            rmt_ScopedCPUSample(AppFluidSim, 0);

            const bool do_advection =
                !g_simulation_settings.step_by_step || g_simulation_settings.do_advection_step;
            const bool do_pressure =
                !g_simulation_settings.step_by_step || g_simulation_settings.do_pressure_step;
            if (do_advection) {
                rmt_ScopedCPUSample(AppFluidSimAdvect, 0);

                static float time = 0;
                const float phi = 0.6f;
                const float angle = 4 * time;
                fluid_sim.solidCells()[0].u = 16 * cos(phi);
                fluid_sim.solidCells()[0].v = 16 * sin(phi) * sin(angle);
                fluid_sim.solidCells()[0].w = 16 * sin(phi) * cos(angle);
                time += fluid_sim.dt();
                fluid_sim.grid().cell(source_pos).concentration += fluid_sim.dt() * sim::Float(20);
                fluid_sim.advect();
            }
            if (do_pressure) {
                rmt_ScopedCPUSample(AppFluidSimPressureSolve, 0);

                fluid_sim.pressureSolve();
                fluid_sim.pressureUpdate();
            }
        }

        // Update fluid voxel texture.
        {
            rmt_ScopedCPUSample(AppFluidVoxelUpdateCPU, 0);
            rmt_ScopedOpenGLSample(AppFluidVoxelUpdateGL);

            for (int i = 0; i < fluid_sim.grid().cellCount(); ++i) {
                const float c = fluid_sim.grid().cell(i).concentration;
                voxel_storage[i] = uint8_t(glm::clamp(c, 0.0f, 1.0f) * 255.0);
            }
            glTextureSubImage3D(
                voxels, 0, 0, 0, 0, grid_dim.x, grid_dim.y, grid_dim.z, GL_RED, GL_UNSIGNED_BYTE,
                voxel_storage.data());
            GL_CHECK();
        }

        // Draw.
        {
            rmt_ScopedOpenGLSample(AppFrameGL);

            glBindFramebuffer(GL_FRAMEBUFFER, g_framebuffer.fbo);

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            // Fill common uniform data.
            glm::mat4 view_proj;
            {
                const float aspect_ratio = g_framebuffer.width / (float)g_framebuffer.height;

                void *ubo_ptr = glMapNamedBuffer(common_ubo, GL_WRITE_ONLY);
                CommonUniforms &uniforms = *reinterpret_cast<CommonUniforms *>(ubo_ptr);
                uniforms.eye_pos = g_camera.eye_pos;
                uniforms.eye_dir = g_camera.getForwardVector();
                const auto view =
                    glm::translate(glm::toMat4(glm::inverse(g_camera.orientation)), -uniforms.eye_pos);
                const auto proj = perspectiveInvZ(cam_fov_x, aspect_ratio, cam_near);
                model = glm::rotate(model, glm::radians(60.0f) * g_time_delta, glm::vec3(1, 1, 1));
                uniforms.mvp = proj * view * model;
                const auto sz_x = 2 * tan(cam_fov_x / 2);
                uniforms.view_size = glm::vec2(sz_x, sz_x / aspect_ratio);
                uniforms.eye_orientation = glm::toMat3(g_camera.orientation);
                uniforms.cam_nearz = cam_near;
                uniforms.time = float(glfwGetTime());
                glUnmapNamedBuffer(common_ubo);

                view_proj = proj * view;
            }

            // Visualize velocity field.
            if (g_render_settings.visualize_velocity_field) {
                const auto dx = fluid_sim.dx();
                const auto renderFromGrid = grid_data.size / glm::vec3(grid_data.grid_dim);
                const auto renderFromPhys = renderFromGrid / dx;

                // U.
                const auto &uGrid = fluid_sim.grid().uGrid();
                for (int idx = 0; idx < uGrid.cellCount(); ++idx) {
                    const auto idx3 = uGrid.indexGridFromLinear(idx);
                    const auto u = uGrid.cell(idx3);
                    if (std::abs(u) < 1e-1)
                        continue;

                    const auto grid_offset = glm::vec3(idx3) + glm::vec3(0, 0.5, 0.5);
                    const auto pos = grid_data.origin + grid_offset * renderFromGrid;
                    const auto length = u * renderFromPhys.x;
                    drawArrow(pos, pos + glm::vec3(length, 0, 0), glm::vec3(0, 0, 0), view_proj);
                }

                // V.
                const auto &vGrid = fluid_sim.grid().vGrid();
                for (int idx = 0; idx < vGrid.cellCount(); ++idx) {
                    const auto idx3 = vGrid.indexGridFromLinear(idx);
                    const auto v = vGrid.cell(idx3);
                    if (std::abs(v) < 1e-1)
                        continue;

                    const auto grid_offset = glm::vec3(idx3) + glm::vec3(0.5, 0, 0.5);
                    const auto pos = grid_data.origin + grid_offset * renderFromGrid;
                    const auto length = v * renderFromPhys.y;
                    drawArrow(pos, pos + glm::vec3(0, length, 0), glm::vec3(0, 0, 0), view_proj);
                }

                // W.
                const auto &wGrid = fluid_sim.grid().wGrid();
                for (int idx = 0; idx < wGrid.cellCount(); ++idx) {
                    const auto idx3 = wGrid.indexGridFromLinear(idx);
                    const auto w = wGrid.cell(idx3);
                    if (std::abs(w) < 1e-1)
                        continue;

                    const auto grid_offset = glm::vec3(idx3) + glm::vec3(0.5, 0.5, 0);
                    const auto pos = grid_data.origin + grid_offset * renderFromGrid;
                    const auto length = w * renderFromPhys.z;
                    drawArrow(pos, pos + glm::vec3(0, 0, length), glm::vec3(0, 0, 0), view_proj);
                }
            }

            // Draw cube.
            glUseProgram(color_cube_program);
            glBindVertexArray(cube_vao);
            glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, nullptr);
            GL_CHECK();

            // Disable and detach z buffer.
            glDisable(GL_DEPTH_TEST);
            glDepthMask(GL_FALSE);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
            GL_CHECK();

            // Bind depth texture.
            glBindTextureUnit(otr_depth_texture_unit, g_framebuffer.depth_texture);

            // Bind voxel texture.
            glBindTextureUnit(otr_voxels_texture_unit, voxels);

            // Set up multiplicative blending.
            glEnable(GL_BLEND);
            glBlendFunc(GL_ZERO, GL_SRC_COLOR);

            // Set up the octree renderer program.
            glUseProgram(otr_program);
            // Set depth texture unit.
            glUniform1i(otr_depth_uniform_loc, otr_depth_texture_unit);
            // Set voxel texture unit.
            glUniform1i(otr_voxels_uniform_loc, otr_voxels_texture_unit);
            // Set dithering_enabled uniform.
            {
                const auto dithering_enabled_uniform_loc =
                    glGetUniformLocation(otr_program, "dithering_enabled");
                glUniform1i(dithering_enabled_uniform_loc, g_render_settings.dither_voxels);
            }
            // Set voxel_extinction uniform.
            {
                const auto extinction = g_render_settings.voxel_extinction_intensity *
                                        (glm::vec3(1, 1, 1) - g_render_settings.voxel_transmit_color);
                const auto voxel_extinction_uniform_loc =
                    glGetUniformLocation(otr_program, "voxel_extinction");
                glUniform3f(voxel_extinction_uniform_loc, extinction.r, extinction.g, extinction.b);
            }

            // Draw fluid volume.
            {
                rmt_ScopedOpenGLSample(DrawVolumeGL);

                glBindVertexArray(quad_vao);
                glDrawArrays(GL_TRIANGLES, 0, 6);
                GL_CHECK();
            }

            // Unbind depth texture.
            glBindTextureUnit(otr_depth_texture_unit, 0);

            // Re-enable and re-attach z buffer.
            glEnable(GL_DEPTH_TEST);
            glDepthMask(GL_TRUE);
            glFramebufferTexture2D(
                GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, g_framebuffer.depth_texture, 0);
            GL_CHECK();

            // Disable blending.
            glDisable(GL_BLEND);

            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            GL_CHECK();

            // Copy framebuffer contents to the display framebuffer (window).
            glBindFramebuffer(GL_READ_FRAMEBUFFER, g_framebuffer.fbo);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0); // default FBO
            glBlitFramebuffer(
                0, 0, g_framebuffer.width, g_framebuffer.height, 0, 0, g_framebuffer.width,
                g_framebuffer.height, GL_COLOR_BUFFER_BIT, GL_LINEAR);
        }

        // Draw GUI.
        {
            rmt_ScopedOpenGLSample(AppDrawGuiGL);
            ImGui::Render();
            ImGui_ImplGlfwGL3_RenderDrawData(ImGui::GetDrawData());
        }

        // Present.
        {
            rmt_ScopedOpenGLSample(AppPresentGL);
            glfwSwapBuffers(window);
        }
    }

    return 0;
}
