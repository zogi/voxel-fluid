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

typedef std::string Path;

std::string readFileContents(const Path &file_path)
{
    std::ifstream ifs(file_path.c_str());
    if (!ifs) {
        g_logger->error("readFileContents: {}: cannot open file", file_path);
        return {};
    }
    return std::string(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
}

// === Shaders' codes used by the application ===

typedef glm::vec3 RGBColor;

#pragma pack(push, 1)
struct CommonUniforms {
    glm::mat4 mvp;
    glm::mat3x4 eye_orientation; // Such that eye_orientation * (0, 0, -1) = eye_dir
    glm::vec3 eye_pos;
    float cam_nearz;
    glm::vec3 eye_dir;
    float time;
    glm::vec2 view_size; // 2 * vec2(tan(fov_x/2), tan(fov_y/2))
};

struct GridData {
    glm::vec3 origin;
    uint32_t _pad1;
    glm::vec3 size;
    uint32_t _pad2;
    glm::ivec3 grid_dim;
    uint32_t _pad3;
    glm::vec3 voxel_size;
    uint32_t _pad4;

    RGBColor voxel_extinction;
    uint32_t grid_flags;

    RGBColor surface_color;
    float surface_opacity;
    float surface_roughness;
    float surface_level;
};
#pragma pack(pop)

inline uint32_t packGridFlags(int8_t density_quantization, bool dithering_enabled)
{
    return uint32_t(density_quantization) | (uint32_t(dithering_enabled) << 8);
}

static std::string common_shader_code = R"glsl(
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
    #version 430
    uniform mat4 mvp;
    in vec3 pos;
    void main() {
        gl_Position = mvp * vec4(pos, 1.0);
    }
    )glsl";
static std::string arrow_fs_code = R"glsl(
    #version 430
    uniform vec4 color;
    void main() {
        gl_FragColor = color;
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

// === Render resources ===

const std::string kShadersPath = "shader";

typedef std::unordered_map<std::string, std::string> DefineMap;

GLProgram loadShader(const std::string &json_path, const DefineMap &defines = {})
{
    std::ifstream fin(json_path);
    if (!fin) {
        g_logger->error("loadShaderResource: cannot open {}", json_path);
        return {};
    }
    Json::Value root;
    fin >> root;
    if (!root) {
        g_logger->error("loadShaderResource: cannot parse json file {}", json_path);
        return {};
    }

    // Verify and collect shader stage entries.
    constexpr int kShaderTypeCount = 2;
    std::vector<Path> source_path_lists[kShaderTypeCount];

    GLenum shader_types[kShaderTypeCount] = { GL_VERTEX_SHADER, GL_FRAGMENT_SHADER };
    const char *json_keys[kShaderTypeCount] = { "vertex_shader", "fragment_shader" };
    for (int i = 0; i < kShaderTypeCount; ++i) {
        const auto key = json_keys[i];
        std::vector<Path> &path_list = source_path_lists[i];

        const auto &json_entry = root[key];
        if (!json_entry) {
            g_logger->error("loadShaderResource: {}: \"{}\" not found", json_path, key);
            return {};
        }
        if (json_entry.isArray()) {
            // Push elements in the json array into path_list.
            path_list.reserve(json_entry.size());
            for (const auto &elem : json_entry) {
                if (!elem.isString()) {
                    g_logger->error(
                        "loadShaderResource: {}: elements of \"{}\" must be strings strings",
                        json_path, key);
                    return {};
                }
                path_list.push_back(elem.asString());
            }
        } else if (json_entry.isString()) {
            // Set path_list to contain solely the value of json.
            path_list.push_back(json_entry.asString());
        } else {
            g_logger->error("loadShaderResource: {}: \"{}\" must be either array or string", json_path, key);
            return {};
        }
    };

    // Concatenate GLSL code from the defines.
    std::stringstream ss_defines;
    ss_defines << "#define JSON_SHADER\n";
    for (const auto &define : defines) {
        ss_defines << "#define " << define.first << " " << define.second << "\n";
    }

    // Load shader sources, then compile and link the shaders.
    std::vector<GLShader> shaders;
    for (int i = 0; i < kShaderTypeCount; ++i) {
        const auto shader_type = shader_types[i];
        const auto &source_path_list = source_path_lists[i];
        std::stringstream ss_source;
        ss_source << "#version 430\n";
        ss_source << ss_defines.str();
        for (const auto &source_path : source_path_list) {
            ss_source << readFileContents(kShadersPath + "/" + source_path) << "\n";
        }
        shaders.push_back(createAndCompileShader(shader_type, ss_source.str().c_str()));
    }
    auto program = createAndLinkProgram(shaders);
    if (!program) {
        g_logger->error("loadShaderResource: {}: failed to link shader", json_path);
        return {};
    }
    g_logger->info("loadShaderResource: shader successfully loaded from {}", json_path);
    return program;
}

struct ShaderResource {
    GLProgram program;
    Path path;
};

struct VoxelRenderer {
    ShaderResource shader;
    GLuint depth_uniform_loc = 0;
    GLuint grid_data_binding = 0;
};

struct RenderResources {
    VoxelRenderer vxr;
};

const GLuint kCommonUBOBindSlot = 1;
static RenderResources g_render_resources;

void loadVoxelShader()
{
    auto vxr_program = loadShader(g_render_resources.vxr.shader.path);
    if (!vxr_program)
        return;

    // Common uniforms.
    const auto common_ubo_index = glGetUniformBlockIndex(vxr_program, "CommonUniforms");
    if (common_ubo_index == GL_INVALID_INDEX) {
        g_logger->error("loadVoxelShader: no CommonUniforms UBO defined in the voxel shader.");
        return;
    }
    glUniformBlockBinding(vxr_program, common_ubo_index, kCommonUBOBindSlot);
    GL_CHECK();

    // Grid data.
    const auto grid_ubo_index = glGetUniformBlockIndex(vxr_program, "GridData");
    if (grid_ubo_index == GL_INVALID_INDEX) {
        g_logger->error("loadVoxelShader: no GridData UBO defined in the voxel shader.");
        return;
    }
    GLint grid_ubo_binding = 0;
    glGetActiveUniformBlockiv(vxr_program, grid_ubo_index, GL_UNIFORM_BLOCK_BINDING, &grid_ubo_binding);
    if (grid_ubo_binding == 0) {
        grid_ubo_binding = 2;
        glUniformBlockBinding(vxr_program, grid_ubo_index, grid_ubo_binding);
        g_logger->warn(
            "loadVoxelShader: no binding specified for GridData UBO in the voxel shader. Trying "
            "fallback index {}.",
            grid_ubo_binding);
    }
    GL_CHECK();

    // Get depth sampler uniform locaction.
    const auto depth_uniform_loc = glGetUniformLocation(vxr_program, "depth");
    GL_CHECK();

    auto &vxr = g_render_resources.vxr;
    vxr.depth_uniform_loc = depth_uniform_loc;
    vxr.grid_data_binding = grid_ubo_binding;
    vxr.shader.program = std::move(vxr_program);
}

void reloadShaders() { loadVoxelShader(); }

// === Global State ===

struct GUIState {
    bool test_window_open = false;
    bool console_open = false;
    bool overlay_open = false;
    bool settings_open = true;
};

struct RenderSettings {
    RGBColor voxel_transmit_color = { glm::vec3(60, 127, 222) / 255.0f };
    float voxel_extinction_intensity = 200.0f;
    RGBColor surface_color = glm::vec3(1, 1, 1);
    float surface_opacity = 0.008;
    float surface_roughness = 0.425;
    float surface_level = 0;
    int voxel_density_quantization = 8;
    glm::vec3 volume_origin = { 0, 0, 0 };
    glm::vec3 volume_size = { 4, 4, 4 };

    bool show_spinning_cube = false;
    bool dither_voxels = true;
    bool visualize_velocity_field[3] = { false, false, false };
    float velocity_arrow_opacity = 0.0f;
    bool show_velocity_arrow = true;
};

const float kInitialEmissionRate = 40.0f;
const float kInitialEmissionSpeed = 50.0f;

struct SimulationSettings {
    bool step_by_step = false;
    int max_solver_iterations = 50;
    float fluid_density = 1.0f;

    // For step-by-step simulation.
    bool do_advection_step = false;
    bool do_pressure_step = false;

    float curl_noise_strength = 0.0f;
    float curl_noise_frequency = 3.0f;

    // Fluid grid dimensions.
    sim::GridSize3 grid_dim = { 16, 16, 16 };

    // Fluid source.
    sim::SmokeData source_emission = sim::SmokeData(kInitialEmissionRate, 0);
    sim::GridIndex3 source_pos = sim::GridIndex3(0, 0, 0);
    SphericalCoords<sim::Float> source_velocity_spherical =
        sphericalFromEuclidean(kInitialEmissionSpeed * glm::normalize(sim::Vector3(1, 1, 1)));
};

struct OverlayData {
    int fluid_cell_count = 0;
    float voxel_draw_ms = 0;
    float advect_ms = 0;
    float pressure_projection_ms = 0;
};

static Framebuffer g_framebuffer;
static Camera g_camera;
static float g_time_delta = 0;
static GUIState g_gui_state;
static OverlayData g_overlay_data;
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

    constexpr float CAMERA_SPEED = 8.0f; // units per second
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
        ImGui::Spacing();
        ImGui::Text("Fluid cells: %d/%d", g_overlay_data.fluid_cell_count, sim::FluidSim::MAX_FLUID_CELL_COUNT);
        ImGui::Spacing();
        ImGui::Text("Fluid draw ms: %.2f", g_overlay_data.voxel_draw_ms);
        ImGui::Text("Advect fluid ms: %.2f", g_overlay_data.advect_ms);
        ImGui::Text("Pressure projection ms: %.2f", g_overlay_data.pressure_projection_ms);
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

bool ImGuiSlider(const char *label, int &var, int min, int max)
{
    return ImGui::SliderInt(label, &var, min, max);
};

bool ImGuiSlider(const char *label, float &var, float min, float max)
{
    return ImGui::SliderFloat(label, &var, min, max);
};

bool ImGuiSlider(const char *label, glm::vec3 &var, float min, float max)
{
    return ImGui::SliderFloat3(label, &var[0], min, max);
};

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

    const auto newTreeNode = [](const char *label, bool open_by_default = true) {
        ImGui::SetNextTreeNodeOpen(open_by_default, ImGuiCond_FirstUseEver);
        const bool node_open = ImGui::TreeNode(label, label);
        ImGui::NextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::NextColumn();
        return node_open;
    };

    const auto colorControl = [](const char *label, RGBColor &var) {
        ImGui::PushID(label);
        ImGui::AlignTextToFramePadding();
        ImGui::Text(label);
        ImGui::NextColumn();
        const auto res = ImGui::ColorPicker3("", &var[0]);
        ImGui::NextColumn();
        ImGui::PopID();
        return res;
    };

    const auto sliderControl = [](const char *label, auto &var, auto min, auto max) {
        ImGui::PushID(label);
        ImGui::AlignTextToFramePadding();
        ImGui::Text(label);
        ImGui::NextColumn();
        const auto res = ImGuiSlider("", var, min, max);
        ImGui::NextColumn();
        ImGui::PopID();
        return res;
    };

    // The control presents angles in degrees, but operates on variables containing radians.
    const auto sphericalAnglesControl = [](const char *label, float &azimuthal_rad, float &polar_rad) {
        ImGui::PushID(label);
        ImGui::AlignTextToFramePadding();
        ImGui::Text(label);
        ImGui::NextColumn();

        // Flip azimuthal so increasing the user-facing value rotates clockwise.
        float angles[2];
        angles[0] = glm::degrees(polar_rad);
        angles[1] = -glm::degrees(azimuthal_rad);
        const auto res = ImGui::DragFloat2("", angles, 0.2f);
        polar_rad = glm::radians(angles[0]);
        azimuthal_rad = -glm::radians(angles[1]);

        ImGui::NextColumn();
        ImGui::PopID();
        return res;
    };

    const auto checkboxControl = [](const char *label, bool &var) {
        ImGui::PushID(label);
        ImGui::AlignTextToFramePadding();
        ImGui::Text(label);
        ImGui::NextColumn();
        const auto res = ImGui::Checkbox("", &var);
        ImGui::NextColumn();
        ImGui::PopID();
        return res;
    };

    // Rendering settings.
    {
        ImGui::AlignTextToFramePadding();

        if (newTreeNode("Rendering")) {

            if (newTreeNode("Volume params")) {

                // Voxel transmittance.
                colorControl("transmit color", g_render_settings.voxel_transmit_color);
                sliderControl(
                    "extinction intensity", g_render_settings.voxel_extinction_intensity, 0.0f, 100.0f);

                // Density quantization.
                sliderControl("density quantization", g_render_settings.voxel_density_quantization, 1, 255);

                ImGui::TreePop();
            }

            if (newTreeNode("Surface params", false)) {

                // Surface opacity.
                colorControl("color", g_render_settings.surface_color);
                sliderControl("opacity", g_render_settings.surface_opacity, 0.0f, 1.0f);

                // Surface level.
                sliderControl("level", g_render_settings.surface_level, 0.0f, 1.0f);

                ImGui::TreePop();
            }

            if (newTreeNode("Volume placement", false)) {

                // Volume origin.
                sliderControl("volume origin", g_render_settings.volume_origin, -4.0f, 4.0f);

                // Volume size.
                sliderControl("volume size", g_render_settings.volume_size, 0.1f, 10.0f);

                ImGui::TreePop();
            }

            if (newTreeNode("Misc", false)) {

                // Reload shaders.
                ImGui::AlignTextToFramePadding();
                ImGui::NextColumn();
                if (ImGui::Button("Reload shaders")) {
                    reloadShaders();
                }
                ImGui::NextColumn();

                // Dithering.
                checkboxControl("enable dithering", g_render_settings.dither_voxels);

                ImGui::Spacing();

                // Show cube.
                checkboxControl("show opaque object", g_render_settings.show_spinning_cube);

                ImGui::TreePop();
            }

            ImGui::TreePop();
        }
    }

    // Simulation settings.
    {
        ImGui::AlignTextToFramePadding();

        if (newTreeNode("Simulation")) {

            if (newTreeNode("Solver", false)) {

                // Simulate step-by-step.
                checkboxControl("simulate step-by-step", g_simulation_settings.step_by_step);

                // Advance simulation by one step.
                ImGui::AlignTextToFramePadding();
                ImGui::Text("advance simulation");
                ImGui::NextColumn();
                g_simulation_settings.do_advection_step = ImGui::Button("Advect");
                ImGui::SameLine();
                g_simulation_settings.do_pressure_step = ImGui::Button("Pressure Solve");
                ImGui::NextColumn();

                // Set max solver iterations.
                sliderControl("max solver iterations", g_simulation_settings.max_solver_iterations, 1, 200);

                // Visualize velocity field.
                ImGui::AlignTextToFramePadding();
                ImGui::Text("visualize velocity field");
                ImGui::NextColumn();
                ImGui::Checkbox("U", &g_render_settings.visualize_velocity_field[0]);
                ImGui::SameLine();
                ImGui::Checkbox("V", &g_render_settings.visualize_velocity_field[1]);
                ImGui::SameLine();
                ImGui::Checkbox("W", &g_render_settings.visualize_velocity_field[2]);
                ImGui::Spacing();
                ImGui::SameLine();
                ImGui::NextColumn();

                ImGui::TreePop();
            }

            if (newTreeNode("Noise", false)) {
                sliderControl("curl noise strength", g_simulation_settings.curl_noise_strength, 0.0f, 20.0f);
                sliderControl(
                    "curl noise frequency", g_simulation_settings.curl_noise_frequency, 1.0f, 16.0f);

                ImGui::TreePop();
            }

            // Fluid density.
            // sliderControl("fluid density", g_simulation_settings.fluid_density, 0.1f, 100.0f);

            ImGui::Spacing();

            const auto positionControl = [](const char *label, sim::GridIndex3 &var) {
                ImGui::AlignTextToFramePadding();
                ImGui::Text(label);
                ImGui::NextColumn();
                ImGui::DragInt3(fmt::format("##slider-{}", label).c_str(), &var.x, 0.1f);
                var = glm::max(var, sim::GridIndex3(0));
                var = glm::min(var, g_simulation_settings.grid_dim - 1);
                ImGui::NextColumn();
            };

            // Fluid source.
            if (newTreeNode("Fluid source")) {

                // Source position.
                positionControl("position", g_simulation_settings.source_pos);

                // Concentration emission.
                auto &emission = g_simulation_settings.source_emission;
                sliderControl("emission rate", emission.concentration, 0.0f, 100.0f);

                // Temperature rate.
                // sliderControl("temperature rate", g_simulation_settings.source.rate.temperature, 0.0f, 100.0f);

                // Velocity.
                auto &vel = g_simulation_settings.source_velocity_spherical;
                bool dragging = false;
                dragging |= sliderControl("velocity magnitude", vel.radius, 0.0f, 70.0f);
                vel.radius = std::max(vel.radius, 0.0f);
                dragging |= sphericalAnglesControl("velocity direction", vel.azimuthal, vel.polar);
                if (dragging) {
                    g_render_settings.velocity_arrow_opacity = 1.0f;
                }

                checkboxControl("show arrow on change", g_render_settings.show_velocity_arrow);

                ImGui::TreePop();
            }

            ImGui::TreePop();
        }
    }

    ImGui::Columns(1);
    ImGui::Separator();
    ImGui::PopStyleVar();
    ImGui::End();
}

class ScopedTimer {
public:
    ScopedTimer(float &out_ms) : m_start(Clock::now()), m_out_ms(out_ms) {}
    ~ScopedTimer()
    {
        const auto duration = Clock::now() - m_start;
        const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(duration);
        m_out_ms = 1e-3f * float(duration_us.count());
    }

private:
    typedef std::chrono::steady_clock Clock;
    typedef Clock::time_point TimePoint;
    typedef Clock::duration Duration;
    TimePoint m_start;
    float &m_out_ms;
};

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
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_DEPTH_BITS, 0);
    glfwWindowHint(GLFW_STENCIL_BITS, 0);
    glfwWindowHint(GLFW_MAXIMIZED, true);
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

        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
            g_logger->critical("Failed to initialize OpenGL");
            return 1;
        }
        g_logger->info("OpenGL Version {}.{} loaded", GLVersion.major, GLVersion.minor);
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

    // Set up a buffer for common uniforms.
    GLUBO common_ubo = GLUBO::create();
    glBindBufferBase(GL_UNIFORM_BUFFER, kCommonUBOBindSlot, common_ubo);
    // Allocate storage.
    glBufferData(GL_UNIFORM_BUFFER, sizeof(CommonUniforms), nullptr, GL_STATIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    GL_CHECK();

    // Set up the voxel shader program.
    auto &vxr = g_render_resources.vxr;
    vxr.shader.path = kShadersPath + "/voxel.json";
    reloadShaders();
    GLProgram &vxr_program = vxr.shader.program;
    const GLuint &vxr_depth_uniform_loc = vxr.depth_uniform_loc;
    const GLuint vxr_depth_texture_unit = 1;

    // Set up a buffer for grid data.
    GLUBO grid_data_ubo = GLUBO::create();
    glBindBufferBase(GL_UNIFORM_BUFFER, vxr.grid_data_binding, grid_data_ubo);
    // Allocate storage.
    glBufferData(GL_UNIFORM_BUFFER, sizeof(GridData), nullptr, GL_STATIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    GL_CHECK();

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
            glUniformBlockBinding(color_cube_program, common_ubo_index, kCommonUBOBindSlot);
        GL_CHECK();
    }

    // Set up quad vertex data for the voxel renderer program.
    auto quad_vao = GLVAO::create();
    {
        const glm::vec2 quad_vertices[6] = { { -1.0f, -1.0f }, { 1.0f, 1.0f },   { -1.0f, 1.0f },
                                             { 1.0f, 1.0f },   { -1.0f, -1.0f }, { 1.0f, -1.0f } };
        glBindVertexArray(quad_vao);
        GLuint quad_vertex_buffer;
        glGenBuffers(1, &quad_vertex_buffer);
        glBindBuffer(GL_ARRAY_BUFFER, quad_vertex_buffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quad_vertices), quad_vertices, GL_STATIC_DRAW);
        const auto vpos_location = glGetAttribLocation(vxr_program, "pos");
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
    g_camera.eye_pos = glm::vec3(-1.6, 4.1, 4.4);
    g_camera.orientation = glm::quatLookAt(glm::vec3(0.66, -0.48, -0.58), glm::vec3(0, 1, 0));
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
    const auto grid_dim = g_simulation_settings.grid_dim;
    GridData grid_data;
    grid_data.grid_dim = grid_dim;
    grid_data.origin = g_render_settings.volume_origin;
    grid_data.size = g_render_settings.volume_size;
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
                    grid.u(i, j, k) = 0;
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
    const GLuint vxr_voxels_texture_unit = 2;
    const GLuint vxr_voxels_uniform_loc = glGetUniformLocation(vxr_program, "voxels");

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
                               const glm::mat4 &view_proj, float opacity = 1.0f) {
        glUseProgram(arrow_program);
        const float scale_factor = glm::length(to - from);
        const auto rotate = glm::mat4(glm::rotation({ 1, 0, 0 }, (to - from) / scale_factor));
        const auto scale = glm::scale(glm::vec3(scale_factor, scale_factor, scale_factor));
        const auto translate = glm::translate(from);
        const auto model = translate * rotate * scale;
        const auto mvp = view_proj * model;
        glUniformMatrix4fv(arrow_program_mvp_loc, 1, GL_FALSE, &mvp[0][0]);
        glUniform4f(arrow_program_color_loc, color.r, color.g, color.b, opacity);
        glBindVertexArray(arrow_vao);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDrawArrays(GL_LINES, 0, 6);
        glDisable(GL_BLEND);
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
            g_overlay_data.fluid_cell_count = fluid_sim.fluidCellCount();

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

            fluid_sim.densityModel().set(g_simulation_settings.fluid_density, 0.1f, 0.0f);

            const bool do_advection =
                !g_simulation_settings.step_by_step || g_simulation_settings.do_advection_step;
            const bool do_pressure =
                !g_simulation_settings.step_by_step || g_simulation_settings.do_pressure_step;
            if (do_advection) {
                rmt_ScopedCPUSample(AppFluidSimAdvect, 0);
                ScopedTimer timer(g_overlay_data.advect_ms);

                // Track time for animation.
                static float time = 0;
                const float dt = fluid_sim.dt();
                time += dt;

                // Wall.
                auto &wall = fluid_sim.solidCells().at(0);
                wall.pos = g_simulation_settings.source_pos;
                wall.velocity =
                    euclideanFromSpherical(g_simulation_settings.source_velocity_spherical);

                // Source.
                const auto &source_pos = g_simulation_settings.source_pos;
                const auto &source_rate = g_simulation_settings.source_emission;
                auto &source = fluid_sim.grid().cell(source_pos);
                source.concentration += dt * source_rate.concentration;
                source.temperature += dt * source_rate.temperature;

                // Advect.
                fluid_sim.advect();
            }

            // Add curl noise.
            {
                siv::PerlinNoise perlin(0);

                const float noiseSize = g_simulation_settings.curl_noise_frequency;
                auto &grid = fluid_sim.grid();
                const auto size = grid.size();
                const auto delta = 1e-1f * glm::vec3(1, 1, 1);
                const auto dx = delta.x, dy = delta.y, dz = delta.z;
                const auto norm = glm::vec3(1, 1, 1) / (2.0f * delta);
                for (int i = 0; i < size.x; ++i) {
                    const float x = noiseSize * float(i) / float(size.x);
                    for (int j = 0; j < size.y; ++j) {
                        const float y = noiseSize * float(j) / float(size.y);
                        for (int k = 0; k < size.z; ++k) {
                            const float z = noiseSize * float(k) / float(size.z);
                            glm::vec3 curl =
                                glm::vec3{ perlin.noise(x + dx, y, z) - perlin.noise(x - dx, y, z),
                                           perlin.noise(x, y + dy, z) - perlin.noise(x, y - dy, z),
                                           perlin.noise(x, y, z + dz) - perlin.noise(x, y, z - dz) };
                            curl *= norm;
                            const auto v =
                                g_simulation_settings.curl_noise_strength * g_time_delta * curl;
                            grid.u(i, j, k) += 0.5f * v.x;
                            grid.u(i + 1, j, k) += 0.5f * v.x;
                            grid.v(i, j, k) += 0.5f * v.y;
                            grid.v(i, j + 1, k) += 0.5f * v.y;
                            grid.w(i, j, k) += 0.5f * v.z;
                            grid.w(i, j, k + 1) += 0.5f * v.z;
                        }
                    }
                }
            }

            // Slowly decrease concentration in fluid cells.
            {
                for (auto &cell : fluid_sim.grid()) {
                    if (cell.concentration > 0) {
                        cell.concentration = std::max(0.0f, cell.concentration - g_time_delta * 1e-2f);
                    }
                }
            }

            // Pressure projection.
            if (do_pressure) {
                rmt_ScopedCPUSample(AppFluidSimPressureSolve, 0);
                ScopedTimer timer(g_overlay_data.pressure_projection_ms);

                fluid_sim.setSolverMaxIterations(g_simulation_settings.max_solver_iterations);
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

            // Draw arrows.
            {
                const auto dx = fluid_sim.dx();
                const auto renderFromGrid = grid_data.size / glm::vec3(grid_data.grid_dim);
                const auto renderFromPhys = renderFromGrid / dx;

                // U.
                if (g_render_settings.visualize_velocity_field[0]) {
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
                }

                // V.
                if (g_render_settings.visualize_velocity_field[1]) {
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
                }

                // W.
                if (g_render_settings.visualize_velocity_field[2]) {
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
            }

            if (g_render_settings.show_spinning_cube) {
                // Draw cube.
                glUseProgram(color_cube_program);
                glBindVertexArray(cube_vao);
                glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, nullptr);
                GL_CHECK();
            }

            // Disable and detach z buffer.
            glDisable(GL_DEPTH_TEST);
            glDepthMask(GL_FALSE);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
            GL_CHECK();

            // Bind depth texture.
            glBindTextureUnit(vxr_depth_texture_unit, g_framebuffer.depth_texture);

            // Bind voxel texture.
            glBindTextureUnit(vxr_voxels_texture_unit, voxels);

            // Set up multiplicative blending.
            glEnable(GL_BLEND);
            glBlendFunc(GL_ZERO, GL_SRC_COLOR);

            // Upload voxel data.
            {
                grid_data.origin = g_render_settings.volume_origin;
                grid_data.size = g_render_settings.volume_size;
                grid_data.voxel_size = grid_data.size / glm::vec3(grid_data.grid_dim);

                const auto extinction = g_render_settings.voxel_extinction_intensity *
                                        (glm::vec3(1, 1, 1) - g_render_settings.voxel_transmit_color);
                grid_data.voxel_extinction = extinction;
                grid_data.grid_flags = packGridFlags(
                    g_render_settings.voxel_density_quantization, g_render_settings.dither_voxels);

                grid_data.surface_color = g_render_settings.surface_color;
                grid_data.surface_opacity = g_render_settings.surface_opacity;
                grid_data.surface_roughness = g_render_settings.surface_roughness;
                grid_data.surface_level = g_render_settings.surface_level;

                void *ubo_ptr = glMapNamedBuffer(grid_data_ubo, GL_WRITE_ONLY);
                memcpy(ubo_ptr, &grid_data, sizeof(grid_data));
                glUnmapNamedBuffer(grid_data_ubo);
                GL_CHECK();
            }

            // Set up the voxel renderer program.
            glUseProgram(vxr_program);
            // Set depth texture unit.
            glUniform1i(vxr_depth_uniform_loc, vxr_depth_texture_unit);
            // Set voxel texture unit.
            glUniform1i(vxr_voxels_uniform_loc, vxr_voxels_texture_unit);

            // Draw fluid volume.
            {
                rmt_ScopedOpenGLSample(DrawVolumeGL);
                GLuint queries[2];
                glGenQueries(2, queries);
                glQueryCounter(queries[0], GL_TIMESTAMP);

                glBindVertexArray(quad_vao);
                glDrawArrays(GL_TRIANGLES, 0, 6);
                GL_CHECK();

                glQueryCounter(queries[1], GL_TIMESTAMP);
                GLuint64 start, end;
                glGetQueryObjectui64v(queries[0], GL_QUERY_RESULT, &start);
                glGetQueryObjectui64v(queries[1], GL_QUERY_RESULT, &end);
                glDeleteQueries(2, queries);
                g_overlay_data.voxel_draw_ms = float((end - start) / 1000ULL) * 1e-3f;
            }

            // Unbind depth texture.
            glBindTextureUnit(vxr_depth_texture_unit, 0);

            // Re-enable and re-attach z buffer.
            glEnable(GL_DEPTH_TEST);
            glDepthMask(GL_TRUE);
            glFramebufferTexture2D(
                GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, g_framebuffer.depth_texture, 0);
            GL_CHECK();

            // Disable blending.
            glDisable(GL_BLEND);

            // Draw velocity arrow.
            {
                auto &opacity = g_render_settings.velocity_arrow_opacity;
                if (g_render_settings.show_velocity_arrow && opacity > 1e-5f) {
                    const auto dx = fluid_sim.dx();
                    const auto renderFromGrid = grid_data.size / glm::vec3(grid_data.grid_dim);
                    const auto renderFromPhys = renderFromGrid / dx;
                    const auto source_pos_grid =
                        glm::vec3(g_simulation_settings.source_pos) + glm::vec3(0.5, 0.5, 0.5);
                    const auto pos = grid_data.origin + source_pos_grid * renderFromGrid;
                    const auto &source_vel_phys =
                        euclideanFromSpherical(g_simulation_settings.source_velocity_spherical);
                    const auto vel = renderFromPhys * source_vel_phys;
                    drawArrow(pos, pos + 0.1f * vel, glm::vec3(1, 1, 1), view_proj, opacity);
                    opacity *= std::pow(0.002f, g_time_delta);
                }
            }

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
