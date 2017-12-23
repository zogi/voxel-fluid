#include <GL/glew.h>
#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>
#include <spdlog/spdlog.h>
#include <variant.hpp>

#include <array>
#include <cstdio>
#include <memory>


using namespace glm;

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
    glm::vec3 eye_pos;
    float _pad1;
    glm::vec3 eye_dir;
    float _pad2;
    glm::mat4 mvp;
};
#pragma pack(pop)

static std::string common_shader_code = R"glsl(
    #version 430
    layout(shared) uniform CommonUniforms {
        vec3 eye_pos;
        vec3 eye_dir;
        mat4 mvp;
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
struct OctreeRendererTextures {
    GLuint depth;
};
static std::string octree_renderer_fs_code = common_shader_code + R"glsl(
    uniform sampler2D depth;
    uniform sampler3D octree_bricks;
    readonly buffer octree_nodes {
        uint array[];
    };

    in vec2 uv;
    void main() {
        float depth = texture(depth, uv).x;
        if (depth != 0.0)
            discard;
        // TODO: transform back stuff to view space
        // Now eye is at the origin and looking towards -z.
        gl_FragColor = vec4(uv.x, uv.y, 0.0, 1.0);
    }
    )glsl";

static std::string simple_vs_code = common_shader_code + R"glsl(
    in vec3 pos;
    out vec3 color;
    void main() {
        gl_Position = mvp * vec4(pos, 1.0);
        color = pos + 0.5;
    }
    )glsl";
static std::string simple_fs_code = common_shader_code + R"glsl(
    in vec3 color;
    void main() {
        gl_FragColor = vec4(color, 1.0);
    }
    )glsl";

// === Quad vertices ===

static const glm::vec2 vertices[6] = { { -1.0f, -1.0f }, { -1.0f, 1.0f }, { 1.0f, 1.0f },
                                       { 1.0f, 1.0f },   { 1.0f, -1.0f }, { -1.0f, -1.0f } };

// === Cube VAO ===

static const uint16_t cube_indices[] = { 0, 2, 4, 4, 2, 6, 1, 5, 3, 3, 5, 7, 0, 1, 2, 2, 1, 3,
                                         4, 6, 5, 5, 6, 7, 0, 4, 1, 1, 4, 5, 2, 3, 6, 6, 3, 7 };
static GLVAO createCubeVAO()
{
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
    std::vector<float> vertices;
    vertices.reserve(3 * 8);
    for (int i = 0; i < 8; ++i) {
        vertices.push_back(float((i & 1) >> 0) - 0.5f);
        vertices.push_back(float((i & 2) >> 1) - 0.5f);
        vertices.push_back(float((i & 4) >> 2) - 0.5f);
    }
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
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

    // Setup fbo with floating-point depth.

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

    Camera() : pivot_distance(1) {}
    glm::vec3 getForwardVector() const;
};

glm::vec3 Camera::getForwardVector() const { return glm::rotate(orientation, glm::vec3(0, 0, -1)); }

// Modified version of cinder's CameraUI class (https://github.com/cinder/Cinder).

class CameraUI {
public:
    CameraUI(GLFWwindow *window = nullptr)
        : mWindow(window)
        , mCamera(nullptr)
        , mInitialPivotDistance(1)
        , mMouseWheelMultiplier(1.1f)
        , mMinimumPivotDistance(0.01f)
        , mLastAction(ACTION_NONE)
        , mEnabled(false)
    {
    }
    void setWindow(GLFWwindow *window) { mWindow = window; }
    void setCamera(Camera *camera) { mCamera = camera; }
    void setEnabled(bool enable) { mEnabled = enable; }

    void cursorPositionCallback(GLFWwindow *window, double xpos, double ypos);
    void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods);
    void scrollCallback(GLFWwindow *window, double xoffset, double yoffset);

private:
    enum { ACTION_NONE, ACTION_ZOOM, ACTION_PAN, ACTION_TUMBLE };

    ivec2 getWindowSize() const
    {
        if (!mWindow)
            return {};
        int w, h;
        glfwGetWindowSize(mWindow, &w, &h);
        return { w, h };
    }

    vec2 mInitialMousePos;
    Camera mInitialCam;
    Camera *mCamera;
    float mInitialPivotDistance;
    float mMouseWheelMultiplier, mMinimumPivotDistance;
    int mLastAction;

    GLFWwindow *mWindow;
    bool mEnabled;
};

void CameraUI::cursorPositionCallback(GLFWwindow *window, double xpos, double ypos)
{
    if (!mCamera || !mEnabled)
        return;

    const auto mousePos = glm::vec2(xpos, ypos);

    const bool leftDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    const bool middleDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
    const bool rightDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    const bool altPressed = glfwGetKey(window, GLFW_KEY_LEFT_ALT);
    const bool ctrlPressed = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL);

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
    const auto world_up = vec3(0, 1, 0);
    const auto window_size = getWindowSize();

    if (action == ACTION_ZOOM) { // zooming
        auto mouseDelta = (mousePos.x - mInitialMousePos.x) + (mousePos.y - mInitialMousePos.y);

        float newPivotDistance =
            powf(2.71828183f, 2 * -mouseDelta / length(vec2(window_size))) * mInitialPivotDistance;
        vec3 oldTarget = mInitialCam.eye_pos + initial_forward * mInitialPivotDistance;
        vec3 newEye = oldTarget - initial_forward * newPivotDistance;
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
        vec3 mW = normalize(initial_forward);

        vec3 mU = normalize(cross(world_up, mW));

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
    vec3 newEye = mCamera->eye_pos + eye_dir * (mCamera->pivot_distance * (1 - multiplier));
    mCamera->eye_pos = newEye;
    mCamera->pivot_distance =
        std::max<float>(mCamera->pivot_distance * multiplier, mMinimumPivotDistance);
}

// === Global State ===

static Framebuffer g_framebuffer;
static Camera g_camera;
static CameraUI g_camera_ui;

// === GLFW input callbacks ===

int main()
{
    g_logger = spdlog::stderr_logger_mt("fluid");
    g_logger->set_level(spdlog::level::debug);

    glfwSetErrorCallback([](int error, const char *description) {
        g_logger->error("GLFW Error {}: {}", error, description);
    });


    // Init GLFW and create a window.

    if (!glfwInit()) {
        g_logger->critical("Failed to initialize GLFT.");
        abort();
    }
    const auto terminate_glfw = finally([]() { glfwTerminate(); });

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

    // Enable vsync.
    glfwSwapInterval(1);
    GL_CHECK();

    // Setup reverse-Z.
    glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);
    GL_CHECK();
    glClearDepth(0.0);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_GEQUAL);
    const auto perspectiveInvZ = [](float fov_y, float aspect_ratio, float z_near) {
        float f = 1.0f / tan(fov_y / 2.0f);
        return glm::mat4(
            f / aspect_ratio, 0.0f, 0.0f, 0.0f, 0.0f, f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f,
            0.0f, z_near, 0.0f);
    };
    const auto cam_fov = glm::radians(100.0f);
    const auto cam_near = 0.01f;

    // Setup common uniform buffers.
    GLUBO common_ubo = GLUBO::create();
    constexpr GLuint common_ubo_bind_point = 1;
    GL_CHECK();

    // Setup the octree renderer program.
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

        // Get depth sampler uniform locaction.
        otr_depth_uniform_loc = glGetUniformLocation(otr_program, "depth");
        GL_CHECK();
    }

    // Setup the simple program.
    GLProgram simple_program;
    {
        std::vector<GLShader> shaders;
        shaders.push_back(createAndCompileShader(GL_VERTEX_SHADER, simple_vs_code.c_str()));
        shaders.push_back(createAndCompileShader(GL_FRAGMENT_SHADER, simple_fs_code.c_str()));
        simple_program = createAndLinkProgram(shaders);
        GL_CHECK();

        // Common uniforms.
        auto common_ubo_index = glGetUniformBlockIndex(simple_program, "CommonUniforms");
        if (common_ubo_index != GL_INVALID_INDEX)
            glUniformBlockBinding(simple_program, common_ubo_index, common_ubo_bind_point);
        GL_CHECK();
    }

    // Setup quad vertex data for the octree renderer program.
    auto quad_vao = GLVAO::create();
    {
        glBindVertexArray(quad_vao);
        GLuint quad_vertex_buffer;
        glGenBuffers(1, &quad_vertex_buffer);
        glBindBuffer(GL_ARRAY_BUFFER, quad_vertex_buffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        const auto vpos_location = glGetAttribLocation(otr_program, "pos");
        glEnableVertexAttribArray(vpos_location);
        glVertexAttribPointer(vpos_location, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);
        glBindVertexArray(0);
        GL_CHECK();
    }

    // Setup cube vertex data for the simple program.
    auto cube_vao = createCubeVAO();
    {
        glBindVertexArray(cube_vao);
        const auto vpos_location = glGetAttribLocation(simple_program, "pos");
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
    g_camera_ui.setWindow(window);
    g_camera_ui.setCamera(&g_camera);
    g_camera_ui.setEnabled(true);

    // Event callbacks.
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow *, int width, int height) {
        glViewport(0, 0, width, height);
        g_framebuffer.init(width, height);
    });

    glfwSetKeyCallback(window, [](GLFWwindow *window, int key, int scancode, int action, int mods) {
        if (key == GLFW_KEY_ESCAPE)
            glfwSetWindowShouldClose(window, GLFW_TRUE);
    });
    glfwSetCursorPosCallback(window, [](GLFWwindow *window, double xpos, double ypos) {
        g_camera_ui.cursorPositionCallback(window, xpos, ypos);
    });
    glfwSetMouseButtonCallback(window, [](GLFWwindow *window, int button, int action, int mods) {
        g_camera_ui.mouseButtonCallback(window, button, action, mods);
    });
    glfwSetScrollCallback(window, [](GLFWwindow *window, double xoffset, double yoffset) {
        g_camera_ui.scrollCallback(window, xoffset, yoffset);
    });

    // Main loop.
    glm::mat4 model;
    float time_prev_start = float(glfwGetTime()) - 1.f / 60.f;
    while (!glfwWindowShouldClose(window)) {
        const float time_start = float(glfwGetTime());
        const float time_delta = time_start - time_prev_start;
        time_prev_start = time_start;

        const int width = g_framebuffer.width;
        const int height = g_framebuffer.height;
        const float aspect_ratio = width / (float)height;

        // Draw
        {
            glBindFramebuffer(GL_FRAMEBUFFER, g_framebuffer.fbo);
            const auto unbind_fbo = finally([]() { glBindFramebuffer(GL_FRAMEBUFFER, 0); });

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            // Fill and bind common uniforms.
            {
                CommonUniforms uniforms;
                uniforms.eye_pos = g_camera.eye_pos;
                uniforms.eye_dir = g_camera.getForwardVector();
                // const auto view = glm::lookAt(
                //    uniforms.eye_pos, uniforms.eye_pos + uniforms.eye_dir, glm::vec3(0, 1, 0));
                const auto view =
                    glm::translate(glm::toMat4(glm::inverse(g_camera.orientation)), -uniforms.eye_pos);
                const auto proj = perspectiveInvZ(cam_fov, aspect_ratio, cam_near);
                // model = glm::rotate(glm::mat4(), float(glfwGetTime()), glm::vec3());
                model = glm::rotate(model, glm::radians(60.0f) * time_delta, glm::vec3(1, 1, 1));
                uniforms.mvp = proj * view * model;
                // glm::mat4 m, p, mvp;
                // m = glm::rotate(glm::mat4(), (float)glfwGetTime(), glm::vec3(0.0, 0.0, 1.0));
                // p = glm::ortho(-aspect_ratio, aspect_ratio, -1.f, 1.f, 1.f, -1.f);
                // mvp = p * m;
                glBindBuffer(GL_UNIFORM_BUFFER, common_ubo);
                glBufferData(GL_UNIFORM_BUFFER, sizeof(CommonUniforms), &uniforms, GL_STATIC_DRAW);
                glBindBufferBase(GL_UNIFORM_BUFFER, common_ubo_bind_point, common_ubo);
            }

            // Draw cube.
            glUseProgram(simple_program);
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

            // Draw quad.
            glUseProgram(otr_program);
            glUniform1i(otr_depth_uniform_loc, otr_depth_texture_unit);
            glBindVertexArray(quad_vao);
            glDrawArrays(GL_TRIANGLES, 0, 6);
            GL_CHECK();

            // Unbind depth texture.
            glBindTextureUnit(otr_depth_texture_unit, 0);

            // Re-enable and re-attach z buffer.
            glEnable(GL_DEPTH_TEST);
            glDepthMask(GL_TRUE);
            glFramebufferTexture2D(
                GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, g_framebuffer.depth_texture, 0);
            GL_CHECK();
        }

        // Copy from framebuffer to window.
        {
            glBindFramebuffer(GL_READ_FRAMEBUFFER, g_framebuffer.fbo);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0); // default FBO
            glBlitFramebuffer(
                0, 0, g_framebuffer.width, g_framebuffer.height, 0, 0, width, height,
                GL_COLOR_BUFFER_BIT, GL_LINEAR);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }

        // Present.
        glfwSwapBuffers(window);

        // Process messages.
        glfwPollEvents();
    }

    return 0;
}
