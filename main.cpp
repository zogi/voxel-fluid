#include <GL/glew.h>
#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>
#include <spdlog/spdlog.h>

#include <array>
#include <cstdio>
#include <memory>


using namespace glm;

std::shared_ptr<spdlog::logger> g_logger;

#define GL_CHECK() checkLastGLError(__FILE__, __LINE__)
void checkLastGLError(const char *file, int line)
{
    GLenum status = glGetError();
    if (status != GL_NO_ERROR) {
        g_logger->error("{} ({}): GL error {}: {}", file, line, status, gluErrorString(status));
    }
}

void errorCallback(int error, const char *description)
{
    g_logger->error("GLFW Error: {}", description);
}

struct CameraUIState {
    enum { ACTION_NONE, ACTION_ZOOM, ACTION_PAN, ACTION_TUMBLE };
};

// === OpenGL RAII wrapper template ===

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

// === Shaders and programs ===

struct GLShaderTraits {
    typedef GLuint value_type;
    static value_type create(GLenum shader_type) { return glCreateShader(shader_type); }
    static void destroy(value_type shader) { glDeleteShader(shader); }
};
typedef GLObject<GLShaderTraits> GLShader;

GLShader createAndCompileShader(GLenum shader_type, const char *source, const char *defines = nullptr)
{
    auto shader = GLShader::create(shader_type);
    if (defines) {
        const char *sources[] = { defines, source };
        glShaderSource(shader, 2, sources, NULL);
    } else {
        glShaderSource(shader, 1, &source, NULL);
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

struct GLProgramTraits {
    typedef GLuint value_type;
    static value_type create() { return glCreateProgram(); }
    static void destroy(value_type program) { glDeleteProgram(program); }
};
typedef GLObject<GLProgramTraits> GLProgram;

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

// === Uniform Buffer Object wrapper ===

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

// === Vertex Array Object wrapper ===

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
    layout(std140) uniform CommonUniforms {
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
static std::string octree_renderer_fs_code = common_shader_code + R"glsl(
    uniform sampler3D octree_bricks;
    readonly buffer octree_nodes {
        uint array[];
    };

    in vec2 uv;
    void main() {
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

// === Framebuffer state ===

struct FramebufferState {
    int width, height;
    GLuint fbo;
    FramebufferState() : width(0), height(0), fbo(0) {}
};
static FramebufferState g_fbstate;

static void framebufferResizedCallback(GLFWwindow *, int width, int height)
{
    glViewport(0, 0, width, height);

    // Setup fbo with floating-point depth.
    GLuint fbo;
    {
        GLuint color, depth;

        glGenTextures(1, &color);
        glBindTexture(GL_TEXTURE_2D, color);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_SRGB8_ALPHA8, width, height);
        glBindTexture(GL_TEXTURE_2D, 0);

        glGenTextures(1, &depth);
        glBindTexture(GL_TEXTURE_2D, depth);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH_COMPONENT32F, width, height);
        glBindTexture(GL_TEXTURE_2D, 0);

        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth, 0);
        GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (status != GL_FRAMEBUFFER_COMPLETE) {
            g_logger->error("glCheckFramebufferStatus: {}", status);
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    g_fbstate.width = width;
    g_fbstate.height = height;
    g_fbstate.fbo = fbo;
}

// === Camera state ===

struct CameraState {
    glm::vec3 eye_pos;
    glm::vec3 eye_dir;
    float pivot_distance;
};

class CameraController {
public:
    CameraController(GLFWwindow *window) : mWindow(window), mCamera(nullptr) {}

    void mouseDrag(const vec2 &mousePos, bool leftDown, bool middleDown, bool rightDown);

private:
    enum { ACTION_NONE, ACTION_ZOOM, ACTION_PAN, ACTION_TUMBLE };

    ivec2 getWindowSize() const
    {
        int w, h;
        glfwGetWindowSize(mWindow, &w, &h);
        return { w, h };
    }

    vec2 mInitialMousePos;
    CameraState mInitialCam;
    CameraState *mCamera;
    float mInitialPivotDistance;
    float mMouseWheelMultiplier, mMinimumPivotDistance;
    int mLastAction;

    ivec2 mWindowSize; // used when mWindow is null
    GLFWwindow *mWindow;
    bool mEnabled;
    int mSignalPriority;
};

void CameraController::mouseDrag(const vec2 &mousePos, bool leftDown, bool middleDown, bool rightDown)
{
    if (!mCamera || !mEnabled)
        return;

    int action;
    if (rightDown || (leftDown && middleDown))
        action = ACTION_ZOOM;
    else if (middleDown)
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

    const auto up = vec3(0, 1, 0);

    if (action == ACTION_ZOOM) { // zooming
        auto mouseDelta = (mousePos.x - mInitialMousePos.x) + (mousePos.y - mInitialMousePos.y);

        float newPivotDistance = powf(2.71828183f, 2 * -mouseDelta / length(vec2(getWindowSize()))) *
                                 mInitialPivotDistance;
        vec3 oldTarget = mInitialCam.eye_pos + mInitialCam.eye_dir * mInitialPivotDistance;
        vec3 newEye = oldTarget - mInitialCam.eye_dir * newPivotDistance;
        mCamera->eye_pos = newEye;
        mCamera->pivot_distance = std::max<float>(newPivotDistance, mMinimumPivotDistance);

    } else if (action == ACTION_PAN) { // panning
        float deltaX =
            (mousePos.x - mInitialMousePos.x) / (float)getWindowSize().x * mInitialPivotDistance;
        float deltaY =
            (mousePos.y - mInitialMousePos.y) / (float)getWindowSize().y * mInitialPivotDistance;
        const auto right = glm::cross(mInitialCam.eye_dir, up);
        mCamera->eye_pos = mInitialCam.eye_pos - right * deltaX + up * deltaY;

    } else { // tumbling
        float deltaX = (mousePos.x - mInitialMousePos.x) / -100.0f;
        float deltaY = (mousePos.y - mInitialMousePos.y) / 100.0f;
        vec3 mW = normalize(mInitialCam.eye_dir);
        bool invertMotion = false;
        // bool invertMotion = (mInitialCam.getOrientation() * mInitialCam.getWorldUp()).y < 0.0f;

        vec3 mU = normalize(cross(up, mW));

        if (invertMotion) {
            deltaX = -deltaX;
            deltaY = -deltaY;
        }

        glm::vec3 rotatedVec =
            glm::angleAxis(deltaY, mU) * (-mInitialCam.eye_dir * mInitialPivotDistance);
        rotatedVec = glm::angleAxis(deltaX, up) * rotatedVec;

        mCamera->eye_pos =
            mInitialCam.eye_pos + mInitialCam.eye_dir * mInitialPivotDistance + rotatedVec;
        mCamera->setOrientation(
            glm::angleAxis(deltaX, mInitialCam.getWorldUp()) * glm::angleAxis(deltaY, mU) *
            mInitialCam.getOrientation());
    }
}
static CameraUIState g_camera;

// === GLFW input callbacks ===

static void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {}

static void cursorPositionCallback(GLFWwindow *window, double xpos, double ypos)
{
    bool left_btn_down = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    bool mid_btn_down = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
    bool right_btn_down = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    bool alt_pressed = glfwGetKey(window, GLFW_KEY_LEFT_ALT);
    bool ctrl_pressed = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL);
    mid_btn_down |= alt_pressed;
    right_btn_down |= ctrl_pressed;
    g_camera.mouseMove({ xpos, ypos }, left_btn_down, mid_btn_down, right_btn_down);
}

static void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
{
    if (action == GLFW_PRESS) {
        double x, y;
        glfwGetCursorPos(window, &x, &y);
        g_camera.mouseDown({ x, y });

    } else if (action == GLFW_RELEASE) {
        g_camera.mouseUp();
    }
}

static void scrollCallback(GLFWwindow *window, double xoffset, double yoffset) {}

int main()
{
    g_logger = spdlog::stderr_logger_mt("fluid");

    glfwSetErrorCallback(errorCallback);

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

    // Resize callback.
    glfwSetFramebufferSizeCallback(window, framebufferResizedCallback);

    // Input callbacks.
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, cursorPositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetScrollCallback(window, scrollCallback);

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

    // Setup reverse-Z.
    glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);
    glClearDepth(0.0);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_GREATER);
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

    // Setup the octree renderer program.
    GLProgram octree_renderer_program;
    {
        std::vector<GLShader> shaders;
        shaders.push_back(createAndCompileShader(GL_VERTEX_SHADER, octree_renderer_vs_code.c_str()));
        shaders.push_back(createAndCompileShader(GL_FRAGMENT_SHADER, octree_renderer_fs_code.c_str()));
        octree_renderer_program = createAndLinkProgram(shaders);

        auto common_ubo_index = glGetUniformBlockIndex(octree_renderer_program, "CommonUniforms");
        glUniformBlockBinding(octree_renderer_program, common_ubo_index, common_ubo_bind_point);
    }

    // Setup the simple program.
    GLProgram simple_program;
    {
        std::vector<GLShader> shaders;
        shaders.push_back(createAndCompileShader(GL_VERTEX_SHADER, simple_vs_code.c_str()));
        shaders.push_back(createAndCompileShader(GL_FRAGMENT_SHADER, simple_fs_code.c_str()));
        simple_program = createAndLinkProgram(shaders);

        auto common_ubo_index = glGetUniformBlockIndex(simple_program, "CommonUniforms");
        glUniformBlockBinding(simple_program, common_ubo_index, common_ubo_bind_point);
    }

    // Setup quad vertex data for the octree renderer program.
    auto quad_vao = GLVAO::create();
    {
        glBindVertexArray(quad_vao);
        GLuint quad_vertex_buffer;
        glGenBuffers(1, &quad_vertex_buffer);
        glBindBuffer(GL_ARRAY_BUFFER, quad_vertex_buffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        const auto vpos_location = glGetAttribLocation(octree_renderer_program, "pos");
        glEnableVertexAttribArray(vpos_location);
        glVertexAttribPointer(vpos_location, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);
        glBindVertexArray(0);
        GL_CHECK();
    }

    // Setup cube vertex data for the simple program.
    auto cube_vao = createCubeVAO();
    {
        // glBindVertexArray(cube_vao);
        const auto vpos_location = glGetAttribLocation(simple_program, "pos");
        glEnableVertexAttribArray(vpos_location);
        glVertexAttribPointer(vpos_location, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);
        // glBindVertexArray(0);
        GL_CHECK();
    }

    // Init framebuffer.
    {
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        framebufferResizedCallback(window, width, height);
    }

    glm::mat4 model;
    float time_prev_start = glfwGetTime() - 1.f / 60.f;
    while (!glfwWindowShouldClose(window)) {
        const float time_start = glfwGetTime();
        const float time_delta = time_start - time_prev_start;
        time_prev_start = time_start;

        const int width = g_fbstate.width;
        const int height = g_fbstate.height;
        const float aspect_ratio = width / (float)height;

        // Draw
        {
            glBindFramebuffer(GL_FRAMEBUFFER, g_fbstate.fbo);
            const auto unbind_fbo = finally([]() { glBindFramebuffer(GL_FRAMEBUFFER, 0); });

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            // Fill and bind common uniforms.
            {
                CommonUniforms uniforms;
                uniforms.eye_pos = { 0.0, 0.0, -3.0f };
                uniforms.eye_dir = { 0.0, 0.0, 1.0f };
                const auto view = glm::lookAt(
                    uniforms.eye_pos, uniforms.eye_pos + uniforms.eye_dir, glm::vec3(0, 1, 0));
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

            // Draw quad.
            //{
            //    glUseProgram(octree_renderer_program);
            //    glBindVertexArray(quad_vao);
            //    // glDrawArrays(GL_TRIANGLES, 0, 6);
            //}

            // Draw cube.
            {
                glUseProgram(simple_program);
                GL_CHECK();
                glBindVertexArray(cube_vao);
                GL_CHECK();
                glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, nullptr);
                GL_CHECK();
            }

            GL_CHECK();
        }

        // Copy from framebuffer to window.
        {
            glBindFramebuffer(GL_READ_FRAMEBUFFER, g_fbstate.fbo);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0); // default FBO
            glBlitFramebuffer(
                0, 0, g_fbstate.width, g_fbstate.height, 0, 0, width, height, GL_COLOR_BUFFER_BIT,
                GL_LINEAR);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }

        // Present.
        glfwSwapBuffers(window);

        // Process messages.
        glfwPollEvents();
    }

    return 0;
}
