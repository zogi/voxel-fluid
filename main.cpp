#include <GL/glew.h>
#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <spdlog/spdlog.h>

#include <array>
#include <cstdio>
#include <memory>

std::shared_ptr<spdlog::logger> g_logger;

#define GL_CHECK() check_last_gl_error(__FILE__, __LINE__)
void check_last_gl_error(const char *file, int line)
{
    GLenum status = glGetError();
    if (status != GL_NO_ERROR) {
        g_logger->error("{} ({}): GL error {}: {}", file, line, status, gluErrorString(status));
    }
}

void error_callback(int error, const char *description)
{
    g_logger->error("GLFW Error: {}", description);
}

static const struct {
    float x, y;
    float r, g, b;
} vertices[3] = { { -0.6f, -0.4f, 1.f, 0.f, 0.f },
                  { 0.6f, -0.4f, 0.f, 1.f, 0.f },
                  { 0.f, 0.6f, 0.f, 0.f, 1.f } };
static const char *vertex_shader_text = "uniform mat4 MVP;\n"
                                        "attribute vec3 vCol;\n"
                                        "attribute vec2 vPos;\n"
                                        "varying vec3 color;\n"
                                        "void main()\n"
                                        "{\n"
                                        "    gl_Position = MVP * vec4(vPos, 0.0, 1.0);\n"
                                        "    color = vCol;\n"
                                        "}\n";
static const char *fragment_shader_text = "varying vec3 color;\n"
                                          "void main()\n"
                                          "{\n"
                                          "    gl_FragColor = vec4(color, 1.0);\n"
                                          "}\n";

template <typename Traits>
class GLObject {
public:
    typedef typename Traits::value_type value_type;

    GLObject() : m_obj(value_type()) {}
    ~GLObject()
    {
        if (m_obj != value_type())
            Traits::destroy(m_obj);
    }

    template <typename... Args>
    static GLObject create(Args &&... args)
    {
        GLObject<Traits> res;
        res.m_obj = Traits::create(std::forward<Args>(args)...);
        return res;
    }

    GLObject(const GLObject &rhs) = delete;
    GLObject &operator=(const GLObject &rhs) = delete;
    GLObject(GLObject &&rhs) : m_obj(rhs.m_obj) { rhs.m_obj = value_type(); }
    GLObject &operator=(GLObject &&rhs)
    {
        m_obj = rhs.m_obj;
        rhs.m_obj = value_type();
        return *this;
    }

    operator value_type() const { return m_obj; }

private:
    value_type m_obj;
};

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

int main()
{
    g_logger = spdlog::stderr_logger_mt("fluid");

    glfwSetErrorCallback(error_callback);

    if (!glfwInit()) {
        g_logger->critical("Failed to initialize GLFT.");
        return 1;
    }
    const auto terminate_glfw = finally([]() { glfwTerminate(); });

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    GLFWwindow *window = glfwCreateWindow(640, 480, "", nullptr, nullptr);
    if (!window) {
        return 1;
    }
    const auto destroy_window = finally([&window]() { glfwDestroyWindow(window); });

    glfwMakeContextCurrent(window);

    const auto err = glewInit();
    if (GLEW_OK != err) {
        g_logger->critical("GLEW Error: {}", glewGetErrorString(err));
        return 1;
    }
    g_logger->info("Using GLEW {}", glewGetString(GLEW_VERSION));

    // Enable vsync.
    glfwSwapInterval(1);

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint vertex_buffer;
    GLint mvp_location, vpos_location, vcol_location;
    glGenBuffers(1, &vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    GLProgram program;
    {
        std::vector<GLShader> shaders;
        shaders.push_back(createAndCompileShader(GL_VERTEX_SHADER, vertex_shader_text));
        shaders.push_back(createAndCompileShader(GL_FRAGMENT_SHADER, fragment_shader_text));
        program = createAndLinkProgram(shaders);
    }

    mvp_location = glGetUniformLocation(program, "MVP");
    vpos_location = glGetAttribLocation(program, "vPos");
    vcol_location = glGetAttribLocation(program, "vCol");
    glEnableVertexAttribArray(vpos_location);
    glVertexAttribPointer(vpos_location, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (void *)0);
    glEnableVertexAttribArray(vcol_location);
    glVertexAttribPointer(
        vcol_location, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (void *)(sizeof(float) * 2));
    GL_CHECK();

    while (!glfwWindowShouldClose(window)) {
        float ratio;
        int width, height;
        // mat4x4 m, p, mvp;
        glm::mat4 m, p, mvp;
        glfwGetFramebufferSize(window, &width, &height);
        ratio = width / (float)height;
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);
        // mat4x4_identity(m);
        // mat4x4_rotate_Z(m, m, (float) glfwGetTime());
        m = glm::rotate(glm::mat4(), (float)glfwGetTime(), glm::vec3(0.0, 0.0, 1.0));
        // mat4x4_ortho(p, -ratio, ratio, -1.f, 1.f, 1.f, -1.f);
        p = glm::ortho(-ratio, ratio, -1.f, 1.f, 1.f, -1.f);
        // mat4x4_mul(mvp, p, m);
        mvp = p * m;
        glUseProgram(program);
        glUniformMatrix4fv(mvp_location, 1, GL_FALSE, (const GLfloat *)&mvp[0][0]);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        GL_CHECK();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    return 0;
}
