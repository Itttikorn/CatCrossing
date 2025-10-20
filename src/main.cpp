#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <learnopengl/filesystem.h>
#include <learnopengl/shader_m.h>
#include <learnopengl/camera.h>
#include <learnopengl/model.h>
#include <stb_image.h>

#include <iostream>
#include <string>
#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>

// ===================== Config =====================
static const unsigned int SCR_WIDTH = 1280;
static const unsigned int SCR_HEIGHT = 720;

static const bool  FLIP_IMAGES_WITH_STB = false;


static const float TURN_SENSITIVITY = 0.25f;


static const float CAT_SPEED = 3.0f;   
static const float CAT_RADIUS = 0.35f;  
static const float CAM_DISTANCE = 5.0f;   
static const float CAM_HEIGHT = 1.5f;   
static const float CAT_SCALE = 1.0f;   
static const float WALK_HEIGHT = 1.5f;   

static const float SIDEWALK_OFFSET_Z = 20.0f;

static const float LANE_OFFSET_Z = 8;

static const float CAR_SPAWN_PERIOD_S = 10.0f; 
static const float CAR_SPEED = 10.0f; 
// South lane
static const float CAR_SPAWN_X_POSDIR = +100.0f; 
static const float CAR_DESPAWN_X_NEG = -120.0f;
// North lane
static const float CAR_SPAWN_X_NEGDIR = -100.0f;
static const float CAR_DESPAWN_X_POS = +120.0f;

static const float CAR_COLLISION_RADIUS = 5.0f;

// ===================== State =====================
Camera camera(glm::vec3(0.0f, 1.5f, 8.0f));

float deltaTime = 0.f, lastFrame = 0.f;
float lastX = SCR_WIDTH * 0.5f, lastY = SCR_HEIGHT * 0.5f;
bool  firstMouse = true;

// Cat orientation (degrees). Spawn 180° from your previous 90° → -90°
static float catYaw = -90.0f;

// ===================== Decls =====================
void framebuffer_size_callback(GLFWwindow*, int w, int h);
void mouse_callback(GLFWwindow*, double x, double y);
void scroll_callback(GLFWwindow*, double xoff, double yoff);

static void updateCameraVectorsFromAngles(Camera& cam);
static void processInputForCat(GLFWwindow* window, glm::vec3& catPos);

static glm::vec3 closestPointOnTriangle(const glm::vec3& p, const glm::vec3& a, const glm::vec3& b, const glm::vec3& c);
static glm::vec3 triNormal(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c);

static glm::vec3 buildCollisionCorrectionFromScene(const Model& scene, const glm::vec3& candidateCenter, float radius, float ignoreHorizontalNormalThreshold = 0.7f);
static void resolveSphereCollisionTriangle(const Model& scene, glm::vec3& center, const glm::vec3& prevCenter, float radius, float fixedHeight);

static glm::vec3 computeModelAABBCenter(const Model& model);
static GLuint loadCubemap(const std::vector<std::string>& faces);

// ===================== Cars =====================
enum class LaneDir { NegX, PosX };
struct Car {
    glm::vec3 pos;   
    float     yaw;   
    float     speed; 
    LaneDir   dir;   
};

static void spawnCarPair(std::vector<Car>& cars, float centerZ)
{
    // South lane
    {
        Car c;
        c.pos = glm::vec3(CAR_SPAWN_X_POSDIR, WALK_HEIGHT - 1.0f, centerZ + LANE_OFFSET_Z);
        c.yaw = 0.0f;         
        c.speed = CAR_SPEED;
        c.dir = LaneDir::NegX;
        cars.push_back(c);
    }
    // North lane 
    {
        Car c;
        c.pos = glm::vec3(CAR_SPAWN_X_NEGDIR, WALK_HEIGHT - 1.0f, centerZ - LANE_OFFSET_Z);
        c.yaw = 180.0f;            
        c.speed = CAR_SPEED;
        c.dir = LaneDir::PosX;
        cars.push_back(c);
    }
}

static void updateCars(std::vector<Car>& cars, float dt)
{
    for (auto& c : cars) {
        if (c.dir == LaneDir::NegX) c.pos.x -= c.speed * dt;
        else                        c.pos.x += c.speed * dt;
    }
    // Despawn
    cars.erase(std::remove_if(cars.begin(), cars.end(), [](const Car& c) {
        return (c.dir == LaneDir::NegX && c.pos.x < CAR_DESPAWN_X_NEG) ||
            (c.dir == LaneDir::PosX && c.pos.x > CAR_DESPAWN_X_POS);
        }), cars.end());
}

// 2D circle test (XZ)
static bool catHitAnyCar(const glm::vec3& catPos, const std::vector<Car>& cars)
{
    const float sumR = CAT_RADIUS + CAR_COLLISION_RADIUS;
    const float sumR2 = sumR * sumR;
    for (const auto& c : cars) {
        float dx = catPos.x - c.pos.x;
        float dz = catPos.z - c.pos.z;
        if (dx * dx + dz * dz <= sumR2) return true;
    }
    return false;
}

// ============================================================
// Callbacks & Helpers
// ============================================================
void framebuffer_size_callback(GLFWwindow*, int w, int h) { glViewport(0, 0, w, h); }

static void updateCameraVectorsFromAngles(Camera& cam)
{
    glm::vec3 front;
    front.x = cos(glm::radians(cam.Yaw)) * cos(glm::radians(cam.Pitch));
    front.y = sin(glm::radians(cam.Pitch));
    front.z = sin(glm::radians(cam.Yaw)) * cos(glm::radians(cam.Pitch));
    cam.Front = glm::normalize(front);
    cam.Right = glm::normalize(glm::cross(cam.Front, cam.WorldUp));
    cam.Up = glm::normalize(glm::cross(cam.Right, cam.Front));
}

void mouse_callback(GLFWwindow*, double x, double y)
{
    float xpos = (float)x, ypos = (float)y;
    if (firstMouse) { lastX = xpos; lastY = ypos; firstMouse = false; }

    float xoffset = xpos - lastX;
    lastX = xpos; lastY = ypos;

    catYaw += xoffset * TURN_SENSITIVITY;

    camera.Pitch = 0.0f;      // keep locked
    camera.Yaw = catYaw;    // keep internal yaw coherent
    updateCameraVectorsFromAngles(camera);
}

void scroll_callback(GLFWwindow*, double, double yoff)
{
    camera.ProcessMouseScroll((float)yoff);
}
static void processInputForCat(GLFWwindow* window, glm::vec3& catPos)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) catYaw -= 90.f * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) catYaw += 90.f * deltaTime;

    float yawRad = glm::radians(catYaw);
    glm::vec3 forward = glm::normalize(glm::vec3(cos(yawRad), 0.0f, sin(yawRad)));
    glm::vec3 right = glm::normalize(glm::cross(forward, camera.WorldUp));

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) catPos += forward * CAT_SPEED * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) catPos -= forward * CAT_SPEED * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) catPos -= right * CAT_SPEED * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) catPos += right * CAT_SPEED * deltaTime;
}

// =================== Collision helpers (R-TCD) ===================
static glm::vec3 closestPointOnTriangle(const glm::vec3& p, const glm::vec3& a, const glm::vec3& b, const glm::vec3& c)
{
    glm::vec3 ab = b - a;
    glm::vec3 ac = c - a;
    glm::vec3 ap = p - a;
    float d1 = glm::dot(ab, ap);
    float d2 = glm::dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;

    glm::vec3 bp = p - b;
    float d3 = glm::dot(ab, bp);
    float d4 = glm::dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / (d1 - d3);
        return a + v * ab;
    }

    glm::vec3 cp = p - c;
    float d5 = glm::dot(ab, cp);
    float d6 = glm::dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / (d2 - d6);
        return a + w * ac;
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + w * (c - b);
    }

    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    return a + ab * v + ac * w;
}

static glm::vec3 triNormal(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c)
{
    return glm::cross(b - a, c - a);
}

static glm::vec3 buildCollisionCorrectionFromScene(const Model& scene, const glm::vec3& candidateCenter, float radius, float ignoreHorizontalNormalThreshold)
{
    glm::vec3 totalCorrection(0.0f);
    for (const auto& mesh : scene.meshes) {
        if (mesh.indices.empty()) continue;
        for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
            unsigned int ia = mesh.indices[i];
            unsigned int ib = mesh.indices[i + 1];
            unsigned int ic = mesh.indices[i + 2];
            if (ia >= mesh.vertices.size() || ib >= mesh.vertices.size() || ic >= mesh.vertices.size()) continue;
            glm::vec3 A = mesh.vertices[ia].Position;
            glm::vec3 B = mesh.vertices[ib].Position;
            glm::vec3 C = mesh.vertices[ic].Position;

            glm::vec3 n = triNormal(A, B, C);
            float nlen = glm::length(n);
            if (nlen < 1e-6f) continue;
            glm::vec3 nn = n / nlen;

            if (nn.y > ignoreHorizontalNormalThreshold) continue;

            glm::vec3 closest = closestPointOnTriangle(candidateCenter, A, B, C);
            glm::vec3 diff = candidateCenter - closest;
            float dist = glm::length(diff);

            if (dist < 1e-6f) {
                float penetration = radius;
                totalCorrection += nn * penetration;
            }
            else if (dist < radius) {
                float penetration = radius - dist;
                glm::vec3 dir = diff / dist;
                totalCorrection += dir * penetration;
            }
        }
    }
    return totalCorrection;
}

static void resolveSphereCollisionTriangle(const Model& scene, glm::vec3& center, const glm::vec3& prevCenter, float radius, float fixedHeight)
{
    center.y = fixedHeight;
    glm::vec3 desired = center;
    glm::vec3 candidate = desired;

    const int maxIterations = 4;
    for (int it = 0; it < maxIterations; ++it) {
        glm::vec3 correction = buildCollisionCorrectionFromScene(scene, candidate, radius, 0.7f);
        if (glm::length(correction) < 1e-4f) { center = candidate; return; }
        candidate += correction;
        candidate.y = fixedHeight;
    }

    glm::vec3 move = desired - prevCenter;
    glm::vec3 candX = prevCenter + glm::vec3(move.x, 0.0f, 0.0f); candX.y = fixedHeight;
    glm::vec3 corrX = buildCollisionCorrectionFromScene(scene, candX, radius, 0.7f);
    if (glm::length(corrX) < 1e-4f) { center = candX; return; }

    glm::vec3 candZ = prevCenter + glm::vec3(0.0f, 0.0f, move.z); candZ.y = fixedHeight;
    glm::vec3 corrZ = buildCollisionCorrectionFromScene(scene, candZ, radius, 0.7f);
    if (glm::length(corrZ) < 1e-4f) { center = candZ; return; }

    center = prevCenter;
}

static glm::vec3 computeModelAABBCenter(const Model& model)
{
    const float INF = std::numeric_limits<float>::infinity();
    glm::vec3 mn(INF), mx(-INF);
    for (const auto& mesh : model.meshes) {
        for (const auto& v : mesh.vertices) {
            mn = glm::min(mn, v.Position);
            mx = glm::max(mx, v.Position);
        }
    }
    return (mn + mx) * 0.5f;
}

static GLuint loadCubemap(const std::vector<std::string>& faces)
{
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);

    int width, height, nrChannels;
    stbi_set_flip_vertically_on_load(false);
    for (GLuint i = 0; i < faces.size(); i++) {
        const std::string& path = faces[i];
        unsigned char* data = stbi_load(path.c_str(), &width, &height, &nrChannels, STBI_rgb_alpha);
        if (data) {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
            stbi_image_free(data);
        }
        else {
            std::cerr << "Cubemap texture failed to load at path: " << path << "\n";
            stbi_image_free(data);
        }
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    return textureID;
}

// main
int main()
{
    if (!glfwInit()) { std::cerr << "Failed to init GLFW\n"; return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Cat Crossing", nullptr, nullptr);
    if (!window) { std::cerr << "Failed to create window\n"; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) { std::cerr << "Failed to init GLAD\n"; return -1; }

    stbi_set_flip_vertically_on_load(FLIP_IMAGES_WITH_STB);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glEnable(GL_DEPTH_TEST);

    // Shaders
    Shader shader(
        FileSystem::getPath("src/3.model_loading/1.model_loading/1.model_loading.vs").c_str(),
        FileSystem::getPath("src/3.model_loading/1.model_loading/1.model_loading.fs").c_str()
    );
    shader.use();
    shader.setInt("texture_diffuse1", 0);

    Shader skyboxShader(
        FileSystem::getPath("src/3.model_loading/1.model_loading/skybox.vs").c_str(),
        FileSystem::getPath("src/3.model_loading/1.model_loading/skybox.fs").c_str()
    );
    skyboxShader.use();
    skyboxShader.setInt("skybox", 0);

    // Models
    const std::string base = "resources/objects";
    Model scene(FileSystem::getPath((base + "/scene/scene.obj").c_str()));
    Model catModel(FileSystem::getPath((base + "/cat/cat.obj").c_str()));
    Model carModel(FileSystem::getPath((base + "/car/car.obj").c_str()));

    // Spawn positions
    glm::vec3 sceneCenter = computeModelAABBCenter(scene);

    // Cat spawn
    const glm::vec3 catSpawnPos(sceneCenter.x, WALK_HEIGHT, sceneCenter.z + SIDEWALK_OFFSET_Z);
    glm::vec3 catPosition = catSpawnPos;
    catYaw = -90.0f; // 180° flipped from your earlier 90°

    // Camera initial alignment
    camera.Yaw = catYaw;
    camera.Pitch = 0.0f;
    updateCameraVectorsFromAngles(camera);

    // Skybox
    std::vector<std::string> faces{
        FileSystem::getPath("resources/textures/skybox/right.jpg"),
        FileSystem::getPath("resources/textures/skybox/left.jpg"),
        FileSystem::getPath("resources/textures/skybox/top.jpg"),
        FileSystem::getPath("resources/textures/skybox/bottom.jpg"),
        FileSystem::getPath("resources/textures/skybox/front.jpg"),
        FileSystem::getPath("resources/textures/skybox/back.jpg")
    };
    GLuint cubemapTexture = loadCubemap(faces);

    float skyboxVertices[] = {
        -1.0f,  1.0f, -1.0f,  -1.0f, -1.0f, -1.0f,   1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,   1.0f,  1.0f, -1.0f,  -1.0f,  1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,  -1.0f, -1.0f, -1.0f,  -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,  -1.0f,  1.0f,  1.0f,  -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f, -1.0f,   1.0f, -1.0f,  1.0f,   1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,   1.0f,  1.0f, -1.0f,   1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,  -1.0f,  1.0f,  1.0f,   1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,   1.0f, -1.0f,  1.0f,  -1.0f, -1.0f,  1.0f,
        -1.0f,  1.0f, -1.0f,   1.0f,  1.0f, -1.0f,   1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,  -1.0f,  1.0f,  1.0f,  -1.0f,  1.0f, -1.0f,
        -1.0f, -1.0f, -1.0f,  -1.0f, -1.0f,  1.0f,   1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,  -1.0f, -1.0f,  1.0f,   1.0f, -1.0f,  1.0f
    };

    GLuint skyboxVAO, skyboxVBO;
    glGenVertexArrays(1, &skyboxVAO);
    glGenBuffers(1, &skyboxVBO);
    glBindVertexArray(skyboxVAO);
    glBindBuffer(GL_ARRAY_BUFFER, skyboxVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVertices), skyboxVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glBindVertexArray(0);

    // Cars
    std::vector<Car> cars;
    float carSpawnTimer = CAR_SPAWN_PERIOD_S;

    while (!glfwWindowShouldClose(window)) {
        float t = (float)glfwGetTime();
        deltaTime = t - lastFrame; lastFrame = t;

        // Spawn cars 
        carSpawnTimer += deltaTime;
        if (carSpawnTimer >= CAR_SPAWN_PERIOD_S) {
            spawnCarPair(cars, sceneCenter.z);
            carSpawnTimer = 0.0f;
        }

        glm::vec3 prevCatPos = catPosition;

        // Move the cat 
        processInputForCat(window, catPosition);

        // Constrain to ground and resolve against scene triangles (horizontal sphere)
        resolveSphereCollisionTriangle(scene, catPosition, prevCatPos, CAT_RADIUS, WALK_HEIGHT);

        // Camera lock
        float yawRad = glm::radians(catYaw);
        glm::vec3 catForward = glm::normalize(glm::vec3(cos(yawRad), 0.0f, sin(yawRad)));
        glm::vec3 camPos = catPosition - catForward * CAM_DISTANCE + glm::vec3(0.0f, CAM_HEIGHT, 0.0f);
        glm::vec3 camFront = catForward;

        camera.Position = camPos;
        camera.Front = glm::normalize(camFront);
        camera.Right = glm::normalize(glm::cross(camera.Front, camera.WorldUp));
        camera.Up = glm::normalize(glm::cross(camera.Right, camera.Front));
        camera.Yaw = catYaw;
        camera.Pitch = 0.0f;

        // Update cars
        updateCars(cars, deltaTime);

        // Cat vs car hit -> respawn cat
        if (catHitAnyCar(catPosition, cars)) {
            std::cout << "Cat hit! Respawning...\n";
            catPosition = catSpawnPos;
            catYaw = -90.0f; 
        }

        // Matrices
        glm::mat4 P = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 2000.0f);
        glm::mat4 V = glm::lookAt(camera.Position, camera.Position + camera.Front, camera.WorldUp);

        // ---------- render ----------
        glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // skybox
        glDepthFunc(GL_LEQUAL);
        skyboxShader.use();
        glm::mat4 viewSky = glm::mat4(glm::mat3(V));
        skyboxShader.setMat4("projection", P);
        skyboxShader.setMat4("view", viewSky);
        glBindVertexArray(skyboxVAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glBindVertexArray(0);
        glDepthFunc(GL_LESS);

        // environment
        shader.use();
        shader.setMat4("projection", P);
        shader.setMat4("view", V);
        glm::mat4 M_env(1.0f);
        shader.setMat4("model", M_env);
        shader.setVec3("viewPos", camera.Position);
        shader.setVec3("lightPos", camera.Position + glm::vec3(0.0f, 2.0f, -2.0f));
        shader.setVec3("lightColor", glm::vec3(1.0f));
        scene.Draw(shader.ID);

        // cat
        glm::mat4 M_cat = glm::translate(glm::mat4(1.0f), catPosition);
        M_cat = glm::rotate(M_cat, glm::radians(-catYaw), glm::vec3(0, 1, 0));
        M_cat = glm::scale(M_cat, glm::vec3(CAT_SCALE));
        shader.setMat4("model", M_cat);
        catModel.Draw(shader.ID);

        // cars
        for (const auto& c : cars) {
            glm::mat4 M_car = glm::translate(glm::mat4(1.0f), c.pos);
            M_car = glm::rotate(M_car, glm::radians(c.yaw), glm::vec3(0, 1, 0));
            shader.setMat4("model", M_car);
            carModel.Draw(shader.ID);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &skyboxVAO);
    glDeleteBuffers(1, &skyboxVBO);
    glfwTerminate();
    return 0;
}
