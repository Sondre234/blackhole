cmake_minimum_required(VERSION 3.20)
project(blackhole3d)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

# GLFW
FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.3.8
)
FetchContent_MakeAvailable(glfw)

# GLAD (OpenGL 4.3 core)
FetchContent_Declare(
        glad
        GIT_REPOSITORY https://github.com/Dav1dde/glad.git
        GIT_TAG v0.1.36
)
FetchContent_MakeAvailable(glad)

add_executable(blackhole3d main.cpp)

target_link_libraries(blackhole3d
        glfw
        glad
        opengl32
)
