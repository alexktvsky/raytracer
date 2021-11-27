cmake_minimum_required(VERSION 3.14.0)

project(raytracer)

if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE release)
else()
    string(TOLOWER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE)
endif()

include(cmake/find_opengl.cmake)
include(cmake/find_glut.cmake)
include(cmake/find_cuda.cmake)
include(cmake/find_fbxsdk.cmake)

add_executable(${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
PRIVATE
    src
    src/cuda
    third-party/happly/include
    third-party/mapbox/include
    ${CMAKE_REQUIRED_INCLUDE_PATHS}
)

target_sources(${PROJECT_NAME}
PRIVATE
    src/main.cpp
    src/raytracer.cpp
    src/camera.cpp
    src/scene.cpp
    src/sphere.cpp
    src/plane.cpp
    src/triangle.cpp
    src/ambient_light.cpp
    src/directional_light.cpp
    src/point_light.cpp
    src/ply_parser.cpp
    src/fbx_parser.cpp
    src/exception.cpp
    src/cuda_exception.cpp
    # CUDA Sources
    src/cuda/render_kernel.cu
    src/cuda/camera.cu
    src/cuda/sphere.cu
    src/cuda/plane.cu
    src/cuda/triangle.cu
    src/cuda/ambient_light.cu
    src/cuda/directional_light.cu
    src/cuda/point_light.cu
    src/cuda/scene.cu
)

# set(CUDA_FOUND 0)

if (CUDA_FOUND)
    enable_language(CUDA)
    target_compile_definitions(${PROJECT_NAME} PRIVATE HAVE_CUDA)
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    # set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "35;50;72")
    # set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES 70-real 72-virtual)
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES OFF)
    target_compile_options(${PROJECT_NAME}
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored
        -use_fast_math
    >)
endif()

target_compile_options(${PROJECT_NAME}
PRIVATE
    -Wall
    -D_GLIBCXX_DEBUG
    -D_ITERATOR_DEBUG_LEVEL=1
)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)


if (CMAKE_BUILD_TYPE STREQUAL debug)
    target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG)
    target_compile_options(${PROJECT_NAME} PRIVATE -g3 -O0 -ggdb3)
elseif (CMAKE_BUILD_TYPE STREQUAL release)
    target_compile_options(${PROJECT_NAME} PRIVATE -g0 -O3)
endif()

if (NOT WIN32)
    list(APPEND CMAKE_REQUIRED_LIBRARIES dl)
endif()

target_link_libraries(${PROJECT_NAME}
    ${CMAKE_REQUIRED_LIBRARIES}
)
