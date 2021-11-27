#include <cstdio>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>
#include <string>
#include <memory>
#include <new>
#include <GL/freeglut.h>

#if defined(__APPLE__)
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#if defined(HAVE_CUDA)
#include <cuda_runtime.h>
#endif

#include "raytracer.h"
#include "sphere.h"
#include "plane.h"
#include "triangle.h"
#include "point_light.h"
#include "ambient_light.h"
#include "directional_light.h"
#include "scene_parser.h"
#include "ply_parser.h"
#include "fbx_parser.h"


size_t image_width = 640;
size_t image_height = 480;

Scene scene;

void init(void)
{
    // Set color of clear
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
}

void changeSize(GLsizei w, GLsizei h)
{
    if (h == 0) {
        h = 1;
    }

    image_width = w;
    image_height = h;

    glViewport(0, 0, w, h);
        
    // Reset the coordinate system before modifying
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    
    // Set the clipping volume
    gluOrtho2D(0.0f, static_cast<GLfloat>(w), 0.0, static_cast<GLfloat>(h));
        
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();    
}

void display(void)
{
    // Clear the window with current clearing color
    glClear(GL_COLOR_BUFFER_BIT);

    glPixelStorei(GL_UNPACK_SWAP_BYTES, true); // GL_PACK_SWAP_BYTES

    // Use Window coordinates to set raster position
    glRasterPos2i(0, 0);

    Camera camera(glm::vec3(0.f, 0.f, 0.f), image_height);

    Raytracer rt(image_width, image_height, scene, camera);

    std::vector<Color> frame = rt.render();

    glDrawPixels(image_width, image_height, GL_RGB, GL_UNSIGNED_BYTE, frame.data());

    glutSwapBuffers();
}

std::string getFileExtension(const std::string &filename)
{
    return filename.substr(filename.find_last_of(".") + 1);
}


int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Missed input file\n");
        return 1;
    }

#if defined (HAVE_CUDA)
    cudaDeviceProp prop;
    int dev;

    std::memset(&prop, 0, sizeof(prop));
    prop.major = 5;
    prop.minor = 0;

    checkCudaError(cudaChooseDevice(&dev, &prop));
    checkCudaError(cudaSetDevice(dev));
#endif

    std::string file_extension = getFileExtension(argv[1]);

    std::unique_ptr<SceneParser> parser;

    if (file_extension == "ply") {
        parser = std::make_unique<PlyParser>();
        scene = parser->parse(argv[1]);
        // We explicitly add light sources because ply file format doesn't contain them.
        // Warning: Total light intensity have to be less then 1.0.
        scene.addLightSource(std::make_shared<AmbientLight>(0.4f));
        scene.addLightSource(std::make_shared<PointLight>(0.4f, glm::vec3(1.f, 1.f, 5.f)));
        scene.addLightSource(std::make_shared<DirectionalLight>(0.2f, glm::vec3(1.f, 1.f, -1.f)));
    }
    else if (file_extension == "fbx") {
        parser = std::make_unique<FbxParser>();
        scene = parser->parse(argv[1]);
        // Fixme: remove light sources for fbx file format.
        // Warning: Total light intensity have to be less then 1.0.
        scene.addLightSource(std::make_shared<AmbientLight>(0.4f));
        scene.addLightSource(std::make_shared<PointLight>(0.4f, glm::vec3(1.f, 1.f, 5.f)));
        scene.addLightSource(std::make_shared<DirectionalLight>(0.2f, glm::vec3(1.f, 1.f, -1.f)));
    }
    else {
        fprintf(stderr, "Unsupported input file extension\n");
        return 1;
    }

    printf("Faces: %zu\n", parser->getPolygonCount());
    printf("Triangles: %zu\n", parser->getTrianglesCount());

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGB);
    glutInitWindowSize(image_width, image_height);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("raytracer");
    glutDisplayFunc(display);
    glutReshapeFunc(changeSize);
    init();
    glutMainLoop();

    return 0;
}
