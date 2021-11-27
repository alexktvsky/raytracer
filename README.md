# Raytracer

This is a training project aimed at learning ray tracing algorithm and practicing convert sequential CPU code into a parallelized GPU code using CUDA.

Ray tracing is a rendering technique for generating an image by tracing the path of light as pixels in an image plane and simulating the effects of its encounters with virtual objects. The technique is capable of producing a high degree of visual realism, more so than typical scanline rendering methods, but at a greater computational cost [[1]][wiki_link].

Happily, the rendering of images is a highly parallelizable activity, as color of every pixels can be calculated independently of the others.

Since graphics APIs such as OpenGL and DirectX are not designed for ray-traced rendering, implementing a GPU raytracer seems like a suitable project to practice with CUDA C++.

Application supports PLY and FBX file formats. The first may contain only polygon meshes, for the last one meshes, light sources and materials are supported.

OpenGL and freeglut libraries were used to display the resulting image.


## Rendering results

![][preview_image1]

![][preview_image2]



The source code is published under BSD 2-clause, the license is available [here][license].

## How to run
```
git clone https://github.com/alexktvsky/raytracer.git
cd raytracer
mkdir build
cd build
cmake .. && cmake --build . --parallel
./raytracer ../assets/fbx/ak74m.fbx
```


## Supported systems
* GNU/Linux
* macOS
* Windows 7 and above


## Third-party
* GLM ([MIT License](https://github.com/g-truc/glm/blob/master/copying.txt))
* freeglut ([MIT License](http://freeglut.sourceforge.net/))
* CUDA Toolkit ([NVIDIA Software License Agreement](https://docs.nvidia.com/cuda/eula/index.html))
* hapPLY ([MIT License](https://github.com/nmwsharp/happly/blob/master/LICENSE))
* Mapbox Earcut ([ISC License](https://github.com/mapbox/earcut.hpp/blob/master/LICENSE))
* Autodesk FBX SDK ([Autodesk FBX SDK License](https://damassets.autodesk.net/content/dam/autodesk/www/Company/docs/pdf/legal-notices-&-trademarks/Autodesk_FBX_SDK_2015_License_and_Services_Agreement.pdf))


[//]: # (LINKS)
[wiki_link]: https://en.wikipedia.org/wiki/Ray_tracing_(graphics)
[license]: LICENSE
[preview_image1]: https://github.com/alexktvsky/raytracer/blob/main/docs/images/preview1.png "Preview Image 1"
[preview_image2]: https://github.com/alexktvsky/raytracer/blob/main/docs/images/preview2.png "Preview Image 2"
