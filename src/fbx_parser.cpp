#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <earcut.hpp>

#include "exception.h"
#include "fbx_parser.h"


FbxParser::FbxParser(void)
{
}

Scene FbxParser::parse(const std::string &filename)
{
    FbxManager *fbx_manager = FbxManager::Create();
    FbxIOSettings *iosettings = FbxIOSettings::Create(fbx_manager, IOSROOT);
    fbx_manager->SetIOSettings(iosettings);
    FbxImporter *fbx_importer = FbxImporter::Create(fbx_manager, "");

    if (!fbx_importer->Initialize(filename.c_str(), -1, fbx_manager->GetIOSettings())) {
        throw ExceptionFromHere(fbx_importer->GetStatus().GetErrorString());
    }

    FbxScene *scene = FbxScene::Create(fbx_manager, "scene");

    fbx_importer->Import(scene);
    fbx_importer->Destroy();

    // Note that we are not printing the root node because it should
    // not contain any attributes.
    FbxNode *node = scene->GetRootNode();
    if (!node) {
        throw ExceptionFromHere("Root node of FBX scene is empty.");
    }

    m_polygon_counter = 0;
    m_triangles_counter = 0;

    for (int i = 0; i < node->GetChildCount(); ++i) {
        processNode(node->GetChild(i));
    }

    fbx_manager->Destroy();

    return m_scene;
}

void FbxParser::processNode(const FbxNode *node)
{
    for (int i = 0; i < node->GetNodeAttributeCount(); ++i) {
        processAttribute(node->GetNodeAttributeByIndex(i));
    }

    for (int j = 0; j < node->GetChildCount(); j++) {
        processNode(node->GetChild(j));
    }
}

void FbxParser::processAttribute(const FbxNodeAttribute *attr)
{
    FbxNodeAttribute::EType type = attr->GetAttributeType();

    if (type == FbxNodeAttribute::eMesh) {
        processMesh(reinterpret_cast<const FbxMesh*>(attr));
    }
    else if (type == FbxNodeAttribute::eLight) {
        processLightSource(reinterpret_cast<const FbxLight*>(attr));
    }
}

void FbxParser::processMesh(const FbxMesh *mesh)
{
    size_t n_polygons = mesh->GetPolygonCount();

    for (size_t i = 0; i < n_polygons; ++i) {
        processPolygon(mesh, i);
    }
    m_polygon_counter += n_polygons;
}

FbxAMatrix FbxParser::calculateGlobalTransform(FbxNode *node) 
{
    FbxAMatrix matrix_geo;
    matrix_geo.SetIdentity();
    if (node->GetNodeAttribute()) {
        const FbxVector4 t = node->GetGeometricTranslation(FbxNode::eSourcePivot);
        const FbxVector4 r = node->GetGeometricRotation(FbxNode::eSourcePivot);
        const FbxVector4 s = node->GetGeometricScaling(FbxNode::eSourcePivot);
        matrix_geo.SetT(t);
        matrix_geo.SetR(r);
        matrix_geo.SetS(s);
    }

    FbxAMatrix global_matrix = node->EvaluateLocalTransform();

    return global_matrix * matrix_geo;
}

void FbxParser::processPolygon(const FbxMesh *mesh, int polygon_index)
{
    Material red(Color(255, 0, 0), Color(255, 0, 0), Color(255, 0, 0), 10.f);
    Material blue(Color(0, 0, 255), Color(0, 0, 255), Color(0, 0, 255), 5.f);
    Material grey(Color(128, 128, 128), Color(128, 128, 128), Color(128, 128, 128));

    constexpr int x = 0;
    constexpr int y = 1;
    constexpr int z = 2;

    FbxAMatrix matrix = calculateGlobalTransform(mesh->GetNode());

    glm::dvec4 r0 = glm::make_vec4(reinterpret_cast<double*>(matrix.GetRow(0).Buffer()));
    glm::dvec4 r1 = glm::make_vec4(reinterpret_cast<double*>(matrix.GetRow(1).Buffer()));
    glm::dvec4 r2 = glm::make_vec4(reinterpret_cast<double*>(matrix.GetRow(2).Buffer()));
    glm::dvec4 r3 = glm::make_vec4(reinterpret_cast<double*>(matrix.GetRow(3).Buffer()));
    glm::mat4 convert_matrix = glm::mat4(r0, r1, r2, r3);

    FbxVector4 *points = mesh->GetControlPoints();
    int n_vertices = mesh->GetPolygonSize(polygon_index);

    size_t index_point1 = mesh->GetPolygonVertex(polygon_index, 0);
    size_t index_point2 = mesh->GetPolygonVertex(polygon_index, 1);
    size_t index_point3 = mesh->GetPolygonVertex(polygon_index, 2);

    glm::vec3 a = glm::vec3(points[index_point1][x], points[index_point1][y], points[index_point1][z]);
    glm::vec3 b = glm::vec3(points[index_point2][x], points[index_point2][y], points[index_point2][z]);
    glm::vec3 c = glm::vec3(points[index_point3][x], points[index_point3][y], points[index_point3][z]);

    glm::vec3 norm = glm::normalize(glm::cross(a - b, a - c));
    glm::vec3 z_basis = norm;
    glm::vec3 x_basis = glm::normalize(glm::cross(norm, b - c));
    glm::vec3 y_basis = glm::normalize(glm::cross(z_basis, x_basis));

    glm::mat4 transform_matrix = {
        {x_basis[x], y_basis[x], z_basis[x], a[x]},
        {x_basis[y], y_basis[y], z_basis[y], a[y]},
        {x_basis[z], y_basis[z], z_basis[z], a[z]},
        {0.f, 0.f, 0.f, 1.f},
    };

    std::vector<std::array<double, 2>> positions2d;

    for (int i = 0; i < n_vertices; ++i) {
        int point_index = mesh->GetPolygonVertex(polygon_index, i);
        FbxVector4 vec = points[point_index];
        glm::vec4 point2d = transform_matrix * glm::vec4(vec[x], vec[y], vec[z], 1.f);
        positions2d.push_back({point2d[x], point2d[y]});
    }

    std::vector<std::vector<std::array<double, 2>>> polygon;
    polygon.push_back(positions2d);

    std::vector<size_t> indices = mapbox::earcut<size_t>(polygon);

    for (size_t i = 0; i < indices.size(); i += 3) {

        index_point1 = mesh->GetPolygonVertex(polygon_index, indices[i]);
        index_point2 = mesh->GetPolygonVertex(polygon_index, indices[i + 1]);
        index_point3 = mesh->GetPolygonVertex(polygon_index, indices[i + 2]);

        glm::vec4 point1 = convert_matrix * glm::vec4(points[index_point1][x], points[index_point1][y], points[index_point1][z], 1.f);
        glm::vec4 point2 = convert_matrix * glm::vec4(points[index_point2][x], points[index_point2][y], points[index_point2][z], 1.f);
        glm::vec4 point3 = convert_matrix * glm::vec4(points[index_point3][x], points[index_point3][y], points[index_point3][z], 1.f);

        // Notes: there is dirty hack. We replace y and z to get valid points' coordinates.
        float x1 = point1[x] / 100.f;
        float y1 = -point1[z] / 100.f;
        float z1 = point1[y] / 100.f;

        float x2 = point2[x] / 100.f;
        float y2 = -point2[z] / 100.f;
        float z2 = point2[y] / 100.f;

        float x3 = point3[x] / 100.f;
        float y3 = -point3[z] / 100.f;
        float z3 = point3[y] / 100.f;
 
        m_triangles_counter += 1;

        size_t n_materials = mesh->GetNode()->GetMaterialCount();
        if (n_materials < 1) {
            m_scene.addObject(std::make_shared<Triangle>(glm::vec3(x1, y1, z1), glm::vec3(x2, y2, z2), glm::vec3(x3, y3, z3), grey));
            continue;
        }

        int material_index = 0;

        FbxSurfacePhong *fbx_material = reinterpret_cast<FbxSurfacePhong*>(mesh->GetNode()->GetMaterial(material_index));
        glm::dvec3 ambient = glm::make_vec3(reinterpret_cast<double*>(fbx_material->Ambient.Get().Buffer()));
        ambient[0] *= 255;
        ambient[1] *= 255;
        ambient[2] *= 255;

        glm::dvec3 diffuse = glm::make_vec3(reinterpret_cast<double*>(fbx_material->Diffuse.Get().Buffer()));
        diffuse[0] *= 255;
        diffuse[1] *= 255;
        diffuse[2] *= 255;

        glm::dvec3 specular = glm::make_vec3(reinterpret_cast<double*>(fbx_material->Specular.Get().Buffer()));
        specular[0] *= 255;
        specular[1] *= 255;
        specular[2] *= 255;

        double shininess = fbx_material->Shininess.Get();
        double reflection = fbx_material->ReflectionFactor.Get();

        Material material(Color(ambient[0], ambient[1], ambient[2]), Color(diffuse[0], diffuse[1], diffuse[2]), Color(specular[0], specular[1], specular[2]), shininess, reflection);
        m_scene.addObject(std::make_shared<Triangle>(glm::vec3(x1, y1, z1), glm::vec3(x2, y2, z2), glm::vec3(x3, y3, z3), material));
    }
}

void FbxParser::processLightSource(const FbxLight *light)
{
    FbxLight::EType type = light->LightType.Get();

    // Convert value to Watts.
    float intensity = light->Intensity.Get() / 100.f;

    constexpr int x = 0;
    constexpr int y = 1;
    constexpr int z = 2;

    // Warning: Total light intensity have to be less then 1.0.
    if (type == FbxLight::ePoint) {

        FbxAMatrix matrix = calculateGlobalTransform(light->GetNode());

        glm::dvec4 r0 = glm::make_vec4(reinterpret_cast<double*>(matrix.GetRow(0).Buffer()));
        glm::dvec4 r1 = glm::make_vec4(reinterpret_cast<double*>(matrix.GetRow(1).Buffer()));
        glm::dvec4 r2 = glm::make_vec4(reinterpret_cast<double*>(matrix.GetRow(2).Buffer()));
        glm::dvec4 r3 = glm::make_vec4(reinterpret_cast<double*>(matrix.GetRow(3).Buffer()));
        glm::mat4 convert_matrix = glm::mat4(r0, r1, r2, r3);

        glm::vec4 temp = convert_matrix * glm::vec4(0.f, 0.f, 0.f, 1.f);
        glm::vec3 position(temp[x] / 100.f, temp[z] / 100.f, temp[y] / 100.f);

        m_scene.addLightSource(std::make_shared<PointLight>(intensity, position));
    }
    else if (type == FbxLight::eDirectional) {
        // printf("Directional: %f, %f, %f\n", position[x], position[y], position[z]);


        // FbxAMatrix matrix = calculateGlobalTransform(target_node);

        // glm::dvec4 r0 = glm::make_vec4(reinterpret_cast<double*>(matrix.GetRow(0).Buffer()));
        // glm::dvec4 r1 = glm::make_vec4(reinterpret_cast<double*>(matrix.GetRow(1).Buffer()));
        // glm::dvec4 r2 = glm::make_vec4(reinterpret_cast<double*>(matrix.GetRow(2).Buffer()));
        // glm::dvec4 r3 = glm::make_vec4(reinterpret_cast<double*>(matrix.GetRow(3).Buffer()));
        // glm::mat4 convert_matrix = glm::mat4(r0, r1, r2, r3);

        // printf("%.2f, %.2f, %.2f, %.2f\n", r0[0], r0[1], r0[2], r0[3]);
        // printf("%.2f, %.2f, %.2f, %.2f\n", r1[0], r1[1], r1[2], r1[3]);
        // printf("%.2f, %.2f, %.2f, %.2f\n", r2[0], r2[1], r2[2], r2[3]);
        // printf("%.2f, %.2f, %.2f, %.2f\n", r3[0], r3[1], r3[2], r3[3]);

        // glm::vec4 temp = convert_matrix * glm::vec4(0.f, 0.f, 0.f, 1.f);
        // glm::vec3 position(temp[x] / 100.f, temp[z] / 100.f, temp[y] / 100.f);
        // printf("Directional: %f, %f, %f\n", position[x], position[y], position[z]);
        // m_scene.addLightSource(std::make_shared<DirectionalLight>(intensity, position));
    }
}
