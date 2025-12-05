#include "renderer.hpp"
#include "pbr.hpp"
#include "json.hpp"

#include <thread>
#include <atomic>

#ifdef _OPENMP
#include <omp.h>
#endif
using json = nlohmann::json;
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cmath>
#include <map>

/**
 * ========== DIY Ray Tracer - Main Entry Point ==========
 * 
 * This is the C++ backend for the Blender DIY Renderer addon.
 * It receives scene data as JSON, renders using path tracing,
 * and outputs pixel data to stdout for Python to capture.
 * 
 * Data Flow:
 * 1. Blender Python exports scene → JSON file
 * 2. C++ reads JSON → constructs Scene data structures
 * 3. Path tracing renders tiles → accumulates samples
 * 4. Outputs linear RGB pixels → Python reads stdout
 * 5. Python Y-flips pixels → passes to Blender display
 * 
 * Command Line Arguments:
 * --scene <path>     : Path to JSON scene file
 * --tile <x> <y> <w> <h> : Tile region to render
 * --samples <n>      : Number of samples per pixel
 * --depth <n>        : Maximum ray bounce depth (default: 8)
 * --mode <mode>      : Render mode (raytrace/normal/albedo/emission)
 * 
 * Output Format:
 * Pixel data as binary: [r g b r g b ...] (float32, linear color space)
 */

// ========== Node Graph Parsing ==========

/**
 * parseSocketValue: Convert JSON value to SocketValue union
 * 
 * Handles multiple types:
 * - Single number → FLOAT
 * - Array[3] → VEC3 (RGB or vector)
 * - Array[4] → VEC4 (RGBA)
 * - Boolean → BOOL
 * - String → STRING
 * 
 * CRITICAL: VEC4 is stored in v4 field, VEC3 in v3 field.
 * This distinction was the source of the "all black materials" bug.
 */
SocketValue parseSocketValue(const json &j) {
    if(j.is_number()) {
        return SocketValue::makeFloat(j.get<float>());
    } else if(j.is_array()) {
        if(j.size() == 3) {
            return SocketValue::makeVec3(j[0], j[1], j[2]);
        } else if(j.size() == 4) {
            float x = j[0], y = j[1], z = j[2], w = j[3];
            static int debugCount = 0;
            if(++debugCount <= 3) {
                std::cerr << "[parseSocketValue] VEC4: [" << x << ", " << y << ", " << z << ", " << w << "]\n";
            }
            return SocketValue::makeVec4(x, y, z, w);
        }
    } else if(j.is_boolean()) {
        SocketValue sv;
        sv.type = SocketValue::BOOL;
        sv.b = j.get<bool>();
        return sv;
    } else if(j.is_string()) {
        SocketValue sv;
        sv.type = SocketValue::STRING;
        sv.s = j.get<std::string>();
        return sv;
    }
    return SocketValue();  // NONE
}

/**
 * parseNodeTree: Construct node graph from JSON
 * 
 * Builds the material node tree by:
 * 1. Creating all MaterialNode objects
 * 2. Parsing input/output sockets for each node
 * 3. Recording socket connections (linked_node, linked_socket)
 * 4. Storing default values for unconnected sockets
 * 
 * @param nodeTreeJson JSON object containing "nodes" array
 * @return NodeTree structure ready for evaluation
 */
NodeTree parseNodeTree(const json &nodeTreeJson) {
    NodeTree tree;
    
    if(!nodeTreeJson.contains("nodes") || !nodeTreeJson.is_object()) {
        return tree;  // Invalid
    }
    
    const auto &nodesArray = nodeTreeJson["nodes"];
    const auto &linksArray = nodeTreeJson.contains("links") ? nodeTreeJson["links"] : json::array();
    
    // Build a map of node connections: (to_node, to_socket) -> (from_node, from_socket)
    std::map<std::pair<std::string, std::string>, std::pair<std::string, std::string>> connections;
    for(const auto &link : linksArray) {
        std::string from_node = link["from_node"];
        std::string from_socket = link["from_socket"];
        std::string to_node = link["to_node"];
        std::string to_socket = link["to_socket"];
        connections[{to_node, to_socket}] = {from_node, from_socket};
    }
    
    // Parse nodes
    for(const auto &nodeJson : nodesArray) {
        MaterialNode node;
        node.name = nodeJson.value("name", "");
        node.type = nodeJson.value("type", "");
        node.label = nodeJson.value("label", "");
        
        // Parse inputs
        if(nodeJson.contains("inputs")) {
            for(const auto &inp : nodeJson["inputs"]) {
                NodeSocket socket;
                socket.name = inp.value("name", "");
                socket.type = inp.value("type", "");
                if(inp.contains("default_value")) {
                    socket.default_value = parseSocketValue(inp["default_value"]);
                }
                
                // Check if this socket is connected
                auto connKey = std::make_pair(node.name, socket.name);
                if(connections.find(connKey) != connections.end()) {
                    socket.is_linked = true;
                    socket.linked_node = connections[connKey].first;
                    socket.linked_socket = connections[connKey].second;
                }
                
                node.inputs.push_back(socket);
            }
        }
        
        // Parse outputs
        if(nodeJson.contains("outputs")) {
            for(const auto &outp : nodeJson["outputs"]) {
                NodeSocket socket;
                socket.name = outp.value("name", "");
                socket.type = outp.value("type", "");
                if(outp.contains("default_value")) {
                    socket.default_value = parseSocketValue(outp["default_value"]);
                }
                node.outputs.push_back(socket);
            }
        }
        
        tree.nodes.push_back(node);
    }
    
    tree.valid = !tree.nodes.empty();
    return tree;
}

/*
Simple protocol (prototype):
Scene file format (plain text):
---
mesh <name> <vertex_count> <triangle_count>
# vertex_count lines: v x y z
# triangle_count lines: t i0 i1 i2
mesh ... (repeat)
(end)
---

Invocation:
./diyrt --scene scene.txt --tile X Y W H --full FULL_W FULL_H --campos cx cy cz --camdir dx dy dz --camup ux uy uz --fov degrees

Output:
ASCII lines RGBA per pixel row-major: r g b a\n
(For speed later switch to binary)
*/

struct Camera {
    Vec3 pos; Vec3 dir; Vec3 up; float fovDeg; float aspect; Vec3 right; Vec3 forward;
};

Scene loadScene(const std::string &path){
    Scene scene; std::ifstream in(path); if(!in){ std::cerr << "Failed to open scene file: " << path << "\n"; return scene; }
    std::string tok;
    while(in >> tok){
        // コメント行をスキップ (# で始まるトークンはその行残りを捨てる)
        if(!tok.empty() && tok[0] == '#'){ std::string dummy; std::getline(in, dummy); continue; }
        if(tok == "mesh"){
            std::string name; int vcount, tcount; in >> name >> vcount >> tcount;
            Mesh m; m.vertices.reserve(vcount); m.triangles.reserve(tcount);
            
            // Check for material line
            std::streampos pos = in.tellg();
            std::string next_tok;
            in >> next_tok;
            if(next_tok == "material"){
                // Read material properties: r g b metallic roughness
                float r, g, b, metallic, roughness;
                in >> r >> g >> b >> metallic >> roughness;
                m.material = Material(Vec3(r, g, b), metallic, roughness);
            } else {
                // No material line, use default and restore position
                in.seekg(pos);
                m.material = Material();
            }
            
            for(int i=0;i<vcount;i++){ std::string vt; in >> vt; if(vt != "v"){ std::cerr << "Expected v" << std::endl; return scene; } float x,y,z; in >> x >> y >> z; m.vertices.emplace_back(x,y,z); }
            for(int i=0;i<tcount;i++){ std::string tt; in >> tt; if(tt != "t"){ std::cerr << "Expected t" << std::endl; return scene; } int a,b,c; in >> a >> b >> c; Triangle tri; tri.i0=a; tri.i1=b; tri.i2=c; tri.smooth=false; m.triangles.push_back(tri); }
            finalizeMeshBounds(m); scene.meshes.push_back(std::move(m));
        } else if(tok == "(end)"){
            break;
        } else {
            std::cerr << "Unknown token: " << tok << "\n"; break;
        }
    }
    return scene;
}

Camera makeCamera(float cx,float cy,float cz,float dx,float dy,float dz,float ux,float uy,float uz,float fovDeg,int fullW,int fullH){
    Camera cam; cam.pos = {cx,cy,cz}; cam.dir = {dx,dy,dz}; cam.dir.normalize(); cam.up = {ux,uy,uz}; cam.up.normalize();
    cam.fovDeg = fovDeg; cam.aspect = (float)fullW / (float)fullH;
    cam.forward = cam.dir; // world forward
    cam.right = Vec3::cross(cam.forward, cam.up); cam.right.normalize();
    cam.up = Vec3::cross(cam.right, cam.forward); cam.up.normalize();
    return cam;
}

Scene loadSceneFromJson(const std::string &path) {
    Scene scene;
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Failed to open JSON scene file: " << path << "\n";
        return scene;
    }
    json j;
    in >> j;
    if (!j.contains("meshes")) return scene;
    for (const auto &meshj : j["meshes"]) {
        Mesh m;
        // Vertices
        if (meshj.contains("vertices")) {
            for (const auto &v : meshj["vertices"]) {
                m.vertices.emplace_back(v[0], v[1], v[2]);
            }
        }
        // Per-triangle vertex normals (for smooth shading)
        std::vector<std::array<Vec3, 3>> triangleNormals;
        bool hasTriangleNormals = false;
        if (meshj.contains("triangle_normals")) {
            hasTriangleNormals = true;
            for (const auto &tn : meshj["triangle_normals"]) {
                std::array<Vec3, 3> normals;
                normals[0] = Vec3(tn[0][0], tn[0][1], tn[0][2]);
                normals[1] = Vec3(tn[1][0], tn[1][1], tn[1][2]);
                normals[2] = Vec3(tn[2][0], tn[2][1], tn[2][2]);
                triangleNormals.push_back(normals);
            }
        }
        
        // Smooth shading flag
        bool smoothShading = meshj.contains("smooth") && meshj["smooth"].get<bool>();
        
        // Triangles
        int triIndex = 0;
        if (meshj.contains("triangles")) {
            for (const auto &t : meshj["triangles"]) {
                Triangle tri;
                tri.i0 = t[0];
                tri.i1 = t[1];
                tri.i2 = t[2];
                
                // Per-triangle vertex normals for smooth shading
                if (hasTriangleNormals && (size_t)triIndex < triangleNormals.size()) {
                    tri.n0 = triangleNormals[triIndex][0];
                    tri.n1 = triangleNormals[triIndex][1];
                    tri.n2 = triangleNormals[triIndex][2];
                    tri.smooth = smoothShading;
                } else {
                    tri.smooth = false;
                }
                
                m.triangles.push_back(tri);
                triIndex++;
            }
        }
        // Material
        if (meshj.contains("material")) {
            const auto &mat = meshj["material"];
            Vec3 albedo = Vec3(0.8f, 0.8f, 0.8f);
            float metallic = 0.0f, roughness = 0.5f;
            Vec3 emission = Vec3(0.0f, 0.0f, 0.0f);
            float transmission = 0.0f, ior = 1.45f;
            if (mat.contains("base_color")) {
                albedo = Vec3(mat["base_color"][0], mat["base_color"][1], mat["base_color"][2]);
            }
            if (mat.contains("metallic")) metallic = mat["metallic"];
            if (mat.contains("roughness")) roughness = mat["roughness"];
            if (mat.contains("emission")) {
                emission = Vec3(mat["emission"][0], mat["emission"][1], mat["emission"][2]);
            }
            if (mat.contains("transmission")) transmission = mat["transmission"];
            if (mat.contains("ior")) ior = mat["ior"];
            
            m.material = Material(albedo, metallic, roughness, emission);
            m.material.transmission = transmission;
            m.material.ior = ior;
            
            // Parse node tree if present
            if (mat.contains("node_tree") && !mat["node_tree"].is_null()) {
                m.material.nodeTree = parseNodeTree(mat["node_tree"]);
                m.material.useNodes = mat.value("use_nodes", false) && m.material.nodeTree.valid;
                
                std::cerr << "[Mesh: " << meshj["name"] << "] node_tree parsed: "
                          << "nodes=" << m.material.nodeTree.nodes.size()
                          << ", useNodes=" << (m.material.useNodes ? "true" : "false") << "\n";
            }
            
            // Debug print material
            std::cerr << "[Mesh: " << meshj["name"] << "] material: "
                      << "color=(" << albedo.x << "," << albedo.y << "," << albedo.z << "), "
                      << "metallic=" << metallic << ", roughness=" << roughness
                      << ", emission=(" << emission.x << "," << emission.y << "," << emission.z << ")"
                      << ", transmission=" << transmission << ", ior=" << ior << "\n";
        }
        // ...attributes (vertex color, uv, custom) can be parsed here as needed...
        if (meshj.contains("attributes")) {
            std::cerr << "[Mesh: " << meshj["name"] << "] attributes:\n";
            for (auto it = meshj["attributes"].begin(); it != meshj["attributes"].end(); ++it) {
                std::cerr << "  " << it.key() << ": ";
                if (it.value().is_object()) {
                    auto obj = it.value();
                    std::cerr << "domain=" << obj["domain"] << ", type=" << obj["data_type"] << ", data=[..." << obj["data"].size() << "]\n";
                } else if (it.value().is_array()) {
                    std::cerr << "array size=" << it.value().size() << "\n";
                } else {
                    std::cerr << it.value() << "\n";
                }
            }
        }
        finalizeMeshBounds(m);
        scene.meshes.push_back(std::move(m));
    }
    return scene;
}

/**
 * ========== Main Entry Point ==========
 * 
 * Parses command-line arguments, loads scene from JSON,
 * and renders the specified tile region.
 */
int main(int argc, char** argv){
    // Default values
    std::string scenePath;
    int tileX=0, tileY=0, tileW=64, tileH=64, fullW=512, fullH=512;
    float cx=0, cy=0, cz=5, dx=0, dy=0, dz=-1, ux=0, uy=1, uz=0, fovDeg=60;
    bool debugFlag=false; bool disableAABB=false;
    std::string mode = "raytrace";  // Render mode: raytrace/normal/albedo/emission
    std::string algorithm = "nee";  // Sampling algorithm: simple/nee/mis
    int samples = 1;                 // Samples per pixel
    int maxDepth = 8;                // Max ray bounce depth (increased from 3 for better quality)
    int sampleOffset = 0;            // Sample offset for progressive rendering
    int passId = -1;                 // Interleaved pass ID (-1 = render all pixels)
    int numPasses = 16;              // Total number of passes (e.g., 16 for 4x4 grid)
    
    // ===== Parse Command-Line Arguments =====
    for(int i=1;i<argc;i++){
        std::string a = argv[i];
        auto need = [&](const char* msg){ if(i+1>=argc){ std::cerr << "Missing value for " << msg << "\n"; std::exit(1);} };
        if(a=="--scene"){ need("--scene"); scenePath = argv[++i]; }
        else if(a=="--tile"){ need("--tile"); tileX = std::atoi(argv[++i]); need("--tile"); tileY = std::atoi(argv[++i]); need("--tile"); tileW = std::atoi(argv[++i]); need("--tile"); tileH = std::atoi(argv[++i]); }
        else if(a=="--full"){ need("--full"); fullW = std::atoi(argv[++i]); need("--full"); fullH = std::atoi(argv[++i]); }
        else if(a=="--campos"){ need("--campos"); cx = std::atof(argv[++i]); need("--campos"); cy = std::atof(argv[++i]); need("--campos"); cz = std::atof(argv[++i]); }
        else if(a=="--camdir"){ need("--camdir"); dx = std::atof(argv[++i]); need("--camdir"); dy = std::atof(argv[++i]); need("--camdir"); dz = std::atof(argv[++i]); }
        else if(a=="--camup"){ need("--camup"); ux = std::atof(argv[++i]); need("--camup"); uy = std::atof(argv[++i]); need("--camup"); uz = std::atof(argv[++i]); }
        else if(a=="--fov"){ need("--fov"); fovDeg = std::atof(argv[++i]); }
        else if(a=="--mode"){ need("--mode"); mode = argv[++i]; }         // Debug or render mode
        else if(a=="--algorithm"){ need("--algorithm"); algorithm = argv[++i]; }  // Sampling algorithm
        else if(a=="--samples"){ need("--samples"); samples = std::atoi(argv[++i]); }  // Samples per pixel
        else if(a=="--depth"){ need("--depth"); maxDepth = std::atoi(argv[++i]); }     // Ray bounce limit
        else if(a=="--sample-offset"){ need("--sample-offset"); sampleOffset = std::atoi(argv[++i]); }  // For progressive rendering
        else if(a=="--pass"){ need("--pass"); passId = std::atoi(argv[++i]); }  // Interleaved pass ID
        else if(a=="--num-passes"){ need("--num-passes"); numPasses = std::atoi(argv[++i]); }  // Total passes
        else if(a=="--debug"){ debugFlag = true; }
        else if(a=="--disable-aabb"){ disableAABB = true; }
    }
    
    std::srand(42);  // Legacy seed (kept for compatibility)
    // Note: Per-pixel seeding is done inside the render loop
    
    // ===== Load Scene =====
    Scene scene;
    if (scenePath.size() > 5 && scenePath.substr(scenePath.size()-5) == ".json") {
        scene = loadSceneFromJson(scenePath);  // JSON format with node trees
    } else {
        scene = loadScene(scenePath);  // Legacy text format
    }
    
    // ===== Setup Camera =====
    Camera cam = makeCamera(cx,cy,cz,dx,dy,dz,ux,uy,uz,fovDeg, fullW, fullH);
    float fovRad = cam.fovDeg * (float)M_PI / 180.0f;
    float scale = std::tan(fovRad * 0.5f);
    
    // ===== Debug Output =====
    bool debug = true;  // Always enable debug for now
    if(debug){
        std::cerr << "[diyrt] SCENE meshes=" << scene.meshes.size() << " aspect=" << cam.aspect << " fovDeg=" << cam.fovDeg << "\n";
        std::cerr << "[diyrt] CAMERA pos=(" << cam.pos.x << "," << cam.pos.y << "," << cam.pos.z << ") dir=(" << cam.forward.x << "," << cam.forward.y << "," << cam.forward.z << ") up=(" << cam.up.x << "," << cam.up.y << "," << cam.up.z << ") right=(" << cam.right.x << "," << cam.right.y << "," << cam.right.z << ")\n";
        for(size_t i=0;i<scene.meshes.size();++i){
            const auto &m = scene.meshes[i];
            std::cerr << "[diyrt] mesh" << i << " bbox [" << m.bmin.x << "," << m.bmin.y << "," << m.bmin.z << "] -> [" << m.bmax.x << "," << m.bmax.y << "," << m.bmax.z << "] verts=" << m.vertices.size() << " tris=" << m.triangles.size() << "\n";
            if(i==0){
                int show = std::min<int>((int)m.vertices.size(),5);
                for(int v=0; v<show; ++v){ const auto &vv = m.vertices[v]; std::cerr << "[diyrt] v"<<v<<"=("<<vv.x<<","<<vv.y<<","<<vv.z<<")\n"; }
                // Test AABB intersection manually for first mesh
                Ray testRay{cam.pos, cam.forward};
                bool aabbHit = rayAABB(testRay, m.bmin, m.bmax);
                std::cerr << "[diyrt] AABB test: ray from " << testRay.o.x << "," << testRay.o.y << "," << testRay.o.z 
                          << " dir " << testRay.d.x << "," << testRay.d.y << "," << testRay.d.z 
                          << " -> hit=" << (aabbHit ? "YES" : "NO") << "\n";
                // Manual AABB calculation for Y axis
                float invY = (testRay.d.y != 0.0f) ? 1.0f / testRay.d.y : 1e30f;
                float t1y = (m.bmin.y - testRay.o.y) * invY;
                float t2y = (m.bmax.y - testRay.o.y) * invY;
                std::cerr << "[diyrt] Y-axis slab: bmin.y=" << m.bmin.y << " bmax.y=" << m.bmax.y 
                          << " ray.o.y=" << testRay.o.y << " ray.d.y=" << testRay.d.y
                          << " t1y=" << t1y << " t2y=" << t2y << "\n";
                // Test first triangle intersection directly
                if(m.triangles.size() > 0) {
                    const auto &tri = m.triangles[0];
                    const Vec3 &a = m.vertices[tri.i0];
                    const Vec3 &b = m.vertices[tri.i1];
                    const Vec3 &c = m.vertices[tri.i2];
                    float t = rayTriangle(testRay, a, b, c);
                    std::cerr << "[diyrt] Triangle test: tri0 indices=" << tri.i0 << "," << tri.i1 << "," << tri.i2
                              << " normal=(" << tri.faceNormal.x << "," << tri.faceNormal.y << "," << tri.faceNormal.z << ")"
                              << " t=" << t << "\n";
                }
            }
        }
    }
    
    // Progress tracking
    int totalPixels = tileW * tileH;
    
    // Allocate output buffer for parallel rendering
    // Each pixel stores RGBA (4 floats)
    // Initialize with -1 to mark unrendered pixels (for interleaved mode)
    std::vector<float> pixelBuffer(totalPixels * 4, -1.0f);
    
    // Build scene lights for MIS
    SceneLights sceneLights;
    sceneLights.buildFromScene(scene);
    std::cerr << "[diyrt] Found " << sceneLights.lights.size() << " light triangles, total area=" << sceneLights.totalArea << "\n";
    
    // Report thread count and pass info
    #ifdef _OPENMP
    int numThreads = omp_get_max_threads();
    std::cerr << "[diyrt] OpenMP enabled with " << numThreads << " threads\n";
    #else
    std::cerr << "[diyrt] Single-threaded mode\n";
    #endif
    
    if (passId >= 0) {
        std::cerr << "[diyrt] Interleaved pass " << passId << "/" << numPasses << "\n";
    }
    
    // Calculate grid size for interleaved pattern
    // For numPasses=16, gridSize=4 (4x4 pattern)
    // For numPasses=4, gridSize=2 (2x2 pattern)
    int gridSize = (int)std::sqrt((float)numPasses);
    if (gridSize * gridSize != numPasses) {
        gridSize = (int)std::ceil(std::sqrt((float)numPasses));
    }
    
    // Parallel rendering loop
    #pragma omp parallel for schedule(dynamic, 16) collapse(2)
    for(int py=0; py<tileH; ++py){
        for(int px=0; px<tileW; ++px){
            int y = tileY + py;
            int x = tileX + px;
            int pixelIdx = py * tileW + px;
            
            // Interleaved rendering: only render pixels belonging to this pass
            if (passId >= 0) {
                // Compute which pass this pixel belongs to
                // Using modulo to create interleaved pattern
                int pixelPassX = x % gridSize;
                int pixelPassY = y % gridSize;
                int pixelPass = pixelPassY * gridSize + pixelPassX;
                
                if (pixelPass != passId) {
                    // Skip this pixel - it belongs to a different pass
                    // Leave buffer at -1 to indicate unrendered
                    continue;
                }
            }
            
            float ndcX = (2.0f * (x + 0.5f) / fullW - 1.0f) * cam.aspect;
            float ndcY = 1.0f - 2.0f * (y + 0.5f) / fullH;  // Y-axis flip for screen coordinates
            // Construct ray direction: forward + offset from image plane
            // Image plane is at distance 1 from camera, with size (2*scale*aspect, 2*scale)
            Vec3 worldDir = cam.forward + cam.right * (ndcX * scale) + cam.up * (ndcY * scale);
            worldDir.normalize();
            Ray ray{cam.pos, worldDir};
            Vec3 color{0,0,0};
            
            if(mode == "debug" || mode == "normal"){
                // Debug mode: show normals
                Vec3 n = traceNormal(scene, ray, !disableAABB);
                color.x = n.x * 0.5f + 0.5f;
                color.y = n.y * 0.5f + 0.5f;
                color.z = n.z * 0.5f + 0.5f;
            } else if(mode == "albedo") {
                // Debug mode: show base color (albedo)
                color = traceAlbedo(scene, ray, !disableAABB);
                if(py < 2 && px < 2) {
                    Hit hit = intersectScene(scene, ray, !disableAABB);
                    std::cerr << "[diyrt] albedo x=" << x << " y=" << y 
                              << " hit=" << (hit.hit ? "YES" : "NO");
                    if(hit.hit) {
                        std::cerr << " useNodes=" << hit.material.useNodes
                                  << " treeValid=" << hit.material.nodeTree.valid
                                  << " legacyAlbedo=(" << hit.material.albedo.x << "," << hit.material.albedo.y << "," << hit.material.albedo.z << ")";
                    }
                    std::cerr << " color=(" << color.x << "," << color.y << "," << color.z << ")\n";
                }
            } else if(mode == "emission") {
                // Debug mode: show emission
                color = traceEmission(scene, ray, !disableAABB);
            } else {
                // Raytrace mode: full path tracing with MIS
                // Output is SUM of all samples (not averaged)
                // Python side will accumulate and divide by total samples
                for(int s = 0; s < samples; ++s){
                    // Seed RNG with x, y, and sample index for independent random streams
                    // sampleOffset allows progressive rendering to continue with unique seeds
                    seed_random_xyz((uint32_t)x, (uint32_t)y, (uint32_t)(sampleOffset + s));
                    
                    Ray sampleRay = ray;
                    // Add slight jitter for anti-aliasing if samples > 1
                    if(samples > 1){
                        float jitterX = (randf() - 0.5f) / fullW;
                        float jitterY = (randf() - 0.5f) / fullH;
                        Vec3 jitteredDir = cam.forward + cam.right * ((ndcX + jitterX) * scale) + cam.up * ((ndcY + jitterY) * scale);
                        jitteredDir.normalize();
                        sampleRay.d = jitteredDir;
                    }
                    // Path tracing with selected algorithm:
                    // - simple: BSDF sampling only (no NEE, slow convergence)
                    // - nee: NEE for direct light, BSDF for indirect (fast)
                    // - mis: Both with MIS weights (best quality)
                    if (algorithm == "simple") {
                        color = color + traceSimple(scene, sampleRay, maxDepth);
                    } else if (algorithm == "mis") {
                        color = color + traceMIS(scene, sceneLights, sampleRay, maxDepth);
                    } else {
                        // Default: NEE
                        color = color + traceNEE(scene, sceneLights, sampleRay, maxDepth);
                    }
                }
                // NOTE: Do NOT divide by samples here!
                // Output is raw sum. Python accumulates sums and divides by total at display time.
            }
            
            // Store in buffer (clamp negative values only)
            color.x = std::max(0.0f, color.x);
            color.y = std::max(0.0f, color.y);
            color.z = std::max(0.0f, color.z);
            
            pixelBuffer[pixelIdx * 4 + 0] = color.x;
            pixelBuffer[pixelIdx * 4 + 1] = color.y;
            pixelBuffer[pixelIdx * 4 + 2] = color.z;
            pixelBuffer[pixelIdx * 4 + 3] = 1.0f;
        }
    }
    
    // Progress report (after parallel section)
    std::cerr << "[diyrt] Rendering complete, outputting " << totalPixels << " pixels (binary, Y-flipped)\n";
    
    // Output pixels as binary data (much faster than text)
    // Format: raw float32 array [r,g,b,a, r,g,b,a, ...]
    // Output Y-flipped: Blender expects bottom-to-top, we rendered top-to-bottom
    for(int y = tileH - 1; y >= 0; --y) {
        int rowStart = y * tileW * 4;
        std::cout.write(reinterpret_cast<const char*>(&pixelBuffer[rowStart]), 
                        tileW * 4 * sizeof(float));
    }
    std::cout.flush();
    
    return 0;
}
