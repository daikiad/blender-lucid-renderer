#include "renderer.hpp"
#include "json.hpp"
using json = nlohmann::json;
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cmath>

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
            for(int i=0;i<tcount;i++){ std::string tt; in >> tt; if(tt != "t"){ std::cerr << "Expected t" << std::endl; return scene; } int a,b,c; in >> a >> b >> c; m.triangles.push_back({a,b,c,{}}); }
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
        // Triangles
        if (meshj.contains("triangles")) {
            for (const auto &t : meshj["triangles"]) {
                m.triangles.push_back({t[0], t[1], t[2], {}});
            }
        }
        // Material
        if (meshj.contains("material")) {
            const auto &mat = meshj["material"];
            Vec3 albedo = Vec3(0.8f, 0.8f, 0.8f);
            float metallic = 0.0f, roughness = 0.5f;
            if (mat.contains("base_color")) {
                albedo = Vec3(mat["base_color"][0], mat["base_color"][1], mat["base_color"][2]);
            }
            if (mat.contains("metallic")) metallic = mat["metallic"];
            if (mat.contains("roughness")) roughness = mat["roughness"];
            m.material = Material(albedo, metallic, roughness);
            
            // Debug print material
            std::cerr << "[Mesh: " << meshj["name"] << "] material: "
                      << "color=(" << albedo.x << "," << albedo.y << "," << albedo.z << "), "
                      << "metallic=" << metallic << ", roughness=" << roughness << "\n";
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

int main(int argc, char** argv){
    std::string scenePath; int tileX=0,tileY=0,tileW=64,tileH=64, fullW=512, fullH=512; float cx=0,cy=0,cz=5, dx=0,dy=0,dz=-1, ux=0,uy=1,uz=0, fovDeg=60;
    bool debugFlag=false; bool disableAABB=false;
    std::string mode = "raytrace";  // "debug" or "raytrace"
    int samples = 1;
    int maxDepth = 3;
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
        else if(a=="--mode"){ need("--mode"); mode = argv[++i]; }
        else if(a=="--samples"){ need("--samples"); samples = std::atoi(argv[++i]); }
        else if(a=="--depth"){ need("--depth"); maxDepth = std::atoi(argv[++i]); }
        else if(a=="--debug"){ debugFlag = true; }
        else if(a=="--disable-aabb"){ disableAABB = true; }
    }
    std::srand(42);  // Fixed seed for consistent results
    // Scene loading
    Scene scene;
    if (scenePath.size() > 5 && scenePath.substr(scenePath.size()-5) == ".json") {
        scene = loadSceneFromJson(scenePath);
    } else {
        scene = loadScene(scenePath);
    }
    Camera cam = makeCamera(cx,cy,cz,dx,dy,dz,ux,uy,uz,fovDeg, fullW, fullH);
    float fovRad = cam.fovDeg * (float)M_PI / 180.0f;
    float scale = std::tan(fovRad * 0.5f);
    bool debug = debugFlag || (std::getenv("DIYRT_DEBUG") != nullptr);
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
    for(int py=0; py<tileH; ++py){
        int y = tileY + py;
        for(int px=0; px<tileW; ++px){
            int x = tileX + px;
            float ndcX = (2.0f * (x + 0.5f) / fullW - 1.0f) * cam.aspect;
            float ndcY = (2.0f * (y + 0.5f) / fullH - 1.0f);  // Fix: Remove the 1.0f - inversion
            // Construct ray direction: forward + offset from image plane
            // Image plane is at distance 1 from camera, with size (2*scale*aspect, 2*scale)
            Vec3 worldDir = cam.forward + cam.right * (ndcX * scale) + cam.up * (ndcY * scale);
            worldDir.normalize();
            Ray ray{cam.pos, worldDir};
            Vec3 color{0,0,0};
            
            if(mode == "debug"){
                // Debug mode: show normals
                Vec3 n = traceNormal(scene, ray, !disableAABB);
                color.x = n.x * 0.5f + 0.5f;
                color.y = n.y * 0.5f + 0.5f;
                color.z = n.z * 0.5f + 0.5f;
            } else {
                // Raytrace mode: full path tracing with samples
                for(int s = 0; s < samples; ++s){
                    Ray sampleRay = ray;
                    // Add slight jitter for anti-aliasing if samples > 1
                    if(samples > 1){
                        float jitterX = (randf() - 0.5f) / fullW;
                        float jitterY = (randf() - 0.5f) / fullH;
                        Vec3 jitteredDir = cam.forward + cam.right * ((ndcX + jitterX) * scale) + cam.up * ((ndcY + jitterY) * scale);
                        jitteredDir.normalize();
                        sampleRay.d = jitteredDir;
                    }
                    color = color + trace(scene, sampleRay, maxDepth, !disableAABB);
                }
                color = color / (float)samples;
            }
            
            // Clamp and gamma correct
            color.x = std::min(1.0f, std::max(0.0f, color.x));
            color.y = std::min(1.0f, std::max(0.0f, color.y));
            color.z = std::min(1.0f, std::max(0.0f, color.z));
            color.x = std::sqrt(color.x);  // Simple gamma correction
            color.y = std::sqrt(color.y);
            color.z = std::sqrt(color.z);
            
            std::cout << color.x << ' ' << color.y << ' ' << color.z << " 1.0\n";
            if(debug && py==0 && px<5){
                // Test AABB with actual pixel ray
                bool pixelAABBHit = false;
                for(const auto &m: scene.meshes) {
                    if(rayAABB(ray, m.bmin, m.bmax)) {
                        pixelAABBHit = true;
                        break;
                    }
                }
                std::cerr << "[diyrt] sampleRay x="<<x<<" y="<<y<<" dir=("<<worldDir.x<<","<<worldDir.y<<","<<worldDir.z<<") color=("<<color.x<<","<<color.y<<","<<color.z<<")"<< (disableAABB?" noAABB":" AABB=") << (pixelAABBHit?"HIT":"MISS") <<" mode="<<mode<<"\n";
            }
        }
    }
    return 0;
}
