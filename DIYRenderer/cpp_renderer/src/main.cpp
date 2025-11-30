#include "renderer.hpp"
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

int main(int argc, char** argv){
    std::string scenePath; int tileX=0,tileY=0,tileW=64,tileH=64, fullW=512, fullH=512; float cx=0,cy=0,cz=5, dx=0,dy=0,dz=-1, ux=0,uy=1,uz=0, fovDeg=60;
    bool debugFlag=false; bool disableAABB=false;
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
        else if(a=="--debug"){ debugFlag = true; }
        else if(a=="--disable-aabb"){ disableAABB = true; }
    }
    Scene scene = loadScene(scenePath);
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
            }
        }
    }
    for(int py=0; py<tileH; ++py){
        int y = tileY + py;
        for(int px=0; px<tileW; ++px){
            int x = tileX + px;
            
            // Test mode: output simple patterns instead of raytracing
            // Mode 1: Solid red
            // float r = 1.0f, g = 0.0f, b = 0.0f, a = 1.0f;
            
            // Mode 2: Horizontal gradient (left=red, right=green)
            // float r = (float)x / fullW;
            // float g = 1.0f - (float)x / fullW;
            // float b = 0.0f;
            // float a = 1.0f;
            
            // Mode 3: 2D gradient (x=red, y=green)
            float r = (float)x / fullW;
            float g = (float)y / fullH;
            float b = 0.5f;
            float a = 1.0f;
            
            std::cout << r << ' ' << g << ' ' << b << ' ' << a << '\n';
            
            if(debug && py==0 && px<5){
                std::cerr << "[diyrt] testMode pixel x="<<x<<" y="<<y<<" rgba=("<<r<<","<<g<<","<<b<<","<<a<<")\n";
            }
        }
    }
    return 0;
}
