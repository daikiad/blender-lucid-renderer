/**
 * server.cpp - Server mode implementation
 * 
 * Handles persistent rendering with command-based protocol.
 * Avoids process startup and scene loading overhead on each frame.
 */

#include "server.hpp"
#include "protocol.hpp"
#include "renderer.hpp"
#include "pbr.hpp"
#include "json.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

using json = nlohmann::json;

// Forward declarations from main.cpp (will be refactored later)
extern Scene loadSceneFromJson(const std::string &path);
extern NodeTree parseNodeTree(const json &nodeTreeJson);

// Helper to load scene from JSON string (not file)
static Scene loadSceneFromJsonString(const std::string& jsonStr) {
    Scene scene;
    
    try {
        json j = json::parse(jsonStr);
        
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
            
            bool smoothShading = meshj.contains("smooth") && meshj["smooth"].get<bool>();
            
            // Triangles
            int triIndex = 0;
            if (meshj.contains("triangles")) {
                for (const auto &t : meshj["triangles"]) {
                    Triangle tri;
                    tri.i0 = t[0];
                    tri.i1 = t[1];
                    tri.i2 = t[2];
                    
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
                }
            }
            
            finalizeMeshBounds(m);
            scene.meshes.push_back(std::move(m));
        }
    } catch (const std::exception& e) {
        std::cerr << "[Server] JSON parse error: " << e.what() << "\n";
    }
    
    return scene;
}

RenderServer::RenderServer() {
    // Ensure binary mode for stdout
    #ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stdin), _O_BINARY);
    #endif
    
    response_ = std::make_unique<protocol::ResponseWriter>(std::cout);
    
    log("Server initialized");
    
    #ifdef _OPENMP
    log("OpenMP enabled with " + std::to_string(omp_get_max_threads()) + " threads");
    #else
    log("Single-threaded mode");
    #endif
}

RenderServer::~RenderServer() {
    log("Server shutting down");
}

void RenderServer::log(const std::string& message) {
    std::cerr << "[Server] " << message << "\n";
    std::cerr.flush();
}

bool RenderServer::readPayload(uint32_t size, std::vector<uint8_t>& payload) {
    payload.resize(size);
    if (size == 0) return true;
    
    // Read in chunks to avoid pipe buffer deadlock with large payloads
    const size_t CHUNK_SIZE = 32768;  // 32KB chunks
    size_t bytesRead = 0;
    
    while (bytesRead < size) {
        size_t toRead = std::min(CHUNK_SIZE, (size_t)(size - bytesRead));
        std::cin.read(reinterpret_cast<char*>(payload.data() + bytesRead), toRead);
        
        if (!std::cin.good()) {
            log("Read error at byte " + std::to_string(bytesRead) + " of " + std::to_string(size));
            return false;
        }
        
        bytesRead += toRead;
    }
    
    return true;
}

void RenderServer::run() {
    log("Entering server loop");
    
    while (true) {
        // Read command header
        protocol::CommandHeader header;
        if (!protocol::CommandHeader::read(std::cin, header)) {
            if (std::cin.eof()) {
                log("EOF received, exiting");
                break;
            }
            log("Failed to read command header");
            response_->writeError(protocol::StatusCode::ERROR_INVALID_COMMAND, "Invalid header");
            continue;
        }
        
        // Read payload
        std::vector<uint8_t> payload;
        if (header.payload_size > 0) {
            log("Reading payload: " + std::to_string(header.payload_size) + " bytes");
        }
        if (!readPayload(header.payload_size, payload)) {
            log("Failed to read payload");
            response_->writeError(protocol::StatusCode::ERROR_INVALID_COMMAND, "Failed to read payload");
            continue;
        }
        if (header.payload_size > 0) {
            log("Payload read complete");
        }
        
        // Dispatch command
        switch (header.type) {
            case protocol::CommandType::INIT:
                handleInit(payload);
                break;
                
            case protocol::CommandType::UPDATE_SCENE:
                handleUpdateScene(payload);
                break;
                
            case protocol::CommandType::UPDATE_CAMERA:
                handleUpdateCamera(payload);
                break;
                
            case protocol::CommandType::RENDER_TILE:
                handleRenderTile(payload);
                break;
                
            case protocol::CommandType::CANCEL:
                handleCancel();
                break;
                
            case protocol::CommandType::QUERY_CAPS:
                handleQueryCaps();
                break;
                
            case protocol::CommandType::SET_BACKEND:
                handleSetBackend(payload);
                break;
                
            case protocol::CommandType::SET_ALGORITHM:
                handleSetAlgorithm(payload);
                break;
                
            case protocol::CommandType::SHUTDOWN:
                log("Shutdown command received");
                response_->writeAck();
                return;
                
            default:
                log("Unknown command: " + std::to_string(static_cast<uint32_t>(header.type)));
                response_->writeError(protocol::StatusCode::ERROR_INVALID_COMMAND, "Unknown command");
                break;
        }
    }
}

void RenderServer::handleInit(const std::vector<uint8_t>& payload) {
    protocol::PayloadReader reader(payload);
    backend_ = reader.readString();
    algorithm_ = reader.readString();
    
    log("Init: backend=" + backend_ + ", algorithm=" + algorithm_);
    
    // Reset state
    scene_loaded_ = false;
    camera_set_ = false;
    scene_ = Scene();
    
    response_->writeAck();
}

void RenderServer::handleUpdateScene(const std::vector<uint8_t>& payload) {
    // Payload is JSON scene data
    std::string jsonStr(payload.begin(), payload.end());
    
    log("UpdateScene: " + std::to_string(payload.size()) + " bytes");
    
    scene_ = loadSceneFromJsonString(jsonStr);
    scene_loaded_ = !scene_.meshes.empty();
    
    if (scene_loaded_) {
        log("Scene loaded: " + std::to_string(scene_.meshes.size()) + " meshes");
        response_->writeAck();
    } else {
        log("Scene load failed");
        response_->writeError(protocol::StatusCode::ERROR_SCENE_NOT_LOADED, "Failed to load scene");
    }
}

void RenderServer::handleUpdateCamera(const std::vector<uint8_t>& payload) {
    if (payload.size() < 40) {
        response_->writeError(protocol::StatusCode::ERROR_INVALID_COMMAND, "Camera payload too small");
        return;
    }
    
    protocol::CameraParams params = protocol::CameraParams::fromBytes(payload.data());
    
    // Build camera structure
    camera_.pos = Vec3(params.pos_x, params.pos_y, params.pos_z);
    camera_.dir = Vec3(params.dir_x, params.dir_y, params.dir_z);
    camera_.dir.normalize();
    camera_.up = Vec3(params.up_x, params.up_y, params.up_z);
    camera_.up.normalize();
    camera_.fovDeg = params.fov;
    
    // Compute derived vectors
    camera_.forward = camera_.dir;
    camera_.right = Vec3::cross(camera_.forward, camera_.up);
    camera_.right.normalize();
    camera_.up = Vec3::cross(camera_.right, camera_.forward);
    camera_.up.normalize();
    
    camera_set_ = true;
    
    response_->writeAck();
}

void RenderServer::handleRenderTile(const std::vector<uint8_t>& payload) {
    if (!scene_loaded_) {
        response_->writeError(protocol::StatusCode::ERROR_SCENE_NOT_LOADED, "Scene not loaded");
        return;
    }
    
    if (!camera_set_) {
        response_->writeError(protocol::StatusCode::ERROR_SCENE_NOT_LOADED, "Camera not set");
        return;
    }
    
    if (payload.size() < 36) {
        response_->writeError(protocol::StatusCode::ERROR_INVALID_COMMAND, "RenderTile payload too small");
        return;
    }
    
    protocol::RenderTileParams params = protocol::RenderTileParams::fromBytes(payload.data());
    
    int tileX = params.tile_x;
    int tileY = params.tile_y;
    int tileW = params.tile_w;
    int tileH = params.tile_h;
    int fullW = params.full_w;
    int fullH = params.full_h;
    int samples = params.samples;
    int sampleOffset = params.sample_offset;
    int maxDepth = params.max_depth;
    
    // Update camera aspect
    camera_.aspect = (float)fullW / (float)fullH;
    
    float fovRad = camera_.fovDeg * (float)M_PI / 180.0f;
    float scale = std::tan(fovRad * 0.5f);
    
    // Reset cancel flag
    cancel_requested_ = false;
    
    // Build scene lights for NEE/MIS
    SceneLights sceneLights;
    sceneLights.buildFromScene(scene_);
    
    // Allocate output buffer
    int totalPixels = tileW * tileH;
    std::vector<float> pixelBuffer(totalPixels * 4, 0.0f);
    
    // Parallel rendering
    #pragma omp parallel for schedule(dynamic, 16) collapse(2)
    for (int py = 0; py < tileH; ++py) {
        for (int px = 0; px < tileW; ++px) {
            if (cancel_requested_) continue;
            
            int y = tileY + py;
            int x = tileX + px;
            int pixelIdx = py * tileW + px;
            
            float ndcX = (2.0f * (x + 0.5f) / fullW - 1.0f) * camera_.aspect;
            float ndcY = 1.0f - 2.0f * (y + 0.5f) / fullH;
            
            Vec3 worldDir = camera_.forward + camera_.right * (ndcX * scale) + camera_.up * (ndcY * scale);
            worldDir.normalize();
            Ray ray{camera_.pos, worldDir};
            
            Vec3 color{0, 0, 0};
            
            for (int s = 0; s < samples; ++s) {
                seed_random_xyz((uint32_t)x, (uint32_t)y, (uint32_t)(sampleOffset + s));
                
                Ray sampleRay = ray;
                if (samples > 1) {
                    float jitterX = (randf() - 0.5f) / fullW;
                    float jitterY = (randf() - 0.5f) / fullH;
                    Vec3 jitteredDir = camera_.forward + camera_.right * ((ndcX + jitterX) * scale) + camera_.up * ((ndcY + jitterY) * scale);
                    jitteredDir.normalize();
                    sampleRay.d = jitteredDir;
                }
                
                if (algorithm_ == "simple") {
                    color = color + traceSimple(scene_, sampleRay, maxDepth);
                } else if (algorithm_ == "mis") {
                    color = color + traceMIS(scene_, sceneLights, sampleRay, maxDepth);
                } else {
                    // Default: NEE
                    color = color + traceNEE(scene_, sceneLights, sampleRay, maxDepth);
                }
            }
            
            color.x = std::max(0.0f, color.x);
            color.y = std::max(0.0f, color.y);
            color.z = std::max(0.0f, color.z);
            
            pixelBuffer[pixelIdx * 4 + 0] = color.x;
            pixelBuffer[pixelIdx * 4 + 1] = color.y;
            pixelBuffer[pixelIdx * 4 + 2] = color.z;
            pixelBuffer[pixelIdx * 4 + 3] = 1.0f;
        }
    }
    
    if (cancel_requested_) {
        response_->writeError(protocol::StatusCode::ERROR_CANCELLED, "Render cancelled");
        return;
    }
    
    // Y-flip the output
    std::vector<float> flippedBuffer(totalPixels * 4);
    for (int y = 0; y < tileH; ++y) {
        int srcRow = y * tileW * 4;
        int dstRow = (tileH - 1 - y) * tileW * 4;
        std::copy(pixelBuffer.begin() + srcRow, 
                  pixelBuffer.begin() + srcRow + tileW * 4,
                  flippedBuffer.begin() + dstRow);
    }
    
    response_->writePixels(flippedBuffer, tileW, tileH);
}

void RenderServer::handleCancel() {
    log("Cancel requested");
    cancel_requested_ = true;
    response_->writeAck();
}

void RenderServer::handleQueryCaps() {
    std::vector<std::string> backends = {"cpu"};
    // TODO: Add "webgpu" when available
    
    std::vector<std::string> algorithms = {"simple", "nee", "mis"};
    
    response_->writeCapabilities(backends, algorithms);
}

void RenderServer::handleSetBackend(const std::vector<uint8_t>& payload) {
    backend_ = std::string(payload.begin(), payload.end());
    log("Backend set to: " + backend_);
    response_->writeAck();
}

void RenderServer::handleSetAlgorithm(const std::vector<uint8_t>& payload) {
    algorithm_ = std::string(payload.begin(), payload.end());
    log("Algorithm set to: " + algorithm_);
    response_->writeAck();
}
