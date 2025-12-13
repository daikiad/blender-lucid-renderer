/**
 * pybind_renderer.hpp - pybind11 用レンダラークラス
 * ===================================================
 * 
 * pybind11 を通じて Python から直接呼び出し可能なレンダラークラス。
 * 協調キャンセル（Cooperative Cancellation）パターンを実装し、
 * Python スレッドからのキャンセル要求に即座に応答できます。
 * 
 * 使用例 (Python):
 *   import diyrenderer
 *   renderer = diyrenderer.Renderer()
 *   renderer.load_scene_json(json_string)
 *   renderer.set_camera(pos, dir, up, fov)
 *   pixels = renderer.render_tile(0, 0, 800, 600, 800, 600, samples=16)
 *   renderer.cancel()  # 別スレッドから呼び出し可能
 * 
 * キャンセルの仕組み:
 *   - cancel() は std::atomic<bool> をセットするだけ（即座に返る）
 *   - render_tile() 内部のループでフラグをチェック
 *   - キャンセルされたら部分的な結果を返して早期終了
 */

#pragma once

#include <atomic>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>

#include "renderer.hpp"
#include "pbr.hpp"
#include "json.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

// node_evaluator.cpp で定義されている関数
extern NodeTree parseNodeTree(const nlohmann::json &nodeTreeJson);

/**
 * PyRenderer - Python バインディング用レンダラークラス
 * 
 * 特徴:
 * - 協調キャンセル対応
 * - GIL 解放状態で動作可能
 * - スレッドセーフなキャンセル機構
 */
class PyRenderer {
public:
    PyRenderer()
        : cancel_requested_(false)
        , scene_loaded_(false)
        , camera_set_(false)
        , algorithm_("nee")
    {
        #ifdef _OPENMP
        std::cerr << "[PyRenderer] OpenMP enabled with " 
                  << omp_get_max_threads() << " threads\n";
        #else
        std::cerr << "[PyRenderer] Single-threaded mode\n";
        #endif
    }

    // =========================================================================
    // キャンセル制御
    // =========================================================================

    /**
     * キャンセルを要求
     * 別スレッドから呼び出し可能。即座に返る。
     */
    void cancel() {
        cancel_requested_.store(true, std::memory_order_relaxed);
    }

    /**
     * キャンセル状態をチェック
     */
    bool is_cancelled() const {
        return cancel_requested_.load(std::memory_order_relaxed);
    }

    /**
     * キャンセルフラグをリセット（render_tile の開始時に自動で呼ばれる）
     */
    void reset_cancel() {
        cancel_requested_.store(false, std::memory_order_relaxed);
    }

    // =========================================================================
    // シーン管理
    // =========================================================================

    /**
     * JSON 文字列からシーンを読み込み
     */
    bool load_scene_json(const std::string& json_str) {
        try {
            scene_ = loadSceneFromJsonString(json_str);
            scene_loaded_ = true;
            std::cerr << "[PyRenderer] Scene loaded: " 
                      << scene_.meshes.size() << " meshes\n";
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[PyRenderer] Failed to load scene: " << e.what() << "\n";
            scene_loaded_ = false;
            return false;
        }
    }

    /**
     * カメラを設定
     */
    void set_camera(float pos_x, float pos_y, float pos_z,
                    float dir_x, float dir_y, float dir_z,
                    float up_x, float up_y, float up_z,
                    float fov_deg) {
        // Convert input floats to typed vectors
        camera_.pos = render::make_position(pos_x, pos_y, pos_z);
        
        // Normalize direction and up
        render::Vec3f dir_vec(dir_x, dir_y, dir_z);
        auto dir = render::make_direction_or_default(dir_vec, render::direction_from_unit_vector(render::Vec3f(0.0f, 0.0f, -1.0f)));
        camera_.dir = dir;
        
        render::Vec3f up_vec(up_x, up_y, up_z);
        auto up = render::make_direction_or_default(up_vec, render::direction_from_unit_vector(render::Vec3f(0.0f, 1.0f, 0.0f)));
        
        camera_.fov = render::degrees(fov_deg);
        
        // Calculate right and forward using cross product
        auto forward = dir;
        auto right_vec = render::cross(forward.vec(), up.vec());
        auto right = render::make_direction_or_default(right_vec, render::direction_from_unit_vector(render::Vec3f(1.0f, 0.0f, 0.0f)));
        auto up_corrected_vec = render::cross(right.vec(), forward.vec());
        auto up_corrected = render::make_direction_or_default(up_corrected_vec, render::direction_from_unit_vector(render::Vec3f(0.0f, 1.0f, 0.0f)));
        
        camera_.forward = forward;
        camera_.right = right;
        camera_.up = up_corrected;
        
        camera_set_ = true;
    }

    /**
     * アルゴリズムを設定 ("simple", "nee", "mis")
     */
    void set_algorithm(const std::string& algo) {
        algorithm_ = algo;
    }

    // =========================================================================
    // レンダリング
    // =========================================================================

    /**
     * タイルをレンダリング
     * 
     * @param tile_x, tile_y タイルの開始位置
     * @param tile_w, tile_h タイルのサイズ
     * @param full_w, full_h 画像全体のサイズ
     * @param samples サンプル数
     * @param sample_offset RNG シード用オフセット
     * @param max_depth 最大バウンス数
     * @return RGBA float 配列（tile_w * tile_h * 4 要素）
     *         キャンセル時は部分的な結果を返す
     */
    std::vector<float> render_tile(
        int tile_x, int tile_y,
        int tile_w, int tile_h,
        int full_w, int full_h,
        int samples,
        int sample_offset,
        int max_depth)
    {
        // キャンセルフラグをリセット
        reset_cancel();
        
        // 出力バッファを確保
        std::vector<float> pixels(tile_w * tile_h * 4, 0.0f);
        
        // シーンまたはカメラが未設定の場合
        if (!scene_loaded_ || !camera_set_) {
            std::cerr << "[PyRenderer] Scene or camera not set\n";
            return pixels;
        }
        
        // アスペクト比を更新
        camera_.aspect = static_cast<float>(full_w) / static_cast<float>(full_h);
        
        // FOV からスケール係数を計算
        float fov_rad = camera_.fovRad();
        float scale = std::tan(fov_rad * 0.5f);
        
        // シーンライトを構築（NEE/MIS 用）
        SceneLights scene_lights;
        scene_lights.buildFromScene(scene_);
        
        // マイクロタイルサイズ（キャンセルチェック頻度を上げる）
        const int micro_tile = 8;
        
        // マイクロタイル単位でレンダリング
        for (int my = 0; my < tile_h; my += micro_tile) {
            for (int mx = 0; mx < tile_w; mx += micro_tile) {
                
                // ★ キャンセルチェックポイント ★
                if (cancel_requested_.load(std::memory_order_relaxed)) {
                    std::cerr << "[PyRenderer] Render cancelled at micro-tile ("
                              << mx << ", " << my << ")\n";
                    return pixels;  // 部分的な結果を返す
                }
                
                int mh = std::min(micro_tile, tile_h - my);
                int mw = std::min(micro_tile, tile_w - mx);
                
                // このマイクロタイル内を OpenMP で並列化
                #pragma omp parallel for collapse(2) schedule(dynamic, 1)
                for (int j = 0; j < mh; ++j) {
                    for (int i = 0; i < mw; ++i) {
                        // OpenMP ループ内でもキャンセルをチェック（continue でスキップ）
                        if (cancel_requested_.load(std::memory_order_relaxed)) {
                            continue;
                        }
                        
                        int px = mx + i;
                        int py = my + j;
                        int global_x = tile_x + px;
                        int global_y = tile_y + py;
                        int pixel_idx = py * tile_w + px;
                        
                        // NDC 座標を計算
                        float ndc_x = (2.0f * (global_x + 0.5f) / full_w - 1.0f) * camera_.aspect;
                        float ndc_y = 1.0f - 2.0f * (global_y + 0.5f) / full_h;
                        
                        // レイ方向を計算 (use typed vectors directly)
                        render::Vec3f world_dir_vec(
                            camera_.forward.vec().x + camera_.right.vec().x * (ndc_x * scale) + camera_.up.vec().x * (ndc_y * scale),
                            camera_.forward.vec().y + camera_.right.vec().y * (ndc_x * scale) + camera_.up.vec().y * (ndc_y * scale),
                            camera_.forward.vec().z + camera_.right.vec().z * (ndc_x * scale) + camera_.up.vec().z * (ndc_y * scale)
                        );
                        auto world_dir = render::make_direction_or_default(world_dir_vec, camera_.forward);
                        Ray ray(camera_.pos, world_dir);
                        
                        render::ColorRGB color(0, 0, 0);
                        
                        // 複数サンプルの平均
                        for (int s = 0; s < samples; ++s) {
                            seed_random_xyz(
                                static_cast<uint32_t>(global_x),
                                static_cast<uint32_t>(global_y),
                                static_cast<uint32_t>(sample_offset + s)
                            );
                            
                            Ray sample_ray = ray;
                            
                            // アンチエイリアシング用ジッター
                            if (samples > 1) {
                                float jitter_x = (randf() - 0.5f) / full_w;
                                float jitter_y = (randf() - 0.5f) / full_h;
                                render::Vec3f jittered_dir_vec(
                                    camera_.forward.vec().x + camera_.right.vec().x * ((ndc_x + jitter_x) * scale) + camera_.up.vec().x * ((ndc_y + jitter_y) * scale),
                                    camera_.forward.vec().y + camera_.right.vec().y * ((ndc_x + jitter_x) * scale) + camera_.up.vec().y * ((ndc_y + jitter_y) * scale),
                                    camera_.forward.vec().z + camera_.right.vec().z * ((ndc_x + jitter_x) * scale) + camera_.up.vec().z * ((ndc_y + jitter_y) * scale)
                                );
                                auto jittered_dir = render::make_direction_or_default(jittered_dir_vec, camera_.forward);
                                sample_ray = Ray(sample_ray.origin, jittered_dir);
                            }
                            
                            // アルゴリズムに応じてパストレーシング
                            if (algorithm_ == "simple") {
                                color = color + traceSimple(scene_, sample_ray, max_depth);
                            } else if (algorithm_ == "mis") {
                                color = color + traceMIS(scene_, scene_lights, sample_ray, max_depth);
                            } else {
                                color = color + traceNEE(scene_, scene_lights, sample_ray, max_depth);
                            }
                        }
                        
                        // 負の値をクランプ
                        float r = std::max(0.0f, color.r);
                        float g = std::max(0.0f, color.g);
                        float b = std::max(0.0f, color.b);
                        
                        // RGBA としてバッファに格納
                        pixels[pixel_idx * 4 + 0] = r;
                        pixels[pixel_idx * 4 + 1] = g;
                        pixels[pixel_idx * 4 + 2] = b;
                        pixels[pixel_idx * 4 + 3] = 1.0f;
                    }
                }
            }
        }
        
        // Y軸反転（Blender は左下が原点）
        std::vector<float> flipped(pixels.size());
        for (int y = 0; y < tile_h; ++y) {
            int src_row = y * tile_w * 4;
            int dst_row = (tile_h - 1 - y) * tile_w * 4;
            std::copy(pixels.begin() + src_row,
                      pixels.begin() + src_row + tile_w * 4,
                      flipped.begin() + dst_row);
        }
        
        return flipped;
    }

    // =========================================================================
    // デバッグレンダリング
    // =========================================================================

    /**
     * デバッグモードでレンダリング
     * @param mode "normal", "albedo", "emission"
     */
    std::vector<float> render_debug(
        int tile_x, int tile_y,
        int tile_w, int tile_h,
        int full_w, int full_h,
        const std::string& mode)
    {
        reset_cancel();
        std::vector<float> pixels(tile_w * tile_h * 4, 0.0f);
        
        if (!scene_loaded_ || !camera_set_) {
            return pixels;
        }
        
        camera_.aspect = static_cast<float>(full_w) / static_cast<float>(full_h);
        float fov_rad = camera_.fovRad();
        float scale = std::tan(fov_rad * 0.5f);
        
        #pragma omp parallel for collapse(2) schedule(dynamic, 16)
        for (int py = 0; py < tile_h; ++py) {
            for (int px = 0; px < tile_w; ++px) {
                if (cancel_requested_.load(std::memory_order_relaxed)) {
                    continue;
                }
                
                int global_x = tile_x + px;
                int global_y = tile_y + py;
                int pixel_idx = py * tile_w + px;
                
                float ndc_x = (2.0f * (global_x + 0.5f) / full_w - 1.0f) * camera_.aspect;
                float ndc_y = 1.0f - 2.0f * (global_y + 0.5f) / full_h;
                
                // Calculate ray direction using typed vectors
                render::Vec3f world_dir_vec(
                    camera_.forward.vec().x + camera_.right.vec().x * (ndc_x * scale) + camera_.up.vec().x * (ndc_y * scale),
                    camera_.forward.vec().y + camera_.right.vec().y * (ndc_x * scale) + camera_.up.vec().y * (ndc_y * scale),
                    camera_.forward.vec().z + camera_.right.vec().z * (ndc_x * scale) + camera_.up.vec().z * (ndc_y * scale)
                );
                auto world_dir = render::make_direction_or_default(world_dir_vec, camera_.forward);
                Ray ray(camera_.pos, world_dir);
                
                render::ColorRGB color(0, 0, 0);
                if (mode == "normal") {
                    color = traceNormal(scene_, ray);
                    // -1..1 を 0..1 にマッピング
                    color = color * 0.5f + render::ColorRGB(0.5f, 0.5f, 0.5f);
                } else if (mode == "albedo") {
                    color = traceAlbedo(scene_, ray);
                } else if (mode == "emission") {
                    color = traceEmission(scene_, ray);
                } else {
                    color = render::ColorRGB(1, 0, 1);  // マゼンタ（エラー表示）
                }
                
                // Y反転を考慮したインデックス
                int flipped_y = tile_h - 1 - py;
                int flipped_idx = flipped_y * tile_w + px;
                
                pixels[flipped_idx * 4 + 0] = color.r;
                pixels[flipped_idx * 4 + 1] = color.g;
                pixels[flipped_idx * 4 + 2] = color.b;
                pixels[flipped_idx * 4 + 3] = 1.0f;
            }
        }
        
        return pixels;
    }

    // =========================================================================
    // 情報取得
    // =========================================================================

    bool is_scene_loaded() const { return scene_loaded_; }
    bool is_camera_set() const { return camera_set_; }
    std::string get_algorithm() const { return algorithm_; }
    
    int get_mesh_count() const {
        return scene_loaded_ ? static_cast<int>(scene_.meshes.size()) : 0;
    }

private:
    std::atomic<bool> cancel_requested_;
    Scene scene_;
    Camera camera_;
    bool scene_loaded_;
    bool camera_set_;
    std::string algorithm_;
    
    // JSON からシーンを読み込むヘルパー（server.cpp と共通化）
    Scene loadSceneFromJsonString(const std::string& json_str) {
        using json = nlohmann::json;
        Scene scene;
        
        json j = json::parse(json_str);
        
        if (!j.contains("meshes")) return scene;
        
        for (const auto& mesh_j : j["meshes"]) {
            Mesh m;
            
            // Vertices
            if (mesh_j.contains("vertices")) {
                for (const auto& v : mesh_j["vertices"]) {
                    m.vertices.push_back(render::make_position(
                        v[0].get<float>(),
                        v[1].get<float>(),
                        v[2].get<float>()
                    ));
                }
            }
            
            // Per-triangle vertex normals
            std::vector<std::array<render::Normal, 3>> triangle_normals;
            bool has_triangle_normals = false;
            if (mesh_j.contains("triangle_normals")) {
                has_triangle_normals = true;
                for (const auto& tn : mesh_j["triangle_normals"]) {
                    std::array<render::Normal, 3> normals;
                    auto default_up = render::normal_from_unit_vector(render::Vec3f(0.0f, 0.0f, 1.0f));
                    normals[0] = render::make_normal_or_default(render::Vec3f(static_cast<float>(tn[0][0]), static_cast<float>(tn[0][1]), static_cast<float>(tn[0][2])), default_up);
                    normals[1] = render::make_normal_or_default(render::Vec3f(static_cast<float>(tn[1][0]), static_cast<float>(tn[1][1]), static_cast<float>(tn[1][2])), default_up);
                    normals[2] = render::make_normal_or_default(render::Vec3f(static_cast<float>(tn[2][0]), static_cast<float>(tn[2][1]), static_cast<float>(tn[2][2])), default_up);
                    triangle_normals.push_back(normals);
                }
            }
            
            // Per-triangle UV coordinates
            std::vector<std::array<render::Vec2f, 3>> triangle_uvs;
            bool has_triangle_uvs = false;
            if (mesh_j.contains("triangle_uvs")) {
                has_triangle_uvs = true;
                for (const auto& tuv : mesh_j["triangle_uvs"]) {
                    std::array<render::Vec2f, 3> uvs;
                    uvs[0] = render::Vec2f(tuv[0][0], tuv[0][1]);
                    uvs[1] = render::Vec2f(tuv[1][0], tuv[1][1]);
                    uvs[2] = render::Vec2f(tuv[2][0], tuv[2][1]);
                    triangle_uvs.push_back(uvs);
                }
            }
            
            bool smooth_shading = mesh_j.contains("smooth") && mesh_j["smooth"].get<bool>();
            
            // Triangles
            int tri_index = 0;
            if (mesh_j.contains("triangles")) {
                for (const auto& t : mesh_j["triangles"]) {
                    Triangle tri;
                    tri.i0 = t[0];
                    tri.i1 = t[1];
                    tri.i2 = t[2];
                    
                    if (has_triangle_normals && static_cast<size_t>(tri_index) < triangle_normals.size()) {
                        tri.n0 = triangle_normals[tri_index][0];
                        tri.n1 = triangle_normals[tri_index][1];
                        tri.n2 = triangle_normals[tri_index][2];
                        tri.smooth = smooth_shading;
                    } else {
                        tri.smooth = false;
                    }
                    
                    // Parse UV coordinates
                    if (has_triangle_uvs && static_cast<size_t>(tri_index) < triangle_uvs.size()) {
                        tri.uv0 = triangle_uvs[tri_index][0];
                        tri.uv1 = triangle_uvs[tri_index][1];
                        tri.uv2 = triangle_uvs[tri_index][2];
                        tri.hasUV = true;
                    } else {
                        tri.hasUV = false;
                    }
                    
                    m.triangles.push_back(tri);
                    tri_index++;
                }
            }
            
            // Material
            if (mesh_j.contains("material")) {
                const auto& mat = mesh_j["material"];
                render::ColorRGB albedo(0.8f, 0.8f, 0.8f);
                float metallic = 0.0f, roughness = 0.5f;
                render::RadianceRGB emission = render::make_radiance_rgb(0.0f, 0.0f, 0.0f);
                float transmission = 0.0f, ior = 1.45f;
                
                if (mat.contains("base_color")) {
                    albedo = render::ColorRGB(
                        mat["base_color"][0].get<float>(), 
                        mat["base_color"][1].get<float>(), 
                        mat["base_color"][2].get<float>()
                    );
                }
                if (mat.contains("metallic")) metallic = mat["metallic"];
                if (mat.contains("roughness")) roughness = mat["roughness"];
                if (mat.contains("emission")) {
                    emission = render::make_radiance_rgb(
                        mat["emission"][0].get<float>(), 
                        mat["emission"][1].get<float>(), 
                        mat["emission"][2].get<float>()
                    );
                }
                if (mat.contains("transmission")) transmission = mat["transmission"];
                if (mat.contains("ior")) ior = mat["ior"];
                
                m.material = Material(
                    albedo, 
                    metallic, 
                    roughness, 
                    emission
                );
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
        
        // Parse native Blender lights (exported as "native_lights" in JSON)
        if (j.contains("native_lights") && j["native_lights"].is_array()) {
            parseNativeLights(j["native_lights"], scene);
        }
        // Also check for "lights" for backward compatibility
        else if (j.contains("lights") && j["lights"].is_array()) {
            parseNativeLights(j["lights"], scene);
        }
        
        // Parse environment settings
        if (j.contains("environment") && j["environment"].is_object()) {
            const auto& envJson = j["environment"];
            
            if (envJson.contains("color") && envJson["color"].is_array()) {
                scene.environment.color = render::ColorRGB(
                    envJson["color"][0].get<float>(),
                    envJson["color"][1].get<float>(),
                    envJson["color"][2].get<float>()
                );
            }
            
            if (envJson.contains("strength")) {
                scene.environment.strength = envJson["strength"].get<float>();
            }
            
            std::cerr << "[Environment] color=(" << scene.environment.color.r << ", "
                      << scene.environment.color.g << ", " << scene.environment.color.b 
                      << "), strength=" << scene.environment.strength << std::endl;
        }
        
        return scene;
    }
    
    static void parseNativeLights(const nlohmann::json& lightsJson, Scene& scene) {
        for (const auto& lightJson : lightsJson) {
            Light light;
            
            std::string typeStr = lightJson.value("type", "POINT");
            
            // Position
            if (lightJson.contains("position")) {
                light.position = render::make_position(
                    lightJson["position"][0].get<float>(),
                    lightJson["position"][1].get<float>(),
                    lightJson["position"][2].get<float>()
                );
            }
            
            // Direction (normal)
            if (lightJson.contains("direction")) {
                auto default_down = render::normal_from_unit_vector(render::Vec3f(0.0f, 0.0f, -1.0f));
                light.normal = render::make_normal_or_default(render::Vec3f(
                    lightJson["direction"][0].get<float>(),
                    lightJson["direction"][1].get<float>(),
                    lightJson["direction"][2].get<float>()
                ), default_down);
            }
            
            // Color and energy
            render::ColorRGB color(1.0f, 1.0f, 1.0f);
            if (lightJson.contains("color")) {
                color = render::ColorRGB(
                    lightJson["color"][0].get<float>(),
                    lightJson["color"][1].get<float>(),
                    lightJson["color"][2].get<float>()
                );
            }
            float energy = lightJson.value("energy", 1.0f);
            
            // =====================================================
            // Blender Light Units (Physical, since 2.8+):
            // - Point/Spot: Watts (W) - total radiant flux
            // - Sun: W/m² - irradiance (directional)
            // - Area: Watts (W) - total radiant flux over the surface
            //
            // Our renderer needs Radiance (W/m²/sr) for integration.
            // Conversion depends on light type and PDF.
            // =====================================================
            
            light.energy = energy * mp_units::si::watt;
            
            // Set light type and type-specific properties
            if (typeStr == "POINT") {
                light.type = LightType::POINT;
                float radius = lightJson.value("radius", 0.0f);
                light.radius = radius * mp_units::si::metre;
                
                // For spherical area light (soft shadows)
                float area = 4.0f * M_PI * radius * radius;
                if (area < EPSILON) area = 1.0f;
                light.area = area * mp_units::square(mp_units::si::metre);
                
                // Point light: I = Power / (4π) [W/sr]
                // In NEE, contribution = I / r² = Power / (4π * r²)
                // We store Power / (4π) as emission, apply 1/r² in integration
                // For soft point (sphere), emission is radiance: Power / (π * area)
                if (radius > EPSILON) {
                    // Spherical light: Lambertian emitter
                    // L = Power / (π * surfaceArea) where surfaceArea = 4πr²
                    // L = Power / (4π²r²)
                    light.emission = render::make_radiance_rgb(
                        color.r * (energy / (M_PI * area)),
                        color.g * (energy / (M_PI * area)),
                        color.b * (energy / (M_PI * area))
                    );
                } else {
                    // True point light: store intensity I = Power / (4π)
                    // 1/r² applied during sampling
                    float factor = energy / (4.0f * M_PI);
                    light.emission = render::make_radiance_rgb(
                        color.r * factor,
                        color.g * factor,
                        color.b * factor
                    );
                }
                
            } else if (typeStr == "SUN") {
                light.type = LightType::SUN;
                light.area = 1.0f * mp_units::square(mp_units::si::metre);  // Sun is directional, area is symbolic
                
                // Sun: energy is already irradiance (W/m²)
                // Use directly as we treat it as parallel rays
                light.emission = render::make_radiance_rgb(
                    color.r * energy,
                    color.g * energy,
                    color.b * energy
                );
                
            } else if (typeStr == "SPOT") {
                light.type = LightType::SPOT;
                float radius = lightJson.value("radius", 0.0f);
                light.radius = radius * mp_units::si::metre;
                float spotAngle = lightJson.value("spot_size", static_cast<float>(M_PI / 4.0f));
                light.spotAngle = spotAngle * mp_units::si::radian;
                light.spotBlend = lightJson.value("spot_blend", 0.0f);
                light.area = 1.0f * mp_units::square(mp_units::si::metre);
                
                // Spot light: same as point but concentrated in cone
                // Blender's spot energy is total power, distributed in cone
                // Solid angle of cone = 2π(1 - cos(θ/2))
                float halfAngle = spotAngle * 0.5f;
                float solidAngle = 2.0f * M_PI * (1.0f - std::cos(halfAngle));
                
                // Intensity in the cone direction = Power / solidAngle
                float factor = energy / solidAngle;
                light.emission = render::make_radiance_rgb(
                    color.r * factor,
                    color.g * factor,
                    color.b * factor
                );
                
            } else if (typeStr == "AREA") {
                light.type = LightType::AREA;
                float sizeX = lightJson.value("size", 1.0f);
                float sizeY = lightJson.value("size_y", sizeX);
                light.sizeX = sizeX * mp_units::si::metre;
                light.sizeY = sizeY * mp_units::si::metre;
                
                float area;
                
                // Parse shape
                std::string shapeStr = lightJson.value("shape", "SQUARE");
                if (shapeStr == "DISK") {
                    light.shape = AreaLightShape::DISK;
                    // Disk area = π * r², where r = size/2
                    float radius = sizeX * 0.5f;
                    area = M_PI * radius * radius;
                } else if (shapeStr == "ELLIPSE") {
                    light.shape = AreaLightShape::ELLIPSE;
                    // Ellipse area = π * a * b, where a = sizeX/2, b = sizeY/2
                    float radiusX = sizeX * 0.5f;
                    float radiusY = sizeY * 0.5f;
                    area = M_PI * radiusX * radiusY;
                } else if (shapeStr == "RECTANGLE") {
                    light.shape = AreaLightShape::RECTANGLE;
                    area = sizeX * sizeY;
                } else {
                    // SQUARE or default
                    light.shape = AreaLightShape::SQUARE;
                    area = sizeX * sizeY;
                }
                
                light.area = area * mp_units::square(mp_units::si::metre);
                
                // Get orientation vectors
                if (lightJson.contains("right")) {
                    auto right = render::Vec3f(
                        lightJson["right"][0].get<float>(),
                        lightJson["right"][1].get<float>(),
                        lightJson["right"][2].get<float>()
                    );
                    light.right = render::make_direction_or_default(right, render::direction_from_unit_vector(render::Vec3f(1.0f, 0.0f, 0.0f)));
                }
                if (lightJson.contains("up")) {
                    auto up = render::Vec3f(
                        lightJson["up"][0].get<float>(),
                        lightJson["up"][1].get<float>(),
                        lightJson["up"][2].get<float>()
                    );
                    light.up = render::make_direction_or_default(up, render::direction_from_unit_vector(render::Vec3f(0.0f, 1.0f, 0.0f)));
                }
                
                // Area light: Lambertian emitter
                // Radiance L = Power / (π * Area) [W/m²/sr]
                // The π factor accounts for Lambertian cosine distribution
                float factor = energy / (M_PI * area);
                light.emission = render::make_radiance_rgb(
                    color.r * factor,
                    color.g * factor,
                    color.b * factor
                );
                
                std::cerr << "[SceneParser] Area light shape=" << shapeStr 
                          << " area=" << area << std::endl;            }
            
            light.meshIndex = -1;
            light.triangleIndex = -1;
            
            // Add to scene's native lights collection
            scene.nativeLights.push_back(light);
            
            std::cerr << "[SceneParser] Added " << typeStr << " light: "
                      << lightJson.value("name", "unnamed") 
                      << " energy=" << energy << std::endl;
        }
    }
};