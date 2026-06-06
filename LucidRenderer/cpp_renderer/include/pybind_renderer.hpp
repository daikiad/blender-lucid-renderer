/**
 * pybind_renderer.hpp - pybind11 用レンダラークラス
 * ===================================================
 * 
 * pybind11 を通じて Python から直接呼び出し可能なレンダラークラス。
 * 協調キャンセル（Cooperative Cancellation）パターンを実装し、
 * Python スレッドからのキャンセル要求に即座に応答できます。
 * 
 * 使用例 (Python):
 *   import lucidrenderer
 *   renderer = lucidrenderer.Renderer()
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
#include "diagnostics/diagnostic_film.hpp"
#include "diagnostics/diagnostic_export.hpp"
#include "diagnostics/diagnostic_integrator.hpp"
#include <nlohmann/json.hpp>

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
        
        // 診断フィルムを初期化（診断が有効な場合）
        if (diagnostics_enabled_) {
            diagnostic_film_ = std::make_unique<render::diagnostics::DiagnosticFilm>(tile_w, tile_h, diagnostic_config_);
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
                        
                        render::RadianceRGB radiance(
                            render::Radiance::zero(),
                            render::Radiance::zero(),
                            render::Radiance::zero()
                        );
                        
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
                            
                            // アルゴリズムに応じてパストレーシング（診断機能付きまたは通常）
                            if (diagnostics_enabled_ && diagnostic_film_) {
                                render::diagnostics::PathDiagnosticRecorder recorder;
                                recorder.begin_path();
                                
                                render::RadianceRGB sample_radiance;
                                if (algorithm_ == "simple") {
                                    sample_radiance = traceSimpleWithDiagnostics(scene_, sample_ray, max_depth, recorder);
                                } else if (algorithm_ == "mis") {
                                    sample_radiance = traceMISWithDiagnostics(scene_, scene_lights, sample_ray, max_depth, recorder);
                                } else {
                                    sample_radiance = traceNEEWithDiagnostics(scene_, scene_lights, sample_ray, max_depth, recorder);
                                }
                                
                                radiance = radiance + sample_radiance;
                                
                                // Plan E: Record all completed paths (paths that reached light sources)
                                // Each path has its own MIS-weighted contribution, so no double counting
                                for (const auto& path : recorder.get_completed_paths()) {
                                    diagnostic_film_->record_path(px, py, path);
                                }
                            } else {
                                if (algorithm_ == "simple") {
                                    radiance = radiance + traceSimple(scene_, sample_ray, max_depth);
                                } else if (algorithm_ == "mis") {
                                    radiance = radiance + traceMIS(scene_, scene_lights, sample_ray, max_depth);
                                } else {
                                    radiance = radiance + traceNEE(scene_, scene_lights, sample_ray, max_depth);
                                }
                            }
                        }
                        
                        // カメラ感度を適用（平均化は Python 側で行う）
                        // 累積値をサンプル数で割らず、そのまま返す
                        // Python 側で total_samples で正規化する
                        render::PixelRGB color = render::apply_camera_sensitivity(radiance, camera_.sensitivity());

                        // 負の値をクランプして float に抽出
                        color = render::attenuation_clamp_min_zero(color);
                        auto [r, g, b] = render::color_to_floats(color);
                        
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
                
                render::AttenuationRGB color = render::zero_attenuation_rgb();
                if (mode == "normal") {
                    color = traceNormal(scene_, ray);
                    // -1..1 を 0..1 にマッピング
                    color = color * 0.5f + render::make_attenuation_rgb(0.5f, 0.5f, 0.5f);
                } else if (mode == "albedo") {
                    color = traceAlbedo(scene_, ray);
                } else if (mode == "emission") {
                    color = traceEmission(scene_, ray);
                } else {
                    color = render::make_attenuation_rgb(1.0f, 0.0f, 1.0f);  // マゼンタ（エラー表示）
                }
                
                // Y反転を考慮したインデックス
                int flipped_y = tile_h - 1 - py;
                int flipped_idx = flipped_y * tile_w + px;
                
                auto [r, g, b] = render::color_to_floats(color);
                pixels[flipped_idx * 4 + 0] = r;
                pixels[flipped_idx * 4 + 1] = g;
                pixels[flipped_idx * 4 + 2] = b;
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

    // =========================================================================
    // 診断機能 (Path Variance Analyzer)
    // =========================================================================

    /**
     * 診断機能を有効化
     * @param config 記録設定
     */
    void enable_diagnostics(const render::diagnostics::PathRecordingConfig& config) {
        diagnostic_config_ = config;
        diagnostics_enabled_ = true;
        
        // DiagnosticFilm は render_tile 内で初期化する
        // (サイズが分かってから)
        diagnostic_film_.reset();
    }
    
    /**
     * 診断機能を無効化
     */
    void disable_diagnostics() {
        diagnostics_enabled_ = false;
        diagnostic_film_.reset();
    }
    
    /**
     * 診断が有効かどうか
     */
    bool is_diagnostics_enabled() const {
        return diagnostics_enabled_;
    }
    
    /**
     * 診断データをクリア
     */
    void clear_diagnostics() {
        if (diagnostic_film_) {
            diagnostic_film_->clear();
        }
    }
    
    /**
     * 診断統計を取得
     */
    render::diagnostics::GlobalDiagnosticStats get_diagnostic_stats() const {
        if (!diagnostic_film_) {
            return render::diagnostics::GlobalDiagnosticStats{};
        }
        render::diagnostics::DiagnosticExporter exporter(*diagnostic_film_);
        return exporter.get_global_stats();
    }
    
    /**
     * ピクセルごとの分散マップを取得
     * @return width * height の float 配列（輝度分散）
     */
    std::vector<float> get_diagnostic_variance_map() const {
        if (!diagnostic_film_) {
            return {};
        }
        
        size_t w = diagnostic_film_->width();
        size_t h = diagnostic_film_->height();
        std::vector<float> variance_map(w * h, 0.0f);
        
        for (size_t y = 0; y < h; ++y) {
            for (size_t x = 0; x < w; ++x) {
                const auto* pixel_data = diagnostic_film_->pixel_data(x, y);
                if (pixel_data) {
                    variance_map[y * w + x] = pixel_data->total_variance();
                }
            }
        }
        
        return variance_map;
    }
    
    /**
     * 診断データをJSON形式でエクスポート
     */
    std::string export_diagnostic_json() const {
        if (!diagnostic_film_) {
            return "{}";
        }
        render::diagnostics::DiagnosticExporter exporter(*diagnostic_film_);
        return exporter.export_metadata_json();
    }
    
    /**
     * 上位分散パスグループを取得
     */
    std::vector<render::diagnostics::ExportedGroupInfo> get_top_variance_groups(size_t count) const {
        if (!diagnostic_film_) {
            return {};
        }
        render::diagnostics::DiagnosticExporter exporter(*diagnostic_film_);
        return exporter.get_top_variance_groups(count);
    }
    
    /**
     * ピクセル単位の診断データを取得
     * @param x ピクセルX座標
     * @param y ピクセルY座標
     * @return ピクセル診断情報 (variance, sample_count, top_groups)
     */
    struct PixelDiagnosticInfo {
        bool valid = false;
        size_t sample_count = 0;
        float variance = 0.0f;
        size_t group_count = 0;
        size_t outlier_count = 0;
        std::array<float, 3> mean_rgb = {0.0f, 0.0f, 0.0f};
        std::vector<render::diagnostics::ExportedGroupInfo> top_groups;
    };
    
    /**
     * Pixel diagnostic info with top groups sorted by variance or mean
     */
    enum class TopGroupSortBy {
        Variance,  // Sort by variance (high variance = noisy)
        Mean       // Sort by mean contribution (high mean = bright)
    };
    
    PixelDiagnosticInfo get_pixel_diagnostic(size_t x, size_t y, 
                                              TopGroupSortBy sort_by = TopGroupSortBy::Variance) const {
        PixelDiagnosticInfo info;
        
        if (!diagnostic_film_) {
            return info;
        }
        
        // Note: We need non-const access to sort
        // DiagnosticFilm::pixel_data returns const pointer, but we need to sort
        // Using const_cast is safe here because we're just sorting the internal array
        auto* pixel_data = const_cast<render::diagnostics::DiagnosticFilm::PixelData*>(
            diagnostic_film_->pixel_data(x, y));
        if (!pixel_data) {
            return info;
        }
        
        // Sort groups by requested criteria
        if (sort_by == TopGroupSortBy::Mean) {
            pixel_data->sort_by_mean();
        } else {
            pixel_data->sort_by_variance();
        }
        
        info.valid = true;
        info.sample_count = pixel_data->total_samples();
        info.variance = pixel_data->total_variance();
        info.group_count = pixel_data->group_count();
        info.outlier_count = pixel_data->outlier_count();
        
        auto mean = pixel_data->total_mean();
        info.mean_rgb[0] = mean.r;
        info.mean_rgb[1] = mean.g;
        info.mean_rgb[2] = mean.b;
        
        // トップグループの情報を変換（全グループを取得、最大16）
        for (size_t i = 0; i < pixel_data->group_count(); ++i) {
            const auto* group = pixel_data->top_group(i);
            if (!group) continue;
            
            render::diagnostics::ExportedGroupInfo ginfo;
            ginfo.pixel_x = x;
            ginfo.pixel_y = y;
            ginfo.signature = group->signature();
            ginfo.signature_heckbert = group->signature_heckbert();
            ginfo.object_ids = group->object_ids();
            ginfo.sample_count = group->stats.count;
            ginfo.mean_luminance = group->mean_contribution();
            ginfo.variance_luminance = group->total_variance();
            ginfo.mean_rgb[0] = group->stats.mean.r;
            ginfo.mean_rgb[1] = group->stats.mean.g;
            ginfo.mean_rgb[2] = group->stats.mean.b;
            ginfo.variance_rgb[0] = group->stats.variance.r;
            ginfo.variance_rgb[1] = group->stats.variance.g;
            ginfo.variance_rgb[2] = group->stats.variance.b;
            ginfo.depth = group->depth;
            ginfo.coarse_type = group->coarse_type;
            ginfo.coarse_type_name = render::diagnostics::coarse_type_name(group->coarse_type);
            
            // Plan E: Export sampling strategy
            ginfo.strategy = group->strategy;
            ginfo.strategy_name = render::diagnostics::sampling_strategy_name(group->strategy);
            
            // オブジェクト名パスを生成（light_typeを含む）
            ginfo.object_path = object_path_string(ginfo.object_ids, group->light_type);
            
            // Export path geometry for visualization
            // First position is camera origin
            ginfo.positions.reserve(group->depth + 1);
            ginfo.normals.reserve(group->depth);
            
            // Add camera position as first point
            auto cam_disp = render::displacement_from_origin(camera_.pos);
            auto cam_pos = cam_disp.numerical_value_in(render::si::metre);
            ginfo.positions.push_back(std::array<float, 3>{cam_pos.x, cam_pos.y, cam_pos.z});
            
            // Add vertex positions and normals
            for (size_t vi = 0; vi < group->depth; ++vi) {
                ginfo.positions.push_back(std::array<float, 3>{
                    group->positions[vi].x,
                    group->positions[vi].y,
                    group->positions[vi].z
                });
                ginfo.normals.push_back(std::array<float, 3>{
                    group->normals[vi].x,
                    group->normals[vi].y,
                    group->normals[vi].z
                });
            }
            
            info.top_groups.push_back(ginfo);
        }
        
        return info;
    }

private:
    std::atomic<bool> cancel_requested_;
    Scene scene_;
    Camera camera_;
    bool scene_loaded_;
    bool camera_set_;
    std::string algorithm_;
    
    // 診断機能用メンバー
    bool diagnostics_enabled_ = false;
    render::diagnostics::PathRecordingConfig diagnostic_config_;
    std::unique_ptr<render::diagnostics::DiagnosticFilm> diagnostic_film_;
    
    // オブジェクト名マッピング（診断表示用）
    std::vector<std::string> object_names_;
    
public:
    /**
     * オブジェクト名を設定（load_scene後に呼び出し）
     * @param names オブジェクト名のリスト（メッシュインデックス順）
     */
    void set_object_names(const std::vector<std::string>& names) {
        object_names_ = names;
    }
    
    /**
     * オブジェクトIDから名前を取得
     * @param object_id オブジェクトID
     * @return オブジェクト名（見つからない場合は "Object_N"）
     */
    std::string get_object_name(int32_t object_id) const {
        if (object_id < 0) {
            return "Environment";
        }
        if (static_cast<size_t>(object_id) < object_names_.size()) {
            return object_names_[object_id];
        }
        return "Object_" + std::to_string(object_id);
    }
    
    /**
     * LightSourceTypeから表示用ライト名を取得（Blenderネイティブライト用）
     */
    static std::string light_source_name(render::diagnostics::LightSourceType type) {
        switch (type) {
            case render::diagnostics::LightSourceType::Point:
                return "PointLight";
            case render::diagnostics::LightSourceType::Area:
                return "AreaLight";
            case render::diagnostics::LightSourceType::Directional:
                return "SunLight";
            case render::diagnostics::LightSourceType::Spot:
                return "SpotLight";
            default:
                return "";  // Environment/Unknown は object_ids から判定
        }
    }
    
    /**
     * オブジェクトIDリストから名前パスを生成
     * object_idsはカメラ→ライトの順で記録されているので、
     * Heckbert表記（ライト→カメラ）に合わせて逆順で表示
     * 
     * 光源の判定：
     * - light_type が Environment または object_ids の最後が -1 → Environment
     * - light_type が Point/Area/Sun/Spot → Blenderネイティブライト名を使用
     * - light_type が Emissive → Emissive mesh（オブジェクト名を使用）
     * - light_type が Unknown → 通常のオブジェクト（光源判定せず）
     * 
     * 例: "Environment → Plane → Camera"
     *     "AreaLight → Plane → Camera"
     *     "EmissiveSphere → Plane → Camera" (Emissive mesh)
     */
    std::string object_path_string(const std::vector<int32_t>& object_ids,
                                    render::diagnostics::LightSourceType light_type = 
                                        render::diagnostics::LightSourceType::Unknown) const {
        if (object_ids.empty()) {
            return "Camera";
        }
        
        std::string path;
        size_t start_idx = object_ids.size();
        int32_t last_obj_id = object_ids[start_idx - 1];
        
        // 光源の判定（light_typeを優先）
        if (light_type == render::diagnostics::LightSourceType::Environment) {
            // Environment light
            path = "Environment";
            // Environment の object_id (-1) をスキップ
            if (last_obj_id == -1) {
                start_idx--;
            }
        } else if (light_type == render::diagnostics::LightSourceType::Emissive) {
            // Emissive mesh - オブジェクト名を使用
            path = get_object_name(last_obj_id);
            start_idx--;
        } else if (light_type == render::diagnostics::LightSourceType::Point ||
                   light_type == render::diagnostics::LightSourceType::Area ||
                   light_type == render::diagnostics::LightSourceType::Directional ||
                   light_type == render::diagnostics::LightSourceType::Spot) {
            // Blender native light (Point, Area, Sun, Spot)
            path = light_source_name(light_type);
            // ライトのobject_idは負の値 (<= -2) なのでスキップ
            if (last_obj_id <= -2) {
                start_idx--;
            }
        } else {
            // Unknown light type - 光源に到達していないパス
            // (Russian Rouletteやmax depthでterminate)
            // または light_type が正しく設定されていないケース
            if (last_obj_id <= -2) {
                // Native light ID but light_type not set - should not happen
                path = "Light(?)";
                start_idx--;
            } else if (last_obj_id == -1) {
                // Environment ID but light_type not set
                path = "Environment(?)";
                start_idx--;
            } else {
                // 通常オブジェクトで終了 = 光源に未到達
                // このパスは表示しない方がいいが、とりあえず "→ (terminated)" として表示
                path = "(terminated)";
                // start_idx は変更しない（最後のオブジェクトも表示する）
            }
        }
        
        // 残りのオブジェクトを追加
        for (size_t i = start_idx; i > 0; --i) {
            path += " → ";
            path += get_object_name(object_ids[i - 1]);
        }
        
        path += " → Camera";
        return path;
    }

private:
    
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
                render::AttenuationRGB albedo = render::make_attenuation_rgb(0.8f, 0.8f, 0.8f);
                float metallic = 0.0f, roughness = 0.5f;
                render::RadianceRGB emission = render::make_radiance_rgb(0.0f, 0.0f, 0.0f);
                float transmission = 0.0f, ior = 1.45f;
                
                if (mat.contains("base_color")) {
                    albedo = render::make_attenuation_rgb(
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
                scene.environment.color = render::make_attenuation_rgb(
                    envJson["color"][0].get<float>(),
                    envJson["color"][1].get<float>(),
                    envJson["color"][2].get<float>()
                );
            }
            
            if (envJson.contains("strength")) {
                scene.environment.strength = envJson["strength"].get<float>();
            }
            
            auto [er, eg, eb] = render::color_to_floats(scene.environment.color);
            std::cerr << "[Environment] color=(" << er << ", "
                      << eg << ", " << eb 
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
            render::AttenuationRGB color = render::make_attenuation_rgb(1.0f, 1.0f, 1.0f);
            if (lightJson.contains("color")) {
                color = render::make_attenuation_rgb(
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
                
                // Extract color components as floats
                auto [cr, cg, cb] = render::color_to_floats(color);
                
                // Point light: I = Power / (4π) [W/sr]
                // In NEE, contribution = I / r² = Power / (4π * r²)
                // We store Power / (4π) as emission, apply 1/r² in integration
                // For soft point (sphere), emission is radiance: Power / (π * area)
                if (radius > EPSILON) {
                    // Spherical light: Lambertian emitter
                    // L = Power / (π * surfaceArea) where surfaceArea = 4πr²
                    // L = Power / (4π²r²)
                    light.emission = render::make_radiance_rgb(
                        cr * (energy / (M_PI * area)),
                        cg * (energy / (M_PI * area)),
                        cb * (energy / (M_PI * area))
                    );
                } else {
                    // True point light: store intensity I = Power / (4π)
                    // 1/r² applied during sampling
                    float factor = energy / (4.0f * M_PI);
                    light.emission = render::make_radiance_rgb(
                        cr * factor,
                        cg * factor,
                        cb * factor
                    );
                }
                
            } else if (typeStr == "SUN") {
                light.type = LightType::SUN;
                light.area = 1.0f * mp_units::square(mp_units::si::metre);  // Sun is directional, area is symbolic
                
                // Extract color components as floats
                auto [cr, cg, cb] = render::color_to_floats(color);
                
                // Sun: energy is already irradiance (W/m²)
                // Use directly as we treat it as parallel rays
                light.emission = render::make_radiance_rgb(
                    cr * energy,
                    cg * energy,
                    cb * energy
                );
                
            } else if (typeStr == "SPOT") {
                light.type = LightType::SPOT;
                float radius = lightJson.value("radius", 0.0f);
                light.radius = radius * mp_units::si::metre;
                float spotAngle = lightJson.value("spot_size", static_cast<float>(M_PI / 4.0f));
                light.spotAngle = spotAngle * mp_units::si::radian;
                light.spotBlend = lightJson.value("spot_blend", 0.0f);
                light.area = 1.0f * mp_units::square(mp_units::si::metre);
                
                // Extract color components as floats
                auto [cr, cg, cb] = render::color_to_floats(color);
                
                // Spot light: same as point but concentrated in cone
                // Blender's spot energy is total power, distributed in cone
                // Solid angle of cone = 2π(1 - cos(θ/2))
                float halfAngle = spotAngle * 0.5f;
                float solidAngle = 2.0f * M_PI * (1.0f - std::cos(halfAngle));
                
                // Intensity in the cone direction = Power / solidAngle
                float factor = energy / solidAngle;
                light.emission = render::make_radiance_rgb(
                    cr * factor,
                    cg * factor,
                    cb * factor
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
                auto [cr, cg, cb] = render::color_to_floats(color);
                float factor = energy / (M_PI * area);
                light.emission = render::make_radiance_rgb(
                    cr * factor,
                    cg * factor,
                    cb * factor
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