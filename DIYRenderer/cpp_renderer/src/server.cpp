/**
 * server.cpp - サーバーモード実装
 * ================================
 * 
 * このファイルは、持続するレンダラープロセスを実装しています。
 * コマンドベースのプロトコルで Python 側と通信し、
 * プロセス起動とシーン読み込みのオーバーヘッドを削減します。
 * 
 * 動作フロー:
 * 1. RenderServer::run() がメインループを開始
 * 2. stdin からコマンドヘッダーを読み取り
 * 3. ペイロードを読み取り
 * 4. コマンドタイプに応じて handleXxx() を呼び出し
 * 5. 結果を stdout に書き込み
 * 6. SHUTDOWN コマンドまで繰り返し
 * 
 * プロトコル:
 * - Command:  [Magic 4B][Type 4B][PayloadSize 4B][Payload...]
 * - Response: [Magic 4B][Type 4B][Status 4B][PayloadSize 4B][Payload...]
 * 
 * 注意: ログは stderr に出力（stdout はバイナリプロトコル専用）
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

// main.cpp で定義されている関数の前方宣言
// TODO: 将来的にリファクタリングしてヘッダーに移動
extern Scene loadSceneFromJson(const std::string &path);
extern NodeTree parseNodeTree(const json &nodeTreeJson);

// =============================================================================
// JSON文字列からシーンを読み込むヘルパー関数
// =============================================================================

/**
 * loadSceneFromJsonString - JSON文字列からシーンを構築
 * 
 * ファイルパスではなく、JSON文字列から直接シーンを読み込みます。
 * サーバーモードでは、Python側から stdin 経由で JSON が送られてくるため、
 * この関数でメモリ内の文字列を直接パースします。
 * 
 * @param jsonStr UTF-8エンコードされたJSON文字列
 * @return 構築されたシーン（エラー時は空のシーン）
 */
static Scene loadSceneFromJsonString(const std::string& jsonStr) {
    Scene scene;
    
    try {
        // nlohmann/json でパース
        json j = json::parse(jsonStr);
        
        // "meshes" 配列がなければ空のシーンを返す
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

/**
 * RenderServer コンストラクタ
 * 
 * サーバーの初期化を行います:
 * 1. Windows環境ではstdin/stdoutをバイナリモードに設定
 * 2. レスポンスライターを初期化
 * 3. OpenMPのスレッド数をログ出力
 */
RenderServer::RenderServer() {
    // Windows ではデフォルトでテキストモード（改行変換あり）なので、
    // バイナリプロトコルのために明示的にバイナリモードに設定
    #ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stdin), _O_BINARY);
    #endif
    
    // stdout にバイナリレスポンスを書き込むライターを作成
    response_ = std::make_unique<protocol::ResponseWriter>(std::cout);
    
    log("Server initialized");
    
    // OpenMP が有効な場合、使用可能なスレッド数をログ出力
    #ifdef _OPENMP
    log("OpenMP enabled with " + std::to_string(omp_get_max_threads()) + " threads");
    #else
    log("Single-threaded mode");
    #endif
}

/**
 * RenderServer デストラクタ
 */
RenderServer::~RenderServer() {
    log("Server shutting down");
}

/**
 * log - デバッグログを stderr に出力
 * 
 * 重要: stdout はバイナリプロトコル専用なので、
 * すべてのログは stderr に出力する必要があります。
 * 
 * @param message ログメッセージ
 */
void RenderServer::log(const std::string& message) {
    std::cerr << "[Server] " << message << "\n";
    std::cerr.flush();  // バッファをフラッシュして即座に表示
}

/**
 * readPayload - stdin からペイロードを読み取り
 * 
 * 大きなペイロード（数百KB〜数MB）を安全に読み取るため、
 * チャンク単位で読み取ります。これは、パイプのバッファサイズ
 * （通常64KB）を超えるデータでもデッドロックを防ぎます。
 * 
 * @param size 読み取るバイト数
 * @param payload 読み取り先のバッファ
 * @return 成功時 true
 */
bool RenderServer::readPayload(uint32_t size, std::vector<uint8_t>& payload) {
    payload.resize(size);
    if (size == 0) return true;
    
    // 32KB チャンクで読み取り（パイプバッファのデッドロック回避）
    const size_t CHUNK_SIZE = 32768;
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

/**
 * run - サーバーメインループ
 * 
 * 無限ループでコマンドを受信し、処理し、レスポンスを返します。
 * SHUTDOWN コマンドを受信するか、stdin が EOF になると終了します。
 * 
 * 処理フロー:
 * 1. コマンドヘッダー（12バイト）を読み取り
 * 2. ペイロードを読み取り
 * 3. コマンドタイプに応じて適切なハンドラを呼び出し
 * 4. ハンドラがレスポンスを stdout に書き込み
 * 5. 繰り返し
 */
void RenderServer::run() {
    log("Entering server loop");
    
    while (true) {
        // ===== ステップ1: コマンドヘッダー読み取り =====
        protocol::CommandHeader header;
        if (!protocol::CommandHeader::read(std::cin, header)) {
            if (std::cin.eof()) {
                // Python側がパイプを閉じた（正常終了）
                log("EOF received, exiting");
                break;
            }
            log("Failed to read command header");
            response_->writeError(protocol::StatusCode::ERROR_INVALID_COMMAND, "Invalid header");
            continue;
        }
        
        // ===== ステップ2: ペイロード読み取り =====
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
        
        // ===== ステップ3: コマンドディスパッチ =====
        // コマンドタイプに応じて適切なハンドラを呼び出す
        switch (header.type) {
            case protocol::CommandType::INIT:
                // レンダラー初期化（バックエンド、アルゴリズム設定）
                handleInit(payload);
                break;
                
            case protocol::CommandType::UPDATE_SCENE:
                // シーンデータ更新（JSON形式）
                handleUpdateScene(payload);
                break;
                
            case protocol::CommandType::UPDATE_CAMERA:
                // カメラパラメータ更新
                handleUpdateCamera(payload);
                break;
                
            case protocol::CommandType::RENDER_TILE:
                // タイルレンダリング実行 → PIXELS レスポンス
                handleRenderTile(payload);
                break;
                
            case protocol::CommandType::CANCEL:
                // 現在のレンダリングをキャンセル
                handleCancel();
                break;
                
            case protocol::CommandType::QUERY_CAPS:
                // レンダラーの機能を問い合わせ
                handleQueryCaps();
                break;
                
            case protocol::CommandType::SET_BACKEND:
                // バックエンド変更
                handleSetBackend(payload);
                break;
                
            case protocol::CommandType::SET_ALGORITHM:
                // アルゴリズム変更
                handleSetAlgorithm(payload);
                break;
                
            case protocol::CommandType::SHUTDOWN:
                // サーバー終了
                log("Shutdown command received");
                response_->writeAck();
                return;  // メインループを抜けてプロセス終了
                
            default:
                // 未知のコマンド
                log("Unknown command: " + std::to_string(static_cast<uint32_t>(header.type)));
                response_->writeError(protocol::StatusCode::ERROR_INVALID_COMMAND, "Unknown command");
                break;
        }
    }
}

/**
 * handleInit - INIT コマンド処理
 * 
 * レンダラーの初期設定を行います。
 * - バックエンド（cpu/webgpu）の設定
 * - アルゴリズム（simple/nee/mis）の設定
 * - 既存のシーン/カメラ状態をリセット
 * 
 * @param payload [backend_len][backend][algo_len][algo] 形式のバイト列
 */
void RenderServer::handleInit(const std::vector<uint8_t>& payload) {
    // ペイロードから文字列を読み取る
    protocol::PayloadReader reader(payload);
    backend_ = reader.readString();
    algorithm_ = reader.readString();
    
    log("Init: backend=" + backend_ + ", algorithm=" + algorithm_);
    
    // 状態をリセット
    scene_loaded_ = false;
    camera_set_ = false;
    scene_ = Scene();
    
    response_->writeAck();
}

/**
 * handleUpdateScene - UPDATE_SCENE コマンド処理
 * 
 * JSON形式のシーンデータを受け取り、内部のシーン構造体に変換します。
 * シーンが変更された場合に呼び出されます。
 * 
 * @param payload UTF-8エンコードされたJSON文字列
 */
void RenderServer::handleUpdateScene(const std::vector<uint8_t>& payload) {
    // バイト列をUTF-8文字列に変換
    std::string jsonStr(payload.begin(), payload.end());
    
    log("UpdateScene: " + std::to_string(payload.size()) + " bytes");
    
    // JSON をパースしてシーン構造体に変換
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

/**
 * handleUpdateCamera - UPDATE_CAMERA コマンド処理
 * 
 * カメラパラメータを更新します。シーンデータは再送信不要なので、
 * カメラを動かすだけの場合は高速に更新できます。
 * 
 * @param payload 40バイトのカメラパラメータ（10 × float）
 */
void RenderServer::handleUpdateCamera(const std::vector<uint8_t>& payload) {
    // サイズチェック
    if (payload.size() < 40) {
        response_->writeError(protocol::StatusCode::ERROR_INVALID_COMMAND, "Camera payload too small");
        return;
    }
    
    // バイト列からカメラパラメータを読み取り
    protocol::CameraParams params = protocol::CameraParams::fromBytes(payload.data());
    
    // カメラ構造体を構築
    camera_.pos = Vec3(params.pos_x, params.pos_y, params.pos_z);
    camera_.dir = Vec3(params.dir_x, params.dir_y, params.dir_z);
    camera_.dir.normalize();
    camera_.up = Vec3(params.up_x, params.up_y, params.up_z);
    camera_.up.normalize();
    camera_.fovDeg = params.fov;
    
    // 派生ベクトルを計算
    // forward: 視線方向
    // right: forward × up（右方向）
    // up: right × forward（再計算して正規直交系を保証）
    camera_.forward = camera_.dir;
    camera_.right = Vec3::cross(camera_.forward, camera_.up);
    camera_.right.normalize();
    camera_.up = Vec3::cross(camera_.right, camera_.forward);
    camera_.up.normalize();
    
    camera_set_ = true;
    
    response_->writeAck();
}

/**
 * handleRenderTile - RENDER_TILE コマンド処理
 * 
 * 指定されたタイル領域をレンダリングし、ピクセルデータを返します。
 * これがサーバーモードの主要な処理です。
 * 
 * 処理フロー:
 * 1. シーンとカメラの状態チェック
 * 2. パラメータ読み取り
 * 3. OpenMP による並列レンダリング
 * 4. Y軸反転（Blenderの座標系に合わせる）
 * 5. PIXELS レスポンスで結果を返す
 * 
 * @param payload 36バイトのタイルパラメータ（9 × uint32）
 */
void RenderServer::handleRenderTile(const std::vector<uint8_t>& payload) {
    // ===== 事前条件チェック =====
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
    
    // ===== パラメータ読み取り =====
    protocol::RenderTileParams params = protocol::RenderTileParams::fromBytes(payload.data());
    
    int tileX = params.tile_x;        // タイル左上X座標
    int tileY = params.tile_y;        // タイル左上Y座標
    int tileW = params.tile_w;        // タイル幅
    int tileH = params.tile_h;        // タイル高さ
    int fullW = params.full_w;        // 画像全体の幅
    int fullH = params.full_h;        // 画像全体の高さ
    int samples = params.samples;     // サンプル数
    int sampleOffset = params.sample_offset;  // サンプルオフセット（累積用）
    int maxDepth = params.max_depth;  // 最大バウンス数
    
    // カメラのアスペクト比を更新
    camera_.aspect = (float)fullW / (float)fullH;
    
    // FOV からスケール係数を計算
    float fovRad = camera_.fovDeg * (float)M_PI / 180.0f;
    float scale = std::tan(fovRad * 0.5f);
    
    // キャンセルフラグをリセット
    cancel_requested_ = false;
    
    // ===== シーンライトを構築（NEE/MIS用）=====
    SceneLights sceneLights;
    sceneLights.buildFromScene(scene_);
    
    // ===== 出力バッファを確保 =====
    int totalPixels = tileW * tileH;
    std::vector<float> pixelBuffer(totalPixels * 4, 0.0f);  // RGBA
    
    // ===== 並列レンダリング =====
    // OpenMP で各ピクセルを並列処理
    // schedule(dynamic, 16): 16ピクセルずつ動的にスレッドに割り当て
    // collapse(2): 2重ループを1つのループとして扱い、より均等に分散
    #pragma omp parallel for schedule(dynamic, 16) collapse(2)
    for (int py = 0; py < tileH; ++py) {
        for (int px = 0; px < tileW; ++px) {
            // キャンセルされた場合はスキップ
            if (cancel_requested_) continue;
            
            // グローバル座標
            int y = tileY + py;
            int x = tileX + px;
            int pixelIdx = py * tileW + px;
            
            // NDC (Normalized Device Coordinates) を計算
            // X: [-aspect, +aspect]、Y: [-1, +1]
            float ndcX = (2.0f * (x + 0.5f) / fullW - 1.0f) * camera_.aspect;
            float ndcY = 1.0f - 2.0f * (y + 0.5f) / fullH;
            
            // ワールド空間のレイ方向を計算
            Vec3 worldDir = camera_.forward + camera_.right * (ndcX * scale) + camera_.up * (ndcY * scale);
            worldDir.normalize();
            Ray ray{camera_.pos, worldDir};
            
            Vec3 color{0, 0, 0};
            
            // 複数サンプルの平均を計算
            for (int s = 0; s < samples; ++s) {
                // ランダムシードを設定（ピクセル位置とサンプルインデックスから）
                seed_random_xyz((uint32_t)x, (uint32_t)y, (uint32_t)(sampleOffset + s));
                
                Ray sampleRay = ray;
                
                // アンチエイリアシングのためのジッター（サブピクセルサンプリング）
                if (samples > 1) {
                    float jitterX = (randf() - 0.5f) / fullW;
                    float jitterY = (randf() - 0.5f) / fullH;
                    Vec3 jitteredDir = camera_.forward + camera_.right * ((ndcX + jitterX) * scale) + camera_.up * ((ndcY + jitterY) * scale);
                    jitteredDir.normalize();
                    sampleRay.d = jitteredDir;
                }
                
                // アルゴリズムに応じてパストレーシング
                if (algorithm_ == "simple") {
                    // シンプルなBSDFサンプリング（参照実装）
                    color = color + traceSimple(scene_, sampleRay, maxDepth);
                } else if (algorithm_ == "mis") {
                    // Multiple Importance Sampling（最高品質）
                    color = color + traceMIS(scene_, sceneLights, sampleRay, maxDepth);
                } else {
                    // Next Event Estimation（デフォルト、バランスが良い）
                    color = color + traceNEE(scene_, sceneLights, sampleRay, maxDepth);
                }
            }
            
            // 負の値をクランプ（物理的にありえないため）
            color.x = std::max(0.0f, color.x);
            color.y = std::max(0.0f, color.y);
            color.z = std::max(0.0f, color.z);
            
            // RGBA としてバッファに格納
            pixelBuffer[pixelIdx * 4 + 0] = color.x;
            pixelBuffer[pixelIdx * 4 + 1] = color.y;
            pixelBuffer[pixelIdx * 4 + 2] = color.z;
            pixelBuffer[pixelIdx * 4 + 3] = 1.0f;  // アルファは常に1.0
        }
    }
    
    // キャンセルされた場合はエラーを返す
    if (cancel_requested_) {
        response_->writeError(protocol::StatusCode::ERROR_CANCELLED, "Render cancelled");
        return;
    }
    
    // ===== Y軸反転 =====
    // Blender は画像の左下が原点だが、C++ レンダラーは左上が原点
    // Python側でも反転しているが、ここでも反転する必要がある
    std::vector<float> flippedBuffer(totalPixels * 4);
    for (int y = 0; y < tileH; ++y) {
        int srcRow = y * tileW * 4;
        int dstRow = (tileH - 1 - y) * tileW * 4;
        std::copy(pixelBuffer.begin() + srcRow, 
                  pixelBuffer.begin() + srcRow + tileW * 4,
                  flippedBuffer.begin() + dstRow);
    }
    
    // ピクセルデータをレスポンスとして返す
    response_->writePixels(flippedBuffer, tileW, tileH);
}

/**
 * handleCancel - CANCEL コマンド処理
 * 
 * 現在のレンダリング操作をキャンセルします。
 * cancel_requested_ フラグを立てると、レンダリングループが
 * 次のピクセル処理をスキップします。
 */
void RenderServer::handleCancel() {
    log("Cancel requested");
    cancel_requested_ = true;
    response_->writeAck();
}

/**
 * handleQueryCaps - QUERY_CAPS コマンド処理
 * 
 * レンダラーがサポートする機能を返します。
 * - 利用可能なバックエンド（cpu, webgpu等）
 * - 利用可能なアルゴリズム（simple, nee, mis）
 */
void RenderServer::handleQueryCaps() {
    std::vector<std::string> backends = {"cpu"};
    // TODO: Phase 2 で WebGPU を追加
    // backends.push_back("webgpu");
    
    std::vector<std::string> algorithms = {"simple", "nee", "mis"};
    
    response_->writeCapabilities(backends, algorithms);
}

/**
 * handleSetBackend - SET_BACKEND コマンド処理
 * 
 * レンダリングバックエンドを変更します。
 * 現在は CPU のみサポート。
 * 
 * @param payload UTF-8エンコードされたバックエンド名
 */
void RenderServer::handleSetBackend(const std::vector<uint8_t>& payload) {
    backend_ = std::string(payload.begin(), payload.end());
    log("Backend set to: " + backend_);
    response_->writeAck();
}

/**
 * handleSetAlgorithm - SET_ALGORITHM コマンド処理
 * 
 * パストレーシングアルゴリズムを変更します。
 * 
 * @param payload UTF-8エンコードされたアルゴリズム名
 */
void RenderServer::handleSetAlgorithm(const std::vector<uint8_t>& payload) {
    algorithm_ = std::string(payload.begin(), payload.end());
    log("Algorithm set to: " + algorithm_);
    response_->writeAck();
}
