/**
 * pybind_module.cpp - pybind11 バインディング定義
 * =================================================
 * 
 * Python から C++ レンダラーを直接呼び出すためのバインディング。
 * GIL 解放により、レンダリング中も Python スレッドが動作可能。
 * 
 * ビルド方法:
 *   pip install pybind11
 *   cd cpp_renderer && mkdir build && cd build
 *   cmake .. -DBUILD_PYBIND=ON
 *   make
 * 
 * 使用例:
 *   import lucidrenderer
 *   r = lucidrenderer.Renderer()
 *   r.load_scene_json('{"meshes": [...]}')
 *   r.set_camera(0, 0, 5, 0, 0, -1, 0, 1, 0, 50)
 *   pixels = r.render_tile(0, 0, 800, 600, 800, 600, 16, 0, 8)
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "pybind_renderer.hpp"
#include "diagnostics/diagnostic_export.hpp"
#include "diagnostics/diagnostic_integrator.hpp"
#include "diagnostics/path_stats_config.hpp"

#ifdef LUCID_HAS_DAWN
#include "gpu/dawn_context.hpp"
#endif

namespace py = pybind11;
using namespace render::diagnostics;

PYBIND11_MODULE(lucidrenderer, m) {
    m.doc() = "Lucid Path Tracer - Python bindings for the C++ renderer";
    
    py::class_<PyRenderer>(m, "Renderer")
        .def(py::init<>(), "Create a new renderer instance")
        
        // =====================================================================
        // キャンセル制御
        // =====================================================================
        .def("cancel", &PyRenderer::cancel,
             "Request cancellation of current rendering. "
             "Can be called from another thread. Returns immediately.")
        
        .def("is_cancelled", &PyRenderer::is_cancelled,
             "Check if cancellation has been requested")
        
        .def("reset_cancel", &PyRenderer::reset_cancel,
             "Reset the cancellation flag (usually not needed, "
             "render_tile does this automatically)")
        
        // =====================================================================
        // シーン管理
        // =====================================================================
        .def("load_scene_json", &PyRenderer::load_scene_json,
             py::arg("json_str"),
             "Load scene from JSON string")
        
        .def("set_camera", &PyRenderer::set_camera,
             py::arg("pos_x"), py::arg("pos_y"), py::arg("pos_z"),
             py::arg("dir_x"), py::arg("dir_y"), py::arg("dir_z"),
             py::arg("up_x"), py::arg("up_y"), py::arg("up_z"),
             py::arg("fov_deg"),
             "Set camera position, direction, up vector, and field of view")
        
        .def("set_algorithm", &PyRenderer::set_algorithm,
             py::arg("algorithm"),
             "Set path tracing algorithm: 'simple', 'nee', or 'mis'")
        
        .def("set_object_names", &PyRenderer::set_object_names,
             py::arg("names"),
             "Set object names for diagnostic display (list of strings, mesh index order)")
        
        .def("get_object_name", &PyRenderer::get_object_name,
             py::arg("object_id"),
             "Get object name by ID, returns 'Object_N' if not found")
        
        // object_path_stringはPythonから直接呼ばれることはないので、
        // シンプルなラムダでラップしてデフォルト引数問題を回避
        .def("object_path_string", [](const PyRenderer& r, const std::vector<int32_t>& ids) {
                return r.object_path_string(ids);
             },
             py::arg("object_ids"),
             "Convert object ID list to path string like 'Light → Plane → Camera'")
        
        // =====================================================================
        // レンダリング（GIL 解放）
        // =====================================================================
        .def("render_tile", &PyRenderer::render_tile,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("tile_x"), py::arg("tile_y"),
             py::arg("tile_w"), py::arg("tile_h"),
             py::arg("full_w"), py::arg("full_h"),
             py::arg("samples") = 1,
             py::arg("sample_offset") = 0,
             py::arg("max_depth") = 8,
             "Render a tile and return RGBA float array. "
             "GIL is released during rendering, allowing cancel() from another thread.")
        
        .def("render_debug", &PyRenderer::render_debug,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("tile_x"), py::arg("tile_y"),
             py::arg("tile_w"), py::arg("tile_h"),
             py::arg("full_w"), py::arg("full_h"),
             py::arg("mode"),
             "Render debug visualization: 'normal', 'albedo', or 'emission'")

        .def("render_debug_gpu", &PyRenderer::render_debug_gpu,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("tile_x"), py::arg("tile_y"),
             py::arg("tile_w"), py::arg("tile_h"),
             py::arg("full_w"), py::arg("full_h"),
             py::arg("mode"),
             "GPU debug visualization (Phase 1b: 'normal' only; others fall back to CPU)")
        
        // =====================================================================
        // NumPy 配列を直接返すバージョン（オプション）
        // =====================================================================
        .def("render_tile_numpy", [](PyRenderer& self,
                                      int tile_x, int tile_y,
                                      int tile_w, int tile_h,
                                      int full_w, int full_h,
                                      int samples,
                                      int sample_offset,
                                      int max_depth) {
            // GIL を解放してレンダリング
            std::vector<float> pixels;
            {
                py::gil_scoped_release release;
                pixels = self.render_tile(tile_x, tile_y, tile_w, tile_h,
                                          full_w, full_h, samples, sample_offset, max_depth);
            }
            
            // NumPy 配列として返す（形状: [height, width, 4]）
            return py::array_t<float>(
                {tile_h, tile_w, 4},  // shape
                {tile_w * 4 * sizeof(float), 4 * sizeof(float), sizeof(float)},  // strides
                pixels.data()
            );
        },
        py::arg("tile_x"), py::arg("tile_y"),
        py::arg("tile_w"), py::arg("tile_h"),
        py::arg("full_w"), py::arg("full_h"),
        py::arg("samples") = 1,
        py::arg("sample_offset") = 0,
        py::arg("max_depth") = 8,
        "Render a tile and return as NumPy array with shape (height, width, 4)")
        
        // =====================================================================
        // 情報取得
        // =====================================================================
        .def("is_scene_loaded", &PyRenderer::is_scene_loaded,
             "Check if a scene has been loaded")
        
        .def("is_camera_set", &PyRenderer::is_camera_set,
             "Check if camera has been configured")
        
        .def("get_algorithm", &PyRenderer::get_algorithm,
             "Get current path tracing algorithm")
        
        .def("get_mesh_count", &PyRenderer::get_mesh_count,
             "Get number of meshes in the loaded scene")
        
        // =====================================================================
        // 診断機能 (Path Variance Analyzer)
        // =====================================================================
        .def("enable_diagnostics", &PyRenderer::enable_diagnostics,
             py::arg("config"),
             "Enable diagnostic path recording with given configuration")
        
        .def("disable_diagnostics", &PyRenderer::disable_diagnostics,
             "Disable diagnostic recording")
        
        .def("is_diagnostics_enabled", &PyRenderer::is_diagnostics_enabled,
             "Check if diagnostics recording is enabled")
        
        .def("clear_diagnostics", &PyRenderer::clear_diagnostics,
             "Clear all recorded diagnostic data")
        
        .def("get_diagnostic_stats", &PyRenderer::get_diagnostic_stats,
             "Get global diagnostic statistics")
        
        .def("get_diagnostic_variance_map", [](const PyRenderer& self) {
            auto variance_map = self.get_diagnostic_variance_map();
            // Return as NumPy array
            return py::array_t<float>(
                {static_cast<py::ssize_t>(variance_map.size())},
                variance_map.data()
            );
        }, "Get per-pixel variance as 1D NumPy array (width * height)")
        
        .def("export_diagnostic_json", &PyRenderer::export_diagnostic_json,
             "Export diagnostic data as JSON string")
        
        .def("get_top_variance_groups", &PyRenderer::get_top_variance_groups,
             py::arg("count") = 10,
             "Get top N path groups by variance")
        
        .def("get_pixel_diagnostic", 
             [](const PyRenderer& self, size_t x, size_t y, const std::string& sort_by) {
                 PyRenderer::TopGroupSortBy sort_mode = PyRenderer::TopGroupSortBy::Variance;
                 if (sort_by == "mean") {
                     sort_mode = PyRenderer::TopGroupSortBy::Mean;
                 }
                 return self.get_pixel_diagnostic(x, y, sort_mode);
             },
             py::arg("x"), py::arg("y"),
             py::arg("sort_by") = "variance",
             "Get diagnostic data for a specific pixel, sorted by 'variance' (default) or 'mean'")
        
        // =====================================================================
        // Python 的な機能
        // =====================================================================
        .def("__repr__", [](const PyRenderer& r) {
            return "<lucidrenderer.Renderer scene_loaded=" + 
                   std::string(r.is_scene_loaded() ? "True" : "False") +
                   " meshes=" + std::to_string(r.get_mesh_count()) +
                   " algorithm='" + r.get_algorithm() + "'>";
        });
    
    // バージョン情報
    m.attr("__version__") = "1.0.0";
    
    #ifdef _OPENMP
    m.attr("openmp_enabled") = true;
    #else
    m.attr("openmp_enabled") = false;
    #endif

    // =========================================================================
    // GPU Backend (Dawn / WebGPU)
    // =========================================================================
    #ifdef LUCID_HAS_DAWN
    m.attr("dawn_enabled") = true;

    m.def("gpu_adapter_info", []() -> std::string {
        auto ctx = lucid::gpu::DawnContext::create();
        return ctx ? ctx->adapter_info() : std::string("<unavailable>");
    }, "Return a string describing the GPU adapter Dawn selected "
       "(vendor / device / backend).");

    m.def("gpu_run_double_test",
          &lucid::gpu::run_double_test,
          py::arg("n") = 64,
          "Phase 1a smoke test: dispatch a compute shader that doubles each "
          "element of an n-float input buffer and return the result.");
    #else
    m.attr("dawn_enabled") = false;
    #endif

    // =========================================================================
    // Path Variance Analyzer (Diagnostic System)
    // =========================================================================
    
    // PathRecordingConfig - Configuration for path statistics recording
    py::class_<PathRecordingConfig>(m, "PathRecordingConfig")
        .def(py::init<>(), "Create default recording configuration")
        .def_readwrite("subsample_factor", &PathRecordingConfig::subsample_factor,
            "Spatial subsampling (1 = every pixel, 2 = every 2nd pixel)")
        .def_readwrite("max_groups_per_pixel", &PathRecordingConfig::max_groups_per_pixel,
            "Maximum unique path types per pixel")
        .def_readwrite("max_outliers_per_pixel", &PathRecordingConfig::max_outliers_per_pixel,
            "Maximum outlier samples to store per pixel")
        .def_readwrite("max_depth", &PathRecordingConfig::max_depth,
            "Maximum path depth to track")
        .def_readwrite("variance_threshold", &PathRecordingConfig::variance_threshold,
            "Minimum variance to track a group")
        .def_readwrite("outlier_threshold", &PathRecordingConfig::outlier_threshold,
            "Threshold for outlier detection (stddev multiplier)")
        .def_static("minimal", &PathRecordingConfig::minimal,
            "Create minimal preset (~200MB at 1080p)")
        .def_static("standard", &PathRecordingConfig::standard,
            "Create standard preset (~800MB at 1080p)")
        .def_static("detailed", &PathRecordingConfig::detailed,
            "Create detailed preset (~3.2GB at 1080p)")
        .def("estimate_memory_bytes", &PathRecordingConfig::estimate_memory_bytes,
            py::arg("width"), py::arg("height"),
            "Estimate memory usage in bytes for given resolution")
        .def("__repr__", [](const PathRecordingConfig& c) {
            return "<PathRecordingConfig subsample=" + std::to_string(c.subsample_factor) +
                   " max_groups=" + std::to_string(c.max_groups_per_pixel) + ">";
        });
    
    // ExportedGroupInfo - Path group statistics for Python
    py::class_<ExportedGroupInfo>(m, "ExportedGroupInfo")
        .def(py::init<>())
        .def_readonly("pixel_x", &ExportedGroupInfo::pixel_x)
        .def_readonly("pixel_y", &ExportedGroupInfo::pixel_y)
        .def_readonly("signature", &ExportedGroupInfo::signature)
        .def_readonly("signature_heckbert", &ExportedGroupInfo::signature_heckbert)
        .def_readonly("object_ids", &ExportedGroupInfo::object_ids)
        .def_readonly("object_path", &ExportedGroupInfo::object_path)
        .def_readonly("sample_count", &ExportedGroupInfo::sample_count)
        .def_readonly("mean_luminance", &ExportedGroupInfo::mean_luminance)
        .def_readonly("variance_luminance", &ExportedGroupInfo::variance_luminance)
        .def_readonly("mean_rgb", &ExportedGroupInfo::mean_rgb)
        .def_readonly("variance_rgb", &ExportedGroupInfo::variance_rgb)
        .def_readonly("depth", &ExportedGroupInfo::depth)
        .def_readonly("coarse_type", &ExportedGroupInfo::coarse_type)
        .def_readonly("coarse_type_name", &ExportedGroupInfo::coarse_type_name)
        .def_readonly("strategy_name", &ExportedGroupInfo::strategy_name)  // Plan E
        .def_readonly("positions", &ExportedGroupInfo::positions,
            "Path vertex world positions (camera first, then hit points)")
        .def_readonly("normals", &ExportedGroupInfo::normals,
            "Surface normals at each hit point")
        .def("__repr__", [](const ExportedGroupInfo& g) {
            return "<ExportedGroupInfo signature='" + g.signature + 
                   "' heckbert='" + g.signature_heckbert + "'" +
                   " path='" + g.object_path + "'" +
                   " strategy='" + g.strategy_name + "'" +
                   " mean=" + std::to_string(g.mean_luminance) +
                   " var=" + std::to_string(g.variance_luminance) +
                   " positions=" + std::to_string(g.positions.size()) + ">";
        });
    
    // TopGroupSortBy enum for pixel diagnostics
    py::enum_<PyRenderer::TopGroupSortBy>(m, "TopGroupSortBy")
        .value("Variance", PyRenderer::TopGroupSortBy::Variance, "Sort by variance (noisy paths first)")
        .value("Mean", PyRenderer::TopGroupSortBy::Mean, "Sort by mean contribution (bright paths first)")
        .export_values();
    
    // PixelDiagnosticInfo - Per-pixel diagnostic data
    py::class_<PyRenderer::PixelDiagnosticInfo>(m, "PixelDiagnosticInfo")
        .def(py::init<>())
        .def_readonly("valid", &PyRenderer::PixelDiagnosticInfo::valid)
        .def_readonly("sample_count", &PyRenderer::PixelDiagnosticInfo::sample_count)
        .def_readonly("variance", &PyRenderer::PixelDiagnosticInfo::variance)
        .def_readonly("group_count", &PyRenderer::PixelDiagnosticInfo::group_count)
        .def_readonly("outlier_count", &PyRenderer::PixelDiagnosticInfo::outlier_count)
        .def_readonly("mean_rgb", &PyRenderer::PixelDiagnosticInfo::mean_rgb)
        .def_readonly("top_groups", &PyRenderer::PixelDiagnosticInfo::top_groups)
        .def("__repr__", [](const PyRenderer::PixelDiagnosticInfo& p) {
            if (!p.valid) return std::string("<PixelDiagnosticInfo invalid>");
            return "<PixelDiagnosticInfo samples=" + std::to_string(p.sample_count) +
                   " variance=" + std::to_string(p.variance) +
                   " groups=" + std::to_string(p.group_count) + ">";
        });
    
    // DiagnosticSuggestion - Improvement suggestions
    py::class_<DiagnosticSuggestion>(m, "DiagnosticSuggestion")
        .def(py::init<>())
        .def_readonly("category", &DiagnosticSuggestion::category)
        .def_readonly("severity", &DiagnosticSuggestion::severity)
        .def_readonly("message", &DiagnosticSuggestion::message)
        .def_readonly("action", &DiagnosticSuggestion::action)
        .def("__repr__", [](const DiagnosticSuggestion& s) {
            return "<DiagnosticSuggestion [" + s.severity + "] " + s.message + ">";
        });
    
    // GlobalDiagnosticStats - Summary statistics
    py::class_<GlobalDiagnosticStats>(m, "GlobalDiagnosticStats")
        .def(py::init<>())
        .def_readonly("total_samples", &GlobalDiagnosticStats::total_samples)
        .def_readonly("active_pixels", &GlobalDiagnosticStats::active_pixels)
        .def_readonly("total_groups", &GlobalDiagnosticStats::total_groups)
        .def_readonly("total_overflow", &GlobalDiagnosticStats::total_overflow)
        .def_readonly("total_variance", &GlobalDiagnosticStats::total_variance);
    
    // ExportedCoarseStats - Coarse type statistics
    py::class_<ExportedCoarseStats>(m, "ExportedCoarseStats")
        .def(py::init<>())
        .def_readonly("type_id", &ExportedCoarseStats::type_id)
        .def_readonly("name", &ExportedCoarseStats::name)
        .def_readonly("sample_count", &ExportedCoarseStats::sample_count)
        .def_readonly("mean_luminance", &ExportedCoarseStats::mean_luminance)
        .def_readonly("variance_luminance", &ExportedCoarseStats::variance_luminance);
    
    // DiagnosticFilm - Film for path statistics recording
    py::class_<DiagnosticFilm>(m, "DiagnosticFilm")
        .def(py::init<size_t, size_t, const PathRecordingConfig&>(),
            py::arg("width"), py::arg("height"), py::arg("config"),
            "Create diagnostic film for given dimensions")
        .def("width", &DiagnosticFilm::width)
        .def("height", &DiagnosticFilm::height)
        .def("sampled_width", &DiagnosticFilm::sampled_width)
        .def("sampled_height", &DiagnosticFilm::sampled_height)
        .def("total_paths_recorded", &DiagnosticFilm::total_paths_recorded)
        .def("global_stats", &DiagnosticFilm::global_stats)
        .def("metadata_json", &DiagnosticFilm::metadata_json)
        .def("record_path", &DiagnosticFilm::record_path,
            py::arg("x"), py::arg("y"), py::arg("trace"),
            "Record a path trace at pixel (x, y)")
        .def("clear", &DiagnosticFilm::clear);
    
    // DiagnosticPathTracer - Path tracer with diagnostic support
    py::class_<DiagnosticPathTracer>(m, "DiagnosticPathTracer")
        .def(py::init<size_t, size_t, const PathRecordingConfig&>(),
            py::arg("width"), py::arg("height"), py::arg("config"),
            "Create diagnostic path tracer")
        .def("is_enabled", &DiagnosticPathTracer::is_enabled)
        .def("set_enabled", &DiagnosticPathTracer::set_enabled)
        .def("film", static_cast<DiagnosticFilm& (DiagnosticPathTracer::*)()>(&DiagnosticPathTracer::film),
            py::return_value_policy::reference_internal);
    
    // DiagnosticExporter - Export interface for analysis results
    py::class_<DiagnosticExporter>(m, "DiagnosticExporter")
        .def(py::init<const DiagnosticFilm&>(),
            py::arg("film"),
            "Create exporter for a diagnostic film")
        .def("get_global_stats", &DiagnosticExporter::get_global_stats)
        .def("get_top_variance_groups", &DiagnosticExporter::get_top_variance_groups,
            py::arg("n") = 10,
            "Get top N groups by variance")
        .def("get_top_mean_groups", &DiagnosticExporter::get_top_mean_groups,
            py::arg("n") = 10,
            "Get top N groups by mean contribution")
        .def("get_coarse_stats", &DiagnosticExporter::get_coarse_stats)
        .def("generate_suggestions", &DiagnosticExporter::generate_suggestions)
        .def("export_metadata_json", &DiagnosticExporter::export_metadata_json)
        .def("export_binary", [](const DiagnosticExporter& exp) {
            auto data = exp.export_binary();
            return py::bytes(reinterpret_cast<const char*>(data.data()), data.size());
        }, "Export diagnostic data as binary blob");
    
    // BsdfType enum for Python
    py::enum_<BsdfType>(m, "BsdfType")
        .value("Diffuse", BsdfType::Diffuse)
        .value("Glossy", BsdfType::Glossy)
        .value("Mirror", BsdfType::Mirror)
        .value("Glass", BsdfType::Glass)
        .value("Emission", BsdfType::Emission)
        .value("Environment", BsdfType::Environment)
        .export_values();
    
    // LightSourceType enum for Heckbert notation
    py::enum_<LightSourceType>(m, "LightSourceType")
        .value("Unknown", LightSourceType::Unknown)
        .value("Point", LightSourceType::Point)
        .value("Area", LightSourceType::Area)
        .value("Directional", LightSourceType::Directional)
        .value("Spot", LightSourceType::Spot)
        .value("Environment", LightSourceType::Environment)
        .value("Emissive", LightSourceType::Emissive)
        .export_values();
    
    // PathTrace - Raw path data structure
    py::class_<PathTrace>(m, "PathTrace")
        .def(py::init<>(), "Create empty path trace")
        .def_readonly("depth", &PathTrace::depth)
        .def("get_contribution", [](const PathTrace& p) {
            return std::array<float, 3>{p.contribution.r, p.contribution.g, p.contribution.b};
        });
    
    // PathDiagnosticRecorder - Per-thread helper for recording paths
    py::class_<PathDiagnosticRecorder>(m, "PathDiagnosticRecorder")
        .def(py::init<>(), "Create a new path recorder")
        .def("begin_path", &PathDiagnosticRecorder::begin_path,
            "Start recording a new path")
        .def("record_vertex", &PathDiagnosticRecorder::record_vertex,
            py::arg("object_id"), py::arg("material_id"),
            py::arg("bsdf_type"), py::arg("is_delta"), py::arg("is_light_sampled"),
            "Record a surface interaction vertex")
        .def("record_environment_hit", &PathDiagnosticRecorder::record_environment_hit,
            "Record environment map hit")
        .def("record_light_hit", &PathDiagnosticRecorder::record_light_hit,
            py::arg("light_id"),
            py::arg("light_source_type") = LightSourceType::Unknown,
            "Record light hit with optional light source type for Heckbert notation")
        .def("record_emissive_hit", &PathDiagnosticRecorder::record_emissive_hit,
            py::arg("object_id"), py::arg("material_id"),
            "Record emissive mesh hit")
        .def("current_depth", &PathDiagnosticRecorder::current_depth,
            "Get current path depth")
        .def("end_path", 
            [](PathDiagnosticRecorder& recorder, float r, float g, float b) -> PathTrace {
                return recorder.end_path(render::RGB3f{r, g, b});
            }, 
            py::arg("r"), py::arg("g"), py::arg("b"),
            "Finalize path and return PathTrace")
        .def("end_path_lum", 
            [](PathDiagnosticRecorder& recorder, float luminance) -> PathTrace {
                return recorder.end_path(render::RGB3f{luminance, luminance, luminance});
            }, 
            py::arg("luminance"),
            "Finalize path with grayscale luminance");
    
    // Add record_path method to DiagnosticFilm that accepts PathTrace
    // (The class binding is above, we just need the method)
}
