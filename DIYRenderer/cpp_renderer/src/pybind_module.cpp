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
 *   import diyrenderer
 *   r = diyrenderer.Renderer()
 *   r.load_scene_json('{"meshes": [...]}')
 *   r.set_camera(0, 0, 5, 0, 0, -1, 0, 1, 0, 50)
 *   pixels = r.render_tile(0, 0, 800, 600, 800, 600, 16, 0, 8)
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "pybind_renderer.hpp"

namespace py = pybind11;

PYBIND11_MODULE(diyrenderer, m) {
    m.doc() = "DIY Path Tracer - Python bindings for the C++ renderer";
    
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
        // Python 的な機能
        // =====================================================================
        .def("__repr__", [](const PyRenderer& r) {
            return "<diyrenderer.Renderer scene_loaded=" + 
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
}
