#include "prtree/core/prtree.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using T = int64_t; // is a temporary type of template. You can change it and
                   // recompile this.
const int B = 8;   // the number of children of tree.

PYBIND11_MODULE(PRTree, m) {
  m.doc() = R"pbdoc(
        Priority R-Tree with native float32 and float64 precision support
    )pbdoc";

  // ========== 2D float32 version ==========
  py::class_<PRTree<T, B, 2, float>>(m, "_PRTree2D_float32")
      .def(py::init<py::array_t<T>, py::array_t<float>>(), R"pbdoc(
          Construct PRTree with float32 input (native float32 precision).
        )pbdoc")
      .def(py::init<>(), R"pbdoc(
          Construct empty PRTree.
        )pbdoc")
      .def(py::init<std::string>(), R"pbdoc(
          Construct PRTree from saved file.
        )pbdoc")
      .def("query", &PRTree<T, B, 2, float>::find_one, R"pbdoc(
          Find all indexes which intersect with given bounding box.
        )pbdoc")
      .def("batch_query", &PRTree<T, B, 2, float>::find_all, R"pbdoc(
          Parallel query with multi-thread.
        )pbdoc")
      .def("batch_query_array", &PRTree<T, B, 2, float>::find_all_array, R"pbdoc(
          Parallel query with multi-thread with array output.
        )pbdoc")
      .def("erase", &PRTree<T, B, 2, float>::erase, R"pbdoc(
          Delete from prtree.
        )pbdoc")
      .def("set_obj", &PRTree<T, B, 2, float>::set_obj, R"pbdoc(
          Set string by index.
        )pbdoc")
      .def("get_obj", &PRTree<T, B, 2, float>::get_obj, R"pbdoc(
          Get string by index.
        )pbdoc")
      .def("insert", &PRTree<T, B, 2, float>::insert,
           py::arg("idx"), py::arg("bb"), py::arg("obj") = py::none(),
           R"pbdoc(
          Insert one to prtree (float32).
        )pbdoc")
      .def("save", &PRTree<T, B, 2, float>::save, R"pbdoc(
          Save prtree to file.
        )pbdoc")
      .def("load", &PRTree<T, B, 2, float>::load, R"pbdoc(
          Load prtree from file.
        )pbdoc")
      .def("rebuild", &PRTree<T, B, 2, float>::rebuild, R"pbdoc(
          Rebuild prtree.
        )pbdoc")
      .def("size", &PRTree<T, B, 2, float>::size, R"pbdoc(
          Get number of elements.
        )pbdoc")
      .def("query_intersections", &PRTree<T, B, 2, float>::query_intersections, R"pbdoc(
          Find all pairs of intersecting AABBs.
        )pbdoc")
      .def("set_relative_epsilon", &PRTree<T, B, 2, float>::set_relative_epsilon,
           py::arg("epsilon"), R"pbdoc(
          Set relative epsilon for adaptive precision calculation.
        )pbdoc")
      .def("set_absolute_epsilon", &PRTree<T, B, 2, float>::set_absolute_epsilon,
           py::arg("epsilon"), R"pbdoc(
          Set absolute epsilon for precision calculation.
        )pbdoc")
      .def("set_adaptive_epsilon", &PRTree<T, B, 2, float>::set_adaptive_epsilon,
           py::arg("enabled"), R"pbdoc(
          Enable or disable adaptive epsilon calculation.
        )pbdoc")
      .def("set_subnormal_detection", &PRTree<T, B, 2, float>::set_subnormal_detection,
           py::arg("enabled"), R"pbdoc(
          Enable or disable subnormal number detection.
        )pbdoc")
      .def("get_relative_epsilon", &PRTree<T, B, 2, float>::get_relative_epsilon, R"pbdoc(
          Get current relative epsilon value.
        )pbdoc")
      .def("get_absolute_epsilon", &PRTree<T, B, 2, float>::get_absolute_epsilon, R"pbdoc(
          Get current absolute epsilon value.
        )pbdoc")
      .def("get_adaptive_epsilon", &PRTree<T, B, 2, float>::get_adaptive_epsilon, R"pbdoc(
          Check if adaptive epsilon is enabled.
        )pbdoc")
      .def("get_subnormal_detection", &PRTree<T, B, 2, float>::get_subnormal_detection, R"pbdoc(
          Check if subnormal detection is enabled.
        )pbdoc");

  // ========== 2D float64 version ==========
  py::class_<PRTree<T, B, 2, double>>(m, "_PRTree2D_float64")
      .def(py::init<py::array_t<T>, py::array_t<double>>(), R"pbdoc(
          Construct PRTree with float64 input (native double precision).
        )pbdoc")
      .def(py::init<>(), R"pbdoc(
          Construct empty PRTree.
        )pbdoc")
      .def(py::init<std::string>(), R"pbdoc(
          Construct PRTree from saved file.
        )pbdoc")
      .def("query", &PRTree<T, B, 2, double>::find_one, R"pbdoc(
          Find all indexes which intersect with given bounding box.
        )pbdoc")
      .def("batch_query", &PRTree<T, B, 2, double>::find_all, R"pbdoc(
          Parallel query with multi-thread.
        )pbdoc")
      .def("batch_query_array", &PRTree<T, B, 2, double>::find_all_array, R"pbdoc(
          Parallel query with multi-thread with array output.
        )pbdoc")
      .def("erase", &PRTree<T, B, 2, double>::erase, R"pbdoc(
          Delete from prtree.
        )pbdoc")
      .def("set_obj", &PRTree<T, B, 2, double>::set_obj, R"pbdoc(
          Set string by index.
        )pbdoc")
      .def("get_obj", &PRTree<T, B, 2, double>::get_obj, R"pbdoc(
          Get string by index.
        )pbdoc")
      .def("insert", &PRTree<T, B, 2, double>::insert,
           py::arg("idx"), py::arg("bb"), py::arg("obj") = py::none(),
           R"pbdoc(
          Insert one to prtree (float64).
        )pbdoc")
      .def("save", &PRTree<T, B, 2, double>::save, R"pbdoc(
          Save prtree to file.
        )pbdoc")
      .def("load", &PRTree<T, B, 2, double>::load, R"pbdoc(
          Load prtree from file.
        )pbdoc")
      .def("rebuild", &PRTree<T, B, 2, double>::rebuild, R"pbdoc(
          Rebuild prtree.
        )pbdoc")
      .def("size", &PRTree<T, B, 2, double>::size, R"pbdoc(
          Get number of elements.
        )pbdoc")
      .def("query_intersections", &PRTree<T, B, 2, double>::query_intersections, R"pbdoc(
          Find all pairs of intersecting AABBs.
        )pbdoc")
      .def("set_relative_epsilon", &PRTree<T, B, 2, double>::set_relative_epsilon,
           py::arg("epsilon"), R"pbdoc(
          Set relative epsilon for adaptive precision calculation.
        )pbdoc")
      .def("set_absolute_epsilon", &PRTree<T, B, 2, double>::set_absolute_epsilon,
           py::arg("epsilon"), R"pbdoc(
          Set absolute epsilon for precision calculation.
        )pbdoc")
      .def("set_adaptive_epsilon", &PRTree<T, B, 2, double>::set_adaptive_epsilon,
           py::arg("enabled"), R"pbdoc(
          Enable or disable adaptive epsilon calculation.
        )pbdoc")
      .def("set_subnormal_detection", &PRTree<T, B, 2, double>::set_subnormal_detection,
           py::arg("enabled"), R"pbdoc(
          Enable or disable subnormal number detection.
        )pbdoc")
      .def("get_relative_epsilon", &PRTree<T, B, 2, double>::get_relative_epsilon, R"pbdoc(
          Get current relative epsilon value.
        )pbdoc")
      .def("get_absolute_epsilon", &PRTree<T, B, 2, double>::get_absolute_epsilon, R"pbdoc(
          Get current absolute epsilon value.
        )pbdoc")
      .def("get_adaptive_epsilon", &PRTree<T, B, 2, double>::get_adaptive_epsilon, R"pbdoc(
          Check if adaptive epsilon is enabled.
        )pbdoc")
      .def("get_subnormal_detection", &PRTree<T, B, 2, double>::get_subnormal_detection, R"pbdoc(
          Check if subnormal detection is enabled.
        )pbdoc");

  // ========== 3D float32 version ==========
  py::class_<PRTree<T, B, 3, float>>(m, "_PRTree3D_float32")
      .def(py::init<py::array_t<T>, py::array_t<float>>(), R"pbdoc(
          Construct PRTree with float32 input (native float32 precision).
        )pbdoc")
      .def(py::init<>(), R"pbdoc(
          Construct empty PRTree.
        )pbdoc")
      .def(py::init<std::string>(), R"pbdoc(
          Construct PRTree from saved file.
        )pbdoc")
      .def("query", &PRTree<T, B, 3, float>::find_one, R"pbdoc(
          Find all indexes which intersect with given bounding box.
        )pbdoc")
      .def("batch_query", &PRTree<T, B, 3, float>::find_all, R"pbdoc(
          Parallel query with multi-thread.
        )pbdoc")
      .def("batch_query_array", &PRTree<T, B, 3, float>::find_all_array, R"pbdoc(
          Parallel query with multi-thread with array output.
        )pbdoc")
      .def("erase", &PRTree<T, B, 3, float>::erase, R"pbdoc(
          Delete from prtree.
        )pbdoc")
      .def("set_obj", &PRTree<T, B, 3, float>::set_obj, R"pbdoc(
          Set string by index.
        )pbdoc")
      .def("get_obj", &PRTree<T, B, 3, float>::get_obj, R"pbdoc(
          Get string by index.
        )pbdoc")
      .def("insert", &PRTree<T, B, 3, float>::insert,
           py::arg("idx"), py::arg("bb"), py::arg("obj") = py::none(),
           R"pbdoc(
          Insert one to prtree (float32).
        )pbdoc")
      .def("save", &PRTree<T, B, 3, float>::save, R"pbdoc(
          Save prtree to file.
        )pbdoc")
      .def("load", &PRTree<T, B, 3, float>::load, R"pbdoc(
          Load prtree from file.
        )pbdoc")
      .def("rebuild", &PRTree<T, B, 3, float>::rebuild, R"pbdoc(
          Rebuild prtree.
        )pbdoc")
      .def("size", &PRTree<T, B, 3, float>::size, R"pbdoc(
          Get number of elements.
        )pbdoc")
      .def("query_intersections", &PRTree<T, B, 3, float>::query_intersections, R"pbdoc(
          Find all pairs of intersecting AABBs.
        )pbdoc")
      .def("set_relative_epsilon", &PRTree<T, B, 3, float>::set_relative_epsilon,
           py::arg("epsilon"), R"pbdoc(
          Set relative epsilon for adaptive precision calculation.
        )pbdoc")
      .def("set_absolute_epsilon", &PRTree<T, B, 3, float>::set_absolute_epsilon,
           py::arg("epsilon"), R"pbdoc(
          Set absolute epsilon for precision calculation.
        )pbdoc")
      .def("set_adaptive_epsilon", &PRTree<T, B, 3, float>::set_adaptive_epsilon,
           py::arg("enabled"), R"pbdoc(
          Enable or disable adaptive epsilon calculation.
        )pbdoc")
      .def("set_subnormal_detection", &PRTree<T, B, 3, float>::set_subnormal_detection,
           py::arg("enabled"), R"pbdoc(
          Enable or disable subnormal number detection.
        )pbdoc")
      .def("get_relative_epsilon", &PRTree<T, B, 3, float>::get_relative_epsilon, R"pbdoc(
          Get current relative epsilon value.
        )pbdoc")
      .def("get_absolute_epsilon", &PRTree<T, B, 3, float>::get_absolute_epsilon, R"pbdoc(
          Get current absolute epsilon value.
        )pbdoc")
      .def("get_adaptive_epsilon", &PRTree<T, B, 3, float>::get_adaptive_epsilon, R"pbdoc(
          Check if adaptive epsilon is enabled.
        )pbdoc")
      .def("get_subnormal_detection", &PRTree<T, B, 3, float>::get_subnormal_detection, R"pbdoc(
          Check if subnormal detection is enabled.
        )pbdoc");

  // ========== 3D float64 version ==========
  py::class_<PRTree<T, B, 3, double>>(m, "_PRTree3D_float64")
      .def(py::init<py::array_t<T>, py::array_t<double>>(), R"pbdoc(
          Construct PRTree with float64 input (native double precision).
        )pbdoc")
      .def(py::init<>(), R"pbdoc(
          Construct empty PRTree.
        )pbdoc")
      .def(py::init<std::string>(), R"pbdoc(
          Construct PRTree from saved file.
        )pbdoc")
      .def("query", &PRTree<T, B, 3, double>::find_one, R"pbdoc(
          Find all indexes which intersect with given bounding box.
        )pbdoc")
      .def("batch_query", &PRTree<T, B, 3, double>::find_all, R"pbdoc(
          Parallel query with multi-thread.
        )pbdoc")
      .def("batch_query_array", &PRTree<T, B, 3, double>::find_all_array, R"pbdoc(
          Parallel query with multi-thread with array output.
        )pbdoc")
      .def("erase", &PRTree<T, B, 3, double>::erase, R"pbdoc(
          Delete from prtree.
        )pbdoc")
      .def("set_obj", &PRTree<T, B, 3, double>::set_obj, R"pbdoc(
          Set string by index.
        )pbdoc")
      .def("get_obj", &PRTree<T, B, 3, double>::get_obj, R"pbdoc(
          Get string by index.
        )pbdoc")
      .def("insert", &PRTree<T, B, 3, double>::insert,
           py::arg("idx"), py::arg("bb"), py::arg("obj") = py::none(),
           R"pbdoc(
          Insert one to prtree (float64).
        )pbdoc")
      .def("save", &PRTree<T, B, 3, double>::save, R"pbdoc(
          Save prtree to file.
        )pbdoc")
      .def("load", &PRTree<T, B, 3, double>::load, R"pbdoc(
          Load prtree from file.
        )pbdoc")
      .def("rebuild", &PRTree<T, B, 3, double>::rebuild, R"pbdoc(
          Rebuild prtree.
        )pbdoc")
      .def("size", &PRTree<T, B, 3, double>::size, R"pbdoc(
          Get number of elements.
        )pbdoc")
      .def("query_intersections", &PRTree<T, B, 3, double>::query_intersections, R"pbdoc(
          Find all pairs of intersecting AABBs.
        )pbdoc")
      .def("set_relative_epsilon", &PRTree<T, B, 3, double>::set_relative_epsilon,
           py::arg("epsilon"), R"pbdoc(
          Set relative epsilon for adaptive precision calculation.
        )pbdoc")
      .def("set_absolute_epsilon", &PRTree<T, B, 3, double>::set_absolute_epsilon,
           py::arg("epsilon"), R"pbdoc(
          Set absolute epsilon for precision calculation.
        )pbdoc")
      .def("set_adaptive_epsilon", &PRTree<T, B, 3, double>::set_adaptive_epsilon,
           py::arg("enabled"), R"pbdoc(
          Enable or disable adaptive epsilon calculation.
        )pbdoc")
      .def("set_subnormal_detection", &PRTree<T, B, 3, double>::set_subnormal_detection,
           py::arg("enabled"), R"pbdoc(
          Enable or disable subnormal number detection.
        )pbdoc")
      .def("get_relative_epsilon", &PRTree<T, B, 3, double>::get_relative_epsilon, R"pbdoc(
          Get current relative epsilon value.
        )pbdoc")
      .def("get_absolute_epsilon", &PRTree<T, B, 3, double>::get_absolute_epsilon, R"pbdoc(
          Get current absolute epsilon value.
        )pbdoc")
      .def("get_adaptive_epsilon", &PRTree<T, B, 3, double>::get_adaptive_epsilon, R"pbdoc(
          Check if adaptive epsilon is enabled.
        )pbdoc")
      .def("get_subnormal_detection", &PRTree<T, B, 3, double>::get_subnormal_detection, R"pbdoc(
          Check if subnormal detection is enabled.
        )pbdoc");

  // ========== 4D float32 version ==========
  py::class_<PRTree<T, B, 4, float>>(m, "_PRTree4D_float32")
      .def(py::init<py::array_t<T>, py::array_t<float>>(), R"pbdoc(
          Construct PRTree with float32 input (native float32 precision).
        )pbdoc")
      .def(py::init<>(), R"pbdoc(
          Construct empty PRTree.
        )pbdoc")
      .def(py::init<std::string>(), R"pbdoc(
          Construct PRTree from saved file.
        )pbdoc")
      .def("query", &PRTree<T, B, 4, float>::find_one, R"pbdoc(
          Find all indexes which intersect with given bounding box.
        )pbdoc")
      .def("batch_query", &PRTree<T, B, 4, float>::find_all, R"pbdoc(
          Parallel query with multi-thread.
        )pbdoc")
      .def("batch_query_array", &PRTree<T, B, 4, float>::find_all_array, R"pbdoc(
          Parallel query with multi-thread with array output.
        )pbdoc")
      .def("erase", &PRTree<T, B, 4, float>::erase, R"pbdoc(
          Delete from prtree.
        )pbdoc")
      .def("set_obj", &PRTree<T, B, 4, float>::set_obj, R"pbdoc(
          Set string by index.
        )pbdoc")
      .def("get_obj", &PRTree<T, B, 4, float>::get_obj, R"pbdoc(
          Get string by index.
        )pbdoc")
      .def("insert", &PRTree<T, B, 4, float>::insert,
           py::arg("idx"), py::arg("bb"), py::arg("obj") = py::none(),
           R"pbdoc(
          Insert one to prtree (float32).
        )pbdoc")
      .def("save", &PRTree<T, B, 4, float>::save, R"pbdoc(
          Save prtree to file.
        )pbdoc")
      .def("load", &PRTree<T, B, 4, float>::load, R"pbdoc(
          Load prtree from file.
        )pbdoc")
      .def("rebuild", &PRTree<T, B, 4, float>::rebuild, R"pbdoc(
          Rebuild prtree.
        )pbdoc")
      .def("size", &PRTree<T, B, 4, float>::size, R"pbdoc(
          Get number of elements.
        )pbdoc")
      .def("query_intersections", &PRTree<T, B, 4, float>::query_intersections, R"pbdoc(
          Find all pairs of intersecting AABBs.
        )pbdoc")
      .def("set_relative_epsilon", &PRTree<T, B, 4, float>::set_relative_epsilon,
           py::arg("epsilon"), R"pbdoc(
          Set relative epsilon for adaptive precision calculation.
        )pbdoc")
      .def("set_absolute_epsilon", &PRTree<T, B, 4, float>::set_absolute_epsilon,
           py::arg("epsilon"), R"pbdoc(
          Set absolute epsilon for precision calculation.
        )pbdoc")
      .def("set_adaptive_epsilon", &PRTree<T, B, 4, float>::set_adaptive_epsilon,
           py::arg("enabled"), R"pbdoc(
          Enable or disable adaptive epsilon calculation.
        )pbdoc")
      .def("set_subnormal_detection", &PRTree<T, B, 4, float>::set_subnormal_detection,
           py::arg("enabled"), R"pbdoc(
          Enable or disable subnormal number detection.
        )pbdoc")
      .def("get_relative_epsilon", &PRTree<T, B, 4, float>::get_relative_epsilon, R"pbdoc(
          Get current relative epsilon value.
        )pbdoc")
      .def("get_absolute_epsilon", &PRTree<T, B, 4, float>::get_absolute_epsilon, R"pbdoc(
          Get current absolute epsilon value.
        )pbdoc")
      .def("get_adaptive_epsilon", &PRTree<T, B, 4, float>::get_adaptive_epsilon, R"pbdoc(
          Check if adaptive epsilon is enabled.
        )pbdoc")
      .def("get_subnormal_detection", &PRTree<T, B, 4, float>::get_subnormal_detection, R"pbdoc(
          Check if subnormal detection is enabled.
        )pbdoc");

  // ========== 4D float64 version ==========
  py::class_<PRTree<T, B, 4, double>>(m, "_PRTree4D_float64")
      .def(py::init<py::array_t<T>, py::array_t<double>>(), R"pbdoc(
          Construct PRTree with float64 input (native double precision).
        )pbdoc")
      .def(py::init<>(), R"pbdoc(
          Construct empty PRTree.
        )pbdoc")
      .def(py::init<std::string>(), R"pbdoc(
          Construct PRTree from saved file.
        )pbdoc")
      .def("query", &PRTree<T, B, 4, double>::find_one, R"pbdoc(
          Find all indexes which intersect with given bounding box.
        )pbdoc")
      .def("batch_query", &PRTree<T, B, 4, double>::find_all, R"pbdoc(
          Parallel query with multi-thread.
        )pbdoc")
      .def("batch_query_array", &PRTree<T, B, 4, double>::find_all_array, R"pbdoc(
          Parallel query with multi-thread with array output.
        )pbdoc")
      .def("erase", &PRTree<T, B, 4, double>::erase, R"pbdoc(
          Delete from prtree.
        )pbdoc")
      .def("set_obj", &PRTree<T, B, 4, double>::set_obj, R"pbdoc(
          Set string by index.
        )pbdoc")
      .def("get_obj", &PRTree<T, B, 4, double>::get_obj, R"pbdoc(
          Get string by index.
        )pbdoc")
      .def("insert", &PRTree<T, B, 4, double>::insert,
           py::arg("idx"), py::arg("bb"), py::arg("obj") = py::none(),
           R"pbdoc(
          Insert one to prtree (float64).
        )pbdoc")
      .def("save", &PRTree<T, B, 4, double>::save, R"pbdoc(
          Save prtree to file.
        )pbdoc")
      .def("load", &PRTree<T, B, 4, double>::load, R"pbdoc(
          Load prtree from file.
        )pbdoc")
      .def("rebuild", &PRTree<T, B, 4, double>::rebuild, R"pbdoc(
          Rebuild prtree.
        )pbdoc")
      .def("size", &PRTree<T, B, 4, double>::size, R"pbdoc(
          Get number of elements.
        )pbdoc")
      .def("query_intersections", &PRTree<T, B, 4, double>::query_intersections, R"pbdoc(
          Find all pairs of intersecting AABBs.
        )pbdoc")
      .def("set_relative_epsilon", &PRTree<T, B, 4, double>::set_relative_epsilon,
           py::arg("epsilon"), R"pbdoc(
          Set relative epsilon for adaptive precision calculation.
        )pbdoc")
      .def("set_absolute_epsilon", &PRTree<T, B, 4, double>::set_absolute_epsilon,
           py::arg("epsilon"), R"pbdoc(
          Set absolute epsilon for precision calculation.
        )pbdoc")
      .def("set_adaptive_epsilon", &PRTree<T, B, 4, double>::set_adaptive_epsilon,
           py::arg("enabled"), R"pbdoc(
          Enable or disable adaptive epsilon calculation.
        )pbdoc")
      .def("set_subnormal_detection", &PRTree<T, B, 4, double>::set_subnormal_detection,
           py::arg("enabled"), R"pbdoc(
          Enable or disable subnormal number detection.
        )pbdoc")
      .def("get_relative_epsilon", &PRTree<T, B, 4, double>::get_relative_epsilon, R"pbdoc(
          Get current relative epsilon value.
        )pbdoc")
      .def("get_absolute_epsilon", &PRTree<T, B, 4, double>::get_absolute_epsilon, R"pbdoc(
          Get current absolute epsilon value.
        )pbdoc")
      .def("get_adaptive_epsilon", &PRTree<T, B, 4, double>::get_adaptive_epsilon, R"pbdoc(
          Check if adaptive epsilon is enabled.
        )pbdoc")
      .def("get_subnormal_detection", &PRTree<T, B, 4, double>::get_subnormal_detection, R"pbdoc(
          Check if subnormal detection is enabled.
        )pbdoc");

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
