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
        INCOMPLETE Priority R-Tree
        Only supports for construct and find
        insert and delete are not supported.
    )pbdoc";

  py::class_<PRTree<T, B, 2>>(m, "_PRTree2D")
      .def(py::init<py::array_t<T>, py::array_t<double>>(), R"pbdoc(
          Construct PRTree with float64 input (float32 tree + double refinement for precision).
        )pbdoc")
      .def(py::init<py::array_t<T>, py::array_t<float>>(), R"pbdoc(
          Construct PRTree with float32 input (no refinement, pure float32 performance).
        )pbdoc")
      .def(py::init<>(), R"pbdoc(
          Construct PRTree with .
        )pbdoc")
      .def(py::init<std::string>(), R"pbdoc(
          Construct PRTree with load.
        )pbdoc")
      .def("query", &PRTree<T, B, 2>::find_one, R"pbdoc(
          Find all indexes which has intersect with given bounding box.
        )pbdoc")
      .def("batch_query", &PRTree<T, B, 2>::find_all, R"pbdoc(
          parallel query with multi-thread
        )pbdoc")
      .def("batch_query_array", &PRTree<T, B, 2>::find_all_array, R"pbdoc(
          parallel query with multi-thread with array output
        )pbdoc")
      .def("erase", &PRTree<T, B, 2>::erase, R"pbdoc(
          Delete from prtree
        )pbdoc")
      .def("set_obj", &PRTree<T, B, 2>::set_obj, R"pbdoc(
          Set string by index
        )pbdoc")
      .def("get_obj", &PRTree<T, B, 2>::get_obj, R"pbdoc(
          Get string by index
        )pbdoc")
      .def("insert",
           py::overload_cast<const T &, const py::array_t<float> &,
                             const std::optional<std::string>>(
               &PRTree<T, B, 2>::insert),
           py::arg("idx"), py::arg("bb"), py::arg("obj") = py::none(),
           R"pbdoc(
          Insert one to prtree (float32)
        )pbdoc")
      .def("insert",
           py::overload_cast<const T &, const py::array_t<double> &,
                             const std::optional<std::string>>(
               &PRTree<T, B, 2>::insert),
           py::arg("idx"), py::arg("bb"), py::arg("obj") = py::none(),
           R"pbdoc(
          Insert one to prtree (float64 with precision)
        )pbdoc")
      .def("save", &PRTree<T, B, 2>::save, R"pbdoc(
          cereal save
        )pbdoc")
      .def("load", &PRTree<T, B, 2>::load, R"pbdoc(
          cereal load
        )pbdoc")
      .def("rebuild", &PRTree<T, B, 2>::rebuild, R"pbdoc(
          rebuild prtree
        )pbdoc")
      .def("size", &PRTree<T, B, 2>::size, R"pbdoc(
          get n
        )pbdoc")
      .def("query_intersections", &PRTree<T, B, 2>::query_intersections,
           R"pbdoc(
          Find all pairs of intersecting AABBs.
          Returns a numpy array of shape (n_pairs, 2) where each row contains
          a pair of indices (i, j) with i < j representing intersecting AABBs.
        )pbdoc")
      .def("set_relative_epsilon", &PRTree<T, B, 2>::set_relative_epsilon,
           py::arg("epsilon"),
           R"pbdoc(
          Set relative epsilon for adaptive precision calculation.
        )pbdoc")
      .def("set_absolute_epsilon", &PRTree<T, B, 2>::set_absolute_epsilon,
           py::arg("epsilon"),
           R"pbdoc(
          Set absolute epsilon for precision calculation.
        )pbdoc")
      .def("set_adaptive_epsilon", &PRTree<T, B, 2>::set_adaptive_epsilon,
           py::arg("enabled"),
           R"pbdoc(
          Enable or disable adaptive epsilon calculation.
        )pbdoc")
      .def("set_subnormal_detection", &PRTree<T, B, 2>::set_subnormal_detection,
           py::arg("enabled"),
           R"pbdoc(
          Enable or disable subnormal number detection.
        )pbdoc")
      .def("get_relative_epsilon", &PRTree<T, B, 2>::get_relative_epsilon,
           R"pbdoc(
          Get current relative epsilon value.
        )pbdoc")
      .def("get_absolute_epsilon", &PRTree<T, B, 2>::get_absolute_epsilon,
           R"pbdoc(
          Get current absolute epsilon value.
        )pbdoc")
      .def("get_adaptive_epsilon", &PRTree<T, B, 2>::get_adaptive_epsilon,
           R"pbdoc(
          Check if adaptive epsilon is enabled.
        )pbdoc")
      .def("get_subnormal_detection", &PRTree<T, B, 2>::get_subnormal_detection,
           R"pbdoc(
          Check if subnormal detection is enabled.
        )pbdoc");

  py::class_<PRTree<T, B, 3>>(m, "_PRTree3D")
      .def(py::init<py::array_t<T>, py::array_t<double>>(), R"pbdoc(
          Construct PRTree with float64 input (float32 tree + double refinement for precision).
        )pbdoc")
      .def(py::init<py::array_t<T>, py::array_t<float>>(), R"pbdoc(
          Construct PRTree with float32 input (no refinement, pure float32 performance).
        )pbdoc")
      .def(py::init<>(), R"pbdoc(
          Construct PRTree with .
        )pbdoc")
      .def(py::init<std::string>(), R"pbdoc(
          Construct PRTree with load.
        )pbdoc")
      .def("query", &PRTree<T, B, 3>::find_one, R"pbdoc(
          Find all indexes which has intersect with given bounding box.
        )pbdoc")
      .def("batch_query", &PRTree<T, B, 3>::find_all, R"pbdoc(
          parallel query with multi-thread
        )pbdoc")
      .def("batch_query_array", &PRTree<T, B, 3>::find_all_array, R"pbdoc(
          parallel query with multi-thread with array output
        )pbdoc")
      .def("erase", &PRTree<T, B, 3>::erase, R"pbdoc(
          Delete from prtree
        )pbdoc")
      .def("set_obj", &PRTree<T, B, 3>::set_obj, R"pbdoc(
          Set string by index
        )pbdoc")
      .def("get_obj", &PRTree<T, B, 3>::get_obj, R"pbdoc(
          Get string by index
        )pbdoc")
      .def("insert",
           py::overload_cast<const T &, const py::array_t<float> &,
                             const std::optional<std::string>>(
               &PRTree<T, B, 3>::insert),
           py::arg("idx"), py::arg("bb"), py::arg("obj") = py::none(),
           R"pbdoc(
          Insert one to prtree (float32)
        )pbdoc")
      .def("insert",
           py::overload_cast<const T &, const py::array_t<double> &,
                             const std::optional<std::string>>(
               &PRTree<T, B, 3>::insert),
           py::arg("idx"), py::arg("bb"), py::arg("obj") = py::none(),
           R"pbdoc(
          Insert one to prtree (float64 with precision)
        )pbdoc")
      .def("save", &PRTree<T, B, 3>::save, R"pbdoc(
          cereal save
        )pbdoc")
      .def("load", &PRTree<T, B, 3>::load, R"pbdoc(
          cereal load
        )pbdoc")
      .def("rebuild", &PRTree<T, B, 3>::rebuild, R"pbdoc(
          rebuild prtree
        )pbdoc")
      .def("size", &PRTree<T, B, 3>::size, R"pbdoc(
          get n
        )pbdoc")
      .def("query_intersections", &PRTree<T, B, 3>::query_intersections,
           R"pbdoc(
          Find all pairs of intersecting AABBs.
          Returns a numpy array of shape (n_pairs, 2) where each row contains
          a pair of indices (i, j) with i < j representing intersecting AABBs.
        )pbdoc")
      .def("set_relative_epsilon", &PRTree<T, B, 3>::set_relative_epsilon,
           py::arg("epsilon"),
           R"pbdoc(
          Set relative epsilon for adaptive precision calculation.
        )pbdoc")
      .def("set_absolute_epsilon", &PRTree<T, B, 3>::set_absolute_epsilon,
           py::arg("epsilon"),
           R"pbdoc(
          Set absolute epsilon for precision calculation.
        )pbdoc")
      .def("set_adaptive_epsilon", &PRTree<T, B, 3>::set_adaptive_epsilon,
           py::arg("enabled"),
           R"pbdoc(
          Enable or disable adaptive epsilon calculation.
        )pbdoc")
      .def("set_subnormal_detection", &PRTree<T, B, 3>::set_subnormal_detection,
           py::arg("enabled"),
           R"pbdoc(
          Enable or disable subnormal number detection.
        )pbdoc")
      .def("get_relative_epsilon", &PRTree<T, B, 3>::get_relative_epsilon,
           R"pbdoc(
          Get current relative epsilon value.
        )pbdoc")
      .def("get_absolute_epsilon", &PRTree<T, B, 3>::get_absolute_epsilon,
           R"pbdoc(
          Get current absolute epsilon value.
        )pbdoc")
      .def("get_adaptive_epsilon", &PRTree<T, B, 3>::get_adaptive_epsilon,
           R"pbdoc(
          Check if adaptive epsilon is enabled.
        )pbdoc")
      .def("get_subnormal_detection", &PRTree<T, B, 3>::get_subnormal_detection,
           R"pbdoc(
          Check if subnormal detection is enabled.
        )pbdoc");

  py::class_<PRTree<T, B, 4>>(m, "_PRTree4D")
      .def(py::init<py::array_t<T>, py::array_t<double>>(), R"pbdoc(
          Construct PRTree with float64 input (float32 tree + double refinement for precision).
        )pbdoc")
      .def(py::init<py::array_t<T>, py::array_t<float>>(), R"pbdoc(
          Construct PRTree with float32 input (no refinement, pure float32 performance).
        )pbdoc")
      .def(py::init<>(), R"pbdoc(
          Construct PRTree with .
        )pbdoc")
      .def(py::init<std::string>(), R"pbdoc(
          Construct PRTree with load.
        )pbdoc")
      .def("query", &PRTree<T, B, 4>::find_one, R"pbdoc(
          Find all indexes which has intersect with given bounding box.
        )pbdoc")
      .def("batch_query", &PRTree<T, B, 4>::find_all, R"pbdoc(
          parallel query with multi-thread
        )pbdoc")
      .def("batch_query_array", &PRTree<T, B, 4>::find_all_array, R"pbdoc(
          parallel query with multi-thread with array output
        )pbdoc")
      .def("erase", &PRTree<T, B, 4>::erase, R"pbdoc(
          Delete from prtree
        )pbdoc")
      .def("set_obj", &PRTree<T, B, 4>::set_obj, R"pbdoc(
          Set string by index
        )pbdoc")
      .def("get_obj", &PRTree<T, B, 4>::get_obj, R"pbdoc(
          Get string by index
        )pbdoc")
      .def("insert",
           py::overload_cast<const T &, const py::array_t<float> &,
                             const std::optional<std::string>>(
               &PRTree<T, B, 4>::insert),
           py::arg("idx"), py::arg("bb"), py::arg("obj") = py::none(),
           R"pbdoc(
          Insert one to prtree (float32)
        )pbdoc")
      .def("insert",
           py::overload_cast<const T &, const py::array_t<double> &,
                             const std::optional<std::string>>(
               &PRTree<T, B, 4>::insert),
           py::arg("idx"), py::arg("bb"), py::arg("obj") = py::none(),
           R"pbdoc(
          Insert one to prtree (float64 with precision)
        )pbdoc")
      .def("save", &PRTree<T, B, 4>::save, R"pbdoc(
          cereal save
        )pbdoc")
      .def("load", &PRTree<T, B, 4>::load, R"pbdoc(
          cereal load
        )pbdoc")
      .def("rebuild", &PRTree<T, B, 4>::rebuild, R"pbdoc(
          rebuild prtree
        )pbdoc")
      .def("size", &PRTree<T, B, 4>::size, R"pbdoc(
          get n
        )pbdoc")
      .def("query_intersections", &PRTree<T, B, 4>::query_intersections,
           R"pbdoc(
          Find all pairs of intersecting AABBs.
          Returns a numpy array of shape (n_pairs, 2) where each row contains
          a pair of indices (i, j) with i < j representing intersecting AABBs.
        )pbdoc")
      .def("set_relative_epsilon", &PRTree<T, B, 4>::set_relative_epsilon,
           py::arg("epsilon"),
           R"pbdoc(
          Set relative epsilon for adaptive precision calculation.
        )pbdoc")
      .def("set_absolute_epsilon", &PRTree<T, B, 4>::set_absolute_epsilon,
           py::arg("epsilon"),
           R"pbdoc(
          Set absolute epsilon for precision calculation.
        )pbdoc")
      .def("set_adaptive_epsilon", &PRTree<T, B, 4>::set_adaptive_epsilon,
           py::arg("enabled"),
           R"pbdoc(
          Enable or disable adaptive epsilon calculation.
        )pbdoc")
      .def("set_subnormal_detection", &PRTree<T, B, 4>::set_subnormal_detection,
           py::arg("enabled"),
           R"pbdoc(
          Enable or disable subnormal number detection.
        )pbdoc")
      .def("get_relative_epsilon", &PRTree<T, B, 4>::get_relative_epsilon,
           R"pbdoc(
          Get current relative epsilon value.
        )pbdoc")
      .def("get_absolute_epsilon", &PRTree<T, B, 4>::get_absolute_epsilon,
           R"pbdoc(
          Get current absolute epsilon value.
        )pbdoc")
      .def("get_adaptive_epsilon", &PRTree<T, B, 4>::get_adaptive_epsilon,
           R"pbdoc(
          Check if adaptive epsilon is enabled.
        )pbdoc")
      .def("get_subnormal_detection", &PRTree<T, B, 4>::get_subnormal_detection,
           R"pbdoc(
          Check if subnormal detection is enabled.
        )pbdoc");

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
