#include "prtree.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using T = int64_t; // is a temporary type of template. You can change it and
                   // recompile this.
const int B = 6;   // the number of children of tree.

PYBIND11_MODULE(PRTree, m) {
  m.doc() = R"pbdoc(
        INCOMPLETE Priority R-Tree
        Only supports for construct and find
        insert and delete are not supported.
    )pbdoc";

  py::class_<PRTree<T, B, 2>>(m, "PRTree2D")
      .def(py::init<py::array_t<T>, py::array_t<float>>(), R"pbdoc(
          Construct PRTree with init.
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
          parallel query with openmp
        )pbdoc")
      .def("erase", &PRTree<T, B, 2>::erase, R"pbdoc(
          Delete from prtree
        )pbdoc")
      .def("insert", &PRTree<T, B, 2>::insert, R"pbdoc(
          Insert one to prtree
        )pbdoc")
      .def("save", &PRTree<T, B, 2>::save, R"pbdoc(
          cereal save
        )pbdoc")
      .def("load", &PRTree<T, B, 2>::load, R"pbdoc(
          cereal load
        )pbdoc");

  py::class_<PRTree<T, B, 3>>(m, "PRTree3D")
      .def(py::init<py::array_t<T>, py::array_t<float>>(), R"pbdoc(
          Construct PRTree with init.
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
          parallel query with openmp
        )pbdoc")
      .def("erase", &PRTree<T, B, 3>::erase, R"pbdoc(
          Delete from prtree
        )pbdoc")
      .def("insert", &PRTree<T, B, 3>::insert, R"pbdoc(
          Insert one to prtree
        )pbdoc")
      .def("save", &PRTree<T, B, 3>::save, R"pbdoc(
          cereal save
        )pbdoc")
      .def("load", &PRTree<T, B, 3>::load, R"pbdoc(
          cereal load
        )pbdoc");

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
