#include "prtree.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using T = int64_t; // is a temporary type of template. You can change it and
                   // recompile this.
const int B = 2;   // the number of children of tree.

PYBIND11_MODULE(PRTree, m)
{
  m.doc() = R"pbdoc(
        INCOMPLETE Priority R-Tree
        Only supports for construct and find
        insert and delete are not supported.
    )pbdoc";

  py::class_<PRTree<T, B, 2>>(m, "_PRTree2D")
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
          parallel query with multi-thread
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
      .def("insert", &PRTree<T, B, 2>::insert, R"pbdoc(
          Insert one to prtree
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
        )pbdoc");

  py::class_<PRTree<T, B, 3>>(m, "_PRTree3D")
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
          parallel query with multi-thread
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
      .def("insert", &PRTree<T, B, 3>::insert, R"pbdoc(
          Insert one to prtree
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
        )pbdoc");

  py::class_<PRTree<T, B, 4>>(m, "_PRTree4D")
      .def(py::init<py::array_t<T>, py::array_t<float>>(), R"pbdoc(
          Construct PRTree with init.
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
      .def("erase", &PRTree<T, B, 4>::erase, R"pbdoc(
          Delete from prtree
        )pbdoc")
      .def("set_obj", &PRTree<T, B, 4>::set_obj, R"pbdoc(
          Set string by index
        )pbdoc")
      .def("get_obj", &PRTree<T, B, 4>::get_obj, R"pbdoc(
          Get string by index
        )pbdoc")
      .def("insert", &PRTree<T, B, 4>::insert, R"pbdoc(
          Insert one to prtree
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
        )pbdoc");

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
