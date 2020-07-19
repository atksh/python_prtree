#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "prtree.h"

namespace py = pybind11;

using T = int64_t; // is a temporary type of template. You can change it and recompile this.
const int B =  6;  // the number of children of tree.


PYBIND11_MODULE(PRTree, m) {
    m.doc() = R"pbdoc(
        INCOMPLETE Priority R-Tree
        Only supports for construct and find
        insert and delete are not supported.
    )pbdoc";

    py::class_<PRTree<T, B>>(m, "PRTree")
    .def(py::init<py::array_t<T>, py::array_t<float>>(), R"pbdoc(
          Construct PRTree with init.
        )pbdoc")
        .def("query", &PRTree<T, B>::find_one, R"pbdoc(
          Find all indexes which has intersect with given bounding box.
        )pbdoc")
        .def("batch_query", &PRTree<T, B>::find_all, R"pbdoc(
          parallel query with openmp
        )pbdoc")
        .def("erase", &PRTree<T, B>::erase, R"pbdoc(
          Delete from prtree
        )pbdoc")
        .def("insert", &PRTree<T, B>::insert, R"pbdoc(
          Insert one to prtree
        )pbdoc")
    ;

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
