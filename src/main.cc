#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "prtree.h"

namespace py = pybind11;
using T = int64_t; // is a temporary type of template. You can change it and recompile this.


PYBIND11_MODULE(PRTree, m) {
    m.doc() = R"pbdoc(
        INCOMPLETE Priority R-Tree
        Only supports for construct and find
        insert and delete are not supported.
    )pbdoc";

    py::class_<PRTree<T>>(m, "PRTree")
    .def(py::init<py::array_t<T>, py::array_t<double>>(), R"pbdoc(
          Construct PRTree with init.
        )pbdoc")
        .def("find_all", &PRTree<T>::find_all, R"pbdoc(
          Find all indexes which has intersect with given bounding box.
        )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
