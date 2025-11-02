import os
import platform
import re
import subprocess
import sys
from distutils.version import LooseVersion
from multiprocessing import cpu_count

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

version = "v0.7.0"

sys.path.append("./tests")

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


def _requires_from_file(filename):
    return open(filename).read().splitlines()


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            cmake_version = LooseVersion(
                re.search(r"version\s*([\d.]+)", out.decode()).group(1)
            )
            if cmake_version < "3.1.0":
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        target_arch = os.getenv("CIBW_ARCHS", "")
        is_cross_compile_arm64 = platform.system() == "Windows" and target_arch == "ARM64"
        
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=" + extdir,
            "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=" + extdir,
            "-DPYBIND11_FINDPYTHON=ON",
            "-DBUILD_SHARED_LIBS=OFF",
        ]
        
        if not is_cross_compile_arm64:
            cmake_args += [
                "-DPYTHON_EXECUTABLE=" + sys.executable,
                "-DPython_EXECUTABLE=" + sys.executable,
            ]

        debug = os.getenv("DEBUG", 0) in {"1", "y", "yes", "true"}
        cfg = "Debug" if debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += [
                "-G", "Visual Studio 17 2022",
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir),
                "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir),
                "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir),
                "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
            ]
            if target_arch == "ARM64":
                cmake_args += ["-A", "ARM64"]
            elif target_arch == "AMD64" or target_arch == "x86_64":
                cmake_args += ["-A", "x64"]
            elif sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        elif platform.system() == "Darwin":
            cmake_args += [
                "-DCMAKE_BUILD_TYPE=" + cfg,
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir),
                "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
            ]
            
            archflags = os.getenv("ARCHFLAGS", "")
            if archflags:
                archs = []
                parts = archflags.split()
                for i, part in enumerate(parts):
                    if part == "-arch" and i + 1 < len(parts):
                        archs.append(parts[i + 1])
                if archs:
                    cmake_args.append("-DCMAKE_OSX_ARCHITECTURES=" + ";".join(archs))
            
            deployment_target = os.getenv("MACOSX_DEPLOYMENT_TARGET", "")
            if deployment_target:
                cmake_args.append("-DCMAKE_OSX_DEPLOYMENT_TARGET=" + deployment_target)
            
            build_args += ["--", "-j" + str(cpu_count())]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ["--", "-j" + str(cpu_count())]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        if not os.path.exists(extdir):
            os.makedirs(extdir)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )
        
        import glob
        import shutil
        ext_suffix = self.get_ext_filename(ext.name).split('.')[-1]
        if platform.system() == "Windows":
            pattern = f"PRTree*.pyd"
        else:
            pattern = f"PRTree*.so"
        
        expected_file = os.path.join(extdir, os.path.basename(self.get_ext_filename(ext.name)))
        if not os.path.exists(expected_file):
            found_files = glob.glob(os.path.join(self.build_temp, "**", pattern), recursive=True)
            if found_files:
                release_files = [f for f in found_files if "Release" in f or "release" in f]
                if release_files:
                    src_file = max(release_files, key=os.path.getmtime)
                else:
                    src_file = max(found_files, key=os.path.getmtime)
                
                print(f"Copying extension from {src_file} to {extdir}")
                shutil.copy2(src_file, extdir)
                
                if not os.path.exists(expected_file):
                    raise RuntimeError(
                        f"Failed to copy extension module to {expected_file}. "
                        f"Source was {src_file}"
                    )
            else:
                raise RuntimeError(
                    f"Could not find compiled extension module {pattern} in build tree. "
                    f"Build may have failed. Check build logs above."
                )


setup(
    name="python_prtree",
    version=version,
    license="MIT",
    description="Python implementation of Priority R-Tree",
    author="atksh",
    url="https://github.com/atksh/python_prtree",
    ext_modules=[CMakeExtension("python_prtree.PRTree")],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=_requires_from_file("requirements.txt"),
    package_dir={"": "src"},
    packages=find_packages("src"),
    test_suite="test_PRTree.suite",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="priority-rtree r-tree prtree rtree pybind11",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
)
