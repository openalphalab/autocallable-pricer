"""Build script for the local-vol-sub-stepped autocallable pricer.

Builds a single pybind11 extension `autocall_pricer_lv` exposing a
`substeps_per_interval` parameter (K Euler steps within each observation
interval). K=1 collapses to the single-step-per-observation path walker.

The AVX-512 TU is compiled with explicit AVX-512 flags so the rest of the
binary stays AVX2-compatible; runtime dispatch picks the best kernel.
"""
import os
import sys

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


def get_build_flags():
    if sys.platform.startswith("win"):
        common = ["/openmp", "/fp:fast", "/arch:AVX2"]
        return {"common": common, "link": []}
    common = [
        "-O3",
        "-march=native",
        "-mavx2",
        "-mfma",
        "-ffast-math",
        "-fno-math-errno",
        "-funroll-loops",
        "-Wall",
        "-Wno-unused-variable",
        "-Wno-unused-but-set-variable",
        "-Wno-unused-parameter",
    ]
    link = []
    if sys.platform != "darwin":
        common.append("-fopenmp")
        link.append("-fopenmp")
    return {"common": common, "link": link}


def get_avx512_flags():
    if sys.platform.startswith("win"):
        return ["/arch:AVX512"]
    return ["-mavx512f", "-mavx512dq", "-mavx512bw", "-mavx512vl"]


flags = get_build_flags()


class split_build_ext(build_ext):
    """Compile the AVX-512 TU with extra flags so the rest stays portable."""

    def build_extension(self, ext):
        sources = list(ext.sources)
        avx512 = [s for s in sources if os.path.basename(s) == "simd_kernel_prod_avx512.cpp"]
        common = [s for s in sources if s not in avx512]

        macros = ext.define_macros or []
        extra = list(ext.extra_compile_args or [])
        objects = []

        if common:
            objects.extend(self.compiler.compile(
                common, output_dir=self.build_temp, macros=macros,
                include_dirs=ext.include_dirs, debug=self.debug,
                extra_postargs=extra, depends=ext.depends,
            ))
        if avx512:
            objects.extend(self.compiler.compile(
                avx512, output_dir=self.build_temp, macros=macros,
                include_dirs=ext.include_dirs, debug=self.debug,
                extra_postargs=extra + get_avx512_flags(), depends=ext.depends,
            ))

        language = ext.language or self.compiler.detect_language(sources)
        ext_path = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(ext_path), exist_ok=True)
        self.compiler.link_shared_object(
            objects, ext_path,
            libraries=self.get_libraries(ext),
            library_dirs=ext.library_dirs,
            runtime_library_dirs=ext.runtime_library_dirs,
            extra_postargs=ext.extra_link_args,
            export_symbols=self.get_export_symbols(ext),
            debug=self.debug, build_temp=self.build_temp,
            target_lang=language,
        )


ext_prod = Pybind11Extension(
    "autocall_pricer_lv",
    [
        "autocall_pricer_prod.cpp",
        "simd_kernel_prod_avx2.cpp",
        "simd_kernel_prod_avx512.cpp",
    ],
    extra_compile_args=flags["common"],
    extra_link_args=flags["link"],
    cxx_std=17,
)

setup(
    name="autocall_pricer_lv",
    version="1.0.0",
    description="Production autocallable pricer + local-vol Euler sub-stepping",
    ext_modules=[ext_prod],
    cmdclass={"build_ext": split_build_ext},
    zip_safe=False,
)
