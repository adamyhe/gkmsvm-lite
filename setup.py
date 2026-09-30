"""Build configuration for C extensions."""

import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class OptionalBuildExt(build_ext):
    """Allow the build to succeed even if the C extension fails to compile."""

    def run(self):
        try:
            super().run()
        except Exception:
            self._warn_failed()

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception:
            self._warn_failed(ext.name)

    @staticmethod
    def _warn_failed(name="C SMO extension"):
        print(
            f"WARNING: Failed to build {name}. "
            "The SMO solver will fall back to the Python implementation.",
            file=sys.stderr,
        )


setup(
    ext_modules=[
        Extension(
            "gkmsvm._csmo",
            sources=["src/gkmsvm/_csmo.c"],
            extra_compile_args=["-O2"],
        ),
    ],
    cmdclass={"build_ext": OptionalBuildExt},
)
