try:
    from ._impl import *
except ImportError:
    # Fallback para CFFI si el binario PyO3 no está listo o estamos en modo legacy
    __all__ = ["lib", "ffi"]
    import os
    from .ffi import ffi

    lib = ffi.dlopen(os.path.join(os.path.dirname(__file__), "lib_impl.so"))
