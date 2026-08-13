try:
    from . import _impl  # noqa: F401  # import intencional: test de disponibilidad
except ImportError:
    pass
