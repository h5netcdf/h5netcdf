import importlib
import importlib.util

import pytest
from packaging.version import Version


def module_available(name: str) -> bool:
    """Return True if a module can be imported."""
    return importlib.util.find_spec(name) is not None


def _importorskip(
    modname: str, minversion: str | None = None
) -> tuple[bool, pytest.MarkDecorator]:
    try:
        mod = importlib.import_module(modname)
        has = True
        if minversion is not None:
            v = getattr(mod, "__version__", "999")
            if Version(v) < Version(minversion):
                raise ImportError("Minimum version not satisfied")
    except ImportError:
        has = False

    reason = f"requires {modname}"
    if minversion is not None:
        reason += f">={minversion}"
    func = pytest.mark.skipif(not has, reason=reason)
    return has, func


has_h5py, requires_h5py = _importorskip("h5py")
has_h5pyd, requires_h5pyd = _importorskip("h5pyd")
has_hsds, _ = _importorskip("h5sd")
has_h5py_ge_3_7_0, requires_h5py_ge_3_7_0 = _importorskip("h5py", minversion="3.7.0")
has_netCDF4, requires_netCDF4 = _importorskip("netCDF4")
has_netCDF4_ge_1_7_0, requires_netCDF4_ge_1_7_0 = _importorskip(
    "netCDF4", minversion="1.7.0"
)
has_pyfive, requires_pyfive = _importorskip("pyfive")


def _importorskip_h5py_ros3(has_h5py: bool):
    if not has_h5py:
        return has_h5py, pytest.mark.skipif(not has_h5py, reason="requires h5py")

    import h5py

    h5py_with_ros3 = h5py.get_config().ros3

    return h5py_with_ros3, pytest.mark.skipif(
        not h5py_with_ros3,
        reason="requires h5py with ros3 support",
    )


has_h5py_ros3, requires_h5py_ros3 = _importorskip_h5py_ros3(has_h5py)
