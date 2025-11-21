import pytest


def _importorskip(modname, minversion=None):
    try:
        pytest.importorskip(modname, minversion=minversion)
        has = True
    except pytest.skip.Exception:
        has = False

    if minversion:
        reason = f"requires {modname}>={minversion}"
    else:
        reason = f"requires {modname}"

    skip_marker = pytest.mark.skipif(not has, reason=reason)
    return has, skip_marker


def _importorskip_h5py_ros3():
    has_h5py, skip_marker_h5py = _importorskip("h5py", minversion="3.7.0")

    if not has_h5py:
        return False, skip_marker_h5py

    import h5py

    h5py_with_ros3 = h5py.get_config().ros3

    return h5py_with_ros3, pytest.mark.skipif(
        not h5py_with_ros3,
        reason="requires h5py with ros3 support",
    )


has_h5py, requires_h5py = _importorskip("h5py")
has_h5pyd, requires_h5pyd = _importorskip("h5pyd")
has_hsds, _ = _importorskip("hsds")
has_h5py_ge_3_7_0, requires_h5py_ge_3_7_0 = _importorskip("h5py", minversion="3.7.0")
has_h5py_ros3, requires_h5py_ros3 = _importorskip_h5py_ros3()
has_netCDF4, requires_netCDF4 = _importorskip("netCDF4")
has_netCDF4_ge_1_7_0, requires_netCDF4_ge_1_7_0 = _importorskip(
    "netCDF4", minversion="1.7.0"
)
has_pyfive, requires_pyfive = _importorskip("pyfive", minversion="1.0.0")
