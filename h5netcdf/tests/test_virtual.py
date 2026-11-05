"""Native VDS creation, NetCDF consumers, and creation failure recovery."""

import hashlib

import numpy as np
import pytest

import h5netcdf
from h5netcdf.core import Variable
from h5netcdf.dimensions import Dimension
from h5netcdf.utils import CompatibilityError

h5py = pytest.importorskip("h5py")


def make_layout(names, name, shape, dtype):
    layout = h5py.VirtualLayout((len(names),) + shape, dtype)
    for index, filename in enumerate(names):
        layout[index] = h5py.VirtualSource(filename, name, shape=shape)
    return layout


@pytest.mark.parametrize(
    "reader", ["h5netcdf", "netCDF4", "xarray-h5netcdf", "xarray-netcdf4"]
)
def test_virtual_timelapse(tmp_path, reader):
    """Two virtual image arrays, virtual time, direct data and explicit metadata."""
    paths = []
    images = np.arange(72, dtype="uint16").reshape(3, 2, 3, 4)
    masks = np.arange(36, dtype="int16").reshape(3, 3, 4)
    times = np.array([0, 30, 60], dtype="int32")
    for index in range(3):
        path = tmp_path / f"timepoint_{index}.nc"
        with h5netcdf.File(path, "w") as f:
            f.dimensions = {"z": 2, "y": 3, "x": 4, "time": 1}
            f.create_variable("image", ("z", "y", "x"), data=images[index])
            f.create_variable("mask", ("y", "x"), data=masks[index])
            f.create_variable("time", ("time",), data=times[index : index + 1])
        paths.append(path)
    hashes = [hashlib.sha256(p.read_bytes()).digest() for p in paths]
    with h5netcdf.File(tmp_path / "combined.nc", "w") as f:
        f.dimensions = {"time": 3, "z": 2, "y": 3, "x": 4}
        # Also exercise replacement of a dimension scale after dependents exist.
        f.create_variable("direct", ("time",), data=np.array([7, 8, 9], dtype="int32"))
        for key, values, dims in [
            ("image", images, ("time", "z", "y", "x")),
            ("mask", masks, ("time", "y", "x")),
        ]:
            layout = make_layout(
                [p.name for p in paths], key, values.shape[1:], values.dtype
            )
            var = f.create_virtual_variable(key, dims, layout)
            var.attrs["description"] = f"synthetic {key}"
        layout = h5py.VirtualLayout((3,), "int32")
        for index, path in enumerate(paths):
            layout[index : index + 1] = h5py.VirtualSource(
                path.name, "time", shape=(1,)
            )
        f.create_virtual_variable("time", ("time",), layout).attrs["units"] = "seconds"
        for key, size in [("z", 2), ("y", 3), ("x", 4)]:
            f.create_variable(key, (key,), data=np.arange(size, dtype="int32"))
        f.attrs["title"] = "synthetic timelapse"
    assert hashes == [hashlib.sha256(p.read_bytes()).digest() for p in paths]

    # Move every file to a new location, then read from an unrelated working directory.
    moved = tmp_path / "moved"
    moved.mkdir()
    for path in [*paths, tmp_path / "combined.nc"]:
        path.rename(moved / path.name)
    output = moved / "combined.nc"
    expected = {
        "image": (("time", "z", "y", "x"), images),
        "mask": (("time", "y", "x"), masks),
        "time": (("time",), times),
        "direct": (("time",), [7, 8, 9]),
        **{
            key: ((key,), np.arange(size))
            for key, size in [("z", 2), ("y", 3), ("x", 4)]
        },
    }
    if reader.startswith("xarray-"):
        xr = pytest.importorskip("xarray")
        engine = reader.split("-", 1)[1]
        if engine == "netcdf4":
            pytest.importorskip("netCDF4")
        with xr.open_dataset(output, engine=engine) as f:
            assert set(f.coords) == {"time", "z", "y", "x"}
            assert dict(f.sizes) == {"time": 3, "z": 2, "y": 3, "x": 4}
            assert f.time.attrs == {"units": "seconds"}
            assert f.attrs == {"title": "synthetic timelapse"}
            for key, (dims, values) in expected.items():
                assert f[key].dims == dims
                np.testing.assert_array_equal(f[key].values, values)
                if key in ("image", "mask"):
                    assert f[key].attrs == {"description": f"synthetic {key}"}
    else:
        mod = pytest.importorskip(reader)
        opener = mod.File if reader == "h5netcdf" else mod.Dataset
        with opener(output) as f:
            if reader == "netCDF4":
                f.set_auto_mask(False)
            attrs = f.attrs if reader == "h5netcdf" else f.__dict__
            assert dict(attrs) == {"title": "synthetic timelapse"}
            assert {key: len(dim) for key, dim in f.dimensions.items()} == {
                "time": 3,
                "z": 2,
                "y": 3,
                "x": 4,
            }
            for key, (dims, values) in expected.items():
                assert f[key].dimensions == dims
                np.testing.assert_array_equal(f[key][:], values)
            attrs = f["image"].attrs if reader == "h5netcdf" else f["image"].__dict__
            assert dict(attrs) == {"description": "synthetic image"}
            attrs = f["time"].attrs if reader == "h5netcdf" else f["time"].__dict__
            assert dict(attrs) == {"units": "seconds"}
    with h5py.File(output) as f:
        for key in ("image", "mask", "time"):
            assert f[key].is_virtual
            assert f[key].id.get_storage_size() == 0
            assert {source.file_name for source in f[key].virtual_sources()} == {
                p.name for p in paths
            }
    assert hashes == [
        hashlib.sha256((moved / p.name).read_bytes()).digest() for p in paths
    ]


@pytest.mark.parametrize("fillvalue", [None, -99])
def test_virtual_missing_source_and_unmapped_region(tmp_path, fillvalue):
    path = tmp_path / "source.nc"
    with h5netcdf.File(path, "w") as f:
        f.create_variable("values", ("x",), data=np.array([5, 6], dtype="int16"))
    layout = h5py.VirtualLayout((3, 2), "int16")
    layout[0] = h5py.VirtualSource(path.name, "values", shape=(2,))
    layout[1] = h5py.VirtualSource("missing.nc", "values", shape=(2,))
    with h5netcdf.File(tmp_path / "virtual.nc", "w") as f:
        f.dimensions = {"time": 3, "x": 2}
        v = f.create_virtual_variable("values", ("time", "x"), layout, fillvalue)
        np.testing.assert_array_equal(
            v[:], [[5, 6], [fillvalue or 0] * 2, [fillvalue or 0] * 2]
        )
        assert dict(v.attrs) == (
            {} if fillvalue is None else {"_FillValue": np.int16(fillvalue)}
        )
    netcdf4 = pytest.importorskip("netCDF4")
    with netcdf4.Dataset(tmp_path / "virtual.nc") as f:
        f.set_auto_mask(False)
        np.testing.assert_array_equal(
            f["values"][:], [[5, 6], [fillvalue or 0] * 2, [fillvalue or 0] * 2]
        )


@pytest.mark.parametrize(
    "dtype",
    [
        "int8",
        "uint8",
        "int16",
        ">i2",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float32",
        ">f4",
        "float64",
    ],
)
def test_virtual_numeric_types(tmp_path, dtype):
    with h5netcdf.File(tmp_path / "source.nc", "w") as f:
        f.create_variable("values", ("x",), data=np.array([1, 2], dtype=dtype))
    layout = h5py.VirtualLayout((2,), dtype)
    layout[:] = h5py.VirtualSource("source.nc", "values", shape=(2,))
    with h5netcdf.File(tmp_path / "virtual.nc", "w") as f:
        f.dimensions["x"] = 2
        f.create_virtual_variable("values", ("x",), layout)
    netcdf4 = pytest.importorskip("netCDF4")
    with netcdf4.Dataset(tmp_path / "virtual.nc") as f:
        np.testing.assert_array_equal(f["values"][:], [1, 2])


def test_virtual_group_paths_and_noncoordinate_name(tmp_path):
    with h5netcdf.File(tmp_path / "source.nc", "w") as f:
        f.create_variable("g/values", ("y", "y"), data=np.arange(4).reshape(2, 2))
    layout = h5py.VirtualLayout((2, 2), "int64")
    layout[:] = h5py.VirtualSource("source.nc", "g/values", shape=(2, 2))
    with h5netcdf.File(tmp_path / "virtual.nc", "w") as f:
        f.dimensions["y"] = 2
        child = f.create_group("g")
        child.dimensions["values"] = 3
        v = f.create_virtual_variable("/g/values", ("y", "y"), layout)
        assert isinstance(v, h5netcdf.Variable)
        np.testing.assert_array_equal(child["values"][:], np.arange(4).reshape(2, 2))
        assert v.dimensions == ("y", "y")
        assert "_nc4_non_coord_values" in child._h5group
        with pytest.raises(ValueError, match="already exists"):
            child.create_virtual_variable("values", ("y", "y"), layout)
    netcdf4 = pytest.importorskip("netCDF4")
    with netcdf4.Dataset(tmp_path / "virtual.nc") as f:
        assert f["g/values"].dimensions == ("y", "y")
        np.testing.assert_array_equal(f["g/values"][:], np.arange(4).reshape(2, 2))


@pytest.mark.parametrize(
    "dtype",
    [
        "bool",
        "complex64",
        "float16",
        "S1",
        "O",
        np.dtype([("x", "i4")]),
        h5py.enum_dtype({"a": 1}, basetype="i4"),
    ],
)
def test_virtual_rejects_unsupported_dtype(tmp_path, dtype):
    with h5netcdf.File(tmp_path / "virtual.nc", "w") as f:
        f.dimensions["x"] = 2
        with pytest.raises(TypeError, match="primitive netCDF numeric dtype"):
            f.create_virtual_variable("values", ("x",), h5py.VirtualLayout((2,), dtype))
        assert list(f._h5group) == ["x"]
        assert not f.variables


@pytest.mark.parametrize(
    "failure",
    [
        "layout_type",
        "shape",
        "missing_dimension",
        "unlimited",
        "maxshape",
        "fillvalue",
        "fill_array",
        "name",
        "path",
        "coordinate",
        "inherited_coordinate",
    ],
)
def test_virtual_validation_leaves_file_unchanged(tmp_path, failure):
    with h5netcdf.File(tmp_path / "virtual.nc", "w") as f:
        f.dimensions = {"x": 2, "y": 3, "unlimited": None}
        child = f.create_group("child")
        args = dict(
            name="values", dimensions=("x",), layout=h5py.VirtualLayout((2,), "int16")
        )
        group = f
        error = ValueError
        if failure == "layout_type":
            args["layout"] = object()
            error = TypeError
        elif failure == "shape":
            args["dimensions"] = ("y",)
        elif failure == "missing_dimension":
            args["dimensions"] = ("missing",)
            error = KeyError
        elif failure == "unlimited":
            args["dimensions"] = ("unlimited",)
        elif failure == "maxshape":
            args["layout"] = h5py.VirtualLayout((2,), "int16", maxshape=(None,))
        elif failure == "fillvalue":
            args["fillvalue"] = "not-an-integer"
        elif failure == "fill_array":
            args["fillvalue"] = [1, 2]
        elif failure == "name":
            args["name"] = ""
        elif failure == "path":
            args["name"] = "missing/values"
            error = KeyError
        elif failure == "coordinate":
            args.update(
                name="x",
                dimensions=("x", "y"),
                layout=h5py.VirtualLayout((2, 3), "int16"),
            )
        elif failure == "inherited_coordinate":
            args["name"] = "x"
            group = child
        before = {name: obj.id for name, obj in f._h5group.items()}
        with pytest.raises(error):
            group.create_virtual_variable(**args)
        assert {name: obj.id for name, obj in f._h5group.items()} == before
        assert not f.variables
        assert not child.variables
        assert not child.dimensions


@pytest.mark.parametrize("backend", ["h5pyd", "pyfive"])
def test_virtual_backend_gate(tmp_path, monkeypatch, backend):
    # Gate the unsupported backend before any backend operation, without HSDS.
    with h5netcdf.File(tmp_path / "virtual.nc", "w") as f:
        with monkeypatch.context() as patch:
            patch.setattr(f, "_backend", backend)
            with pytest.raises(NotImplementedError, match="h5py backend"):
                f.create_virtual_variable("values", (), object())
        assert len(f._h5group) == 0


def test_virtual_readonly(tmp_path):
    path = tmp_path / "virtual.nc"
    with h5netcdf.File(path, "w"):
        pass
    with h5netcdf.File(path) as f:
        with pytest.raises(RuntimeError, match="read only"):
            f.create_virtual_variable("values", (), h5py.VirtualLayout((), "int16"))


@pytest.mark.parametrize("name", ["values", "x"])
def test_virtual_rejects_unlimited_mapping_inside_fixed_layout(tmp_path, name):
    source = h5py.VirtualSource("growing.nc", "values", shape=(2,), maxshape=(None,))
    layout = h5py.VirtualLayout((2,), "int16")
    layout[: h5py.h5s.UNLIMITED] = source[: h5py.h5s.UNLIMITED]
    assert layout.maxshape is None
    with h5netcdf.File(tmp_path / "virtual.nc", "w") as f:
        f.dimensions["x"] = 2
        f.create_variable("existing", ("x",), data=[1, 2])
        old_scale = f._h5group["x"]
        with pytest.raises(ValueError, match="unlimited mapping selections"):
            f.create_virtual_variable(name, ("x",), layout)
        assert set(f._h5group) == {"x", "existing"}
        assert set(f.variables) == {"existing"}
        assert f._h5group["x"].id == old_scale.id
        assert h5py.h5ds.is_attached(f._h5group["existing"].id, old_scale.id, 0)


def test_virtual_fixed_selection_from_growing_source(tmp_path):
    source_path = tmp_path / "growing.nc"
    with h5py.File(source_path, "w") as f:
        f.create_dataset("values", data=[1, 2], maxshape=(None,))
    layout = h5py.VirtualLayout((2,), "int64")
    layout[:] = h5py.VirtualSource(
        source_path.name, "values", shape=(2,), maxshape=(None,)
    )[:]
    output = tmp_path / "virtual.nc"
    with h5netcdf.File(output, "w") as f:
        f.dimensions["x"] = 2
        f.create_virtual_variable("values", ("x",), layout)
    with h5py.File(source_path, "a") as f:
        f["values"].resize((3,))
        f["values"][2] = 99
    with h5netcdf.File(output) as f:
        np.testing.assert_array_equal(f["values"][:], [1, 2])
        assert f["values"].shape == (2,)


@pytest.mark.parametrize("mapped", [False, True])
def test_virtual_reused_layout_fill_values_are_independent(tmp_path, mapped):
    layout = h5py.VirtualLayout((2,), "int16")
    if mapped:
        layout[:] = h5py.VirtualSource("missing.nc", "values", shape=(2,))
    with h5netcdf.File(tmp_path / "virtual.nc", "w") as f:
        f.dimensions["x"] = 2
        a = f.create_virtual_variable("a", ("x",), layout, -99)
        b = f.create_virtual_variable("b", ("x",), layout)
        c = f.create_virtual_variable("c", ("x",), layout, 7)
        np.testing.assert_array_equal(a[:], [-99, -99])
        np.testing.assert_array_equal(b[:], [0, 0])
        np.testing.assert_array_equal(c[:], [7, 7])
        assert dict(b.attrs) == {}


@pytest.mark.parametrize(
    "stage",
    ["_attach_dim_scales", "_attach_coords", "_ensure_dim_id", "_add_fillvalue"],
)
def test_virtual_failure_removes_variable_and_scale_references(
    tmp_path, monkeypatch, stage
):
    with h5netcdf.File(tmp_path / "virtual.nc", "w") as f:
        f.dimensions = {"x": 2, "y": 3}
        original = getattr(Variable, stage)

        def fail(variable, *args):
            if stage == "_attach_dim_scales":
                variable._h5ds.dims[0].attach_scale(f._h5group["x"])
            else:
                original(variable, *args)
            raise RuntimeError("injected setup failure")

        with monkeypatch.context() as patch:
            patch.setattr(Variable, stage, fail)
            with pytest.raises(RuntimeError, match="injected"):
                f.create_virtual_variable(
                    "values", ("x", "y"), h5py.VirtualLayout((2, 3), "i2"), -1
                )
        assert set(f._h5group) == {"x", "y"}
        assert not f.variables
        for key in ("x", "y"):
            assert len(f._h5group[key].attrs.get("REFERENCE_LIST", [])) == 0
        f.create_virtual_variable(
            "values", ("x", "y"), h5py.VirtualLayout((2, 3), "i2"), -1
        )
    with h5netcdf.File(tmp_path / "virtual.nc") as f:
        assert f["values"].dimensions == ("x", "y")
        np.testing.assert_array_equal(f["values"][:], np.full((2, 3), -1))


@pytest.mark.parametrize("stage", ["detach", "reattach", "scale", "coords", "fill"])
def test_virtual_coordinate_failure_restores_existing_scale(
    tmp_path, monkeypatch, stage
):
    path = tmp_path / "virtual.nc"
    with h5netcdf.File(path, "w") as f:
        f.dimensions["time"] = 3
        f.create_variable("a", ("time",), data=[1, 2, 3])
        f.create_variable("b", ("time",), data=[4, 5, 6])
        old = f._h5group["time"]
        old.attrs["sentinel"] = "preserve me"
        dimid = old.attrs["_Netcdf4Dimid"]

        if stage in ("detach", "reattach"):
            method = "_detach_scale" if stage == "detach" else "_attach_scale"

            def fail(dim, *args):
                refs = dim._scale_refs if stage == "detach" else args[0]
                ref, axis = refs[0]
                target = f._h5file[ref].dims[axis]
                if stage == "detach":
                    target.detach_scale(dim._h5ds)
                else:
                    target.attach_scale(dim._h5ds)
                raise RuntimeError("injected coordinate failure")

            cls = Dimension
        else:
            cls, method = {
                "scale": (Dimension, "_create_scale"),
                "coords": (Variable, "_attach_coords"),
                "fill": (Variable, "_add_fillvalue"),
            }[stage]
            original = getattr(cls, method)

            def fail(obj, *args, **kwargs):
                original(obj, *args, **kwargs)
                raise RuntimeError("injected coordinate failure")

        with monkeypatch.context() as patch:
            patch.setattr(cls, method, fail)
            with pytest.raises(RuntimeError, match="injected"):
                f.create_virtual_variable(
                    "time", ("time",), h5py.VirtualLayout((3,), "i2"), -1
                )
        assert f._h5group["time"].id == old.id
        assert f._h5group["time"].attrs["sentinel"] == "preserve me"
        assert f._h5group["time"].attrs["_Netcdf4Dimid"] == dimid
        assert set(f.variables) == {"a", "b"}
        for key, values in [("a", [1, 2, 3]), ("b", [4, 5, 6])]:
            assert h5py.h5ds.is_attached(f._h5group[key].id, old.id, 0)
            np.testing.assert_array_equal(f[key][:], values)
        f.create_virtual_variable("time", ("time",), h5py.VirtualLayout((3,), "i2"), -1)
    with h5netcdf.File(path) as f:
        assert set(f.variables) == {"time", "a", "b"}
        assert not f.dimensions["time"].isunlimited()
        assert f["a"].dimensions == ("time",)
        np.testing.assert_array_equal(f["time"][:], [-1, -1, -1])


def test_virtual_writes_forward_to_sources_and_netcdf_append(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    with h5netcdf.File(tmp_path / "source.nc", "w") as f:
        f.create_variable("values", ("x",), data=np.array([1, 2], dtype="int16"))
    layout = h5py.VirtualLayout((2,), "int16")
    layout[:] = h5py.VirtualSource("source.nc", "values", shape=(2,))
    output = tmp_path / "virtual.nc"
    with h5netcdf.File(output, "w") as f:
        f.dimensions["x"] = 2
        f.create_virtual_variable("values", ("x",), layout)[:] = [3, 4]
    with h5netcdf.File(tmp_path / "source.nc") as f:
        np.testing.assert_array_equal(f["values"][:], [3, 4])
    for value in ("first", "updated"):
        with netcdf4.Dataset(output, "a") as f:
            for index in range(32):
                f["values"].setncattr(f"attr_{index}", value)
        with netcdf4.Dataset(output) as f:
            assert all(
                f["values"].getncattr(f"attr_{index}") == value for index in range(32)
            )


def test_virtual_classic_and_scalar(tmp_path):
    for classic in (True, False):
        with h5netcdf.File(
            tmp_path / f"{classic}.nc",
            "w",
            format="NETCDF4_CLASSIC" if classic else "NETCDF4",
        ) as f:
            layout = h5py.VirtualLayout((), "int16")
            if classic:
                with pytest.raises(CompatibilityError, match="without dimensions"):
                    f.create_virtual_variable("scalar", (), layout)
                f.dimensions["x"] = 2
                f.create_virtual_variable(
                    "values", ("x",), h5py.VirtualLayout((2,), "i2")
                )
                with pytest.raises(CompatibilityError):
                    f.create_virtual_variable(
                        "unsigned", ("x",), h5py.VirtualLayout((2,), "u2")
                    )
            else:
                assert f.create_virtual_variable("scalar", (), layout, 7)[()] == 7


@pytest.mark.parametrize("dtype", ["u2", ">u2", "u4", ">u4", "u8", ">u8", "i8", ">i8"])
def test_virtual_classic_rejects_nonclassic_numeric_types(tmp_path, dtype):
    with h5netcdf.File(tmp_path / "virtual.nc", "w", format="NETCDF4_CLASSIC") as f:
        f.dimensions["x"] = 2
        old = f._h5group["x"]
        with pytest.raises(CompatibilityError):
            f.create_virtual_variable("values", ("x",), h5py.VirtualLayout((2,), dtype))
        assert set(f._h5group) == {"x"}
        assert not f.variables
        assert f._h5group["x"].id == old.id
