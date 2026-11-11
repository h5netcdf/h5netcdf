.. _creating virtual variables:

Virtual variables
=================

HDF5 virtual datasets (VDS) map regions of source datasets into a single array
without copying those source arrays into the destination file. For example, a
microscopy acquisition saved as one NetCDF file per time point can be exposed
as a single NetCDF time series.

:meth:`h5netcdf.Group.create_virtual_variable` accepts an explicit
``h5py.VirtualLayout`` and maintains the netCDF dimension scales and coordinate
metadata. It returns an ordinary :class:`h5netcdf.Variable`; set its attributes
as usual. The h5py backend and HDF5 1.10 or newer are required to create and read
the underlying VDS. Other backends raise ``NotImplementedError`` on creation.
Reading with netCDF4-python also requires a netCDF-C/HDF5 build supporting VDS.

Example: separate acquisition files
-----------------------------------

This self-contained example uses three small, synthetic images. Coordinates and
attributes are provided explicitly; the source files' metadata is not inferred
or copied. The virtual file stores mappings and its own metadata and coordinates,
so its total size is not zero even though its virtual image stores no raw data.

.. code-block:: python

    from pathlib import Path
    from tempfile import TemporaryDirectory

    import h5netcdf
    import h5py
    import numpy as np

    with TemporaryDirectory() as dirname:
        directory = Path(dirname)
        for index in range(3):
            with h5netcdf.File(directory / f"timepoint_{index}.nc", "w") as source:
                source.dimensions = {"y": 2, "x": 3}
                source.create_variable(
                    "image",
                    ("y", "x"),
                    data=np.full((2, 3), index + 1, dtype="uint16"),
                )

        layout = h5py.VirtualLayout(shape=(3, 2, 3), dtype="uint16")
        for index in range(3):
            layout[index] = h5py.VirtualSource(
                f"timepoint_{index}.nc", "image", shape=(2, 3)
            )

        output = directory / "combined.nc"
        with h5netcdf.File(output, "w") as target:
            target.dimensions = {"time": 3, "y": 2, "x": 3}
            image = target.create_virtual_variable(
                "image", ("time", "y", "x"), layout, fillvalue=0
            )
            image.attrs["description"] = "Synthetic acquisition"
            time = target.create_variable("time", ("time",), data=[0, 30, 60])
            time.attrs["units"] = "seconds"
            target.create_variable("y", ("y",), data=[0, 1])
            target.create_variable("x", ("x",), data=[0, 1, 2])
            target.create_variable("temperature", ("time",), data=[20.0, 20.1, 20.2])

        with h5netcdf.File(output) as dataset:
            assert dataset["image"].dimensions == ("time", "y", "x")
            np.testing.assert_array_equal(dataset["image"][1], np.full((2, 3), 2))

The resulting file can be opened with ``xarray.open_dataset(output,
engine="h5netcdf")`` or ``netCDF4.Dataset(output)`` while the temporary directory
still exists. In real acquisitions, keep the virtual file and its sources in a
persistent directory.

Layouts, dimensions and coordinates
-----------------------------------

The layout determines the dtype, shape, source paths and mapping selections.
Dimensions must already exist in the current group or an ancestor and their
sizes must exactly match the layout shape. This entry point supports primitive
signed and unsigned integers of 1, 2, 4 or 8 bytes and 4- or 8-byte floats.
It does not support unlimited dimensions, resizable layouts, unlimited mapping
selections, strings, enums, compound types or other user-defined dtypes. A fixed
selection may reference a resizable source, but its mapped extent remains fixed.

Create intermediate groups explicitly with ``create_group`` before passing a
path such as ``"acquisition/image"``. Names beginning with ``/`` are relative to
the file root. A virtual variable can also use inherited dimensions in a child
group, just like a regular variable.

A one-dimensional virtual coordinate can replace a local dimension's dummy
scale, including when other variables already use that dimension. For example,
``target.create_virtual_variable("time", ("time",), time_layout)`` uses the same
dimension-scale helpers as ordinary coordinates. The time layout must have the
fixed shape of that local dimension. Multidimensional virtual coordinates and
implicit shadowing of an ancestor's dimension are not supported.

If creation raises while the backend remains accessible, the new variable is
removed and any replaced dummy scale and its dependent scale attachments are
restored. This does not guarantee byte-identical files or recovery after a
process crash, power loss or storage failure.

Source lifetime, fill values and writes
---------------------------------------

Source mapping and path handling are delegated to h5py and HDF5. In particular,
h5py may replace the current file's path with ``.`` for same-file sources. Use
relative paths with respect to the virtual file's directory to move that file
together with its sources. Absolute paths continue to point at their original locations. No source
file is opened or validated by ``create_virtual_variable``; callers must supply
the correct dataset paths, source shapes and mappings. A source may be created
later, as allowed by h5py.

Missing sources and unmapped regions return the HDF5 fill value rather than an
error. Supplying ``fillvalue`` sets both that value and the variable's
``_FillValue`` attribute. Without it, HDF5 uses zero and no ``_FillValue`` is added
by the new API. Readers such as xarray and netCDF4-python may mask explicit fill
values according to their normal decoding rules. Check source availability and
integrity separately when missing acquisitions must be treated as errors.

Opening a virtual file read-only is appropriate for viewing an acquisition.
Writing a mapped region through a virtual variable can modify its source files.
The virtual file is not a snapshot or an independent backup of source data.

Related approaches
-------------------

Raw `h5py VDS`_ provides the layout and source mapping machinery used here, but
requires the caller to maintain netCDF dimension scales. `xarray.open_mfdataset`_
combines multiple files lazily in Python and does not create a persistent native
HDF5 VDS file. `kerchunk`_ and `VirtualiZarr`_ expose virtual arrays through
Zarr-compatible references rather than producing a native NetCDF VDS file.
Choose according to the intended reader and storage workflow; no performance
advantage over these alternatives is implied by this API.

.. _h5py VDS: https://docs.h5py.org/en/stable/vds.html
.. _xarray.open_mfdataset: https://docs.xarray.dev/en/stable/generated/xarray.open_mfdataset.html
.. _kerchunk: https://fsspec.github.io/kerchunk/
.. _VirtualiZarr: https://virtualizarr.readthedocs.io/en/stable/
