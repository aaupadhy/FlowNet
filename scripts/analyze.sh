#!/usr/bin/env bash
#
# Quick script to print sizes and chunk info for your Zarr datasets.

# Paths to your Zarr datasets
SSH_ZARR="/scratch/user/aaupadhy/college/RA/final_data/SSH_Atlantic.zarr"
SST_ZARR="/scratch/user/aaupadhy/college/RA/final_data/SST_Atlantic.zarr"
VNT_ZARR="/scratch/user/aaupadhy/college/RA/final_data/VNT_Atlantic.zarr"

##################################################################
# 1) Print the on-disk sizes using 'du -hs'
##################################################################
echo "=== Checking on-disk size (via du -hs) ==="
for dataset in "$SSH_ZARR" "$SST_ZARR" "$VNT_ZARR"; do
  if [ -d "$dataset" ]; then
    size=$(du -hs "$dataset" | cut -f1)
    echo "Dataset: $dataset | On-disk size: $size"
  else
    echo "Dataset path not found: $dataset"
  fi
done

echo ""

##################################################################
# 2) Run a Python snippet to inspect each dataset with xarray/zarr
##################################################################
python << EOF
import xarray as xr
import zarr
import os

paths = [
    "$SSH_ZARR",
    "$SST_ZARR",
    "$VNT_ZARR"
]

print("=== Python-based inspection of chunking and metadata ===")
for pth in paths:
    if not os.path.exists(pth):
        print(f"[WARN] Path does not exist: {pth}")
        continue

    print("------------------------------------------------------------------")
    print(f"Opening {pth} using xarray.open_dataset(..., chunks=None)")
    ds = xr.open_dataset(pth, engine="zarr", chunks=None)

    print(f"Dataset has variables: {list(ds.data_vars.keys())}")
    print("Dataset dimensions:", ds.dims)
    print("")

    # For each variable, show shape, dtype, and chunk info
    for varname, da in ds.data_vars.items():
        print(f"Variable: {varname}")
        print(f"  Shape: {da.shape}")
        print(f"  Dtype: {da.dtype}")
        print(f"  Attributes: {da.attrs}")
        # Encoding can reveal chunk details if present
        print(f"  Encoding: {da.encoding}")
        # Dask chunks (None if not chunked in memory):
        print(f"  Dask .chunks: {da.chunks}")
        print("")

    ds.close()

    print(f"Now opening the Zarr group directly with zarr.open(...): {pth}")
    zgroup = zarr.open(pth, mode='r')
    print("Zarr group structure:")
    zarr.convenience.tree(zgroup, expand=True)
    print("")
EOF
