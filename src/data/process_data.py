import xarray as xr
import numpy as np
import dask.array as da
from dask.diagnostics import ProgressBar
import logging
from pathlib import Path
from tqdm import tqdm
from ..utils.dask_utils import dask_monitor

class OceanDataProcessor:
    def __init__(self, ssh_path: str, sst_path: str, vnt_path: str):
        """Initialize data processor for ocean data."""
        self.logger = logging.getLogger(__name__)
        
        with dask_monitor.profile_task("data_loading"):
            self.logger.info("Loading datasets...")
            self._load_datasets(ssh_path, sst_path, vnt_path)
    
    def _load_datasets(self, ssh_path, sst_path, vnt_path):
        """Load datasets with consistent chunking"""
        self.logger.info("Opening datasets...")
        
        ssh_ds_raw = xr.open_zarr(ssh_path)
        sst_ds_raw = xr.open_zarr(sst_path)
        vnt_ds_raw = xr.open_zarr(vnt_path)
        
        time_size = len(ssh_ds_raw.time)
        nlat_size = len(ssh_ds_raw.nlat)
        nlon_size = len(ssh_ds_raw.nlon)
        z_t_size = len(vnt_ds_raw.z_t) if 'z_t' in vnt_ds_raw.dims else None
        
        self.logger.info(f"Dataset dimensions - Time: {time_size}, Lat: {nlat_size}, Lon: {nlon_size}")
        if z_t_size:
            self.logger.info(f"Z_t dimension size: {z_t_size}")
        
        time_lengths = [
            len(ds.time) for ds in [ssh_ds_raw, sst_ds_raw, vnt_ds_raw]
        ]
        if not all(t == time_lengths[0] for t in time_lengths):
            raise ValueError(f"Time dimensions don't match: {time_lengths}")
            
        time_chunk = min(500, time_size)
        lat_chunk = min(1000, nlat_size)
        lon_chunk = min(1000, nlon_size)
        
        base_chunks = {
            'time': time_chunk,
            'nlat': lat_chunk,
            'nlon': lon_chunk
        }
        
        vnt_chunks = base_chunks.copy()
        if z_t_size:
            vnt_chunks['z_t'] = z_t_size
        
        self.logger.info(f"Using chunks: {base_chunks}")
        
        self.logger.info("Loading SSH dataset...")
        self.ssh_ds = xr.open_zarr(ssh_path, chunks=base_chunks)
        self.logger.info(f"SSH dataset loaded with chunks: {self.ssh_ds.chunks}")
        
        self.logger.info("Loading SST dataset...")
        self.sst_ds = xr.open_zarr(sst_path, chunks=base_chunks)
        self.logger.info(f"SST dataset loaded with chunks: {self.sst_ds.chunks}")
        
        self.logger.info("Loading VNT dataset...")
        self.vnt_ds = xr.open_zarr(vnt_path, chunks=vnt_chunks)
        self.logger.info(f"VNT dataset loaded with chunks: {self.vnt_ds.chunks}")
        
        self.tarea_conversion = 0.0001
        self.dz_conversion = 0.01
        
        self.logger.info(f"SSH variables: {list(self.ssh_ds.data_vars)}")
        self.logger.info(f"SST variables: {list(self.sst_ds.data_vars)}")
        self.logger.info(f"VNT variables: {list(self.vnt_ds.data_vars)}")
        
        ssh_ds_raw.close()
        sst_ds_raw.close()
        vnt_ds_raw.close()

    def get_spatial_data(self):
        """Get SSH and SST data along with their coordinates"""
        with dask_monitor.profile_task("get_spatial_data"):
            self.logger.info("Getting SSH data...")
            ssh = self.ssh_ds["SSH"]
            
            self.logger.info("Getting SST data...")
            sst = self.sst_ds["SST"]
            
            self.logger.info("Getting TLAT/TLONG coordinates...")
            tlat = self.vnt_ds["TLAT"]
            tlong = self.vnt_ds["TLONG"]
            
            # Compute and log basic statistics
            with ProgressBar():
                ssh_stats = {
                    'min': float(ssh.min().compute()),
                    'max': float(ssh.max().compute()),
                    'mean': float(ssh.mean().compute()),
                    'shape': ssh.shape
                }
                
                sst_stats = {
                    'min': float(sst.min().compute()),
                    'max': float(sst.max().compute()),
                    'mean': float(sst.mean().compute()),
                    'shape': sst.shape
                }
            
            self.logger.info(f"SSH stats: {ssh_stats}")
            self.logger.info(f"SST stats: {sst_stats}")
            
            return ssh, sst, tlat, tlong

    def get_dataset_info(self):
        """Get information about the datasets"""
        info = {
            'ssh_shape': self.ssh_ds["SSH"].shape,
            'sst_shape': self.sst_ds["SST"].shape,
            'vnt_shape': self.vnt_ds["VNT"].shape,
            'time_steps': len(self.ssh_ds["time"]),
            'ssh_chunks': self.ssh_ds["SSH"].chunks,
            'sst_chunks': self.sst_ds["SST"].chunks,
            'vnt_chunks': self.vnt_ds["VNT"].chunks
        }
        return info

    def calculate_heat_transport(self, latitude_index):
        """Calculate heat transport for a given latitude"""
        with dask_monitor.profile_task(f"heat_transport_lat_{latitude_index}"):
            self.logger.info(f"Calculating heat transport for latitude index {latitude_index}")

            with ProgressBar():
                atlantic_mask_slice = self.atlantic_mask.isel(nlat=latitude_index)
                valid_lons = np.argwhere(atlantic_mask_slice.values).flatten()

                vnt = self.vnt_ds["VNT"].isel(
                    nlat=latitude_index,
                    nlon=valid_lons
                )

                tarea = (self.vnt_ds["TAREA"]
                        .isel(nlat=latitude_index, nlon=valid_lons)
                        * self.tarea_conversion)

                dz = self.vnt_ds["dz"] * self.dz_conversion

                heat_transport = ((vnt * tarea * dz)
                                .sum(dim='z_t')
                                .sum(dim='nlon'))

                result = heat_transport.compute()

                self.logger.info(f"Heat transport range: {float(result.min())} to {float(result.max())}")
                return result
