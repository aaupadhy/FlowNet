import xarray as xr
import numpy as np
import dask
import dask.array as da
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import logging
from pathlib import Path
import gc
from tqdm import tqdm
from ..utils.dask_utils import dask_monitor

class OceanDataProcessor:
    def __init__(self, ssh_path: str, sst_path: str, vnt_path: str):
        self.logger = logging.getLogger(__name__)
        
        self.cache_dir = Path('/scratch/user/aaupadhy/college/RA/final_data/cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Created/verified cache directory at {self.cache_dir}")
        
        with dask_monitor.profile_task("data_loading"):
            self.logger.info("Loading datasets...")
            self._load_datasets(ssh_path, sst_path, vnt_path)

    def _compute_mean_std(self, data_array, var_name: str, dim="time"):
        """Compute and cache global statistics using Dask"""
        cache_file = self.cache_dir / f'{var_name}_stats.npy'
        
        if cache_file.exists():
            try:
                stats = np.load(cache_file)
                mean_da = xr.DataArray(stats[0])
                std_da = xr.DataArray(stats[1])
                self.logger.info(f"Loaded cached statistics for {var_name}")
                return mean_da, std_da
            except Exception as e:
                self.logger.warning(f"Failed to load cached stats: {str(e)}. Recomputing...")

    def _cleanup_memory(self):
        gc.collect()
        if hasattr(self, 'client'):
            self.client.cancel(self.client.get_current_task())
            
        torch.cuda.empty_cache()
        self.logger.info(f"Memory cleaned up. Current GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    def log_chunk_stats(self, data_array, array_name: str = "data_array"):
        if not hasattr(data_array, 'chunks') or data_array.chunks is None:
            self.logger.info(f"{array_name} is not dask-backed or has no chunks. Shape: {data_array.shape}")
            return {}

        chunk_tuples = data_array.chunks
        dtype_size = data_array.dtype.itemsize

        total_elems = np.prod(data_array.shape)
        total_mem_gb = (total_elems * dtype_size) / (1024**3)
        
        chunks_per_dim = [len(dim_c) for dim_c in chunk_tuples]
        n_chunks = np.prod(chunks_per_dim)
        
        avg_shape_per_dim = [np.mean(dim_chunk_sizes) for dim_chunk_sizes in chunk_tuples]
        avg_chunk_elems = int(np.prod(avg_shape_per_dim))
        avg_chunk_mem_gb = (avg_chunk_elems * dtype_size) / (1024**3)
        
        stats = {
            "shape": data_array.shape,
            "dtype": str(data_array.dtype),
            "total_size_gb": total_mem_gb,
            "n_chunks": n_chunks,
            "chunks_per_dim": chunks_per_dim,
            "avg_chunk_size_gb": avg_chunk_mem_gb
        }
        
        self.logger.info(f"\nChunk stats for {array_name}:")
        for key, value in stats.items():
            self.logger.info(f"  {key}: {value}")
        
        return stats

    def _compute_mean_std(self, data_array, var_name: str, dim="time"):
        """Compute and cache global statistics using Dask"""
        cache_file = Path(f'/scratch/user/aaupadhy/college/RA/final_data/cache/{var_name}_stats.npy')
        
        if cache_file.exists():
            try:
                stats = np.load(cache_file)
                mean_da = xr.DataArray(stats[0])
                std_da = xr.DataArray(stats[1])
                self.logger.info(f"Loaded cached statistics for {var_name}")
                return mean_da, std_da
            except Exception as e:
                self.logger.warning(f"Failed to load cached stats: {str(e)}. Recomputing...")
        
        sum_chunks = da.sum(data_array.data, axis=data_array.get_axis_num(dim))
        sum_of_squares_chunks = da.sum(data_array.data**2, axis=data_array.get_axis_num(dim))
        count_chunks = da.sum(da.ones_like(data_array.data), axis=data_array.get_axis_num(dim))

        total_sum = sum_chunks.sum().compute()
        total_sum_of_squares = sum_of_squares_chunks.sum().compute()
        total_count = count_chunks.sum().compute()

        global_mean = total_sum / total_count
        global_variance = (total_sum_of_squares / total_count) - (global_mean**2)
        global_std = np.sqrt(global_variance)

        dims = [d for d in data_array.dims if d != dim]
        coords = {k: v for k, v in data_array.coords.items() if k != dim}

        if "time" in coords:
            coords.pop("time")

        mean_da = xr.DataArray(global_mean, dims=dims, coords=coords)
        std_da = xr.DataArray(global_std, dims=dims, coords=coords)
        
        np.save(cache_file, np.array([mean_da.values, std_da.values]))
        
        return mean_da, std_da

    def _load_datasets(self, ssh_path, sst_path, vnt_path):
        try:
            self.logger.info("Opening datasets...")
            
            bounds_cache = Path('/scratch/user/aaupadhy/college/RA/final_data/atlantic_bounds.npz')
            
            if bounds_cache.exists():
                try:
                    cached_bounds = np.load(bounds_cache)
                    self.lat_min, self.lat_max = cached_bounds['lat_bounds']
                    self.lon_min, self.lon_max = cached_bounds['lon_bounds']
                    self.logger.info(f"Loaded cached Atlantic bounds: lat [{self.lat_min}:{self.lat_max}], lon [{self.lon_min}:{self.lon_max}]")
                except Exception as e:
                    self.logger.warning(f"Failed to load cached bounds: {str(e)}. Recomputing...")
            
            if not hasattr(self, 'lat_min'):
                self.logger.info("Computing Atlantic bounds...")
                test_ssh = xr.open_dataset(ssh_path, engine="zarr").isel(time=0)['SSH']
                valid_points = ~np.isnan(test_ssh)
                
                valid_lats, valid_lons = np.where(valid_points)
                self.lat_min, self.lat_max = valid_lats.min(), valid_lats.max()
                self.lon_min, self.lon_max = valid_lons.min(), valid_lons.max()
                
                np.savez(bounds_cache, 
                        lat_bounds=np.array([self.lat_min, self.lat_max]),
                        lon_bounds=np.array([self.lon_min, self.lon_max]))
                
                self.logger.info(f"Computed and cached Atlantic bounds: lat [{self.lat_min}:{self.lat_max}], lon [{self.lon_min}:{self.lon_max}]")
            
            base_chunks = {'time': 180, 'nlat': 'auto', 'nlon': 'auto'}
            
            self.logger.info("Loading SSH dataset...")
            self.ssh_ds = xr.open_dataset(ssh_path, engine="zarr", chunks=None)
            self.ssh_ds = self.ssh_ds.isel(
                nlat=slice(self.lat_min, self.lat_max+1),
                nlon=slice(self.lon_min, self.lon_max+1)
            ).chunk(base_chunks)
            self.log_chunk_stats(self.ssh_ds["SSH"], array_name="SSH")

            self.logger.info("Loading SST dataset...")
            self.sst_ds = xr.open_dataset(sst_path, engine="zarr", chunks=None)
            self.sst_ds = self.sst_ds.isel(
                nlat=slice(self.lat_min, self.lat_max+1),
                nlon=slice(self.lon_min, self.lon_max+1)
            ).chunk(base_chunks)
            self.log_chunk_stats(self.sst_ds["SST"], array_name="SST")
            
            self.shape = self.sst_ds["SST"].shape
            self.logger.info(f"Atlantic region shape: {self.shape}")

            with dask_monitor.profile_task("compute_statistics"):
                self.logger.info("Computing SSH statistics...")
                self.ssh_mean, self.ssh_std = self._compute_mean_std(self.ssh_ds["SSH"], var_name="SSH")
                
                self.logger.info("Computing SST statistics...")
                self.sst_mean, self.sst_std = self._compute_mean_std(self.sst_ds["SST"], var_name="SST")

            self.logger.info("Loading VNT dataset...")
            vnt_chunks = {'time': 12, 'nlat': 300, 'nlon': 450, 'z_t': None}
            self.vnt_ds = xr.open_dataset(vnt_path, engine="zarr", chunks=None)
            self.vnt_ds = self.vnt_ds.chunk(vnt_chunks).unify_chunks()
            self.log_chunk_stats(self.vnt_ds["VNT"], array_name="VNT")

            self.tarea_conversion = 0.0001
            self.dz_conversion = 0.01

            self.logger.info("Successfully loaded all datasets")
            
        except Exception as e:
            self.logger.error(f"Error loading datasets: {str(e)}")
            raise

    def get_spatial_data(self):
        return (
            self.ssh_ds["SSH"],
            self.sst_ds["SST"],
            self.vnt_ds["TLAT"],
            self.vnt_ds["TLONG"]
        )

    def get_dataset_info(self):
        return {
            'ssh_shape': self.ssh_ds["SSH"].shape,
            'sst_shape': self.sst_ds["SST"].shape,
            'vnt_shape': self.vnt_ds["VNT"].shape,
            'time_steps': len(self.ssh_ds["time"]),
            'ssh_chunks': self.ssh_ds["SSH"].chunks,
            'sst_chunks': self.sst_ds["SST"].chunks,
            'vnt_chunks': self.vnt_ds["VNT"].chunks
        }

    def calculate_heat_transport(self, latitude_index):
        cache_file = Path(f'/scratch/user/aaupadhy/college/RA/final_data/cache/heat_transport_lat{latitude_index}.npy')
        cache_mean_file = Path(f'/scratch/user/aaupadhy/college/RA/final_data/cache/heat_transport_lat{latitude_index}_mean.npy')
        
        if cache_file.exists() and cache_mean_file.exists():
            try:
                heat_transport = np.load(cache_file)
                mean_transport = np.load(cache_mean_file)
                
                self.logger.info(f"Loaded cached heat transport data:")
                self.logger.info(f"  Shape: {heat_transport.shape}")
                self.logger.info(f"  Range: {heat_transport.min():.2e} to {heat_transport.max():.2e}")
                self.logger.info(f"  Mean: {mean_transport:.2e}")
                
                return heat_transport, mean_transport
                
            except Exception as e:
                self.logger.warning(f"Failed to load cached data: {str(e)}. Recomputing...")
        
        try:
            self.logger.info(f"Calculating heat transport for latitude index {latitude_index}")
            
            vnt = self.vnt_ds["VNT"].isel(nlat=latitude_index)
            self.logger.info(f"VNT shape after lat selection: {vnt.shape}")
            
            tarea = self.vnt_ds["TAREA"].isel(nlat=latitude_index)
            self.logger.info(f"TAREA shape: {tarea.shape}")
            
            dz = self.vnt_ds["dz"].isel(nlat=latitude_index)
            self.logger.info(f"dz shape: {dz.shape}")
            
            tarea_scaled = tarea * self.tarea_conversion
            dz_scaled = dz * self.dz_conversion
            
            step1 = vnt * tarea_scaled
            self.logger.info(f"Shape after VNT * TAREA: {step1.shape}")
            
            step2 = step1 * dz_scaled
            self.logger.info(f"Shape after multiplying with dz: {step2.shape}")
            
            heat_transport = step2.sum(dim='z_t')
            self.logger.info(f"Shape after summing z_t: {heat_transport.shape}")
            
            heat_transport = heat_transport.sum(dim='nlon')
            self.logger.info(f"Shape after summing nlon: {heat_transport.shape}")
            
            heat_transport = heat_transport.compute()
            self.logger.info(f"Final heat transport shape: {heat_transport.shape}")
            
            mean_transport = float(heat_transport.mean())
            
            np.save(cache_file, heat_transport)
            np.save(cache_mean_file, mean_transport)
            
            self.logger.info(f"Heat transport statistics:")
            self.logger.info(f"  Range: {float(heat_transport.min()):.2e} to {float(heat_transport.max()):.2e}")
            self.logger.info(f"  Mean: {mean_transport:.2e}")
            
            return heat_transport, mean_transport
                
        except Exception as e:
            self.logger.error(f"Error calculating heat transport: {str(e)}")
            raise
            
        finally:
            self._cleanup_memory()

class OceanDataset(Dataset):
    def __init__(self, ssh, sst, heat_transport, heat_transport_mean, 
                 ssh_mean, ssh_std, sst_mean, sst_std, shape):
        self.ssh = ssh
        self.sst = sst
        if isinstance(heat_transport, xr.DataArray):
            self.heat_transport = heat_transport.values - heat_transport_mean
        else:
            self.heat_transport = heat_transport - heat_transport_mean
        self.ssh_mean = ssh_mean
        self.ssh_std = ssh_std
        self.sst_mean = sst_mean
        self.sst_std = sst_std
        self.length = shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        try:
            ssh_data, sst_data = dask.compute(
                self.ssh.isel(time=idx),
                self.sst.isel(time=idx)
            )
            
            valid_mask = ~np.isnan(ssh_data.values)
            
            ssh_norm = (ssh_data.values - self.ssh_mean.values[0,0]) / self.ssh_std.values[0,0]
            sst_norm = (sst_data.values - self.sst_mean.values[0,0]) / self.sst_std.values[0,0]
            
            ssh_tensor = torch.from_numpy(ssh_norm).float().unsqueeze(0)
            sst_tensor = torch.from_numpy(sst_norm).float().unsqueeze(0)
            
            ssh_downsampled = F.avg_pool2d(ssh_tensor, kernel_size=2, stride=2)
            sst_downsampled = F.avg_pool2d(sst_tensor, kernel_size=2, stride=2)

            return (
                ssh_downsampled,  # Already has batch dimension from unsqueeze(0)
                sst_downsampled,  # Already has batch dimension from unsqueeze(0)
                torch.tensor(self.heat_transport[idx]).float()
            )
        except Exception as e:
            logging.error(f"Error in __getitem__ at index {idx}: {str(e)}")
            raise