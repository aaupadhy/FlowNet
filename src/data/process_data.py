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
    
    def _load_datasets(self, ssh_path, sst_path, vnt_path):
        try:
            base_chunks = {'time': 180, 'nlat': 'auto', 'nlon': 'auto'}
            
            self.logger.info("Loading SSH dataset...")
            self.ssh_ds = xr.open_dataset(ssh_path, engine="zarr", chunks=None)
            self.ssh_ds = self.ssh_ds.chunk(base_chunks)
            
            self.logger.info("Loading SST dataset...")
            self.sst_ds = xr.open_dataset(sst_path, engine="zarr", chunks=None)
            self.sst_ds = self.sst_ds.chunk(base_chunks)
            
            self.logger.info("Loading VNT dataset...")
            vnt_chunks = {'time': 12, 'nlat': 300, 'nlon': 300, 'z_t': None}
            self.vnt_ds = xr.open_dataset(vnt_path, engine="zarr", chunks=None)
            self.vnt_ds = self.vnt_ds.chunk(vnt_chunks)

            self.reference_latitude = self._find_reference_latitude()
            
            with dask_monitor.profile_task("compute_statistics"):
                ssh_sample = self.ssh_ds["SSH"].isel(time=0).compute()
                self.logger.info(f"SSH sample stats:")
                self.logger.info(f"Shape: {ssh_sample.shape}")
                self.logger.info(f"Any NaNs: {np.any(np.isnan(ssh_sample))}")
                self.logger.info(f"Min/Max: {float(ssh_sample.min())}/{float(ssh_sample.max())}")
                self.logger.info("Computing SSH statistics...")
                self.ssh_mean, self.ssh_std = self._compute_mean_std(
                    self.ssh_ds["SSH"],
                    var_name="SSH"
                )
                self.logger.info(f"Final SSH stats:")
                self.logger.info(f"Mean: {float(self.ssh_mean.mean())}")
                self.logger.info(f"Std: {float(self.ssh_std.mean())}")
                self.logger.info("Computing SST statistics...")
                self.sst_mean, self.sst_std = self._compute_mean_std(
                    self.sst_ds["SST"],
                    var_name="SST"
                )

            self.tarea_conversion = 0.0001
            self.dz_conversion = 0.01
            self.shape = self.sst_ds["SST"].shape
            
            self.stats = {
                'grid_shape': self.shape,
                'ssh_mean': float(np.nanmean(self.ssh_mean.values)), 
                'ssh_std': float(np.nanmean(self.ssh_std.values)),   
                'sst_mean': float(np.nanmean(self.sst_mean.values)),  
                'sst_std': float(np.nanmean(self.sst_std.values))    
            }
            
            self._validate_statistics()
            self.logger.info("Successfully loaded all datasets")
            
        except Exception as e:
            self.logger.error(f"Error loading datasets: {str(e)}")
            raise

    def _validate_statistics(self):
        stat_checks = {
            'ssh_mean': (-500, 500),
            'sst_mean': (-5, 35),
            'ssh_std': (0, 100),
            'sst_std': (0, 20)
        }
        
        for stat_name, (min_val, max_val) in stat_checks.items():
            value = self.stats[stat_name]
            if not np.isfinite(value):
                raise ValueError(f"Invalid {stat_name}: {value}")
            if value == 0:
                raise ValueError(f"Zero {stat_name} detected")
            if not min_val <= value <= max_val:
                raise ValueError(
                    f"{stat_name} ({value:.2f}) outside reasonable range "
                    f"[{min_val}, {max_val}]"
                )

        self.logger.info("Statistics validation passed:")
        for key, value in self.stats.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"{key}: {value:.4f}")
            else:
                self.logger.info(f"{key}: {value}")
                
    def _find_reference_latitude(self):
        cache_file = Path('/scratch/user/aaupadhy/college/RA/final_data/cache/ref_lat_index.npy')
        
        if cache_file.exists():
            try:
                ref_lat_idx = np.load(cache_file)
                tlat = self.vnt_ds['TLAT'][0]
                actual_lat = float(tlat[ref_lat_idx].mean())
                self.logger.info(f"Loaded cached reference latitude index {ref_lat_idx} corresponding to {actual_lat:.2f}°N")
                return ref_lat_idx
            except Exception as e:
                self.logger.warning(f"Failed to load cached reference latitude: {str(e)}. Recomputing...")
        
        try:
            tlat = self.vnt_ds['TLAT'][0]
            self.logger.info(f"TLAT: {tlat}")
            lat_mask = (tlat >= 39.9) & (tlat <= 40.1)
            lat_mask = lat_mask.compute()
            
            lat_indices = np.where(lat_mask)
            if len(lat_indices) == 0:
                raise ValueError("No points found at 40°N latitude")

            self.logger.info(f"Lat indices: {lat_indices}")
            
            ref_lat_idx = int(np.median(lat_indices))
            actual_lat = float(tlat[ref_lat_idx].mean())
            
            np.save(cache_file, ref_lat_idx)
            
            self.logger.info(f"Found and cached reference latitude index {ref_lat_idx} corresponding to {actual_lat:.2f}°N")
            self.logger.info(f"Number of points found at ~40°N: {len(lat_indices)}")
            
            return ref_lat_idx

        except Exception as e:
            self.logger.error(f"Error finding reference latitude: {str(e)}")
            raise
        
    def _compute_mean_std(self, data_array, var_name: str, dim="time"):
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

        self.logger.info(f"Computing statistics for {var_name}")
        self.logger.info(f"Input shape: {data_array.shape}")

        MISSING_VALUE = 9.96921e+36
        data_masked = data_array.where(data_array != MISSING_VALUE)
        
        mean = data_masked.mean(dim=dim)
        std = data_masked.std(dim=dim)

        mean = mean.compute()
        std = std.compute()
        
        self.logger.info(f"Computed statistics for {var_name}:")
        self.logger.info(f"Mean shape: {mean.shape}")
        self.logger.info(f"Mean overall: {float(mean.mean())}")
        self.logger.info(f"Mean range: {float(mean.min())} to {float(mean.max())}")
        self.logger.info(f"Std range: {float(std.min())} to {float(std.max())}")

        np.save(cache_file, np.array([mean.values, std.values]))
                
        return mean, std

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

    def calculate_heat_transport(self, latitude_index=None):
        if latitude_index is None:
            latitude_index = self.reference_latitude
            
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
            
            dz = self.vnt_ds["dz"]
            self.logger.info(f"dz shape: {dz.shape}")

            chunk_size = 60
            total_times = vnt.shape[0]
            heat_transport_list = []
            
            for start_idx in range(0, total_times, chunk_size):
                end_idx = min(start_idx + chunk_size, total_times)
                vnt_chunk = vnt.isel(time=slice(start_idx, end_idx))
                
                chunk_transport = (
                    vnt_chunk *
                    tarea * self.tarea_conversion *
                    dz * self.dz_conversion
                ).sum(dim=['z_t', 'nlon']).compute()
                
                heat_transport_list.append(chunk_transport)
            
            heat_transport = np.concatenate(heat_transport_list)
            mean_transport = float(np.mean(heat_transport))
            
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
        self.logger = logging.getLogger(__name__)
        heat_transport_std = np.std(heat_transport - heat_transport_mean)
        self.heat_transport = (heat_transport - heat_transport_mean) / heat_transport_std
        
        self.logger.info(f"Heat Transport statistics after standardization:")
        self.logger.info(f"  Min: {np.min(self.heat_transport):.2f}")
        self.logger.info(f"  Max: {np.max(self.heat_transport):.2f}")
        self.logger.info(f"  Mean: {np.mean(self.heat_transport):.2f}")
        self.logger.info(f"  Std: {np.std(self.heat_transport):.2f}")
        
        self.ssh_mean = ssh_mean.values
        
        self.ssh_std = ssh_std.values
        self.sst_mean = sst_mean.values
        self.sst_std = sst_std.values
        self.logger.info(f"SELF SSH MEAN: {self.ssh_mean}")
        self.logger.info(f"SELF SSH STD: {self.ssh_std}")
        self.logger.info(f"SELF SST MEAN: {self.sst_mean}")
        self.logger.info(f"SELF SST STD: {self.ssh_std}")
        
        MISSING_VALUE = 9.96921e+36
        ssh_vals = ssh.isel(time=0).values
        sst_vals = sst.isel(time=0).values
        
        ssh_valid = ssh_vals != MISSING_VALUE
        sst_valid = sst_vals != MISSING_VALUE
        self.valid_mask = ssh_valid & sst_valid
        
        self.length = shape[0]

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        try:
            ssh_data, sst_data = dask.compute(
                self.ssh.isel(time=idx),
                self.sst.isel(time=idx)
            )
            
            MISSING_VALUE = 9.96921e+36
            ssh_vals = ssh_data.values
            sst_vals = sst_data.values
            
            ssh_valid = (ssh_vals != MISSING_VALUE).reshape(ssh_vals.shape)
            sst_valid = (sst_vals != MISSING_VALUE).reshape(sst_vals.shape)
            
            ssh_norm = np.zeros_like(ssh_vals, dtype=np.float32)
            sst_norm = np.zeros_like(sst_vals, dtype=np.float32)
            
            ssh_norm = np.where(ssh_valid, 
                            (ssh_vals - self.ssh_mean) / (self.ssh_std + 1e-6),
                            0)
            sst_norm = np.where(sst_valid,
                            (sst_vals - self.sst_mean) / (self.sst_std + 1e-6),
                            0)
            
            ssh_norm = np.clip(ssh_norm, -3, 3)
            sst_norm = np.clip(sst_norm, -3, 3)
            
            ssh_tensor = torch.from_numpy(ssh_norm).float().unsqueeze(0)
            sst_tensor = torch.from_numpy(sst_norm).float().unsqueeze(0)
            mask_tensor = torch.from_numpy(ssh_valid & sst_valid).float()
            
            ssh_downsampled = F.avg_pool2d(ssh_tensor.unsqueeze(0), kernel_size=2, stride=2).squeeze(0)
            sst_downsampled = F.avg_pool2d(sst_tensor.unsqueeze(0), kernel_size=2, stride=2).squeeze(0)
            mask_downsampled = F.avg_pool2d(mask_tensor.unsqueeze(0).unsqueeze(0), kernel_size=2, stride=2).squeeze(0).squeeze(0)
            
            return (
                ssh_downsampled, 
                sst_downsampled,
                mask_downsampled,
                torch.tensor(self.heat_transport[idx]).float()
            )
        except Exception as e:
            self.logger.error(f"Error in __getitem__ at index {idx}: {str(e)}")
            raise