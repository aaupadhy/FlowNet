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
            self.logger.info("Opening datasets...")
            
            # Step 1: Compute or load Atlantic bounds
            bounds_cache = Path('/scratch/user/aaupadhy/college/RA/final_data/atlantic_bounds.npz')
            recompute_bounds = False
            
            if bounds_cache.exists():
                try:
                    cached_bounds = np.load(bounds_cache)
                    self.lat_min, self.lat_max = cached_bounds['lat_bounds']
                    self.lon_min, self.lon_max = cached_bounds['lon_bounds']
                    self.logger.info(f"Loaded cached Atlantic bounds: lat [{self.lat_min}:{self.lat_max}], lon [{self.lon_min}:{self.lon_max}]")
                except Exception as e:
                    self.logger.warning(f"Failed to load cached bounds: {str(e)}. Will recompute.")
                    recompute_bounds = True
            else:
                recompute_bounds = True

            if recompute_bounds:
                self.logger.info("Computing Atlantic bounds...")
                # Load a single time slice to compute bounds
                test_ssh = xr.open_dataset(ssh_path, engine="zarr").isel(time=0)['SSH']
                test_sst = xr.open_dataset(sst_path, engine="zarr").isel(time=0)['SST']
                
                # Find valid points in both SSH and SST
                ssh_valid = ~np.isnan(test_ssh)
                sst_valid = ~np.isnan(test_sst)
                valid_points = ssh_valid & sst_valid
                
                if not np.any(valid_points):
                    raise ValueError("No valid points found in the intersection of SSH and SST data")
                
                valid_lats, valid_lons = np.where(valid_points)
                self.lat_min, self.lat_max = valid_lats.min(), valid_lats.max()
                self.lon_min, self.lon_max = valid_lons.min(), valid_lons.max()
                
                # Cache the computed bounds
                np.savez(bounds_cache,
                        lat_bounds=np.array([self.lat_min, self.lat_max]),
                        lon_bounds=np.array([self.lon_min, self.lon_max]))
                self.logger.info(f"Computed and cached Atlantic bounds: lat [{self.lat_min}:{self.lat_max}], lon [{self.lon_min}:{self.lon_max}]")

            # Step 2: Load datasets with computed bounds
            base_chunks = {'time': 180, 'nlat': 'auto', 'nlon': 'auto'}
            
            self.logger.info("Loading SSH dataset...")
            self.ssh_ds = xr.open_dataset(ssh_path, engine="zarr", chunks=None)
            self.ssh_ds = self.ssh_ds.isel(
                nlat=slice(self.lat_min, self.lat_max+1),
                nlon=slice(self.lon_min, self.lon_max+1)
            ).chunk(base_chunks)

            # Create valid data mask (non-NaN values)
            ssh_valid = ~np.isnan(self.ssh_ds["SSH"].isel(time=0))
            
            self.logger.info("Loading SST dataset...")
            self.sst_ds = xr.open_dataset(sst_path, engine="zarr", chunks=None)
            self.sst_ds = self.sst_ds.isel(
                nlat=slice(self.lat_min, self.lat_max+1),
                nlon=slice(self.lon_min, self.lon_max+1)
            ).chunk(base_chunks)
            
            sst_valid = ~np.isnan(self.sst_ds["SST"].isel(time=0))
            
            self.valid_mask = ssh_valid & sst_valid
            valid_points_count = np.sum(self.valid_mask)
            total_points = np.prod(self.valid_mask.shape)
            
            # self.logger.info(f"Valid points: {valid_points_count} out of {total_points} {valid_points_count/total_points*100:.2f}%)")
            
            if valid_points_count == 0:
                raise ValueError("No valid data points found in the Atlantic region")
                
            # Step 3: Apply mask and compute statistics
            with dask_monitor.profile_task("compute_statistics"):
                self.logger.info("Computing SSH statistics...")
                self.ssh_mean, self.ssh_std = self._compute_mean_std(
                    self.ssh_ds["SSH"].where(self.valid_mask), 
                    var_name="SSH"
                )
                
                self.logger.info("Computing SST statistics...")
                self.sst_mean, self.sst_std = self._compute_mean_std(
                    self.sst_ds["SST"].where(self.valid_mask),
                    var_name="SST"
                )

            # Step 4: Load and process VNT dataset
            self.logger.info("Loading VNT dataset...")
            vnt_chunks = {'time': 12, 'nlat': 300, 'nlon': 450, 'z_t': None}
            self.vnt_ds = xr.open_dataset(vnt_path, engine="zarr", chunks=None)
            self.vnt_ds = self.vnt_ds.isel(
                nlat=slice(self.lat_min, self.lat_max+1),
                nlon=slice(self.lon_min, self.lon_max+1)
            )
            
            # Apply the same mask to VNT-related variables
            self.vnt_ds["VNT"] = self.vnt_ds["VNT"].where(self.valid_mask)
            self.vnt_ds["TAREA"] = self.vnt_ds["TAREA"].where(self.valid_mask)
            self.vnt_ds["dz"] = self.vnt_ds["dz"].where(self.valid_mask)
            
            self.vnt_ds = self.vnt_ds.chunk(vnt_chunks).unify_chunks()
            
            self.log_chunk_stats(self.vnt_ds["VNT"], array_name="VNT")
            
            self.tarea_conversion = 0.0001
            self.dz_conversion = 0.01
            
            # Store shape after processing
            self.shape = self.sst_ds["SST"].shape
            
            # Step 5: Store statistics for validation
            self.stats = {
                'ssh_valid_points': valid_points_count,
                'total_points': total_points,
                'coverage_percentage': valid_points_count/total_points*100,
                'ssh_mean': float(self.ssh_mean.values.mean()),
                'ssh_std': float(self.ssh_std.values.mean()),
                'sst_mean': float(self.sst_mean.values.mean()),
                'sst_std': float(self.sst_std.values.mean()),
                'grid_shape': self.shape
            }
            
            # Step 6: Validate statistics
            self._validate_statistics()
            
            self.logger.info("Successfully loaded all datasets")
            
        except Exception as e:
            self.logger.error(f"Error loading datasets: {str(e)}")
            raise

    def _validate_statistics(self):
        """Validate the computed statistics to ensure they make sense."""
        
        # Check data coverage
        coverage_threshold = 10  # 10% minimum coverage
        if self.stats['coverage_percentage'] < coverage_threshold:
            raise ValueError(
                f"Insufficient data coverage: {self.stats['coverage_percentage']:.2f}% "
                f"(threshold: {coverage_threshold}%)"
            )
        
        # Check for reasonable ranges in statistics
        stat_checks = {
            'ssh_mean': (-500, 500),    # reasonable SSH range in cm
            'sst_mean': (-5, 35),       # reasonable SST range in °C
            'ssh_std': (0, 100),        # reasonable SSH std in cm
            'sst_std': (0, 20)          # reasonable SST std in °C
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
            


    def _compute_mean_std(self, data_array, var_name: str, dim="time"):
        """Compute mean and standard deviation for valid (non-NaN) data points only."""
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
        
        # Create masks for valid data (excluding both NaNs and zeros where appropriate)
        valid_data = ~np.isnan(data_array)
        if var_name in ['SSH', 'SST']:  # For these variables, zero might be a valid value
            data_mask = valid_data
        else:  # For other variables like VNT, we might want to exclude zeros
            data_mask = valid_data & (data_array != 0)
        
        # Use xarray's where to properly handle the masking
        masked_data = data_array.where(data_mask)
        
        # Compute statistics only for valid data points
        sum_valid = da.sum(masked_data, axis=data_array.get_axis_num(dim))
        count_valid = da.sum(data_mask, axis=data_array.get_axis_num(dim))
        sum_squares_valid = da.sum((masked_data ** 2), axis=data_array.get_axis_num(dim))
        
        # Compute global statistics
        total_sum = sum_valid.sum().compute()
        total_count = count_valid.sum().compute()
        total_sum_squares = sum_squares_valid.sum().compute()
        
        # Calculate mean and std only for valid points
        global_mean = total_sum / total_count
        global_variance = (total_sum_squares / total_count) - (global_mean ** 2)
        global_std = np.sqrt(np.maximum(global_variance, 0))  # Ensure non-negative
        
        dims = [d for d in data_array.dims if d != dim]
        coords = {k: v for k, v in data_array.coords.items() if k != dim}
        if "time" in coords:
            coords.pop("time")
            
        mean_da = xr.DataArray(global_mean, dims=dims, coords=coords)
        std_da = xr.DataArray(global_std, dims=dims, coords=coords)
        
        # Cache the statistics
        np.save(cache_file, np.array([mean_da.values, std_da.values]))
        
        return mean_da, std_da

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
                 ssh_mean, ssh_std, sst_mean, sst_std, shape, valid_mask):
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
        self.valid_mask = valid_mask

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        try:
            ssh_data, sst_data = dask.compute(
                self.ssh.isel(time=idx),
                self.sst.isel(time=idx)
            )
            
            # Create valid data mask
            mask = ~np.isnan(ssh_data.values) & ~np.isnan(sst_data.values)
            
            # Normalize data
            ssh_norm = (ssh_data.values - self.ssh_mean.values) / self.ssh_std.values
            sst_norm = (sst_data.values - self.sst_mean.values) / self.sst_std.values
            
            # Zero out NaN values after normalization
            ssh_norm = np.where(mask, ssh_norm, 0)
            sst_norm = np.where(mask, sst_norm, 0)
            
            ssh_tensor = torch.from_numpy(ssh_norm).float().unsqueeze(0)
            sst_tensor = torch.from_numpy(sst_norm).float().unsqueeze(0)
            mask_tensor = torch.from_numpy(mask).float()
            
            ssh_downsampled = F.avg_pool2d(ssh_tensor, kernel_size=2, stride=2)
            sst_downsampled = F.avg_pool2d(sst_tensor, kernel_size=2, stride=2)
            mask_downsampled = F.avg_pool2d(mask_tensor.unsqueeze(0), kernel_size=2, stride=2).squeeze(0)
            
            return (
                ssh_downsampled,
                sst_downsampled,
                mask_downsampled,
                torch.tensor(self.heat_transport[idx]).float()
            )
        except Exception as e:
            logging.error(f"Error in __getitem__ at index {idx}: {str(e)}")
            raise