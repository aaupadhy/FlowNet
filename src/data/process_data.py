import xarray as xr
import numpy as np
import dask
import dask.array as da
import torch
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
        time_chunk = 180
        base_chunks = {'time': time_chunk, 'nlat': 'auto', 'nlon': 'auto'}

        with dask_monitor.profile_task("load_datasets"):
            self.ssh_ds = xr.open_dataset(ssh_path, engine="zarr", chunks=None)
            self.ssh_ds = self.ssh_ds.chunk(base_chunks)
            self.sst_ds = xr.open_dataset(sst_path, engine="zarr", chunks=None)
            self.sst_ds = self.sst_ds.chunk(base_chunks)
            vnt_chunks = {'time': 12, 'nlat': 300, 'nlon': 300, 'z_t': None}
            self.vnt_ds = xr.open_dataset(vnt_path, engine="zarr", chunks=None)
            self.vnt_ds = self.vnt_ds.chunk(vnt_chunks)

        with dask_monitor.profile_task("compute_statistics"):
            self.ssh_mean, self.ssh_std = self._compute_mean_std(self.ssh_ds["SSH"], "SSH")
            self.sst_mean, self.sst_std = self._compute_mean_std(self.sst_ds["SST"], "SST")
            self.shape = self.ssh_ds["SSH"].shape
            self.reference_latitude = self._find_reference_latitude()

        self.tarea_conversion = 0.0001
        self.dz_conversion = 0.01
        
        self.stats = {
            'grid_shape': self.shape,
            'ssh_mean': self.ssh_mean,
            'ssh_std': self.ssh_std,
            'sst_mean': self.sst_mean,
            'sst_std': self.sst_std,
            'spatial_dims': {'nlat': self.shape[1], 'nlon': self.shape[2]},
            'time_steps': self.shape[0]
        }
        self._validate_statistics()
        self._validate_data_consistency()
        self.logger.info("Successfully loaded all datasets")

    def _cleanup_memory(self):
        gc.collect()
        torch.cuda.empty_cache()

    def _validate_data_consistency(self):
        if not (len(self.ssh_ds.time) == len(self.sst_ds.time) == len(self.vnt_ds.time)):
            raise ValueError("Temporal dimensions do not match across datasets")
        ssh_shape = self.ssh_ds.SSH.shape
        sst_shape = self.sst_ds.SST.shape
        if ssh_shape != sst_shape:
            raise ValueError(f"Spatial dimensions mismatch: SSH {ssh_shape} vs SST {sst_shape}")
        if not np.allclose(self.ssh_ds.nlat, self.sst_ds.nlat) or \
           not np.allclose(self.ssh_ds.nlon, self.sst_ds.nlon):
            raise ValueError("Coordinate systems do not match between datasets")

    def _validate_statistics(self):
        stat_checks = {
            'ssh_mean': (-500, 500),
            'sst_mean': (-5, 35),
            'ssh_std': (0, 100),
            'sst_std': (0, 20)
        }
        for stat_name, (min_val, max_val) in stat_checks.items():
            value = self.stats[stat_name]
            if not isinstance(value, (int, float)):
                raise ValueError(f"{stat_name} is not a scalar value")
            if not np.isfinite(value):
                raise ValueError(f"Invalid {stat_name}: {value}")
            if value == 0:
                raise ValueError(f"Zero {stat_name} detected")
            if not min_val <= value <= max_val:
                raise ValueError(
                    f"{stat_name} ({value:.2f}) outside reasonable range "
                    f"[{min_val}, {max_val}]"
                )

    def _find_reference_latitude(self) -> int:
        cache_file = self.cache_dir / 'ref_lat_index.npy'
        if cache_file.exists():
            try:
                ref_lat_idx = int(np.load(cache_file))
                tlat = self.vnt_ds['TLAT']
                cached_lats = tlat[ref_lat_idx].compute()
                valid_points = ((cached_lats >= 39.9) & (cached_lats <= 40.1)).sum()
                if valid_points > 0:
                    return ref_lat_idx
            except Exception as e:
                self.logger.warning(f"Failed to load cached reference latitude: {str(e)}")

        try:
            tlat = self.vnt_ds['TLAT']
            lat_array = tlat.compute()
            target_mask = (lat_array >= 39.9) & (lat_array <= 40.1)
            valid_rows = np.where(target_mask)[0]
            if len(valid_rows) == 0:
                raise ValueError("No points found between 39.9°N and 40.1°N")
            ref_lat_idx = valid_rows[len(valid_rows)//2]
            np.save(cache_file, ref_lat_idx)
            return ref_lat_idx
        except Exception as e:
            self.logger.error(f"Error finding reference latitude: {str(e)}")
            raise
        finally:
            self._cleanup_memory()

    def _compute_mean_std(self, data_array, var_name: str):
        cache_file = self.cache_dir / f'{var_name}_stats.npy'
        if cache_file.exists():
            try:
                stats = np.load(cache_file)
                return float(stats[0]), float(stats[1])
            except Exception as e:
                self.logger.warning(f"Failed to load cached stats: {str(e)}")

        try:
            data_masked = data_array.where(~np.isnan(data_array))
            with dask_monitor.profile_task(f"compute_{var_name}_stats"):
                mean = float(data_masked.mean().compute())
                std = float(data_masked.std().compute())
                if not np.isfinite(mean) or not np.isfinite(std):
                    raise ValueError(f"Invalid statistics: mean={mean}, std={std}")
                if std < 1e-6:
                    std = 1.0
            try:
                np.save(cache_file, np.array([mean, std]))
            except Exception as e:
                self.logger.warning(f"Failed to cache statistics: {str(e)}")
            return mean, std
        except Exception as e:
            self.logger.error(f"Error computing statistics for {var_name}: {str(e)}")
            raise

    def calculate_heat_transport(self, latitude_index=None):
        if latitude_index is None:
            latitude_index = self.reference_latitude

        cache_file = self.cache_dir / f'heat_transport_lat{latitude_index}.npy'
        cache_mean_file = self.cache_dir / f'heat_transport_lat{latitude_index}_mean.npy'

        if cache_file.exists() and cache_mean_file.exists():
            try:
                heat_transport = np.load(cache_file)
                mean_transport = np.load(cache_mean_file)
                return heat_transport, mean_transport
            except Exception as e:
                self.logger.warning(f"Failed to load cached data: {str(e)}")

        try:
            vnt = self.vnt_ds["VNT"].isel(nlat=latitude_index)
            tarea = self.vnt_ds["TAREA"].isel(nlat=latitude_index)
            dz = self.vnt_ds["dz"]

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

            return heat_transport, mean_transport
        except Exception as e:
            self.logger.error(f"Error calculating heat transport: {str(e)}")
            raise
        finally:
            self._cleanup_memory()

    def get_spatial_data(self):
        return (
            self.ssh_ds["SSH"],
            self.sst_ds["SST"],
            self.vnt_ds["TLAT"],
            self.vnt_ds["TLONG"]
        )

class OceanDataset(torch.utils.data.Dataset):
    def __init__(self, ssh, sst, heat_transport, heat_transport_mean,
                 ssh_mean, ssh_std, sst_mean, sst_std, shape, debug=False):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing OceanDataset")

        if debug:
            self.ssh = ssh.isel(time=slice(0, 32))
            self.sst = sst.isel(time=slice(0, 32))
            self.heat_transport = heat_transport[:32]
            self.length = 32
        else:
            self.logger.info("Computing dask arrays...")
            with dask_monitor.profile_task("compute_dataset_arrays"):
                self.ssh = ssh.compute()
                self.sst = sst.compute()
            self.heat_transport = heat_transport
            self.length = shape[0]
            self.logger.info(f"Dataset arrays computed. Shape: {self.ssh.shape}")

        heat_transport_std = np.std(heat_transport)
        self.heat_transport = (heat_transport - heat_transport_mean) / heat_transport_std

        self.ssh_mean = float(ssh_mean)
        self.ssh_std = float(ssh_std)
        self.sst_mean = float(sst_mean)
        self.sst_std = float(sst_std)

        self.logger.info(f"Normalization parameters:")
        self.logger.info(f"  SSH mean: {self.ssh_mean:.4f}")
        self.logger.info(f"  SSH std:  {self.ssh_std:.4f}")
        self.logger.info(f"  SST mean: {self.sst_mean:.4f}")
        self.logger.info(f"  SST std:  {self.sst_std:.4f}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        try:
            self.logger.debug(f"Fetching item {idx}")
            ssh_vals = self.ssh[idx].values
            sst_vals = self.sst[idx].values

            self.logger.debug(f"Processing masks for item {idx}")
            ssh_valid = ~np.isnan(ssh_vals)
            sst_valid = ~np.isnan(sst_vals)

            self.logger.debug(f"Normalizing data for item {idx}")
            ssh_norm = np.where(ssh_valid,
                              (ssh_vals - self.ssh_mean) / (self.ssh_std + 1e-6),
                              0)
            sst_norm = np.where(sst_valid,
                              (sst_vals - self.sst_mean) / (self.sst_std + 1e-6),
                              0)

            self.logger.debug(f"Creating tensors for item {idx}")
            ssh_tensor = torch.from_numpy(ssh_norm).float().unsqueeze(0)
            sst_tensor = torch.from_numpy(sst_norm).float().unsqueeze(0)
            mask_tensor = torch.from_numpy(ssh_valid & sst_valid).float()

            self.logger.debug(f"Downsampling for item {idx}")
            ssh_downsampled = F.avg_pool2d(ssh_tensor.unsqueeze(0),
                                         kernel_size=2,
                                         stride=2).squeeze(0)
            sst_downsampled = F.avg_pool2d(sst_tensor.unsqueeze(0),
                                         kernel_size=2,
                                         stride=2).squeeze(0)
            mask_downsampled = F.avg_pool2d(mask_tensor.unsqueeze(0).unsqueeze(0),
                                          kernel_size=2,
                                          stride=2).squeeze(0).squeeze(0)

            self.logger.debug(f"Successfully prepared item {idx}")
            return (
                ssh_downsampled,
                sst_downsampled,
                mask_downsampled,
                torch.tensor(self.heat_transport[idx]).float()
            )
        except Exception as e:
            self.logger.error(f"Error in __getitem__ at index {idx}: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise