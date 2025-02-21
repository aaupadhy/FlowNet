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
from time import time
from ..utils.dask_utils import dask_monitor
import pandas as pd

logger = logging.getLogger(__name__)

class OceanDataProcessor:
    def __init__(self, ssh_path: str, sst_path: str, vnt_path: str, preload_data: bool = False, cache_data: bool = True):
        logger.info("Initializing OceanDataProcessor")
        self.cache_dir = Path('/scratch/user/aaupadhy/college/RA/final_data/cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        time_chunk = 180
        base_chunks = {'time': time_chunk, 'nlat': 'auto', 'nlon': 'auto'}
        with dask_monitor.profile_task("load_datasets"):
            logger.info("Opening SSH dataset")
            self.ssh_ds = xr.open_dataset(ssh_path, engine="zarr", chunks=None).chunk(base_chunks)
            logger.info("Opening SST dataset")
            self.sst_ds = xr.open_dataset(sst_path, engine="zarr", chunks=None).chunk(base_chunks)
            logger.info("Opening VNT dataset")
            vnt_chunks = {'time': 12, 'nlat': 300, 'nlon': 300, 'z_t': None}
            self.vnt_ds = xr.open_dataset(vnt_path, engine="zarr", chunks=None).chunk(vnt_chunks)
        if preload_data:
            t_preload = time()
            logger.info("Preloading SSH and SST into memory")
            self.ssh_ds = self.ssh_ds.load()
            self.sst_ds = self.sst_ds.load()
            logger.info("Preloading completed in %.2f seconds", time() - t_preload)
            logger.info("SSH data: min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
                        float(self.ssh_ds["SSH"].min()), float(self.ssh_ds["SSH"].max()),
                        float(self.ssh_ds["SSH"].mean()), float(self.ssh_ds["SSH"].std()))
            logger.info("SST data: min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
                        float(self.sst_ds["SST"].min()), float(self.sst_ds["SST"].max()),
                        float(self.sst_ds["SST"].mean()), float(self.sst_ds["SST"].std()))
            if cache_data:
                self.ssh_array = self.ssh_ds["SSH"].values
                self.sst_array = self.sst_ds["SST"].values
                logger.info("Cached SSH and SST arrays in memory.")
        t0 = time()
        with dask_monitor.profile_task("compute_statistics"):
            logger.info("Computing SSH mean and std")
            self.ssh_mean, self.ssh_std = self._compute_mean_std(self.ssh_ds["SSH"], "SSH")
            logger.info("Computing SST mean and std")
            self.sst_mean, self.sst_std = self._compute_mean_std(self.sst_ds["SST"], "SST")
            self.shape = self.ssh_ds["SSH"].shape
            logger.info("Computing reference latitude")
            self.reference_latitude = self._find_reference_latitude()
        t_elapsed = time() - t0
        logger.info("Statistics computed in %.2f seconds", t_elapsed)
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
        logger.info("Successfully loaded all datasets")

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
        if not np.allclose(self.ssh_ds.nlat, self.sst_ds.nlat) or not np.allclose(self.ssh_ds.nlon, self.sst_ds.nlon):
            raise ValueError("Coordinate systems do not match between datasets")

    def validate_coordinates(self):
        for coord in ['nlat', 'nlon', 'time']:
            ssh_coord = self.ssh_ds.coords.get(coord)
            sst_coord = self.sst_ds.coords.get(coord)
            vnt_coord = self.vnt_ds.coords.get(coord)
            if ssh_coord is None or sst_coord is None or vnt_coord is None:
                raise ValueError(f"Missing coordinate: {coord}")
            if not np.array_equal(ssh_coord.values, sst_coord.values) or not np.array_equal(ssh_coord.values, vnt_coord.values):
                raise ValueError(f"Mismatch found in coordinate {coord}")
        logger.info("All spatial and temporal coordinates match.")

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
                raise ValueError(f"{stat_name} ({value:.2f}) outside reasonable range [{min_val}, {max_val}]")

    def _find_reference_latitude(self) -> int:
        cache_file = self.cache_dir / 'ref_lat_index.npy'
        if cache_file.exists():
            try:
                ref_lat_idx = int(np.load(cache_file))
                tlat = self.vnt_ds['TLAT']
                cached_lats = tlat[ref_lat_idx].compute()
                valid_points = ((cached_lats >= 39.9) & (cached_lats <= 40.1)).sum()
                if valid_points > 0:
                    logger.info("Using cached reference latitude: %d", ref_lat_idx)
                    return ref_lat_idx
            except Exception as e:
                logger.warning("Failed to load cached reference latitude: %s", str(e))
        try:
            tlat = self.vnt_ds['TLAT']
            lat_array = tlat.compute()
            target_mask = (lat_array >= 39.9) & (lat_array <= 40.1)
            valid_rows = np.where(target_mask)[0]
            if len(valid_rows) == 0:
                raise ValueError("No points found between 39.9°N and 40.1°N")
            ref_lat_idx = valid_rows[len(valid_rows)//2]
            np.save(cache_file, ref_lat_idx)
            logger.info("Computed and cached reference latitude: %d", ref_lat_idx)
            return ref_lat_idx
        except Exception as e:
            logger.error("Error finding reference latitude: %s", str(e))
            raise
        finally:
            self._cleanup_memory()

    def _compute_mean_std(self, data_array, var_name: str):
        cache_file = self.cache_dir / f'{var_name}_stats.npy'
        if cache_file.exists():
            try:
                stats = np.load(cache_file)
                logger.info("Loaded cached %s stats", var_name)
                return float(stats[0]), float(stats[1])
            except Exception as e:
                logger.warning("Failed to load cached stats: %s", str(e))
        t0 = time()
        data_masked = data_array.where(~np.isnan(data_array))
        with dask_monitor.profile_task(f"compute_{var_name}_stats"):
            mean = float(data_masked.mean().compute())
            std = float(data_masked.std().compute())
        t_elapsed = time() - t0
        logger.info("%s mean and std computed in %.2f seconds", var_name, t_elapsed)
        if not np.isfinite(mean) or not np.isfinite(std):
            raise ValueError(f"Invalid statistics for {var_name}: mean={mean}, std={std}")
        if std < 1e-6:
            std = 1.0
        try:
            np.save(cache_file, np.array([mean, std]))
            logger.info("Cached %s stats", var_name)
        except Exception as e:
            logger.warning("Failed to cache statistics: %s", str(e))
        return mean, std

    def calculate_heat_transport(self, latitude_index=None):
        if latitude_index is None:
            latitude_index = self.reference_latitude
        cache_file = self.cache_dir / f'heat_transport_lat{latitude_index}.npy'
        cache_mean_file = self.cache_dir / f'heat_transport_lat{latitude_index}_mean.npy'
        cache_std_file = self.cache_dir / f'heat_transport_lat{latitude_index}_std.npy'
        if cache_file.exists() and cache_mean_file.exists() and cache_std_file.exists():
            try:
                heat_transport = np.load(cache_file)
                mean_transport = np.load(cache_mean_file)
                std_transport = np.load(cache_std_file)
                logger.info("Loaded cached heat transport for latitude index %d", latitude_index)
                return heat_transport, mean_transport, std_transport
            except Exception as e:
                logger.warning("Failed to load cached heat transport data: %s", str(e))
        try:
            vnt = self.vnt_ds["VNT"].isel(nlat=latitude_index)
            tarea = self.vnt_ds["TAREA"].isel(nlat=latitude_index)
            dz = self.vnt_ds["dz"]
            chunk_size = 60
            total_times = vnt.shape[0]
            heat_transport_list = []
            logger.info("Calculating heat transport in %d-chunk steps", chunk_size)
            for start_idx in tqdm(range(0, total_times, chunk_size), desc="Heat transport chunks", unit="chunk"):
                end_idx = min(start_idx + chunk_size, total_times)
                t0 = time()
                vnt_chunk = vnt.isel(time=slice(start_idx, end_idx))
                chunk_transport = (
                    vnt_chunk *
                    tarea * self.tarea_conversion *
                    dz * self.dz_conversion
                ).sum(dim=['z_t', 'nlon']).compute()
                elapsed = time() - t0
                logger.info("Processed chunk %d-%d in %.2f seconds", start_idx, end_idx, elapsed)
                heat_transport_list.append(chunk_transport)
            heat_transport = np.concatenate(heat_transport_list)
            mean_transport = float(np.mean(heat_transport))
            std_transport = float(np.std(heat_transport))
            np.save(cache_file, heat_transport)
            np.save(cache_mean_file, mean_transport)
            np.save(cache_std_file, std_transport)
            logger.info("Cached heat transport for latitude index %d", latitude_index)
            return heat_transport, mean_transport, std_transport
        except Exception as e:
            logger.error("Error calculating heat transport: %s", str(e))
            raise
        finally:
            self._cleanup_memory()

    def get_spatial_data(self):
        # Return SSH, SST, and coordinate arrays for TLAT and TLONG.
        return (self.ssh_ds["SSH"], self.sst_ds["SST"],
                self.vnt_ds["TLAT"], self.vnt_ds["TLONG"])

def aggregate_vnt(predicted_vnt, tarea, dz, tarea_conversion=0.0001, dz_conversion=0.01, ref_lat_index=0):
    """
    Aggregate the predicted VNT field.
    predicted_vnt: Tensor of shape (B, 62, nlat, nlon)
    Multiply elementwise with tarea and dz (with conversion factors) and sum over z_t and nlon.
    Select the row corresponding to the reference latitude.
    """
    agg = (predicted_vnt * tarea * tarea_conversion * dz * dz_conversion).sum(dim=[1, 3])
    return agg[:, ref_lat_index]

class OceanDataset(torch.utils.data.Dataset):
    def __init__(self, ssh, sst, heat_transport, heat_transport_mean, heat_transport_std,
                 ssh_mean, ssh_std, sst_mean, sst_std, shape, debug=False, log_target=False, target_scale=10.0):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing OceanDataset")
        self.log_target = log_target
        self.target_scale = target_scale
        if debug:
            self.ssh = ssh.isel(time=slice(0, 32))
            self.sst = sst.isel(time=slice(0, 32))
            self.heat_transport = heat_transport[:32]
            self.length = 32
            units = self.ssh.coords['time'].attrs.get('units')
            calendar = self.ssh.coords['time'].attrs.get('calendar')
            time_vals = self.ssh.coords['time'].values
            try:
                if units is None or calendar is None:
                    self.months = np.array([pd.Timestamp(str(t)).month for t in time_vals])
                else:
                    decoded = xr.conventions.times.decode_cf_datetime(time_vals, units, calendar)
                    if hasattr(decoded[0], "strftime"):
                        self.months = pd.DatetimeIndex(decoded).month.to_numpy()
                    else:
                        self.months = np.array([pd.Timestamp(str(t)).month for t in decoded])
            except Exception as e:
                self.logger.warning("Failed to decode time using units/calendar: %s", str(e))
                self.months = np.array([pd.Timestamp(str(t)).month for t in time_vals])
        else:
            self.logger.info("Using lazy dask arrays for full dataset")
            self.ssh = ssh
            self.sst = sst
            self.heat_transport = heat_transport
            self.length = shape[0]
            self.logger.info("Dataset arrays set. Shape: %s", ssh.shape)
            units = self.ssh.coords['time'].attrs.get('units')
            calendar = self.ssh.coords['time'].attrs.get('calendar')
            time_vals = self.ssh.coords['time'].values
            try:
                if units is None or calendar is None:
                    self.months = np.array([pd.Timestamp(str(t)).month for t in time_vals])
                else:
                    decoded = xr.conventions.times.decode_cf_datetime(time_vals, units, calendar)
                    if hasattr(decoded[0], "strftime"):
                        self.months = pd.DatetimeIndex(decoded).month.to_numpy()
                    else:
                        self.months = np.array([pd.Timestamp(str(t)).month for t in decoded])
            except Exception as e:
                self.logger.warning("Failed to decode time using units/calendar: %s", str(e))
                self.months = np.array([pd.Timestamp(str(t)).month for t in time_vals])
        self.heat_transport_mean = float(heat_transport_mean)
        self.heat_transport_std = float(heat_transport_std)
        if self.log_target:
            self.logger.info("Applying log transform to heat transport")
            self.heat_transport = np.log(self.heat_transport + 1e-8)
            self.logger.info("Using fixed target scaling factor: %.4f", self.target_scale)
            self.heat_transport = self.heat_transport / self.target_scale
            self.logger.info("Transformed heat transport: min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
                             float(np.min(self.heat_transport)), float(np.max(self.heat_transport)),
                             float(np.mean(self.heat_transport)), float(np.std(self.heat_transport)))
        self.ssh_mean = float(ssh_mean)
        self.ssh_std = float(ssh_std)
        self.sst_mean = float(sst_mean)
        self.sst_std = float(sst_std)
        self.logger.info("Normalization parameters:")
        self.logger.info("  SSH mean: %.4f, SSH std: %.4f", self.ssh_mean, self.ssh_std)
        self.logger.info("  SST mean: %.4f, SST std: %.4f", self.sst_mean, self.sst_std)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        try:
            t0 = time()
            self.logger.debug("Fetching item %d", idx)
            ssh_vals = np.asarray(self.ssh[idx].values, dtype=np.float32)
            sst_vals = np.asarray(self.sst[idx].values, dtype=np.float32)
            self.logger.debug("Processing masks for item %d", idx)
            ssh_valid = ~np.isnan(ssh_vals)
            sst_valid = ~np.isnan(sst_vals)
            self.logger.debug("Normalizing data for item %d", idx)
            ssh_norm = np.where(ssh_valid,
                                (ssh_vals - self.ssh_mean) / (self.ssh_std + 1e-8),
                                0)
            sst_norm = np.where(sst_valid,
                                (sst_vals - self.sst_mean) / (self.sst_std + 1e-8),
                                0)
            self.logger.debug("Creating tensors for item %d", idx)
            ssh_tensor = torch.from_numpy(ssh_norm).float().unsqueeze(0)
            sst_tensor = torch.from_numpy(sst_norm).float().unsqueeze(0)
            mask_tensor = torch.from_numpy(ssh_valid & sst_valid).float()
            target = torch.tensor(self.heat_transport[idx], dtype=torch.float32)
            t_elapsed = time() - t0
            self.logger.debug("Item %d processed in %.4f seconds; SSH min=%.2f, max=%.2f", idx, t_elapsed, float(np.min(ssh_vals)), float(np.max(ssh_vals)))
            return (ssh_tensor, sst_tensor, mask_tensor, target)
        except Exception as e:
            self.logger.error("Error in __getitem__ at index %d: %s", idx, str(e))
            raise
