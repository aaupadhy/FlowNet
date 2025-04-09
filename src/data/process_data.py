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
import time
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
            self.ssh_ds = xr.open_zarr(ssh_path, chunks=base_chunks)
            
            logger.info("Opening SST dataset")
            self.sst_ds = xr.open_zarr(sst_path, chunks=base_chunks)
            
            logger.info("Opening VNT dataset")
            vnt_chunks = {'time': time_chunk, 'nlat': 1, 'nlon': 'auto', 'z_t': -1}
            self.vnt_ds = xr.open_zarr(vnt_path, chunks=vnt_chunks)
        
        if preload_data:
            self._preload_data(cache_data)
        
        self._compute_and_cache_stats()
        
        self.tarea_conversion = 0.0001
        self.dz_conversion = 0.01
        
        self._validate_data_consistency()
        logger.info("Successfully loaded all datasets")
    
    def _preload_data(self, cache_data):
        t_preload = time.time()
        logger.info("Preloading SSH data")
        
        self.ssh_ds = self.ssh_ds.load()
        logger.info("SSH data loaded")
        
        logger.info("Preloading SST data")
        self.sst_ds = self.sst_ds.load()
        logger.info("SST data loaded")
        
        logger.info("Preloading completed in %.2f seconds", time.time() - t_preload)
        
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
            
            gc.collect()
    
    def _compute_and_cache_stats(self):
        t0 = time.time()
        
        with dask_monitor.profile_task("compute_statistics"):
            logger.info("Computing SSH mean and std")
            self.ssh_mean, self.ssh_std = self._compute_mean_std(self.ssh_ds["SSH"], "SSH")
            
            logger.info("Computing SST mean and std")
            self.sst_mean, self.sst_std = self._compute_mean_std(self.sst_ds["SST"], "SST")
            
            self.shape = self.ssh_ds["SSH"].shape
            
            logger.info("Computing reference latitude")
            self.reference_latitude = self._find_reference_latitude()
        
        t_elapsed = time.time() - t0
        logger.info("Statistics computed in %.2f seconds", t_elapsed)
        
        self.stats = {
            'grid_shape': self.shape,
            'ssh_mean': self.ssh_mean,
            'ssh_std': self.ssh_std,
            'sst_mean': self.sst_mean,
            'sst_std': self.sst_std,
            'spatial_dims': {'nlat': self.shape[1], 'nlon': self.shape[2]},
            'time_steps': self.shape[0],
            'reference_latitude': self.reference_latitude
        }
        
        self._validate_statistics()
    
    def _cleanup_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _validate_data_consistency(self):
        if not (len(self.ssh_ds.time) == len(self.sst_ds.time) == len(self.vnt_ds.time)):
            raise ValueError("Temporal dimensions do not match across datasets")
        
        ssh_shape = self.ssh_ds.SSH.shape
        sst_shape = self.sst_ds.SST.shape
        if ssh_shape != sst_shape:
            raise ValueError(f"Spatial dimensions mismatch: SSH {ssh_shape} vs SST {sst_shape}")
        
        if not np.allclose(self.ssh_ds.nlat, self.sst_ds.nlat):
            raise ValueError("Latitude coordinates do not match between SSH and SST datasets")
            
        if not np.allclose(self.ssh_ds.nlon, self.sst_ds.nlon):
            raise ValueError("Longitude coordinates do not match between SSH and SST datasets")
    
    def validate_coordinates(self):
        for coord in ['nlat', 'nlon', 'time']:
            ssh_coord = self.ssh_ds.coords.get(coord)
            sst_coord = self.sst_ds.coords.get(coord)
            vnt_coord = self.vnt_ds.coords.get(coord)
            
            if ssh_coord is None or sst_coord is None or vnt_coord is None:
                raise ValueError(f"Missing coordinate: {coord}")
                
        logger.info("All spatial and temporal coordinates are valid.")
    
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
            tlat = self.vnt_ds['TLAT'].compute()
            target_mask = (tlat >= 39.9) & (tlat <= 40.1)
            valid_rows = np.where(target_mask.any(axis=1))[0]
            
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
        
        t0 = time.time()
        data_masked = data_array.where(~np.isnan(data_array))
        
        with dask_monitor.profile_task(f"compute_{var_name}_stats"):
            mean = float(data_masked.mean().compute())
            std = float(data_masked.std().compute())
        
        t_elapsed = time.time() - t0
        logger.info("%s mean and std computed in %.2f seconds", var_name, t_elapsed)
        
        np.save(cache_file, np.array([mean, std]))
        return mean, std
    
    def calculate_heat_transport(self, latitude_index=None):
        if latitude_index is None:
            latitude_index = self.reference_latitude
        
        t0 = time.time()
        logger.info("Calculating heat transport at latitude index %d", latitude_index)
        
        with dask_monitor.profile_task("heat_transport_calculation"):
            vnt_slice = self.get_vnt_slice(lat_indices=[latitude_index])
            
            tarea = self.vnt_ds['TAREA'].isel(nlat=latitude_index)
            dz = self.vnt_ds['dz']
            
            heat_transport = (
                vnt_slice * 
                tarea * self.tarea_conversion * 
                dz * self.dz_conversion
            )
            
            heat_transport = heat_transport.sum(dim=['z_t', 'nlon'])
            
            heat_transport = heat_transport.compute()
            
            mean = float(heat_transport.mean())
            std = float(heat_transport.std())
            
            logger.info("Heat transport statistics - mean: %.4f, std: %.4f", mean, std)
            
            t_elapsed = time.time() - t0
            logger.info("Heat transport calculation completed in %.2f seconds", t_elapsed)
            
            return heat_transport, mean, std
    
    def get_vnt_slice(self, time_indices=None, lat_indices=None, depth_indices=None):
        vnt = self.vnt_ds['VNT']
        
        if time_indices is not None:
            vnt = vnt.isel(time=time_indices)
        if lat_indices is not None:
            vnt = vnt.isel(nlat=lat_indices)
        if depth_indices is not None:
            vnt = vnt.isel(z_t=depth_indices)
            
        return vnt
    
    def get_aggregation_data(self):
        return self.vnt_ds['TAREA'], self.vnt_ds['dz']
    
    def get_spatial_data(self):
        return self.ssh_ds['SSH'], self.sst_ds['SST'], self.ssh_mean, self.ssh_std

def aggregate_vnt(predicted_vnt, tarea, dz, tarea_conversion=0.0001, dz_conversion=0.01, ref_lat_index=0):
    batch_size = predicted_vnt.shape[0]
    
    tarea_expanded = tarea.view(1, 1, 1, -1).expand(batch_size, -1, -1, -1)
    dz_expanded = dz.view(1, -1, 1, 1).expand(batch_size, -1, -1, predicted_vnt.shape[-1])
    
    heat_transport = (
        predicted_vnt * 
        tarea_expanded * tarea_conversion * 
        dz_expanded * dz_conversion
    )
    
    heat_transport = heat_transport.sum(dim=[1, 3])
    
    if ref_lat_index is not None:
        heat_transport = heat_transport[:, ref_lat_index]
        
    return heat_transport

class OceanDataset(torch.utils.data.Dataset):
    def __init__(self, ssh, sst, heat_transport, heat_transport_mean, heat_transport_std,
                 ssh_mean, ssh_std, sst_mean, sst_std, shape, vnt_data=None, 
                 tarea=None, dz=None, ref_lat_index=0, debug=False, 
                 log_target=False, target_scale=10.0):
        self.ssh = ssh
        self.sst = sst
        self.heat_transport = heat_transport
        self.heat_transport_mean = heat_transport_mean
        self.heat_transport_std = heat_transport_std
        self.ssh_mean = ssh_mean
        self.ssh_std = ssh_std
        self.sst_mean = sst_mean
        self.sst_std = sst_std
        self.shape = shape
        self.vnt_data = vnt_data
        self.tarea = tarea
        self.dz = dz
        self.ref_lat_index = ref_lat_index
        self.debug = debug
        self.log_target = log_target
        self.target_scale = target_scale
        
        if debug:
            self.ssh = self.ssh[:100]
            self.sst = self.sst[:100]
            self.heat_transport = self.heat_transport[:100]
            if vnt_data is not None:
                self.vnt_data = self.vnt_data[:100]
    
    def _extract_months(self, time_coord):
        months = []
        for t in time_coord:
            try:
                month = pd.to_datetime(str(t)).month
                months.append(month)
            except:
                months.append(1)
        return np.array(months)
    
    def _process_targets(self):
        if self.log_target:
            self.heat_transport = np.log1p(self.heat_transport * self.target_scale)
            self.heat_transport_mean = np.log1p(self.heat_transport_mean * self.target_scale)
            self.heat_transport_std = self.heat_transport_std * self.target_scale / (self.heat_transport_mean + 1)
    
    def __len__(self):
        return len(self.ssh)
    
    def __getitem__(self, idx):
        ssh = self.ssh[idx]
        sst = self.sst[idx]
        ht = self.heat_transport[idx]
        
        ssh = (ssh - self.ssh_mean) / self.ssh_std
        sst = (sst - self.sst_mean) / self.sst_std
        ht = (ht - self.heat_transport_mean) / self.heat_transport_std
        
        ssh = torch.FloatTensor(ssh).unsqueeze(0)
        sst = torch.FloatTensor(sst).unsqueeze(0)
        ht = torch.FloatTensor(ht)
        
        if self.vnt_data is not None:
            vnt = self.vnt_data[idx]
            vnt = torch.FloatTensor(vnt)
            return ssh, sst, ht, vnt
        
        return ssh, sst, ht