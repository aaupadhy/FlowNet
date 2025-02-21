import argparse
import logging
import sys
import traceback
import numpy as np
import os
import torch
import yaml
from pathlib import Path
from src.train import OceanTrainer, setup_random_seeds
from src.data.process_data import OceanDataProcessor
from src.utils.dask_utils import dask_monitor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='all', help='all|train|predict')
    parser.add_argument('--config', type=str, default='config/data.yml')
    return parser.parse_args()

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def train_model(args):
    logger.info("Entering train_model function")
    cfg = load_config(args.config)
    logger.info("Loading configuration from %s", args.config)
    logger.info("Configuration loaded successfully")
    ensure_dir_exists(cfg['paths']['output_dir'])
    ensure_dir_exists(cfg['paths']['model_dir'])
    ensure_dir_exists(os.path.join(cfg['paths']['model_dir'], 'checkpoints'))
    ensure_dir_exists(os.path.join(cfg['paths']['output_dir'], 'plots'))
    ensure_dir_exists(os.path.join(cfg['paths']['output_dir'], 'attention_maps'))
    dask_enabled = cfg.get("dask", {}).get("enabled", True)
    if dask_enabled:
        logger.info("Setting up Dask cluster")
        dask_monitor.setup_optimal_cluster({
            'n_workers': cfg['dask']['n_workers'],
            'threads_per_worker': cfg['dask']['threads_per_worker'],
            'use_processes': cfg['dask'].get('use_processes', False)
        })
    else:
        logger.info("Dask disabled in configuration; using synchronous scheduler")
        dask.config.set(scheduler='synchronous')
    with dask_monitor.monitor_operation("initialization"):
        logger.info("Initialization phase complete")
    logger.info("Creating OceanDataProcessor")
    preload = cfg.get("preload_data", False)
    dp = OceanDataProcessor(cfg['paths']['ssh_zarr'], cfg['paths']['sst_zarr'], cfg['paths']['vnt_zarr'], preload_data=preload)
    with dask_monitor.monitor_operation("training_data_preparation"):
        logger.info("Calculating heat transport")
        heat_transport, ht_mean, ht_std = dp.calculate_heat_transport()
        logger.info("Using heat transport mean: %s", ht_mean)
    if dask_enabled:
        logger.info("Shutting down Dask cluster")
        dask_monitor.shutdown()
    from src.architecture.transformer import OceanTransformer
    # Set target_nlat and target_nlon equal to the native resolution of SSH/SST
    target_nlat = dp.stats['spatial_dims']['nlat']
    target_nlon = dp.stats['spatial_dims']['nlon']
    model = OceanTransformer(
        spatial_size=(dp.stats['spatial_dims']['nlat'], dp.stats['spatial_dims']['nlon']),
        d_model=cfg['training']['d_model'],
        nhead=cfg['training']['n_heads'],
        num_layers=cfg['training']['num_layers'],
        dim_feedforward=cfg['training']['dim_feedforward'],
        dropout=cfg['training']['dropout'],
        target_nlat=target_nlat,
        target_nlon=target_nlon
    )
    trainer = OceanTrainer(model, cfg, save_dir=cfg['paths']['model_dir'])
    ssh, sst, _, _ = dp.get_spatial_data()
    train_loader, val_loader, test_loader = trainer.create_dataloaders(
        ssh, sst, heat_transport, ht_mean, ht_std,
        dp.ssh_mean, dp.ssh_std, dp.sst_mean, dp.sst_std,
        dp.stats['grid_shape']
    )
    logger.info("Starting training")
    trainer.train(train_loader, val_loader, start_epoch=0)
    logger.info("Generating predictions and attention visualizations")
    try:
        test_metrics, predictions, test_truth, attn_maps = trainer.evaluate(test_loader, ht_mean, ht_std)
        from src.utils.visualization import OceanVisualizer
        viz = OceanVisualizer(cfg['paths']['output_dir'])
        logger.info("Plotting final predictions...")
        viz.plot_predictions(predictions, test_truth, time_indices=np.arange(len(predictions)), save_path='predictions')
        if attn_maps:
            test_iter = iter(test_loader)
            sample = next(test_iter)
            sample_ssh, sample_sst, sample_mask, _ = sample
            attn = trainer.plot_attention_maps(sample_ssh, sample_sst, sample_mask)
            logger.info("Plotting attention maps...")
            _, attn_dict = trainer.model(sample_ssh.to(trainer.device), sample_sst.to(trainer.device), sample_mask.to(trainer.device))
            patch_dims = attn_dict.get('patch_dims', None)
            viz.plot_attention_maps({'attn': attn, 'patch_dims': patch_dims}, save_path='attention_maps')
        else:
            logger.info("No attention maps available for visualization.")
    except Exception as e:
        logger.error("Error during training: %s", e)
        logger.error(traceback.format_exc())
    finally:
        logger.info("Cleaning up resources...")

def main():
    args = parse_args()
    setup_random_seeds()
    try:
        if args.mode in ['all', 'train']:
            train_model(args)
        else:
            logger.info("Mode not implemented")
    except Exception as e:
        logger.error("Error: %s", e)
        logger.error(traceback.format_exc())
    finally:
        logger.info("Cleaning up resources...")

if __name__ == '__main__':
    main()
