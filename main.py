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
from src.utils.visualization import OceanVisualizer
from src.architecture.transformer import OceanTransformer

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
        logger.info("Dask disabled in configuration")
        import dask
        dask.config.set(scheduler='synchronous')
        
    with dask_monitor.monitor_operation("initialization"):
        logger.info("Creating OceanDataProcessor")
        preload = cfg.get("preload_data", False)
        dp = OceanDataProcessor(
            cfg['paths']['ssh_zarr'],
            cfg['paths']['sst_zarr'],
            cfg['paths']['vnt_zarr'],
            preload_data=preload
        )
        
    with dask_monitor.monitor_operation("training_data_preparation"):
        logger.info("Calculating heat transport")
        heat_transport, ht_mean, ht_std = dp.calculate_heat_transport()
        logger.info("Using heat transport mean: %s", ht_mean)
        
    if dask_enabled:
        logger.info("Shutting down Dask cluster")
        dask_monitor.shutdown()

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
    
    logger.info("Training completed. Starting evaluation phase...")
    try:
        viz = OceanVisualizer(cfg['paths']['output_dir'])
        
        test_metrics, predictions, test_truth, attn_dict = trainer.evaluate(test_loader, ht_mean, ht_std)
        
        logger.info("Plotting predictions...")
        viz.plot_predictions(
            predictions, 
            test_truth, 
            time_indices=np.arange(len(predictions)), 
            save_path='final_predictions'
        )
        
        logger.info("Plotting error histogram...")
        viz.plot_error_histogram(
            predictions,
            test_truth,
            save_path='prediction_errors'
        )
        
        if attn_dict and 'attn' in attn_dict:
            logger.info("Plotting attention maps...")
            viz.plot_attention_maps(attn_dict, save_path='final_attention_maps')
        else:
            logger.warning("No attention maps available for visualization.")
        
        logger.info("Plotting temporal trends...")
        viz.plot_temporal_trends(
            np.arange(len(predictions)),
            test_truth,
            predictions,
            save_path='temporal_trends'
        )
        
        logger.info("Final Test Metrics:")
        for metric_name, value in test_metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
            
        if cfg.get('wandb_mode', 'online') != 'disabled':
            wandb.run.summary.update(test_metrics)
            
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        logger.error(traceback.format_exc())
    
    finally:
        logger.info("Cleaning up resources...")
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except Exception as e:
            logger.warning(f"Error while closing wandb: {str(e)}")

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