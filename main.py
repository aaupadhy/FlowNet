import argparse
import logging
import sys
import traceback
import numpy as np
import os
import torch
import yaml
from pathlib import Path
from src.train import OceanTrainer,setup_random_seeds
from src.utils.dask_utils import dask_monitor
from src.data.process_data import OceanDataProcessor
from src.utils.visualization import OceanVisualizer

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',type=str,default='all',help='all|train|predict')
    parser.add_argument('--config',type=str,default='config/data.yml')
    return parser.parse_args()

def load_config(path):
    with open(path,'r') as f:
        return yaml.safe_load(f)

def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def train_model(args):
    cfg=load_config(args.config)
    logging.info(f"Loading configuration from {args.config}")
    logging.info("Configuration loaded successfully")
    ensure_dir_exists(cfg['paths']['output_dir'])
    ensure_dir_exists(cfg['paths']['model_dir'])
    ensure_dir_exists(os.path.join(cfg['paths']['model_dir'],'checkpoints'))
    ensure_dir_exists(os.path.join(cfg['paths']['output_dir'],'plots'))
    ensure_dir_exists(os.path.join(cfg['paths']['output_dir'],'attention_maps'))
    cdata=load_config(args.config)
    dask_monitor.setup_optimal_cluster({'n_workers':cdata['dask']['n_workers'],'threads_per_worker':cdata['dask']['threads_per_worker']})
    with dask_monitor.monitor_operation("initialization"):
        pass
    dp=OceanDataProcessor(cfg['paths']['ssh_zarr'],cfg['paths']['sst_zarr'],cfg['paths']['vnt_zarr'])
    with dask_monitor.monitor_operation("training_data_preparation"):
        heat_transport,ht_mean,ht_std=dp.calculate_heat_transport()
        logging.info(f"Using heat transport mean: {ht_mean}")
    from src.train import OceanTrainer
    from src.architecture.transformer import OceanTransformer
    model=OceanTransformer(spatial_size=(dp.stats['spatial_dims']['nlat'],dp.stats['spatial_dims']['nlon']),d_model=cfg['training']['d_model'],nhead=cfg['training']['n_heads'],num_layers=cfg['training']['num_layers'],dim_feedforward=cfg['training']['dim_feedforward'],dropout=cfg['training']['dropout'])
    trainer=OceanTrainer(model,cfg,save_dir=cfg['paths']['model_dir'])
    ssh, sst, _, _=dp.get_spatial_data()
    train_loader,val_loader,test_loader=trainer.create_dataloaders(ssh,sst,heat_transport,ht_mean,ht_std,dp.ssh_mean,dp.ssh_std,dp.sst_mean,dp.sst_std,dp.stats['grid_shape'])
    trainer.train(train_loader,val_loader,start_epoch=0)
    logging.info("Generating predictions and attention visualizations")
    try:
        test_metrics,predictions,test_truth=trainer.evaluate(test_loader,ht_mean,ht_std)
        viz=OceanVisualizer(cfg['paths']['output_dir'])
        logging.info("Plotting final predictions...")
        viz.plot_predictions(predictions,test_truth,time_indices=np.arange(len(predictions)),save_path='predictions')
    except Exception as e:
        logging.error(f"Error during training: {e}")
        logging.error(traceback.format_exc())
    finally:
        logging.info("Cleaning up resources...")

def main():
    args=parse_args()
    setup_random_seeds()
    try:
        if args.mode=='all' or args.mode=='train':
            train_model(args)
        else:
            pass
    except Exception as e:
        logging.error(f"Error: {e}")
        logging.error(traceback.format_exc())
    finally:
        logging.info("Cleaning up resources...")

if __name__=='__main__':
    main()
