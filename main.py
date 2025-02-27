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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ocean_heat_transport.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Ocean Heat Transport Prediction")
    parser.add_argument('--mode', type=str, default='all', 
                        choices=['all', 'train', 'evaluate', 'visualize'],
                        help='Operation mode (all, train, evaluate, visualize)')
    parser.add_argument('--config', type=str, default='config/data.yml',
                        help='Path to configuration file')
    parser.add_argument('--debug', action='store_true', 
                        help='Enable debug mode for faster iteration')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file for evaluation or resuming training')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use (-1 for CPU)')
    return parser.parse_args()

def load_config(path):
    """Load configuration from YAML file"""
    try:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

def ensure_dir_exists(dir_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.info(f"Created directory: {dir_path}")

def setup_device(gpu_id):
    """Set up compute device (GPU or CPU)"""
    if gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)
        logger.info(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device

def train_model(args, device):
    """
    Train ocean heat transport model
    
    Args:
        args: Command line arguments
        device: Compute device (GPU or CPU)
    
    Returns:
        Trained model, data processor, trainer
    """
    logger.info("Starting model training process")
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Override settings if debug mode is enabled
    if args.debug:
        logger.info("Debug mode enabled - using reduced dataset and faster training")
        cfg['training']['debug_mode'] = True
        cfg['training']['epochs'] = min(5, cfg['training'].get('epochs', 20))
        cfg['training']['batch_size'] = min(16, cfg['training'].get('batch_size', 32))
    
    # Disable wandb if requested
    if args.no_wandb:
        cfg['wandb_mode'] = "disabled"
        logger.info("Weights & Biases logging disabled")
    
    # Create output directories
    ensure_dir_exists(cfg['paths']['output_dir'])
    ensure_dir_exists(cfg['paths']['model_dir'])
    ensure_dir_exists(os.path.join(cfg['paths']['model_dir'], 'checkpoints'))
    ensure_dir_exists(os.path.join(cfg['paths']['output_dir'], 'plots'))
    ensure_dir_exists(os.path.join(cfg['paths']['output_dir'], 'attention_maps'))
    
    # Set up Dask for distributed computing
    dask_enabled = cfg.get("dask", {}).get("enabled", True)
    if dask_enabled:
        logger.info("Setting up Dask distributed computing environment")
        dask_monitor.setup_optimal_cluster({
            'n_workers': cfg['dask'].get('n_workers', 4),
            'threads_per_worker': cfg['dask'].get('threads_per_worker', 4),
            'use_processes': cfg['dask'].get('use_processes', False)
        })
    else:
        logger.info("Dask disabled in configuration")
        import dask
        dask.config.set(scheduler='synchronous')
    
    # Initialize data processor
    with dask_monitor.profile_task("data_processing"):
        logger.info("Initializing OceanDataProcessor")
        preload = cfg.get("preload_data", False)
        data_processor = OceanDataProcessor(
            cfg['paths']['ssh_zarr'],
            cfg['paths']['sst_zarr'],
            cfg['paths']['vnt_zarr'],
            preload_data=preload
        )
    
    # Calculate heat transport and prepare VNT data
    with dask_monitor.profile_task("heat_transport_calculation"):
        logger.info("Calculating heat transport")
        heat_transport, ht_mean, ht_std = data_processor.calculate_heat_transport()
        logger.info(f"Heat transport statistics - mean: {ht_mean:.4f}, std: {ht_std:.4f}")
        
        # Get data for VNT aggregation
        logger.info("Preparing data for heat transport aggregation")
        tarea, dz = data_processor.get_aggregation_data()
        
        # Prepare VNT data for supervision if enabled
        vnt_data = None
        if cfg['training'].get('vnt_supervision', False):
            logger.info("Loading VNT data for direct supervision")
            # Only load reference latitude for efficiency
            vnt_data = data_processor.get_vnt_slice(
                time_indices=None,  # All time steps
                lat_indices=[data_processor.reference_latitude]
            )
            logger.info(f"VNT data shape: {vnt_data.shape}")
    
    # Clean up Dask resources after data loading
    if dask_enabled:
        logger.info("Shutting down Dask cluster")
        dask_monitor.shutdown()
    
    # Initialize model
    target_nlat = data_processor.stats['spatial_dims']['nlat']
    target_nlon = data_processor.stats['spatial_dims']['nlon']
    vnt_depth = cfg['training'].get('vnt_depth', 62)  # Depth levels in VNT
    
    logger.info(f"Creating OceanTransformer model with dimensions: ({target_nlat}, {target_nlon})")
    model = OceanTransformer(
        spatial_size=(data_processor.stats['spatial_dims']['nlat'], 
                      data_processor.stats['spatial_dims']['nlon']),
        d_model=cfg['training']['d_model'],
        nhead=cfg['training']['n_heads'],
        num_layers=cfg['training']['num_layers'],
        dim_feedforward=cfg['training']['dim_feedforward'],
        dropout=cfg['training']['dropout'],
        patch_size=cfg['training'].get('patch_size', 8),
        target_nlat=target_nlat,
        target_nlon=target_nlon,
        vnt_depth=vnt_depth,
        tarea=tarea,
        dz=dz,
        ref_lat_index=data_processor.reference_latitude
    )
    
    # Log model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model initialized with {total_params:,} total parameters")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    trainer = OceanTrainer(model, cfg, save_dir=cfg['paths']['model_dir'], device=device)
    
    # Get data for training
    ssh, sst, _, _ = data_processor.get_spatial_data()
    
    # Create dataloaders
    train_loader, val_loader, test_loader = trainer.create_dataloaders(
        ssh, sst, heat_transport, ht_mean, ht_std,
        data_processor.ssh_mean, data_processor.ssh_std, 
        data_processor.sst_mean, data_processor.sst_std,
        data_processor.stats['grid_shape'],
        vnt_data=vnt_data, 
        tarea=tarea, 
        dz=dz, 
        ref_lat_index=data_processor.reference_latitude
    )
    
    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint is not None:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        start_epoch = trainer.load_checkpoint(args.checkpoint)
    
    # Train the model
    logger.info(f"Starting training from epoch {start_epoch}")
    try:
        trainer.train(train_loader, val_loader, start_epoch=start_epoch)
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        logger.error(traceback.format_exc())
    
    return model, data_processor, trainer, test_loader

def evaluate_model(model, data_processor, trainer, test_loader, cfg):
    """
    Evaluate trained model
    
    Args:
        model: Trained model
        data_processor: Data processor instance
        trainer: Trainer instance
        test_loader: Test data loader
        cfg: Configuration dictionary
    """
    logger.info("Starting model evaluation")
    
    try:
        # Create visualizer
        viz = OceanVisualizer(cfg['paths']['output_dir'])
        
        # Evaluate on test set
        test_metrics, predictions, test_truth, attn_dict = trainer.evaluate(test_loader)
        
        # Visualize results
        logger.info("Creating visualization plots")
        
        # Plot predictions vs ground truth
        viz.plot_predictions(
            predictions,
            test_truth,
            time_indices=np.arange(len(predictions)),
            save_path='final_predictions',
            title='Heat Transport Predictions'
        )
        
        # Plot error distribution
        viz.plot_error_histogram(
            predictions,
            test_truth,
            save_path='prediction_errors',
            title='Prediction Error Distribution'
        )
        
        # Plot attention maps
        if attn_dict and 'attn' in attn_dict:
            logger.info("Plotting attention maps")
            viz.plot_attention_maps(
                attn_dict, 
                save_path='final_attention_maps',
                title='Spatial Attention Patterns'
            )
        else:
            logger.warning("No attention maps available for visualization")
        
        # Plot time series
        viz.plot_temporal_trends(
            np.arange(len(predictions)),
            test_truth,
            predictions,
            save_path='temporal_trends',
            title='Heat Transport Time Series'
        )
        
        # Evaluate VNT-based prediction approach
        if cfg['training'].get('eval_vnt_aggregation', True):
            logger.info("Evaluating heat transport from aggregated VNT")
            vnt_metrics = trainer.calculate_heat_transport_from_vnt(
                test_loader, 
                data_processor.get_aggregation_data()[0],  # tarea 
                data_processor.get_aggregation_data()[1],  # dz
                ref_lat_index=data_processor.reference_latitude
            )
            
            # Compare direct and VNT-based approaches
            if 'predictions' in vnt_metrics:
                logger.info("Plotting comparison between direct and VNT-based predictions")
                vnt_pred = vnt_metrics['predictions']
                
                # Plot comparison
                viz.plot_predictions(
                    vnt_pred,
                    test_truth,
                    time_indices=np.arange(len(vnt_pred)),
                    save_path='vnt_aggregated_predictions',
                    title='VNT-Aggregated Heat Transport Predictions'
                )
                
                # Plot difference between approaches
                viz.plot_method_comparison(
                    predictions,
                    vnt_pred,
                    test_truth,
                    save_path='prediction_method_comparison',
                    title='Direct vs. VNT-Aggregated Predictions'
                )
        
        # Log final metrics
        logger.info("Final Test Metrics:")
        for metric_name, value in test_metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        logger.info("Evaluation completed")
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except Exception as e:
            logger.warning(f"Error while closing wandb: {str(e)}")

def main():
    """Main execution function"""
    # Parse command line arguments
    args = parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")
    
    # Set random seeds for reproducibility
    setup_random_seeds()
    
    # Set up compute device
    device = setup_device(args.gpu)
    
    try:
        if args.mode in ['all', 'train']:
            # Train model
            model, data_processor, trainer, test_loader = train_model(args, device)
            
            # Evaluate model if training was successful
            if args.mode == 'all':
                cfg = load_config(args.config)
                evaluate_model(model, data_processor, trainer, test_loader, cfg)
                
        elif args.mode == 'evaluate':
            # Load configuration
            cfg = load_config(args.config)
            
            # Check for checkpoint
            if args.checkpoint is None:
                logger.error("Checkpoint path must be provided for evaluation mode")
                return
            
            # Initialize data processor
            logger.info("Initializing data processor for evaluation")
            data_processor = OceanDataProcessor(
                cfg['paths']['ssh_zarr'],
                cfg['paths']['sst_zarr'],
                cfg['paths']['vnt_zarr'],
                preload_data=False
            )
            
            # Calculate heat transport
            heat_transport, ht_mean, ht_std = data_processor.calculate_heat_transport()
            tarea, dz = data_processor.get_aggregation_data()
            
            # Initialize model for evaluation
            target_nlat = data_processor.stats['spatial_dims']['nlat']
            target_nlon = data_processor.stats['spatial_dims']['nlon']
            vnt_depth = cfg['training'].get('vnt_depth', 62)
            
            model = OceanTransformer(
                spatial_size=(data_processor.stats['spatial_dims']['nlat'], 
                              data_processor.stats['spatial_dims']['nlon']),
                d_model=cfg['training']['d_model'],
                nhead=cfg['training']['n_heads'],
                num_layers=cfg['training']['num_layers'],
                dim_feedforward=cfg['training']['dim_feedforward'],
                dropout=cfg['training']['dropout'],
                patch_size=cfg['training'].get('patch_size', 8),
                target_nlat=target_nlat,
                target_nlon=target_nlon,
                vnt_depth=vnt_depth,
                tarea=tarea,
                dz=dz,
                ref_lat_index=data_processor.reference_latitude
            )
            
            # Initialize trainer
            trainer = OceanTrainer(model, cfg, save_dir=cfg['paths']['model_dir'], device=device)
            
            # Load checkpoint
            trainer.load_checkpoint(args.checkpoint)
            
            # Create test dataloader
            ssh, sst, _, _ = data_processor.get_spatial_data()
            _, _, test_loader = trainer.create_dataloaders(
                ssh, sst, heat_transport, ht_mean, ht_std,
                data_processor.ssh_mean, data_processor.ssh_std, 
                data_processor.sst_mean, data_processor.sst_std,
                data_processor.stats['grid_shape'],
                vnt_data=None,
                tarea=tarea, 
                dz=dz, 
                ref_lat_index=data_processor.reference_latitude
            )
            
            # Evaluate model
            evaluate_model(model, data_processor, trainer, test_loader, cfg)
            
        elif args.mode == 'visualize':
            logger.info("Visualization mode not yet implemented")
            # To be implemented if needed for standalone visualization
            
        else:
            logger.info(f"Unsupported mode: {args.mode}")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        logger.error(traceback.format_exc())
    finally:
        logger.info("Execution completed")

if __name__ == '__main__':
    main()