import argparse
import yaml
from pathlib import Path
import logging
import sys
from contextlib import contextmanager
import time
from datetime import datetime
import torch
import traceback
import wandb
from src.models.transformer import OceanTransformer
from src.data.process_data import OceanDataProcessor, OceanDataset
from src.utils.visualization import OceanVisualizer
from src.utils.dask_utils import dask_monitor
import torch._dynamo
torch._dynamo.config.disable = True
from src.train import OceanTrainer
from src.utils.ocean_monitor import ocean_monitor
from src.utils.training_monitor import TrainingMonitor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ocean_analysis.log')
    ]
)

logger = logging.getLogger(__name__)

def setup_directories(config):
    """Ensure all required directories exist"""
    dirs = [
        config['paths']['output_dir'],
        config['paths']['model_dir'],
        Path(config['paths']['model_dir']) / 'checkpoints',
        Path(config['paths']['output_dir']) / 'plots',
        Path(config['paths']['output_dir']) / 'attention_maps'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_path}")

@contextmanager
def timing_block(name):
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        logger.info(f"{name} completed in {duration:.2f} seconds")

def load_config(config_path: str) -> dict:
    """Load and validate configuration"""
    logger.info(f"Loading configuration from {config_path}")
    
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        required_keys = ['paths', 'dask', 'data', 'monitoring']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration section: {key}")
        
        for path_key in ['output_dir', 'model_dir']:
            Path(config['paths'][path_key]).mkdir(parents=True, exist_ok=True)
        
        logger.info("Configuration loaded successfully")
        return config
        
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise

def analyze_data(config: dict, data_processor: OceanDataProcessor) -> None:
    logger.info("Starting data analysis phase")
    
    try:
        with timing_block("Data Analysis"):
            with dask_monitor.monitor_operation("spatial_data_loading"):
                ssh, sst, tlat, tlong = data_processor.get_spatial_data()
                
                visualizer = OceanVisualizer(
                    output_dir=config['paths']['output_dir'],
                    fig_size=tuple(config['visualization']['fig_size']),
                    dpi=config['visualization']['dpi']
                )
                
                ocean_monitor.analyze_spatial_coverage(ssh, "SSH")
                ocean_monitor.analyze_spatial_coverage(sst, "SST")
                ocean_monitor.analyze_temporal_patterns(ssh, "SSH")
                ocean_monitor.analyze_temporal_patterns(sst, "SST")
                
                ocean_monitor.validate_data(ssh, "SSH")
                ocean_monitor.validate_data(sst, "SST")
            
            if config['dask'].get('chunk_optimization', True):
                ssh = dask_monitor.auto_optimize_chunks(ssh, "SSH")
                sst = dask_monitor.auto_optimize_chunks(sst, "SST")
            
            visualizer.plot_spatial_pattern(
                ssh.isel(time=0).compute(),
                tlat,
                tlong,
                "SSH Pattern",
                save_path='ssh_pattern'
            )
            visualizer.plot_spatial_pattern(
                sst.isel(time=0).compute(),
                tlat,
                tlong,
                "SST Pattern",
                save_path='sst_pattern'
            )
            
            ocean_monitor.plot_spatial_coverage(ssh, "SSH")
            ocean_monitor.plot_spatial_coverage(sst, "SST")
            ocean_monitor.create_performance_dashboard()
            
            logger.info("Analysis phase completed successfully")
            return ssh, sst
            
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def train_model(config: dict, data_processor: OceanDataProcessor) -> None:
    torch.cuda.set_per_process_memory_fraction(0.9)
    torch.backends.cuda.max_split_size_mb = 512
    logger.info("Starting model training phase")
    
    try:
        with timing_block("Model Training"):
            
            with dask_monitor.monitor_operation("training_data_preparation"):
                ssh, sst, tlat , tlong = data_processor.get_spatial_data()
                config['data'].update({
                'tlat': tlat.compute(),
                'tlong': tlong.compute()
                })
                heat_transport, heat_transport_mean = data_processor.calculate_heat_transport()
                
                logger.info(f"Using heat transport mean: {heat_transport_mean}")
            
            torch.backends.cudnn.benchmark = True
            
            spatial_size = (data_processor.shape[1], data_processor.shape[2])
            
            model = OceanTransformer(
                spatial_size=spatial_size,
                d_model=config['training']['d_model'],
                nhead=config['training']['n_heads'],
                num_layers=config['training']['num_layers'],
                dim_feedforward=config['training']['dim_feedforward'],
                dropout=config['training']['dropout']
            ).cuda()
            
            if hasattr(torch, 'compile'):
                model = torch.compile(model, backend='eager')
            
            trainer = OceanTrainer(
                model=model,
                config=config,
                save_dir=Path(config['paths']['model_dir']),
                device='cuda'
            )
            
            train_loader, val_loader, test_loader = trainer.create_dataloaders(
                ssh, 
                sst,
                heat_transport,
                heat_transport_mean,
                data_processor.ssh_mean,
                data_processor.ssh_std,
                data_processor.sst_mean,
                data_processor.sst_std,
                data_processor.shape
            )
            
            checkpoint_path = Path(config['paths']['model_dir']) / 'checkpoints/latest_checkpoint.pt'
            start_epoch = 0
            if checkpoint_path.exists():
                start_epoch, metrics = trainer.resume_from_checkpoint(str(checkpoint_path))
                logger.info(f"Resumed from epoch {start_epoch} with metrics: {metrics}")
            
            # trainer.profile_model(train_loader)
            
            trainer.train(train_loader, val_loader, start_epoch=start_epoch)
            
            logger.info("Generating predictions and attention visualizations")
            test_metrics, predictions, attention_maps = trainer.evaluate(
                test_loader, 
                heat_transport_mean
            )
            
            visualizer = OceanVisualizer(output_dir=config['paths']['output_dir'])
            
            attention_maps = attention_maps.mean(dim=1)
            attention_maps = attention_maps.reshape((-1, len(ssh.nlat), len(ssh.nlon)))
            
            visualizer.plot_attention_maps(
                attention_maps,
                ssh.nlat, 
                ssh.nlon, 
                save_path='attention_maps'
            )
            
            visualizer.plot_predictions(
                predictions, 
                heat_transport,
                time_indices=np.arange(len(predictions)),
                save_path='predictions'
            )
            
            logger.info(f"Test metrics: {test_metrics}")
            
            results = {
                'test_metrics': test_metrics,
                'predictions': predictions.tolist(),
                'attention_maps': attention_maps.cpu().numpy().tolist()
            }
            
            with open(Path(config['paths']['output_dir']) / 'results.json', 'w') as f:
                json.dump(results, f)
            
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def main():
    parser = argparse.ArgumentParser(description='Ocean Heat Transport Analysis')
    parser.add_argument('--config', default='config/data.yml', help='Path to configuration file')
    parser.add_argument('--mode', choices=['analyze', 'train', 'all'], default='train',
                      help='Mode of operation')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        config = load_config(args.config)
        setup_directories(config)
        run_name = f"FlowNet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project="FlowNet",
            name=run_name,
            config=config,
            settings=wandb.Settings(start_method="thread")
        )
        with dask_monitor.monitor_operation("initialization"):
            client = dask_monitor.setup_optimal_cluster(config['dask'])
            logger.info(f"Dask dashboard available at: {client.dashboard_link}")
            
            data_processor = OceanDataProcessor(
                ssh_path=config['paths']['ssh_zarr'],
                sst_path=config['paths']['sst_zarr'],
                vnt_path=config['paths']['vnt_zarr']
            )
        
        if args.mode in ['train', 'all']:
            train_model(config, data_processor)
        
        ocean_monitor.save_metrics()
        ocean_monitor.create_monitoring_report()
        dask_monitor.save_monitoring_data(config['paths']['output_dir'])
        
        logger.info("All operations completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
        
    finally:
        logger.info("Cleaning up resources...")
        dask_monitor.shutdown()

if __name__ == "__main__":
    main()