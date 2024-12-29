import argparse
import yaml
import logging
import sys
from pathlib import Path
import torch
import numpy as np

from src.data.process_data import OceanDataProcessor
from src.utils.visualization import OceanVisualizer
from src.utils.dask_utils import dask_monitor
from src.models.transformer import OceanTransformer
from src.train import OceanTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration file"""
    logger.info("Loading configuration...")
    config_path = Path("config/data.yml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    logger.info("Configuration loaded successfully")
    return config

def analyze_data(config, data_processor, visualizer):
    """Analyze the data"""
    logger.info("Starting data analysis...")
    
    try:
        logger.info("Processing spatial data...")
        ssh, sst, tlat, tlong = data_processor.get_spatial_data()
        
        logger.info("Creating visualizations...")
        visualizer.plot_spatial_pattern(
            data=ssh.isel(time=0),
            tlat=tlat,
            tlong=tlong,
            title="SSH Pattern",
            save_path="ssh_pattern"
        )
        
        visualizer.plot_spatial_pattern(
            data=sst.isel(time=0),
            tlat=tlat,
            tlong=tlong,
            title="SST Pattern",
            save_path="sst_pattern"
        )
        
        logger.info("Analysis completed successfully")
        return ssh, sst
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        raise

def train_model(config, data_processor):
    """Train the model"""
    logger.info("Starting model training...")
    
    try:
        logger.info("Loading training data...")
        ssh, sst, tlat, tlong = data_processor.get_spatial_data()
        heat_transport = data_processor.calculate_heat_transport(
            config['data']['reference_latitude']
        )
        
        spatial_size = (ssh.shape[1], ssh.shape[2])
        model = OceanTransformer(
            spatial_size=spatial_size,
            d_model=config['model']['d_model'],
            nhead=config['model']['nhead'],
            num_layers=config['model']['num_layers'],
            dim_feedforward=config['model']['dim_feedforward'],
            dropout=config['model']['dropout']
        )
        
        trainer = OceanTrainer(
            model=model,
            config=config,
            save_dir=Path(config['paths']['model_dir']),
            device=config['training']['device']
        )
        
        train_loader, val_loader, test_loader = trainer.create_dataloaders(
            ssh_data=ssh.values,
            sst_data=sst.values,
            heat_transport=heat_transport.values
        )
        
        trainer.train(train_loader, val_loader)
        
        metrics, predictions, attention_maps = trainer.evaluate(test_loader)
        
        logger.info(f"Test metrics: {metrics}")
        return trainer, test_loader, attention_maps
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise

def visualize_results(trainer, test_loader, attention_maps, visualizer):
    """Visualize model results and attention maps"""
    logger.info("Visualizing results...")
    
    try:
        # Plot example predictions
        metrics, predictions, _ = trainer.evaluate(test_loader)
        
        # Plot attention maps
        for i in range(min(3, len(attention_maps))):  # Plot first 3 attention maps
            visualizer.plot_attention_map(
                attention_maps[i],
                title=f"Attention Map {i+1}",
                save_path=f"attention_map_{i+1}"
            )
        
        logger.info("Results visualization completed")
        
    except Exception as e:
        logger.error(f"Error during visualization: {str(e)}", exc_info=True)
        raise

def main():
    parser = argparse.ArgumentParser(description='Ocean Heat Transport Analysis')
    parser.add_argument('--mode', type=str, choices=['analyze', 'train', 'all'],
                      required=True, help='Mode of operation')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode')
    
    args = parser.parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting Ocean Heat Transport Analysis...")
    
    try:
        config = load_config()
        
        logger.info("Setting up Dask client...")
        client = dask_monitor.start_client()
        logger.info(f"Dask dashboard available at: {client.dashboard_link}")
        
        data_processor = OceanDataProcessor(
            ssh_path=config['paths']['ssh_zarr'],
            sst_path=config['paths']['sst_zarr'],
            vnt_path=config['paths']['vnt_zarr']
        )
        
        visualizer = OceanVisualizer(
            output_dir=config['paths']['output_dir']
        )
        
        if args.mode in ['analyze', 'all']:
            ssh, sst = analyze_data(config, data_processor, visualizer)
        
        if args.mode in ['train', 'all']:
            trainer, test_loader, attention_maps = train_model(config, data_processor)
            visualize_results(trainer, test_loader, attention_maps, visualizer)
        
        logger.info("All operations completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("Cleaning up resources...")
        logger.info("Shutting down Dask...")
        dask_monitor.shutdown()
        logger.info("Cleanup completed")

if __name__ == "__main__":
    main()