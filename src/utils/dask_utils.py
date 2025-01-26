import dask
from dask.distributed import Client, LocalCluster, performance_report
import psutil
import numpy as np
import time
from contextlib import contextmanager
import logging
from typing import Dict, Any, Optional
import json
from pathlib import Path
import pandas as pd

class ChunkProfiler:
    def __init__(self, logger):
        self.logger = logger
        self.metrics = []
        self.output_dir = Path('chunk_profiles')
        # self.output_dir.mkdir(exist_ok=True)
    
    def profile_chunks(self, dask_array, operation='compute'):
        """Profile performance of current chunking"""
        start_time = time.time()
        mem_start = psutil.Process().memory_info().rss
        
        # Get chunk info
        chunks = dask_array.chunks
        chunk_sizes = [np.prod(c) * dask_array.dtype.itemsize / 1e6 for c in chunks]  # MB
        
        with dask.config.set({"optimization.fuse.active": False}):  # Disable fusion for accurate measurements
            result = dask_array.compute()
        
        duration = time.time() - start_time
        mem_peak = psutil.Process().memory_info().rss
        mem_used = mem_peak - mem_start
        
        metrics = {
            'operation': operation,
            'duration': duration,
            'memory_used_gb': mem_used / (1024**3),
            'memory_peak_gb': mem_peak / (1024**3),
            'chunk_sizes_mb': chunk_sizes,
            'n_chunks': len(dask_array.chunks[0]) * len(dask_array.chunks[1]),
            'throughput_mbs': result.nbytes / duration / 1e6
        }
        
        self.logger.info(f"Operation: {operation}")
        self.logger.info(f"Duration: {duration:.2f}s")
        self.logger.info(f"Memory Used: {metrics['memory_used_gb']:.2f}GB")
        self.logger.info(f"Throughput: {metrics['throughput_mbs']:.2f}MB/s")
        
        self.metrics.append(metrics)
        return metrics
    
    def suggest_optimal_chunks(self, history_window=5):
        """Suggest optimal chunk sizes based on history"""
        if len(self.metrics) < history_window:
            return None
            
        recent_metrics = self.metrics[-history_window:]
        best_throughput = max(m['throughput_mbs'] for m in recent_metrics)
        best_config = next(m for m in recent_metrics if m['throughput_mbs'] == best_throughput)
        
        return {
            'recommended_chunks': best_config['chunk_sizes_mb'],
            'expected_throughput': best_throughput,
            'confidence': min(1.0, len(self.metrics) / 10)  # Increases with more data
        }
    
    def save_profile(self):
        """Save profiling results"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # Save metrics to CSV
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.output_dir / f'chunk_profile_{timestamp}.csv', index=False)
        
        if len(self.metrics) > 0:
            best = self.suggest_optimal_chunks()
            if best:
                with open(self.output_dir / f'recommendations_{timestamp}.json', 'w') as f:
                    json.dump(best, f, indent=2)

class DaskManager:
    def __init__(self, logger, total_memory: int = 300):
        self.logger = logger
        self.total_memory = total_memory
        self.chunk_profiler = ChunkProfiler(logger)
        self.performance_metrics = []
        self.client = None
        self.cluster = None
        
    @contextmanager
    def profile_task(self, task_name: str):
        """Profile a specific task with timing and memory monitoring"""
        start_time = time.time()
        start_mem = psutil.Process().memory_info().rss
        
        try:
            self.logger.info(f"Starting task: {task_name}")
            yield
        finally:
            end_time = time.time()
            end_mem = psutil.Process().memory_info().rss
            duration = end_time - start_time
            mem_change = (end_mem - start_mem) / (1024**3)  # Convert to GB
            
            metrics = {
                'task': task_name,
                'duration': duration,
                'memory_change_gb': mem_change,
                'peak_memory_gb': psutil.Process().memory_info().rss / (1024**3),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.performance_metrics.append(metrics)
            self.logger.info(f"Task {task_name} completed in {duration:.2f}s")
            self.logger.info(f"Memory change: {mem_change:.2f}GB")
    
    @contextmanager
    def monitor_operation(self, operation_name: str):
        """Context manager for monitoring operations"""
        start_time = time.time()
        start_mem = psutil.Process().memory_info().rss
        
        try:
            self.logger.info(f"Starting operation: {operation_name}")
            yield
        finally:
            end_time = time.time()
            end_mem = psutil.Process().memory_info().rss
            duration = end_time - start_time
            mem_change = (end_mem - start_mem) / 1024**3
            
            metrics = {
                'operation': operation_name,
                'duration': duration,
                'memory_change_gb': mem_change,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'peak_memory_gb': psutil.Process().memory_info().rss / (1024**3)
            }
            
            self.performance_metrics.append(metrics)
            self.logger.info(f"Completed: {operation_name}")
            self.logger.info(f"Duration: {duration:.2f}s")
            self.logger.info(f"Memory Change: {mem_change:.2f}GB")
    
    def setup_optimal_cluster(self, data_analysis: Dict[str, Any]) -> Client:
        """Setup optimized Dask cluster"""
        with self.monitor_operation("cluster_setup"):
            n_workers = 2
            mem_per_worker = int(self.total_memory * 0.95 / n_workers)
            
            self.logger.info(f"Setting up cluster with {n_workers} workers")
            self.logger.info(f"Memory per worker: {mem_per_worker}GB")
            
            dask.config.set({
                'distributed.comm.timeouts.connect': '30s',
                'distributed.comm.timeouts.tcp': '30s',
                'distributed.nanny.daemon.exit-on-closed-stream': False,
                'distributed.worker.memory.target': 0.6,    # Use 60% memory before spilling
                'distributed.worker.memory.spill': 0.70,    # Spill at 70%
                'distributed.worker.memory.pause': 0.75,    # Pause at 75%
                'distributed.worker.memory.terminate': 0.80, # Terminate at 80%
                'distributed.scheduler.work-stealing': False # Prevent task stealing between workers
            })
            
            self.cluster = LocalCluster(
                n_workers=2,
                threads_per_worker=2,
                memory_limit="60GB",
                silence_logs=logging.WARNING,
                dashboard_address=':8787',
            )
            
            self.client = Client(self.cluster)
            
            self.logger.info(f"Dask dashboard available at: {self.client.dashboard_link}")
        
            self.client = Client(self.cluster)
            return self.client 

    def auto_optimize_chunks(self, dask_array, operation_name: str):
        with self.monitor_operation(f"optimize_chunks_{operation_name}"):
            # Add memory safety check
            current_mem = psutil.Process().memory_info().rss / (1024**3)
            if current_mem > self.total_memory * 0.8:
                self.logger.warning("Memory usage too high, forcing garbage collection")
                import gc
                gc.collect()
                
            if current_mem > self.total_memory * 0.7:
                chunk_size = "500MB"
            else:
                chunk_size = "750MB"
                
            dask_array = dask_array.rechunk(chunks=chunk_size)
            return dask_array

    
    def save_monitoring_data(self, output_dir: Optional[str] = None):
        """Save all monitoring data"""
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = Path('monitoring_data')
        
        output_path.mkdir(exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        df = pd.DataFrame(self.performance_metrics)
        df.to_csv(output_path / f'performance_metrics_{timestamp}.csv', index=False)
        
        self.chunk_profiler.save_profile()
        
        # Save summary report
        summary = {
            'total_operations': len(self.performance_metrics),
            'total_duration': sum(m['duration'] for m in self.performance_metrics),
            'peak_memory': max(m['peak_memory_gb'] for m in self.performance_metrics),
            'timestamp': timestamp
        }
        
        with open(output_path / f'summary_{timestamp}.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Monitoring data saved to {output_path}")
    
    def shutdown(self):
        """Cleanup Dask resources"""
        if self.client:
            self.client.close()
        if self.cluster:
            self.cluster.close()

dask_monitor = DaskManager(logging.getLogger(__name__))