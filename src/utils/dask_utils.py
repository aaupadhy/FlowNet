from dask.distributed import Client, LocalCluster, performance_report
import dask
import logging
import os
import psutil
import time
from contextlib import contextmanager
import json
from typing import Dict, Optional, Any


class DaskMonitor:

    def __init__(self, dashboard_port: int = 8787) -> None:
        """
        Initialize Dask monitoring system.

        Args:
            dashboard_port: Port for Dask dashboard
        """
        self.dashboard_port = dashboard_port
        self.client = None
        self.cluster = None
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for Dask monitoring."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger

    def _get_slurm_memory(self) -> Optional[float]:
        """Get memory allocated by SLURM in GB."""
        try:
            return int(os.environ.get('SLURM_MEM_PER_NODE', 0)) / 1024
        except (ValueError, TypeError):
            return None

    def _get_available_memory(self) -> float:
        """Get available system memory in GB."""
        return psutil.virtual_memory().total / (1024**3)

    def _calculate_worker_resources(self) -> Dict[str, Any]:
        """Calculate optimal worker resources based on available memory."""
        total_memory = self._get_slurm_memory() or self._get_available_memory()
        n_cores = int(
            os.environ.get('SLURM_CPUS_PER_TASK',
                           psutil.cpu_count(logical=False)))

        usable_memory = total_memory * 0.9

        n_workers = max(1, n_cores // 2)
        memory_per_worker = int(usable_memory / n_workers)
        threads_per_worker = max(1, n_cores // n_workers)

        self.logger.info(f"Total memory: {total_memory:.2f}GB")
        self.logger.info(f"Number of workers: {n_workers}")
        self.logger.info(f"Memory per worker: {memory_per_worker}GB")
        self.logger.info(f"Threads per worker: {threads_per_worker}")

        return {
            'n_workers': n_workers,
            'threads_per_worker': threads_per_worker,
            'memory_limit': f"{memory_per_worker}GB",
            'memory_target_fraction': 0.75,
            'memory_spill_fraction': 0.85,
            'memory_pause_fraction': 0.95
        }

    def start_client(self) -> Client:
        """Start Dask client with optimized settings."""
        resources = self._calculate_worker_resources()

        self.cluster = LocalCluster(
            n_workers=resources['n_workers'],
            threads_per_worker=resources['threads_per_worker'],
            memory_limit=resources['memory_limit'],
            dashboard_address=f':{self.dashboard_port}',
            scheduler_port=0)

        dask.config.set({
            'distributed.worker.memory.target':
            resources['memory_target_fraction'],
            'distributed.worker.memory.spill':
            resources['memory_spill_fraction'],
            'distributed.worker.memory.pause':
            resources['memory_pause_fraction'],
            'distributed.worker.memory.terminate':
            0.98,
            'distributed.comm.compression':
            'snappy',
            'distributed.scheduler.work-stealing':
            True,
            'array.chunk-size':
            f"{int((float(resources['memory_limit'][:-2]) * 1024 * 0.1))}MiB",
            'distributed.worker.profile.interval':
            '10ms',
            'distributed.worker.profile.cycle':
            '1s'
        })

        self.client = Client(self.cluster)
        self.logger.info(
            f"Dask dashboard available at: {self.client.dashboard_link}")
        return self.client

    @contextmanager
    def profile_task(self, task_name: str):
        """Context manager for task profiling."""
        start_time = time.time()
        profile_path = f"profiling/{task_name}_{int(start_time)}"
        os.makedirs(os.path.dirname(profile_path), exist_ok=True)

        with performance_report(filename=f"{profile_path}.html"):
            yield

        duration = time.time() - start_time
        metrics = {
            'task_name': task_name,
            'duration': duration,
            'peak_memory': psutil.Process().memory_info().rss / 1024**2,
            'cpu_percent': psutil.Process().cpu_percent(),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(f"{profile_path}_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)

        self.logger.info(f"Task {task_name} completed in {duration:.2f}s")

    def shutdown(self) -> None:
        """Shutdown Dask client and cluster gracefully."""
        try:
            if self.client:
                self.logger.info("Shutting down Dask client...")
                self.client.close()
                self.client = None

            if self.cluster:
                self.logger.info("Shutting down Dask cluster...")
                self.cluster.close()
                self.cluster = None

            self.logger.info("Dask shutdown completed successfully")

        except Exception as e:
            self.logger.warning(f"Warning during Dask shutdown: {str(e)}")


dask_monitor = DaskMonitor()
