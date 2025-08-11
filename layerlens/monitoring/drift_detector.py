"""
Module for detecting changes in layer behavior over time.
"""

import numpy as np
from scipy import stats
import pandas as pd
import datetime
import logging

class DriftDetector:
    """Detects drift in layer behavior over time."""
    
    def __init__(self, reference_data, sensitivity=0.05, layer_names=None):
        """
        Initialize the drift detector.
        
        Args:
            reference_data: Baseline data for drift comparison
            sensitivity (float): Sensitivity threshold for drift detection
            layer_names: Names of layers to monitor (if None, monitors all layers)
        """
        self.reference_data = reference_data
        self.sensitivity = sensitivity
        self.layer_names = layer_names
        self.reference_statistics = {}
        self.drift_history = []
        
        # Initialize reference statistics
        self._compute_reference_statistics()
    
    def _compute_reference_statistics(self):
        """Compute statistics on the reference data."""
        # Check if reference data is a dictionary of layer outputs
        if isinstance(self.reference_data, dict):
            for layer_name, layer_data in self.reference_data.items():
                if self.layer_names is None or layer_name in self.layer_names:
                    self.reference_statistics[layer_name] = self._compute_layer_statistics(layer_data)
        else:
            # Assume it's a single dataset
            self.reference_statistics['global'] = self._compute_layer_statistics(self.reference_data)
    
    def _compute_layer_statistics(self, data):
        """Compute statistics for a single layer."""
        # Flatten the data if it's multi-dimensional
        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)
        
        # Compute basic statistics
        statistics = {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0),
            'median': np.median(data, axis=0),
            'kurtosis': stats.kurtosis(data, axis=0),
            'skewness': stats.skew(data, axis=0)
        }
        
        return statistics
    
    def detect(self, new_data):
        """
        Check for drift in new data.
        
        Args:
            new_data: New data to check for drift
            
        Returns:
            Dictionary with drift detection results
        """
        # Check if new data is a dictionary of layer outputs
        if isinstance(new_data, dict):
            results = {}
            for layer_name, layer_data in new_data.items():
                if layer_name in self.reference_statistics:
                    results[layer_name] = self._detect_layer_drift(
                        layer_data, self.reference_statistics[layer_name]
                    )
        else:
            # Assume it's a single dataset
            results = {'global': self._detect_layer_drift(
                new_data, self.reference_statistics.get('global')
            )}
        
        # Log the results
        self._log_drift_results(results)
        
        return results
    
    def _detect_layer_drift(self, data, reference_stats):
        """
        Detect drift for a single layer.
        
        Args:
            data: New data for the layer
            reference_stats: Reference statistics for the layer
            
        Returns:
            Dictionary with drift metrics
        """
        if reference_stats is None:
            return {'drift_detected': False, 'error': 'No reference statistics available'}
        
        # Flatten the data if it's multi-dimensional
        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)
        
        # Compute statistics on the new data
        new_stats = self._compute_layer_statistics(data)
        
        # Compare with reference statistics
        drift_metrics = {}
        
        # Check mean shift
        mean_diff = np.abs(new_stats['mean'] - reference_stats['mean'])
        mean_threshold = self.sensitivity * reference_stats['std']
        mean_drift = np.mean(mean_diff > mean_threshold)
        
        # Check distribution change using Kolmogorov-Smirnov test
        ks_results = []
        for i in range(min(10, data.shape[1])):  # Limit to 10 features for efficiency
            ks_stat, p_value = stats.ks_2samp(
                data[:, i], 
                np.random.normal(
                    reference_stats['mean'][i], 
                    reference_stats['std'][i], 
                    size=data.shape[0]
                )
            )
            ks_results.append(p_value < self.sensitivity)
        
        ks_drift = np.mean(ks_results)
        
        # Combine drift metrics
        drift_metrics = {
            'mean_drift': float(mean_drift),
            'distribution_drift': float(ks_drift),
            'drift_detected': mean_drift > 0.2 or ks_drift > 0.2,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        return drift_metrics
    
    def _log_drift_results(self, results):
        """Log drift detection results."""
        # Add to history
        self.drift_history.append({
            'timestamp': datetime.datetime.now(),
            'results': results
        })
        
        # Log any detected drift
        for layer_name, layer_results in results.items():
            if layer_results.get('drift_detected', False):
                logging.warning(f"Drift detected in layer {layer_name}: {layer_results}")
    
    def get_drift_metrics(self):
        """
        Get detailed drift metrics.
        
        Returns:
            Pandas DataFrame with drift metrics over time
        """
        if not self.drift_history:
            return pd.DataFrame()
        
        # Extract metrics into a DataFrame
        metrics = []
        
        for entry in self.drift_history:
            for layer_name, layer_results in entry['results'].items():
                metrics.append({
                    'timestamp': entry['timestamp'],
                    'layer': layer_name,
                    'mean_drift': layer_results.get('mean_drift', 0),
                    'distribution_drift': layer_results.get('distribution_drift', 0),
                    'drift_detected': layer_results.get('drift_detected', False)
                })
        
        return pd.DataFrame(metrics)
    
    def update_reference(self, new_reference):
        """
        Update the reference data.
        
        Args:
            new_reference: New reference data
        """
        self.reference_data = new_reference
        self._compute_reference_statistics()
        
        logging.info("Reference data updated")
    
    def alert_if_drift(self, new_data, alert_system=None):
        """
        Check for drift and alert if detected.
        
        Args:
            new_data: New data to check
            alert_system: AlertSystem instance for sending alerts
            
        Returns:
            True if drift was detected and alert was sent, False otherwise
        """
        results = self.detect(new_data)
        
        # Check if drift was detected in any layer
        drift_detected = False
        for layer_results in results.values():
            if layer_results.get('drift_detected', False):
                drift_detected = True
                break
        
        # Send alert if drift was detected
        if drift_detected and alert_system is not None:
            alert_system.send_alert(
                "Model Drift Detected",
                f"Drift was detected in the model at {datetime.datetime.now().isoformat()}. "
                f"Please check the monitoring dashboard for details."
            )
            return True
        
        return drift_detected
