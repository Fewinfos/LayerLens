"""
Module for pinpointing which layer caused a model error.
"""

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
import logging

class FailureLocalizer:
    """Pinpoint which layer caused a model error."""
    
    def __init__(self, model, surrogate_models=None):
        """
        Initialize the failure localizer.
        
        Args:
            model: The original model
            surrogate_models: Dictionary of surrogate models for each layer
        """
        self.model = model
        self.surrogate_models = surrogate_models or {}
        self.layer_outputs = {}
        self.failure_stats = {}
    
    def analyze_failure(self, input_data, expected_output, layer_outputs=None):
        """
        Analyze which layer is most responsible for a model failure.
        
        Args:
            input_data: Input data that caused the failure
            expected_output: Expected output for the input
            layer_outputs: Dictionary of layer outputs (if available)
            
        Returns:
            Dictionary with failure analysis results
        """
        # If layer outputs are provided, use them
        if layer_outputs is not None:
            self.layer_outputs = layer_outputs
        else:
            # Extract layer outputs (requires layer extractor)
            try:
                from layerlens.core.layer_extractor import LayerExtractor
                extractor = LayerExtractor(self.model)
                self.layer_outputs = extractor.extract(input_data)
            except Exception as e:
                logging.error(f"Failed to extract layer outputs: {e}")
                return {"error": "Failed to extract layer outputs"}
        
        # Check if we have surrogate models
        if not self.surrogate_models:
            return {"error": "No surrogate models available for failure localization"}
        
        # Get the actual model output
        try:
            if hasattr(self.model, 'predict'):
                actual_output = self.model.predict(input_data)
            else:
                # Assume it's a PyTorch model
                import torch
                with torch.no_grad():
                    actual_output = self.model(
                        torch.tensor(input_data, dtype=torch.float32)
                    ).cpu().numpy()
        except Exception as e:
            logging.error(f"Failed to get model output: {e}")
            return {"error": "Failed to get model output"}
        
        # Calculate baseline error
        baseline_error = self._calculate_error(actual_output, expected_output)
        
        # Analyze each layer's contribution to the error
        layer_contributions = {}
        layer_names = sorted(self.surrogate_models.keys())
        
        for i, layer_name in enumerate(layer_names):
            # Get the surrogate model for this layer
            surrogate = self.surrogate_models[layer_name]
            
            # Create a hybrid model: original up to this layer, then surrogate
            hybrid_output = self._run_hybrid_model(input_data, layer_name, surrogate)
            
            if hybrid_output is None:
                continue
            
            # Calculate error with the hybrid model
            hybrid_error = self._calculate_error(hybrid_output, expected_output)
            
            # Contribution is the reduction in error when using the surrogate
            contribution = baseline_error - hybrid_error
            
            layer_contributions[layer_name] = {
                'contribution': contribution,
                'baseline_error': baseline_error,
                'hybrid_error': hybrid_error
            }
        
        # Identify the layer with the highest contribution
        if layer_contributions:
            max_contribution_layer = max(
                layer_contributions.items(),
                key=lambda x: x[1]['contribution']
            )[0]
        else:
            max_contribution_layer = None
        
        # Update failure statistics
        self._update_failure_stats(max_contribution_layer)
        
        return {
            'primary_failure_layer': max_contribution_layer,
            'layer_contributions': layer_contributions,
            'baseline_error': baseline_error
        }
    
    def _run_hybrid_model(self, input_data, layer_name, surrogate):
        """
        Run a hybrid model: original up to layer_name, then surrogate.
        
        Args:
            input_data: Input data
            layer_name: Name of the layer to replace with surrogate
            surrogate: Surrogate model for the layer
            
        Returns:
            Output of the hybrid model
        """
        # Get the output of the layer before the current one
        prev_layers = self._get_previous_layers(layer_name)
        
        if not prev_layers:
            # If this is the first layer, use the input data
            layer_input = input_data
        else:
            # Use the output of the previous layer
            prev_layer = prev_layers[-1]
            if prev_layer not in self.layer_outputs:
                logging.warning(f"Output for layer {prev_layer} not available")
                return None
            
            layer_input = self.layer_outputs[prev_layer]
        
        # Reshape the input if needed
        if len(layer_input.shape) > 2:
            layer_input_flat = layer_input.reshape(layer_input.shape[0], -1)
        else:
            layer_input_flat = layer_input
        
        # Get the surrogate output
        try:
            surrogate_output = surrogate.predict(layer_input_flat)
        except Exception as e:
            logging.error(f"Failed to get surrogate output for layer {layer_name}: {e}")
            return None
        
        # If this is the final layer, return the surrogate output
        next_layers = self._get_next_layers(layer_name)
        if not next_layers:
            return surrogate_output
        
        # Otherwise, we need to run the rest of the original model
        # This is challenging to implement generically, as we would need to
        # inject the surrogate output back into the original model's computation graph
        # For simplicity, we'll just return the surrogate output
        logging.warning("Running partial hybrid model not fully implemented")
        return surrogate_output
    
    def _get_previous_layers(self, layer_name):
        """Get the layers that come before the given layer."""
        # This is a simplified implementation that assumes layers are in order
        # In a real implementation, we would use the model's graph structure
        layers = sorted(self.surrogate_models.keys())
        try:
            idx = layers.index(layer_name)
            return layers[:idx]
        except ValueError:
            return []
    
    def _get_next_layers(self, layer_name):
        """Get the layers that come after the given layer."""
        # This is a simplified implementation that assumes layers are in order
        # In a real implementation, we would use the model's graph structure
        layers = sorted(self.surrogate_models.keys())
        try:
            idx = layers.index(layer_name)
            return layers[idx+1:]
        except ValueError:
            return []
    
    def _calculate_error(self, actual, expected):
        """Calculate error between actual and expected outputs."""
        # Determine if this is a classification or regression problem
        if len(expected.shape) > 1 and expected.shape[1] > 1:
            # Multi-class classification with one-hot encoding
            return 1.0 - accuracy_score(
                np.argmax(expected, axis=1),
                np.argmax(actual, axis=1)
            )
        elif np.all(np.isin(np.unique(expected), [0, 1])):
            # Binary classification
            return 1.0 - accuracy_score(expected, np.round(actual))
        else:
            # Regression
            return mean_squared_error(expected, actual)
    
    def _update_failure_stats(self, failure_layer):
        """Update statistics on layer failures."""
        if failure_layer is not None:
            if failure_layer not in self.failure_stats:
                self.failure_stats[failure_layer] = 0
            self.failure_stats[failure_layer] += 1
    
    def get_failure_statistics(self):
        """
        Get statistics on which layers are causing failures.
        
        Returns:
            Dictionary with failure counts by layer
        """
        total = sum(self.failure_stats.values())
        if total == 0:
            return {}
        
        # Calculate percentages
        percentages = {
            layer: count / total * 100
            for layer, count in self.failure_stats.items()
        }
        
        return {
            'counts': self.failure_stats,
            'percentages': percentages,
            'total_failures': total
        }
    
    def analyze_batch(self, input_batch, expected_outputs):
        """
        Analyze failures for a batch of inputs.
        
        Args:
            input_batch: Batch of input data
            expected_outputs: Expected outputs for the inputs
            
        Returns:
            Summary of failure analysis
        """
        # Extract layer outputs for the batch
        try:
            from layerlens.core.layer_extractor import LayerExtractor
            extractor = LayerExtractor(self.model)
            batch_layer_outputs = extractor.extract(input_batch)
        except Exception as e:
            logging.error(f"Failed to extract layer outputs for batch: {e}")
            return {"error": "Failed to extract layer outputs for batch"}
        
        # Analyze each example
        results = []
        for i in range(len(input_batch)):
            # Extract this example's data
            example_input = input_batch[i:i+1]
            example_output = expected_outputs[i:i+1]
            
            # Extract this example's layer outputs
            example_layer_outputs = {
                layer: outputs[i:i+1] 
                for layer, outputs in batch_layer_outputs.items()
            }
            
            # Analyze the failure
            result = self.analyze_failure(
                example_input, example_output, example_layer_outputs
            )
            results.append(result)
        
        # Summarize results
        layer_counts = {}
        for result in results:
            layer = result.get('primary_failure_layer')
            if layer is not None:
                if layer not in layer_counts:
                    layer_counts[layer] = 0
                layer_counts[layer] += 1
        
        # Find the most common failure layer
        most_common_layer = max(layer_counts.items(), key=lambda x: x[1])[0] if layer_counts else None
        
        return {
            'most_common_failure_layer': most_common_layer,
            'layer_failure_counts': layer_counts,
            'total_examples': len(input_batch),
            'detailed_results': results
        }
