"""
Module for extracting layer outputs/activations from deep learning models.
"""

import numpy as np
import inspect

class LayerExtractor:
    """Extract intermediate outputs from model layers."""
    
    def __init__(self, model, layers=None):
        """
        Initialize the layer extractor.
        
        Args:
            model: The model to extract layers from
            layers: List of layer names to extract (if None, extracts all)
        """
        self.model = model
        self.layers = layers
        self.hooks = {}
        self.layer_outputs = {}
        self.framework = self._detect_framework()
        
        # Register hooks for layer extraction
        if self.framework == 'pytorch':
            self._register_pytorch_hooks()
        elif self.framework == 'tensorflow':
            self._register_tensorflow_hooks()
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")
    
    def _detect_framework(self):
        """Detect which deep learning framework the model uses."""
        model_class = self.model.__class__.__name__
        module_name = self.model.__class__.__module__
        
        if 'torch' in module_name:
            return 'pytorch'
        elif 'tensorflow' in module_name or 'keras' in module_name:
            return 'tensorflow'
        else:
            # Try to infer from available methods
            model_methods = dir(self.model)
            if 'forward' in model_methods and hasattr(self.model, 'parameters'):
                return 'pytorch'
            elif 'predict' in model_methods and hasattr(self.model, 'layers'):
                return 'tensorflow'
            else:
                raise ValueError("Could not detect framework. Supported frameworks: PyTorch, TensorFlow/Keras")
    
    def _register_pytorch_hooks(self):
        """Register hooks for PyTorch models."""
        import torch
        
        def get_activation(name):
            def hook(model, input, output):
                self.layer_outputs[name] = output.detach().cpu().numpy()
            return hook
        
        # Identify layers to hook
        for name, module in self.model.named_modules():
            if name == '':  # Skip the model itself
                continue
            
            if self.layers is None or name in self.layers:
                # Register the hook
                hook = module.register_forward_hook(get_activation(name))
                self.hooks[name] = hook
    
    def _register_tensorflow_hooks(self):
        """Register hooks for TensorFlow/Keras models."""
        import tensorflow as tf
        
        # Check if model is Sequential or Functional
        if hasattr(self.model, 'layers'):
            # For each layer, create an intermediate model that outputs that layer
            for i, layer in enumerate(self.model.layers):
                layer_name = layer.name
                if self.layers is None or layer_name in self.layers:
                    # Create an intermediate model
                    intermediate_model = tf.keras.Model(
                        inputs=self.model.input,
                        outputs=layer.output,
                        name=f"intermediate_{layer_name}"
                    )
                    self.hooks[layer_name] = intermediate_model
    
    def extract(self, data):
        """
        Extract layer outputs for the given data.
        
        Args:
            data: Input data to pass through the model
            
        Returns:
            Dictionary of layer outputs
        """
        # Clear previous outputs
        self.layer_outputs = {}
        
        if self.framework == 'pytorch':
            import torch
            
            # Convert data to tensor if it's not already
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=torch.float32)
            
            # Move to same device as model
            device = next(self.model.parameters()).device
            data = data.to(device)
            
            # Forward pass to trigger hooks
            with torch.no_grad():
                self.model(data)
            
            return self.layer_outputs
            
        elif self.framework == 'tensorflow':
            # For TensorFlow, run each intermediate model
            layer_outputs = {}
            for layer_name, intermediate_model in self.hooks.items():
                layer_outputs[layer_name] = intermediate_model.predict(data)
            
            return layer_outputs
    
    def register_hook(self, layer_name, hook_fn):
        """
        Register a custom hook function for a layer.
        
        Args:
            layer_name (str): Name of the layer
            hook_fn: Hook function to register
        """
        if self.framework == 'pytorch':
            # Find the module
            for name, module in self.model.named_modules():
                if name == layer_name:
                    # Register the hook
                    hook = module.register_forward_hook(hook_fn)
                    self.hooks[layer_name] = hook
                    return
            
            raise ValueError(f"Layer not found: {layer_name}")
            
        elif self.framework == 'tensorflow':
            # For TensorFlow, we need to create a custom layer wrapper
            # This is more complex and would depend on the specific use case
            raise NotImplementedError("Custom hooks for TensorFlow not implemented yet")
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        if self.framework == 'pytorch':
            for hook in self.hooks.values():
                hook.remove()
            
        self.hooks = {}
