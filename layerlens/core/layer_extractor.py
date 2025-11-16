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
        import numpy as np
        
        # Ensure the model is fully built before creating hooks
        # Check multiple indicators of whether the model is built
        model_is_built = False
        
        # Method 1: Check if first layer has inbound nodes
        if hasattr(self.model, 'layers') and len(self.model.layers) > 0:
            first_layer = self.model.layers[0]
            if hasattr(first_layer, '_inbound_nodes') and len(first_layer._inbound_nodes) > 0:
                model_is_built = True
        
        # Method 2: Try to access model.input
        if not model_is_built:
            try:
                _ = self.model.input
                model_is_built = True
            except (AttributeError, ValueError):
                pass
        
        # If model is not built, try to build it
        if not model_is_built:
            if hasattr(self.model, 'layers') and len(self.model.layers) > 0:
                first_layer = self.model.layers[0]
                
                # Get input shape
                input_shape = None
                if hasattr(first_layer, 'input_shape') and first_layer.input_shape:
                    input_shape = first_layer.input_shape
                elif hasattr(first_layer, 'batch_input_shape') and first_layer.batch_input_shape:
                    input_shape = first_layer.batch_input_shape
                
                if input_shape and None not in input_shape[1:]:
                    try:
                        # Build with a dummy input
                        dummy_input = np.zeros((1,) + input_shape[1:], dtype=np.float32)
                        _ = self.model(dummy_input, training=False)
                        print(f"✓ LayerExtractor auto-built model with input shape: {input_shape}")
                        model_is_built = True
                    except Exception as e:
                        print(f"Warning: Failed to auto-build model: {e}")
        
        # Now try to create intermediate models for each layer
        if hasattr(self.model, 'layers'):
            successful_hooks = 0
            failed_hooks = 0
            
            for i, layer in enumerate(self.model.layers):
                layer_name = layer.name
                if self.layers is None or layer_name in self.layers:
                    try:
                        # Create an intermediate model
                        intermediate_model = tf.keras.Model(
                            inputs=self.model.input,
                            outputs=layer.output,
                            name=f"intermediate_{layer_name}"
                        )
                        self.hooks[layer_name] = intermediate_model
                        successful_hooks += 1
                    except (AttributeError, ValueError) as e:
                        # If model input is not defined, skip this layer with a warning
                        failed_hooks += 1
                        if failed_hooks == 1:  # Only print detailed warning once
                            print(f"Warning: Could not create hooks for layers. Error: {e}")
                            if not model_is_built:
                                print(f"Hint: Model may not be fully built. Trying alternative extraction method...")
                        continue
            
            # If all hooks failed, use alternative extraction method
            if successful_hooks == 0 and len(self.model.layers) > 0:
                print("Using alternative layer extraction method...")
                self._use_alternative_extraction = True
            else:
                self._use_alternative_extraction = False
                if successful_hooks > 0:
                    print(f"✓ Successfully created hooks for {successful_hooks} layers")
    
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
            # Check if we need to use alternative extraction
            if hasattr(self, '_use_alternative_extraction') and self._use_alternative_extraction:
                return self._extract_alternative(data)
            
            # Standard method: run each intermediate model
            layer_outputs = {}
            for layer_name, intermediate_model in self.hooks.items():
                layer_outputs[layer_name] = intermediate_model.predict(data, verbose=0)
            
            return layer_outputs
    
    def _extract_alternative(self, data):
        """
        Alternative extraction method for models where intermediate models can't be created.
        Uses manual forward pass through layers.
        """
        import tensorflow as tf
        
        layer_outputs = {}
        current_output = data
        
        # Pass through each layer sequentially
        for layer in self.model.layers:
            if self.layers is None or layer.name in self.layers:
                try:
                    current_output = layer(current_output, training=False)
                    layer_outputs[layer.name] = current_output.numpy() if hasattr(current_output, 'numpy') else current_output
                except Exception as e:
                    print(f"Warning: Could not extract output for layer '{layer.name}': {e}")
                    continue
            else:
                # Still need to pass through the layer even if we're not extracting its output
                current_output = layer(current_output, training=False)
        
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
