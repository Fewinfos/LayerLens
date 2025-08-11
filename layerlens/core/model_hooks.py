"""
Module for hooking into deep learning models to extract intermediate outputs.
"""

def register_hooks(model, hook_fn, layer_filter=None):
    """
    Register hooks on a model to capture intermediate outputs.
    
    Args:
        model: The model to hook into
        hook_fn: Function to call when a layer is executed
        layer_filter: Function to filter which layers to hook
        
    Returns:
        Dictionary of registered hooks
    """
    hooks = {}
    
    # Detect framework
    framework = _detect_framework(model)
    
    if framework == 'pytorch':
        hooks = _register_pytorch_hooks(model, hook_fn, layer_filter)
    elif framework == 'tensorflow':
        hooks = _register_tensorflow_hooks(model, hook_fn, layer_filter)
    else:
        raise ValueError(f"Unsupported framework: {framework}")
    
    return hooks

def remove_hooks(hooks):
    """
    Remove registered hooks.
    
    Args:
        hooks: Dictionary of hooks to remove
    """
    for hook in hooks.values():
        if hasattr(hook, 'remove'):
            hook.remove()

def _detect_framework(model):
    """
    Detect which deep learning framework the model uses.
    
    Args:
        model: The model to detect the framework for
        
    Returns:
        String indicating the framework ('pytorch', 'tensorflow', etc.)
    """
    model_class = model.__class__.__name__
    module_name = model.__class__.__module__
    
    if 'torch' in module_name:
        return 'pytorch'
    elif 'tensorflow' in module_name or 'keras' in module_name:
        return 'tensorflow'
    else:
        # Try to infer from available methods
        model_methods = dir(model)
        if 'forward' in model_methods and hasattr(model, 'parameters'):
            return 'pytorch'
        elif 'predict' in model_methods and hasattr(model, 'layers'):
            return 'tensorflow'
        else:
            raise ValueError("Could not detect framework. Supported frameworks: PyTorch, TensorFlow/Keras")

def _register_pytorch_hooks(model, hook_fn, layer_filter=None):
    """
    Register hooks for PyTorch models.
    
    Args:
        model: PyTorch model
        hook_fn: Function to call when a layer is executed
        layer_filter: Function to filter which layers to hook
        
    Returns:
        Dictionary of registered hooks
    """
    hooks = {}
    
    # Define a default layer filter if none provided
    if layer_filter is None:
        layer_filter = lambda name, module: True
    
    # Register hooks on all modules that pass the filter
    for name, module in model.named_modules():
        if name == '':  # Skip the model itself
            continue
        
        if layer_filter(name, module):
            # Create a hook function that captures the layer name
            def make_hook(layer_name):
                def hook(module, input, output):
                    return hook_fn(layer_name, module, input, output)
                return hook
            
            # Register the hook
            hook_handle = module.register_forward_hook(make_hook(name))
            hooks[name] = hook_handle
    
    return hooks

def _register_tensorflow_hooks(model, hook_fn, layer_filter=None):
    """
    Register hooks for TensorFlow/Keras models.
    
    Args:
        model: TensorFlow/Keras model
        hook_fn: Function to call when a layer is executed
        layer_filter: Function to filter which layers to hook
        
    Returns:
        Dictionary of registered hooks
    """
    import tensorflow as tf
    
    hooks = {}
    
    # Define a default layer filter if none provided
    if layer_filter is None:
        layer_filter = lambda name, layer: True
    
    # Check if model is Sequential or Functional
    if hasattr(model, 'layers'):
        # Create a subclass that overrides the call method to capture outputs
        class LayerHook(tf.keras.layers.Layer):
            def __init__(self, layer, layer_name, hook_fn, **kwargs):
                super(LayerHook, self).__init__(**kwargs)
                self.layer = layer
                self.layer_name = layer_name
                self.hook_fn = hook_fn
            
            def call(self, inputs, **kwargs):
                outputs = self.layer(inputs, **kwargs)
                self.hook_fn(self.layer_name, self.layer, inputs, outputs)
                return outputs
        
        # Create a new model with hook layers
        input_tensor = model.input
        x = input_tensor
        
        for i, layer in enumerate(model.layers):
            layer_name = layer.name
            
            if layer_filter(layer_name, layer):
                # Wrap the layer with a hook
                x = LayerHook(layer, layer_name, hook_fn)(x)
                hooks[layer_name] = layer
            else:
                # Pass through without a hook
                x = layer(x)
        
        # Create the new model
        hooked_model = tf.keras.Model(inputs=input_tensor, outputs=x)
        
        # Store the hooked model in hooks to prevent garbage collection
        hooks['_model'] = hooked_model
    
    return hooks

def create_activation_hook():
    """
    Create a hook function that captures layer activations.
    
    Returns:
        Hook function and dictionary to store activations
    """
    activations = {}
    
    def hook_fn(layer_name, module, inputs, outputs):
        # Convert PyTorch tensors to numpy arrays if needed
        if hasattr(outputs, 'detach'):
            outputs_np = outputs.detach().cpu().numpy()
        else:
            # Assume it's already a numpy array or TensorFlow tensor
            try:
                outputs_np = outputs.numpy()
            except:
                outputs_np = outputs
        
        # Store the activations
        activations[layer_name] = outputs_np
        
        return outputs
    
    return hook_fn, activations

def create_gradient_hook():
    """
    Create a hook function that captures gradients.
    
    Returns:
        Hook function and dictionary to store gradients
    """
    gradients = {}
    
    def hook_fn(layer_name, module, grad_input, grad_output):
        # Convert PyTorch tensors to numpy arrays if needed
        if hasattr(grad_output, 'detach'):
            grad_output_np = [g.detach().cpu().numpy() if g is not None else None 
                              for g in grad_output]
        else:
            # Assume it's already a numpy array or TensorFlow tensor
            try:
                grad_output_np = [g.numpy() if g is not None else None 
                                  for g in grad_output]
            except:
                grad_output_np = grad_output
        
        # Store the gradients
        gradients[layer_name] = grad_output_np
        
        return grad_input
    
    return hook_fn, gradients
