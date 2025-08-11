"""
Utility functions for working with models in LayerLens.
"""

import numpy as np
import inspect
import importlib
import logging

def get_model_framework(model):
    """
    Determine which deep learning framework a model uses.
    
    Args:
        model: The model to check
        
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
            return 'unknown'

def get_model_layers(model):
    """
    Get a list of layers from a model.
    
    Args:
        model: The model
        
    Returns:
        List of (name, layer) tuples
    """
    framework = get_model_framework(model)
    
    if framework == 'pytorch':
        return list(model.named_modules())
    elif framework == 'tensorflow':
        if hasattr(model, 'layers'):
            return [(layer.name, layer) for layer in model.layers]
        else:
            return []
    else:
        return []

def get_layer_by_name(model, layer_name):
    """
    Get a layer from a model by name.
    
    Args:
        model: The model
        layer_name: Name of the layer to get
        
    Returns:
        The layer or None if not found
    """
    framework = get_model_framework(model)
    
    if framework == 'pytorch':
        for name, module in model.named_modules():
            if name == layer_name:
                return module
    elif framework == 'tensorflow':
        if hasattr(model, 'get_layer'):
            try:
                return model.get_layer(layer_name)
            except:
                pass
    
    return None

def predict_with_model(model, data):
    """
    Make predictions with a model, handling different frameworks.
    
    Args:
        model: The model
        data: Input data
        
    Returns:
        Model predictions
    """
    framework = get_model_framework(model)
    
    if framework == 'pytorch':
        import torch
        
        # Convert data to tensor if needed
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        
        # Move to same device as model
        device = next(model.parameters()).device
        data = data.to(device)
        
        # Make prediction
        with torch.no_grad():
            predictions = model(data)
        
        # Convert to numpy
        return predictions.cpu().numpy()
    
    elif framework == 'tensorflow':
        # Use predict method if available
        if hasattr(model, 'predict'):
            return model.predict(data)
        else:
            # Try to call the model directly
            try:
                import tensorflow as tf
                if isinstance(data, np.ndarray):
                    data = tf.convert_to_tensor(data)
                return model(data).numpy()
            except:
                raise ValueError("Could not make predictions with TensorFlow model")
    
    else:
        # Try a generic approach
        if hasattr(model, 'predict'):
            return model.predict(data)
        elif callable(model):
            return model(data)
        else:
            raise ValueError(f"Don't know how to make predictions with model of type {type(model)}")

def load_model_from_path(model_path):
    """
    Load a model from a file path.
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        Loaded model
    """
    # Determine file type
    if model_path.endswith('.h5') or model_path.endswith('.keras'):
        # TensorFlow/Keras model
        try:
            import tensorflow as tf
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            logging.error(f"Failed to load Keras model: {e}")
            return None
    
    elif model_path.endswith('.pt') or model_path.endswith('.pth'):
        # PyTorch model - need model class
        try:
            import torch
            model_dict = torch.load(model_path)
            
            # Check if it's a state dict or a full model
            if isinstance(model_dict, dict) and 'state_dict' in model_dict:
                # This is a checkpoint with state_dict
                # We need the model class to load it
                raise ValueError("Loading PyTorch state dicts requires a model class. Use load_pytorch_model instead.")
            
            # Assume it's a full model
            return model_dict
        except Exception as e:
            logging.error(f"Failed to load PyTorch model: {e}")
            return None
    
    elif model_path.endswith('.joblib') or model_path.endswith('.pkl'):
        # Scikit-learn model
        try:
            import joblib
            return joblib.load(model_path)
        except Exception as e:
            logging.error(f"Failed to load scikit-learn model: {e}")
            return None
    
    else:
        logging.error(f"Unknown model file type: {model_path}")
        return None

def load_pytorch_model(model_class, model_path, **kwargs):
    """
    Load a PyTorch model from a file path.
    
    Args:
        model_class: PyTorch model class
        model_path (str): Path to the model file
        **kwargs: Arguments to pass to the model constructor
        
    Returns:
        Loaded PyTorch model
    """
    try:
        import torch
        
        # Create model instance
        if inspect.isclass(model_class):
            model = model_class(**kwargs)
        elif isinstance(model_class, str):
            # Parse the string to get the class
            module_name, class_name = model_class.rsplit('.', 1)
            module = importlib.import_module(module_name)
            model = getattr(module, class_name)(**kwargs)
        else:
            model = model_class
        
        # Load state dict
        state_dict = torch.load(model_path)
        
        # Check if it's a checkpoint or just a state dict
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        model.load_state_dict(state_dict)
        model.eval()
        
        return model
    
    except Exception as e:
        logging.error(f"Failed to load PyTorch model: {e}")
        return None

def get_model_summary(model):
    """
    Get a summary of a model's architecture.
    
    Args:
        model: The model
        
    Returns:
        String with model summary
    """
    framework = get_model_framework(model)
    
    if framework == 'pytorch':
        try:
            from torchinfo import summary
            return str(summary(model))
        except ImportError:
            # Fallback to a simple summary
            return _get_pytorch_summary(model)
    
    elif framework == 'tensorflow':
        # Use built-in summary
        try:
            import io
            import contextlib
            
            with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                model.summary()
                return buf.getvalue()
        except:
            # Fallback to a simple summary
            return _get_tensorflow_summary(model)
    
    else:
        # Generic summary
        return f"Model type: {type(model)}\n" + \
               f"Attributes: {dir(model)}"

def _get_pytorch_summary(model):
    """Create a simple summary for PyTorch models."""
    lines = [f"Model: {model.__class__.__name__}"]
    
    # Add parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    lines.append(f"Total parameters: {total_params}")
    lines.append(f"Trainable parameters: {trainable_params}")
    
    # Add model structure
    lines.append("\nModel structure:")
    for name, module in model.named_modules():
        if name == '':  # Skip the model itself
            continue
        lines.append(f"  {name}: {module.__class__.__name__}")
    
    return "\n".join(lines)

def _get_tensorflow_summary(model):
    """Create a simple summary for TensorFlow models."""
    lines = [f"Model: {model.__class__.__name__}"]
    
    # Add parameter count if possible
    if hasattr(model, 'count_params'):
        total_params = model.count_params()
        lines.append(f"Total parameters: {total_params}")
    
    # Add model structure
    lines.append("\nModel structure:")
    if hasattr(model, 'layers'):
        for i, layer in enumerate(model.layers):
            lines.append(f"  Layer {i}: {layer.name} ({layer.__class__.__name__})")
    
    return "\n".join(lines)
