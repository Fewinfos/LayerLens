"""
Utility functions for data processing in LayerLens.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

def preprocess_data(data, target_model, normalize=True, scale_method='standard'):
    """
    Preprocess data for use with LayerLens.
    
    Args:
        data: Input data to preprocess
        target_model: Model the data will be used with
        normalize (bool): Whether to normalize the data
        scale_method (str): Scaling method ('standard' or 'minmax')
        
    Returns:
        Preprocessed data
    """
    # Check data type
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    # Convert to numpy array if not already
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    # Normalize if requested
    if normalize:
        if scale_method == 'standard':
            scaler = StandardScaler()
            if len(data.shape) > 2:  # Image data
                # Reshape to 2D for scaling
                original_shape = data.shape
                data = data.reshape(original_shape[0], -1)
                data = scaler.fit_transform(data)
                # Reshape back
                data = data.reshape(original_shape)
            else:  # Tabular data
                data = scaler.fit_transform(data)
        elif scale_method == 'minmax':
            scaler = MinMaxScaler()
            if len(data.shape) > 2:  # Image data
                # Reshape to 2D for scaling
                original_shape = data.shape
                data = data.reshape(original_shape[0], -1)
                data = scaler.fit_transform(data)
                # Reshape back
                data = data.reshape(original_shape)
            else:  # Tabular data
                data = scaler.fit_transform(data)
        else:
            raise ValueError(f"Unknown scaling method: {scale_method}")
    
    # Check if data shape matches model input
    data = _check_model_compatibility(data, target_model)
    
    return data

def _check_model_compatibility(data, model):
    """
    Check if data is compatible with model input and reshape if needed.
    
    Args:
        data: Input data
        model: Target model
        
    Returns:
        Possibly reshaped data
    """
    # Try to determine model input shape
    input_shape = _get_model_input_shape(model)
    
    if input_shape is None:
        # Can't determine input shape, return data as is
        return data
    
    # Check if batch dimension is included
    if len(input_shape) == len(data.shape) - 1:
        # Input shape doesn't include batch dimension
        required_shape = data.shape[0:1] + input_shape
    else:
        # Input shape includes batch dimension
        required_shape = input_shape
    
    # Check if reshaping is needed
    if data.shape != required_shape:
        try:
            # Try to reshape
            data = data.reshape(required_shape)
        except ValueError:
            # Can't reshape, warn and return original
            print(f"Warning: Data shape {data.shape} doesn't match model input shape {required_shape}")
    
    return data

def _get_model_input_shape(model):
    """
    Try to determine the input shape for a model.
    
    Args:
        model: The model
        
    Returns:
        Tuple with input shape or None if can't be determined
    """
    # Try different frameworks
    # PyTorch
    if hasattr(model, 'forward'):
        # Check if the model has a specific input size attribute
        if hasattr(model, 'input_size'):
            return model.input_size
        # Otherwise, this is more difficult without a sample input
        return None
    
    # TensorFlow/Keras
    if hasattr(model, 'input_shape'):
        return model.input_shape
    
    # If model has a layers attribute (Keras models)
    if hasattr(model, 'layers') and len(model.layers) > 0:
        if hasattr(model.layers[0], 'input_shape'):
            return model.layers[0].input_shape
    
    # Can't determine
    return None

def split_dataset(data, labels, test_size=0.2, validation_size=0.1, random_state=42):
    """
    Split a dataset into training, validation, and test sets.
    
    Args:
        data: Input data
        labels: Target labels
        test_size (float): Proportion of data to use for testing
        validation_size (float): Proportion of data to use for validation
        random_state (int): Random seed for reproducibility
        
    Returns:
        Tuple (train_data, train_labels, val_data, val_labels, test_data, test_labels)
    """
    # First split: separate test set
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=test_size, random_state=random_state
    )
    
    # Second split: separate validation set from training set
    if validation_size > 0:
        # Calculate validation size relative to remaining data
        relative_val_size = validation_size / (1 - test_size)
        
        train_data, val_data, train_labels, val_labels = train_test_split(
            train_data, train_labels, test_size=relative_val_size, 
            random_state=random_state
        )
        
        return train_data, train_labels, val_data, val_labels, test_data, test_labels
    else:
        return train_data, train_labels, None, None, test_data, test_labels

def load_sample_data(dataset_name='mnist', n_samples=1000):
    """
    Load a sample dataset for examples and testing.
    
    Args:
        dataset_name (str): Name of the dataset ('mnist', 'cifar10', 'boston')
        n_samples (int): Number of samples to load
        
    Returns:
        Tuple (data, labels)
    """
    if dataset_name.lower() == 'mnist':
        try:
            from sklearn.datasets import fetch_openml
            mnist = fetch_openml('mnist_784', version=1, as_frame=False)
            data = mnist.data[:n_samples]
            labels = mnist.target[:n_samples]
            
            # Reshape to image format
            data = data.reshape(-1, 28, 28, 1)
            
            return data, labels
        except Exception as e:
            print(f"Error loading MNIST: {e}")
            return _generate_dummy_data(n_samples)
    
    elif dataset_name.lower() == 'cifar10':
        try:
            import tensorflow as tf
            (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
            return x_train[:n_samples], y_train[:n_samples]
        except Exception as e:
            print(f"Error loading CIFAR-10: {e}")
            return _generate_dummy_data(n_samples, shape=(32, 32, 3))
    
    elif dataset_name.lower() == 'boston':
        try:
            from sklearn.datasets import load_boston
            boston = load_boston()
            return boston.data[:n_samples], boston.target[:n_samples]
        except Exception as e:
            print(f"Error loading Boston Housing: {e}")
            return _generate_dummy_data(n_samples, shape=(13,))
    
    else:
        print(f"Unknown dataset: {dataset_name}")
        return _generate_dummy_data(n_samples)

def _generate_dummy_data(n_samples, shape=(28, 28, 1)):
    """Generate dummy data for testing."""
    if len(shape) > 1:
        # Image data
        data = np.random.rand(n_samples, *shape)
        labels = np.random.randint(0, 10, size=n_samples)
    else:
        # Tabular data
        data = np.random.rand(n_samples, shape[0])
        labels = np.random.rand(n_samples)
    
    return data, labels
