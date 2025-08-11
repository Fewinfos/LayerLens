"""
Module for calculating fidelity metrics between original model layers and surrogate models.
"""

import numpy as np
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error

class FidelityScore:
    """Class to calculate and store fidelity scores."""
    
    def __init__(self, local_score=None, global_score=None, compositional_score=None):
        """
        Initialize fidelity score object.
        
        Args:
            local_score (float): Local fidelity score
            global_score (float): Global fidelity score
            compositional_score (float): Compositional fidelity score
        """
        self.local_score = local_score
        self.global_score = global_score
        self.compositional_score = compositional_score
    
    def __repr__(self):
        """String representation of fidelity scores."""
        return (f"FidelityScore(local={self.local_score:.4f}, "
                f"global={self.global_score:.4f}, "
                f"compositional={self.compositional_score:.4f})")
    
    def as_dict(self):
        """Return scores as a dictionary."""
        return {
            'local': self.local_score,
            'global': self.global_score,
            'compositional': self.compositional_score
        }

def calculate_fidelity(surrogate_model, X, y_true, model_type=None):
    """
    Calculate the fidelity of a surrogate model.
    
    Args:
        surrogate_model: The surrogate model
        X: Input data
        y_true: True outputs from the original model
        model_type: Type of model (regression, classification, etc.)
        
    Returns:
        FidelityScore object with different fidelity metrics
    """
    # Determine model type if not provided
    if model_type is None:
        if hasattr(surrogate_model, "predict_proba"):
            model_type = "classification"
        else:
            model_type = "regression"
    
    # Get predictions from surrogate model
    y_pred = surrogate_model.predict(X)
    
    # Calculate global fidelity
    if model_type == "classification":
        # For classification, use accuracy
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            # Multi-class with one-hot encoding
            global_score = accuracy_score(np.argmax(y_true, axis=1), 
                                          np.argmax(y_pred, axis=1))
        else:
            # Binary classification
            global_score = accuracy_score(y_true, y_pred)
    else:
        # For regression, use R-squared
        global_score = r2_score(y_true, y_pred)
    
    # Calculate local fidelity (use MSE for individual predictions)
    local_scores = []
    for i in range(len(X)):
        local_pred = surrogate_model.predict(X[i:i+1])
        local_scores.append(mean_squared_error(y_true[i:i+1], local_pred))
    local_score = 1.0 - np.mean(local_scores) / np.var(y_true)
    
    # Compositional fidelity is calculated when stitching models together
    # For now, just use the global score
    compositional_score = global_score
    
    return FidelityScore(local_score, global_score, compositional_score)

def calculate_compositional_fidelity(surrogate_models, layer_order, X):
    """
    Calculate the compositional fidelity of a chain of surrogate models.
    
    Args:
        surrogate_models (dict): Dictionary of surrogate models by layer name
        layer_order (list): Order of layers to compose
        X: Input data
        
    Returns:
        Compositional fidelity score
    """
    # This is a simplified version - in practice, we would need the actual layer outputs
    # from the original model to compare with
    
    # Forward pass through the original model
    # (This would require access to the original model's layer outputs)
    
    # Forward pass through the chain of surrogate models
    current_X = X
    for layer_name in layer_order:
        if layer_name not in surrogate_models:
            raise ValueError(f"Layer {layer_name} not found in surrogate models")
        surrogate = surrogate_models[layer_name]
        current_X = surrogate.predict(current_X)
    
    # Compare the final output with the original model's output
    # (This would require the original model's final output)
    
    # For now, return a placeholder score
    return 0.9

def fidelity_bounds(surrogate_model, X, confidence_level=0.95):
    """
    Calculate confidence bounds on fidelity scores.
    
    Args:
        surrogate_model: The surrogate model
        X: Input data
        confidence_level: Confidence level for bounds
        
    Returns:
        Lower and upper bounds on fidelity
    """
    # This would be implemented using bootstrap sampling or similar methods
    # to estimate the distribution of fidelity scores
    
    # For now, return placeholder values
    fidelity = calculate_fidelity(surrogate_model, X, surrogate_model.predict(X))
    margin = 0.05  # This would be calculated based on the confidence level
    
    lower_bound = max(0.0, fidelity.global_score - margin)
    upper_bound = min(1.0, fidelity.global_score + margin)
    
    return lower_bound, upper_bound
