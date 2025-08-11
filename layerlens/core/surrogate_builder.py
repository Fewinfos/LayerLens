"""
Module for building interpretable surrogate models for neural network layers.
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

from layerlens.core.fidelity_metrics import calculate_fidelity

class SurrogateBuilder:
    """Builds interpretable surrogate models for neural network layers."""
    
    def __init__(self, surrogate_type='tree', max_samples=1000, fidelity_threshold=0.9):
        """
        Initialize the surrogate builder.
        
        Args:
            surrogate_type (str): Type of surrogate model ('tree', 'linear', 'rule')
            max_samples (int): Maximum number of samples to use for training
            fidelity_threshold (float): Minimum required fidelity score
        """
        self.surrogate_type = surrogate_type
        self.max_samples = max_samples
        self.fidelity_threshold = fidelity_threshold
        self.surrogates = {}
        self.fidelity_scores = {}
        
    def fit(self, layer_name, input_data, layer_output):
        """
        Train a surrogate model for the given layer.
        
        Args:
            layer_name (str): Name of the layer
            input_data: Input data to the model
            layer_output: Output of the layer
            
        Returns:
            Trained surrogate model
        """
        # Sample data if needed
        if len(input_data) > self.max_samples:
            idx = np.random.choice(len(input_data), self.max_samples, replace=False)
            input_data = input_data[idx]
            layer_output = layer_output[idx]
        
        # Flatten the data if needed
        if len(input_data.shape) > 2:
            input_data_flat = input_data.reshape(input_data.shape[0], -1)
        else:
            input_data_flat = input_data
            
        if len(layer_output.shape) > 2:
            layer_output_flat = layer_output.reshape(layer_output.shape[0], -1)
        else:
            layer_output_flat = layer_output
        
        # Split data for training and evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            input_data_flat, layer_output_flat, test_size=0.2, random_state=42
        )
        
        # Build appropriate surrogate model
        surrogate = self._build_surrogate(X_train, y_train)
        
        # Calculate fidelity score
        self.fidelity_scores[layer_name] = calculate_fidelity(
            surrogate, X_test, y_test
        )
        
        # Store the surrogate
        self.surrogates[layer_name] = surrogate
        
        return surrogate
    
    def _build_surrogate(self, X, y):
        """
        Build a surrogate model based on the specified type.
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            Trained surrogate model
        """
        # Determine if regression or classification
        is_regression = True
        if len(y.shape) > 1 and y.shape[1] > 1:
            # Multi-output regression or multi-class classification
            unique_values = np.unique(y)
            if len(unique_values) < 10 and np.all(np.isin(unique_values, [0, 1])):
                is_regression = False
        else:
            # Single output
            unique_values = np.unique(y)
            if len(unique_values) < 10:
                is_regression = False
        
        # Build the appropriate model
        if self.surrogate_type == 'tree':
            if is_regression:
                model = DecisionTreeRegressor(max_depth=5)
            else:
                model = DecisionTreeClassifier(max_depth=5)
        elif self.surrogate_type == 'linear':
            if is_regression:
                model = Ridge(alpha=1.0)
            else:
                model = LogisticRegression(max_iter=1000)
        elif self.surrogate_type == 'rule':
            # Implement rule-based models like RuleFit or similar
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=10, max_depth=5)
        else:
            raise ValueError(f"Unknown surrogate type: {self.surrogate_type}")
        
        # Train the model
        model.fit(X, y)
        
        return model
    
    def evaluate(self, layer_name, X, y, metrics=None):
        """
        Evaluate the surrogate model on new data.
        
        Args:
            layer_name (str): Name of the layer
            X: Input features
            y: True layer outputs
            metrics (list): List of metrics to compute
            
        Returns:
            Dictionary of evaluation metrics
        """
        if metrics is None:
            metrics = ['fidelity', 'accuracy']
            
        surrogate = self.surrogates.get(layer_name)
        if surrogate is None:
            raise ValueError(f"No surrogate model found for layer: {layer_name}")
        
        results = {}
        
        # Make predictions
        y_pred = surrogate.predict(X)
        
        # Calculate metrics
        for metric in metrics:
            if metric == 'fidelity':
                results[metric] = calculate_fidelity(surrogate, X, y)
            elif metric == 'accuracy':
                if hasattr(surrogate, 'predict_proba'):
                    results[metric] = accuracy_score(np.argmax(y, axis=1), 
                                                    np.argmax(y_pred, axis=1))
                else:
                    results[metric] = r2_score(y, y_pred)
        
        return results
    
    def explain(self, layer_name, instance):
        """
        Generate explanation for a specific instance.
        
        Args:
            layer_name (str): Name of the layer
            instance: Input instance to explain
            
        Returns:
            Explanation object specific to the surrogate type
        """
        surrogate = self.surrogates.get(layer_name)
        if surrogate is None:
            raise ValueError(f"No surrogate model found for layer: {layer_name}")
        
        # Flatten the instance if needed
        if len(instance.shape) > 1:
            instance_flat = instance.reshape(1, -1)
        else:
            instance_flat = instance.reshape(1, -1)
        
        # Generate explanation based on surrogate type
        if self.surrogate_type == 'tree':
            # For tree models, extract the path through the tree
            return self._explain_tree(surrogate, instance_flat)
        elif self.surrogate_type == 'linear':
            # For linear models, return coefficients
            return self._explain_linear(surrogate, instance_flat)
        else:
            # Generic explanation using feature importance
            return self._explain_generic(surrogate, instance_flat)
    
    def _explain_tree(self, tree_model, instance):
        """Extract the decision path from a tree model."""
        # Get the decision path
        decision_path = tree_model.decision_path(instance)
        
        # Extract the nodes in the path
        node_indices = decision_path.indices
        
        # Extract feature importances
        feature_importances = tree_model.feature_importances_
        
        return {
            'type': 'tree_path',
            'decision_path': node_indices,
            'feature_importances': feature_importances,
            'prediction': tree_model.predict(instance)
        }
    
    def _explain_linear(self, linear_model, instance):
        """Extract coefficients from a linear model."""
        # Get the coefficients
        if hasattr(linear_model, 'coef_'):
            coefficients = linear_model.coef_
        else:
            coefficients = None
        
        # Calculate the contribution of each feature
        contributions = instance * coefficients if coefficients is not None else None
        
        return {
            'type': 'linear',
            'coefficients': coefficients,
            'contributions': contributions,
            'prediction': linear_model.predict(instance)
        }
    
    def _explain_generic(self, model, instance):
        """Generic explanation using feature importance."""
        # Try to get feature importances if available
        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
        else:
            feature_importances = None
        
        return {
            'type': 'generic',
            'feature_importances': feature_importances,
            'prediction': model.predict(instance)
        }
