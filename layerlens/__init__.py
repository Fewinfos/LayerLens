"""
LayerLens: A library for layer-by-layer explainability of deep learning models.
"""

__version__ = "0.1.0"

from layerlens.core.surrogate_builder import SurrogateBuilder
from layerlens.core.layer_extractor import LayerExtractor
from layerlens.core.stitching_engine import StitchingEngine
from layerlens.core.model_hooks import register_hooks, remove_hooks

class Explainer:
    """Main explainer class that coordinates the explanation process."""
    
    def __init__(self, model, surrogate_type='tree', layers=None):
        """
        Initialize the explainer.
        
        Args:
            model: The model to explain
            surrogate_type: Type of surrogate model to use
            layers: Specific layers to explain (if None, all layers are used)
        """
        self.model = model
        self.surrogate_type = surrogate_type
        
        # Build the model if needed (for Keras 3.x Sequential models)
        self._ensure_model_built()
        
        self.layer_extractor = LayerExtractor(model, layers)
        self.surrogate_builder = SurrogateBuilder(surrogate_type)
        self.stitching_engine = StitchingEngine()
        
    def _ensure_model_built(self):
        """Ensure the model is built (for TensorFlow/Keras models)."""
        try:
            # Check if this is a TensorFlow/Keras model
            model_module = self.model.__class__.__module__
            if 'tensorflow' in model_module or 'keras' in model_module:
                # Check if model is built using multiple methods
                is_built = False
                
                # Method 1: Check if layers have been built
                if hasattr(self.model, 'layers') and len(self.model.layers) > 0:
                    first_layer = self.model.layers[0]
                    # In Keras 3.x, check if the layer has been called (has inbound nodes)
                    if hasattr(first_layer, '_inbound_nodes') and len(first_layer._inbound_nodes) > 0:
                        is_built = True
                    # Also check if the model has built attribute set to True
                    elif hasattr(self.model, 'built') and self.model.built:
                        is_built = True
                
                # Method 2: Try to access model.input or output (for older Keras versions)
                if not is_built:
                    try:
                        _ = self.model.input
                        is_built = True
                    except (AttributeError, ValueError):
                        try:
                            _ = self.model.output
                            is_built = True
                        except (AttributeError, ValueError):
                            pass
                
                if is_built:
                    return  # Model is already built
                
                # Model is not built, try to build it automatically
                if hasattr(self.model, 'layers') and len(self.model.layers) > 0:
                    first_layer = self.model.layers[0]
                    
                    # Try to get input shape from the first layer
                    input_shape = None
                    if hasattr(first_layer, 'input_shape') and first_layer.input_shape:
                        input_shape = first_layer.input_shape
                    elif hasattr(first_layer, 'batch_input_shape') and first_layer.batch_input_shape:
                        input_shape = first_layer.batch_input_shape
                    
                    if input_shape and None not in input_shape[1:]:
                        try:
                            import numpy as np
                            # Create dummy input and call the model to build it
                            dummy_input = np.zeros((1,) + input_shape[1:], dtype=np.float32)
                            _ = self.model(dummy_input, training=False)
                            print(f"✓ Auto-built model with input shape: {input_shape}")
                            return
                        except Exception as e:
                            # Auto-build failed, continue to error message
                            pass
                
                # If we couldn't build it automatically, provide helpful error
                print(f"\n⚠️  Model needs to be built before using LayerLens!")
                print(f"Please run this BEFORE creating the Explainer:")
                print(f"    _ = model.predict(your_data[:1], verbose=0)")
                raise ValueError("Model must be built before creating LayerLens Explainer. "
                               "Call model.predict(sample_data) or model(sample_data) first.")
        except ValueError:
            raise
        except Exception:
            pass  # Not a problem if it's not a TensorFlow model
        
    def explain(self, data, target_layer=None):
        """
        Generate explanations for the model.
        
        Args:
            data: Input data to explain
            target_layer: Specific layer to focus on (if None, explains all layers)
            
        Returns:
            Explanation object containing the generated explanations
        """
        # Extract layer outputs
        layer_outputs = self.layer_extractor.extract(data)
        
        # Build surrogate models for each layer
        surrogates = {}
        for layer_name, outputs in layer_outputs.items():
            if target_layer is None or layer_name == target_layer:
                surrogates[layer_name] = self.surrogate_builder.fit(
                    layer_name, data, outputs
                )
        
        # Stitch explanations together
        explanations = self.stitching_engine.stitch(surrogates)
        
        return explanations
    
    def visualize(self, explanations, visualization_type='dashboard'):
        """
        Visualize the generated explanations.
        
        Args:
            explanations: Explanation object to visualize
            visualization_type: Type of visualization to use
            
        Returns:
            Visualization object
        """
        if visualization_type == 'dashboard':
            from layerlens.visualization.dashboard import create_dashboard
            return create_dashboard(explanations)
        elif visualization_type == 'layer_graph':
            from layerlens.visualization.layer_graph import plot_layer_graph
            return plot_layer_graph(self.model, explanations)
        elif visualization_type == 'feature_flow':
            from layerlens.visualization.feature_flow import plot_feature_flow
            return plot_feature_flow(explanations)
        else:
            raise ValueError(f"Unknown visualization type: {visualization_type}")

# Convenience functions
def explain(model, data, **kwargs):
    """Convenience function to explain a model."""
    explainer = Explainer(model, **kwargs)
    return explainer.explain(data)

def visualize(explanations, **kwargs):
    """Convenience function to visualize explanations."""
    from layerlens.visualization.dashboard import create_dashboard
    return create_dashboard(explanations, **kwargs)

def monitor(model, reference_data, **kwargs):
    """Convenience function to set up monitoring."""
    from layerlens.monitoring.drift_detector import DriftDetector
    return DriftDetector(model, reference_data, **kwargs)
