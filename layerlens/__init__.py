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
        self.layer_extractor = LayerExtractor(model, layers)
        self.surrogate_builder = SurrogateBuilder(surrogate_type)
        self.stitching_engine = StitchingEngine()
        
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
