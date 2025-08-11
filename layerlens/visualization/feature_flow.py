"""
Module for visualizing how feature values evolve layer-by-layer.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def plot_feature_flow(explanations, input_data=None, feature_indices=None, end_layer=None):
    """
    Visualize how feature values evolve through the layers.
    
    Args:
        explanations: Hierarchical explanation object
        input_data: Input data to track through the model
        feature_indices: Indices of features to track (if None, tracks top features)
        end_layer: Name of the final layer to visualize (if None, uses all layers)
        
    Returns:
        Plotly figure
    """
    # Create the figure
    fig = create_feature_flow_figure(explanations, input_data, feature_indices, end_layer)
    
    return fig

def create_feature_flow_figure(explanations, input_data=None, feature_indices=None, end_layer=None):
    """
    Create a Plotly figure for feature flow visualization.
    
    Args:
        explanations: Hierarchical explanation object
        input_data: Input data to track through the model
        feature_indices: Indices of features to track (if None, tracks top features)
        end_layer: Name of the final layer to visualize (if None, uses all layers)
        
    Returns:
        Plotly figure
    """
    # Determine the layers to visualize
    if hasattr(explanations, 'layer_order'):
        layer_order = explanations.layer_order
    else:
        # Assume it's a dictionary - create a linear ordering
        layer_order = list(explanations.keys())
    
    # Filter layers if end_layer is specified
    if end_layer is not None:
        try:
            end_idx = layer_order.index(end_layer)
            layer_order = layer_order[:end_idx + 1]
        except ValueError:
            # end_layer not found, use all layers
            pass
    
    # If no input data provided, create a placeholder visualization
    if input_data is None:
        return _create_placeholder_feature_flow(layer_order)
    
    # Extract feature values for each layer
    feature_values = {}
    feature_importances = {}
    
    for layer_name in layer_order:
        # Get layer explanation
        if hasattr(explanations, 'get_layer_explanation'):
            layer_expl = explanations.get_layer_explanation(layer_name)
        else:
            # Assume it's a dictionary
            layer_expl = explanations.get(layer_name)
        
        if layer_expl is None:
            continue
        
        # Extract feature values for this layer (placeholder implementation)
        # In a real implementation, this would use the surrogate model to
        # extract meaningful feature values
        feature_values[layer_name] = np.random.rand(10)  # Placeholder
        
        # Extract feature importances for this layer
        if hasattr(layer_expl, 'get_feature_importances'):
            importances = layer_expl.get_feature_importances()
        elif isinstance(layer_expl, dict) and 'feature_importances' in layer_expl:
            importances = layer_expl['feature_importances']
        else:
            importances = np.ones(10)  # Placeholder
        
        feature_importances[layer_name] = importances
    
    # Determine which features to track
    if feature_indices is None:
        # Use the most important features from the first layer
        first_layer = layer_order[0]
        if first_layer in feature_importances:
            importances = feature_importances[first_layer]
            feature_indices = np.argsort(importances)[-5:]  # Top 5 features
        else:
            feature_indices = np.arange(5)  # First 5 features
    
    # Create the figure
    fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=True,
                        subplot_titles=("Feature Values", "Feature Importances"),
                        vertical_spacing=0.1)
    
    # Add feature value traces
    for feature_idx in feature_indices:
        feature_name = f"Feature {feature_idx}"
        
        # Extract values for this feature across layers
        x_values = []
        y_values = []
        
        for i, layer_name in enumerate(layer_order):
            if layer_name in feature_values:
                values = feature_values[layer_name]
                if feature_idx < len(values):
                    x_values.append(layer_name)
                    y_values.append(values[feature_idx])
        
        # Add the trace
        fig.add_trace(
            go.Scatter(x=x_values, y=y_values, mode='lines+markers', name=feature_name),
            row=1, col=1
        )
    
    # Add feature importance traces
    for feature_idx in feature_indices:
        feature_name = f"Feature {feature_idx}"
        
        # Extract importances for this feature across layers
        x_values = []
        y_values = []
        
        for i, layer_name in enumerate(layer_order):
            if layer_name in feature_importances:
                importances = feature_importances[layer_name]
                if feature_idx < len(importances):
                    x_values.append(layer_name)
                    y_values.append(importances[feature_idx])
        
        # Add the trace
        fig.add_trace(
            go.Scatter(x=x_values, y=y_values, mode='lines+markers', 
                      name=feature_name, showlegend=False),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title='Feature Flow Across Layers',
        height=800,
        hovermode='closest',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes
    fig.update_xaxes(title_text="Layer", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Importance", row=2, col=1)
    
    return fig

def _create_placeholder_feature_flow(layer_order):
    """Create a placeholder feature flow figure when no input data is provided."""
    fig = make_subplots(rows=1, cols=1)
    
    # Add a text annotation
    fig.add_annotation(
        text="Please provide input data to visualize feature flow",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False
    )
    
    # Update layout
    fig.update_layout(
        title='Feature Flow Across Layers',
        height=400,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig

def compare_feature_flows(explanations, input_samples, sample_labels=None, feature_indices=None):
    """
    Compare feature flows for multiple input samples.
    
    Args:
        explanations: Hierarchical explanation object
        input_samples: List of input samples to compare
        sample_labels: Labels for the samples (if None, uses numbered samples)
        feature_indices: Indices of features to track (if None, tracks top features)
        
    Returns:
        Plotly figure
    """
    # Create a figure with subplots for each sample
    n_samples = len(input_samples)
    
    if sample_labels is None:
        sample_labels = [f"Sample {i+1}" for i in range(n_samples)]
    
    fig = make_subplots(rows=n_samples, cols=1, 
                        subplot_titles=sample_labels,
                        shared_xaxes=True,
                        vertical_spacing=0.1)
    
    # Create a feature flow visualization for each sample
    for i, sample in enumerate(input_samples):
        # Get a single-sample feature flow figure
        sample_fig = create_feature_flow_figure(
            explanations, sample, feature_indices
        )
        
        # Extract the traces and add to the main figure
        for trace in sample_fig.data:
            fig.add_trace(trace, row=i+1, col=1)
    
    # Update layout
    fig.update_layout(
        title='Comparison of Feature Flows',
        height=300 * n_samples,
        hovermode='closest'
    )
    
    # Update y-axis titles
    for i in range(n_samples):
        fig.update_yaxes(title_text="Value", row=i+1, col=1)
    
    # Update x-axis title (only for the bottom subplot)
    fig.update_xaxes(title_text="Layer", row=n_samples, col=1)
    
    return fig
