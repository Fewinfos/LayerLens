"""
Module for generating activation and importance heatmaps.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib import cm

def generate_activation_heatmap(activations, layer_name=None, input_image=None, overlay=False):
    """
    Generate a heatmap visualization of layer activations.
    
    Args:
        activations: Layer activation values
        layer_name: Name of the layer (for the title)
        input_image: Original input image (for overlay)
        overlay: Whether to overlay heatmap on the input image
        
    Returns:
        Plotly figure with the heatmap
    """
    # Prepare the activations for visualization
    prepared_data = _prepare_activation_data(activations)
    
    # Create the title
    if layer_name:
        title = f"Activation Heatmap - {layer_name}"
    else:
        title = "Activation Heatmap"
    
    # Create the heatmap figure
    if overlay and input_image is not None:
        fig = _create_overlay_heatmap(prepared_data, input_image, title)
    else:
        fig = _create_standard_heatmap(prepared_data, title)
    
    return fig

# Alias for backward compatibility
def generate_heatmap(activations, layer_name=None, input_image=None, overlay=False):
    """
    Generate a heatmap visualization of layer activations.
    
    Args:
        activations: Layer activation values
        layer_name: Name of the layer (for the title)
        input_image: Original input image (for overlay)
        overlay: Whether to overlay heatmap on the input image
        
    Returns:
        Plotly figure with the heatmap
    """
    # Prepare the activations for visualization
    prepared_data = _prepare_activation_data(activations)
    
    # Create the title
    if layer_name:
        title = f"Activation Heatmap - {layer_name}"
    else:
        title = "Activation Heatmap"
    
    # Create the heatmap figure
    if overlay and input_image is not None:
        fig = _create_overlay_heatmap(prepared_data, input_image, title)
    else:
        fig = _create_standard_heatmap(prepared_data, title)
    
    return fig

def _prepare_activation_data(activations):
    """Prepare activation data for visualization."""
    # Check dimensionality
    if len(activations.shape) == 4:
        # (batch_size, channels, height, width) or (batch_size, height, width, channels)
        if activations.shape[1] > activations.shape[3]:
            # (batch_size, channels, height, width) - PyTorch style
            activations = np.transpose(activations, (0, 2, 3, 1))
        
        # Take the first example from the batch
        activations = activations[0]
    
    # If 3D (height, width, channels), aggregate across channels
    if len(activations.shape) == 3:
        # For now, use max across channels
        activations = np.max(activations, axis=2)
    
    # Normalize to [0, 1]
    activations = (activations - np.min(activations)) / (np.max(activations) - np.min(activations) + 1e-8)
    
    return activations

def _create_standard_heatmap(data, title):
    """Create a standard heatmap visualization."""
    fig = px.imshow(
        data,
        color_continuous_scale='Viridis',
        title=title
    )
    
    # Update layout
    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Activation"
        )
    )
    
    return fig

def _create_overlay_heatmap(activation_data, input_image, title):
    """Create a heatmap overlaid on an input image."""
    # Ensure input image is properly formatted
    if len(input_image.shape) == 4:
        input_image = input_image[0]
    
    if input_image.shape[0] == 3 and len(input_image.shape) == 3:
        # (channels, height, width) -> (height, width, channels)
        input_image = np.transpose(input_image, (1, 2, 0))
    
    # Normalize input image
    input_image = (input_image - np.min(input_image)) / (np.max(input_image) - np.min(input_image) + 1e-8)
    
    # Resize activation map to match input image if needed
    if activation_data.shape != input_image.shape[:2]:
        from scipy.ndimage import zoom
        zoom_factors = (
            input_image.shape[0] / activation_data.shape[0],
            input_image.shape[1] / activation_data.shape[1]
        )
        activation_data = zoom(activation_data, zoom_factors, order=1)
    
    # Create a composite image
    colormap = cm.viridis
    heatmap = colormap(activation_data)[:, :, :3]  # Remove alpha channel
    
    alpha = 0.7  # Transparency factor
    overlay_img = alpha * heatmap + (1 - alpha) * input_image
    
    # Create figure
    fig = px.imshow(
        overlay_img,
        title=title
    )
    
    return fig

def generate_feature_importance_heatmap(model, layer_name, feature_idx, input_data):
    """
    Generate a heatmap showing the importance of input features for a specific layer feature.
    
    Args:
        model: The model
        layer_name: Name of the layer
        feature_idx: Index of the feature in the layer
        input_data: Input data to analyze
        
    Returns:
        Plotly figure with the importance heatmap
    """
    # This is a placeholder implementation
    # In a real implementation, this would use techniques like:
    # - Gradient-based attribution
    # - Integrated gradients
    # - Layer-wise relevance propagation
    
    # For now, generate random importance scores
    importance_scores = np.random.rand(*input_data.shape[1:3])
    
    # Create the heatmap
    fig = px.imshow(
        importance_scores,
        color_continuous_scale='RdBu_r',
        title=f"Feature {feature_idx} Importance - {layer_name}"
    )
    
    # Update layout
    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Importance"
        )
    )
    
    return fig

def compare_layer_activations(activations_dict, input_image=None, max_layers=6):
    """
    Create a comparison of activations across different layers.
    
    Args:
        activations_dict: Dictionary mapping layer names to activation arrays
        input_image: Original input image (optional)
        max_layers: Maximum number of layers to display
        
    Returns:
        Plotly figure with multiple heatmaps
    """
    # Limit the number of layers
    if len(activations_dict) > max_layers:
        # Choose a subset of layers
        layer_names = list(activations_dict.keys())
        indices = np.linspace(0, len(layer_names) - 1, max_layers, dtype=int)
        selected_layers = [layer_names[i] for i in indices]
    else:
        selected_layers = list(activations_dict.keys())
    
    # Calculate grid layout
    n_cols = min(3, len(selected_layers))
    n_rows = (len(selected_layers) + n_cols - 1) // n_cols
    
    # Create subplot figure
    fig = plt.figure(figsize=(4 * n_cols, 3 * n_rows))
    
    # Add input image if provided
    if input_image is not None:
        if len(input_image.shape) == 4:
            input_image = input_image[0]
        
        if input_image.shape[0] == 3 and len(input_image.shape) == 3:
            # (channels, height, width) -> (height, width, channels)
            input_image = np.transpose(input_image, (1, 2, 0))
        
        # Normalize input image
        input_image = (input_image - np.min(input_image)) / (np.max(input_image) - np.min(input_image) + 1e-8)
        
        # Add input image as first subplot
        ax = fig.add_subplot(n_rows, n_cols, 1)
        ax.imshow(input_image)
        ax.set_title("Input Image")
        ax.axis('off')
        
        # Adjust selected layers to account for input image
        selected_layers = selected_layers[:max_layers - 1]
    
    # Add activation heatmaps
    for i, layer_name in enumerate(selected_layers):
        activations = activations_dict[layer_name]
        prepared_data = _prepare_activation_data(activations)
        
        # Calculate subplot index
        if input_image is not None:
            subplot_idx = i + 2  # +2 because input image is first
        else:
            subplot_idx = i + 1
        
        # Add subplot
        ax = fig.add_subplot(n_rows, n_cols, subplot_idx)
        im = ax.imshow(prepared_data, cmap='viridis')
        ax.set_title(layer_name)
        ax.axis('off')
    
    # Add a colorbar
    plt.colorbar(im, ax=fig.axes, shrink=0.8, label="Activation")
    
    # Adjust layout
    plt.tight_layout()
    
    # Convert matplotlib figure to plotly
    return _matplotlib_to_plotly(fig)

def _matplotlib_to_plotly(fig):
    """Convert a matplotlib figure to plotly."""
    import io
    
    # Save the figure to a buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    # Close the matplotlib figure to free memory
    plt.close(fig)
    
    # Create a plotly figure from the image
    from PIL import Image
    img = Image.open(buf)
    
    fig = px.imshow(img)
    fig.update_layout(
        title="Layer Activation Comparison",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False)
    )
    
    return fig
