"""
Utility functions for plotting and visualization in LayerLens.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io
import base64

def plot_to_image(figure):
    """
    Convert a matplotlib figure to an image.
    
    Args:
        figure: Matplotlib figure
        
    Returns:
        Image as numpy array
    """
    # Save the figure to a buffer
    buf = io.BytesIO()
    figure.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    
    # Read the image and convert to array
    from PIL import Image
    image = Image.open(buf)
    image_array = np.array(image)
    
    # Close the figure to free memory
    plt.close(figure)
    
    return image_array

def plot_to_base64(figure):
    """
    Convert a matplotlib figure to a base64 string.
    
    Args:
        figure: Matplotlib figure
        
    Returns:
        Base64 encoded string of the image
    """
    # Save the figure to a buffer
    buf = io.BytesIO()
    figure.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    
    # Encode as base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    # Close the figure to free memory
    plt.close(figure)
    
    return img_str

def visualize_layer_activations(activations, layer_name=None, cmap='viridis'):
    """
    Visualize layer activations.
    
    Args:
        activations: Layer activation values
        layer_name: Name of the layer (for the title)
        cmap: Colormap to use
        
    Returns:
        Matplotlib figure
    """
    # Determine the type of activations
    if len(activations.shape) == 4:
        # Conv layer activations (batch, channels, height, width) or (batch, height, width, channels)
        return _visualize_conv_activations(activations, layer_name, cmap)
    else:
        # Dense layer activations
        return _visualize_dense_activations(activations, layer_name, cmap)

def _visualize_conv_activations(activations, layer_name=None, cmap='viridis'):
    """Visualize convolutional layer activations."""
    # Take the first example in the batch
    activations = activations[0]
    
    # Determine the format of activations
    if activations.shape[0] < activations.shape[-1]:
        # (height, width, channels)
        n_channels = activations.shape[-1]
        height, width = activations.shape[0], activations.shape[1]
    else:
        # (channels, height, width)
        n_channels = activations.shape[0]
        height, width = activations.shape[1], activations.shape[2]
        # Reorder to (height, width, channels)
        activations = np.transpose(activations, (1, 2, 0))
    
    # Limit the number of channels to display
    n_display = min(16, n_channels)
    
    # Create a grid of subplots
    grid_size = int(np.ceil(np.sqrt(n_display)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    
    # Flatten the axes array if it's multi-dimensional
    if grid_size > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Plot each channel
    for i in range(n_display):
        ax = axes[i]
        channel_data = activations[:, :, i]
        im = ax.imshow(channel_data, cmap=cmap)
        ax.set_title(f'Channel {i}')
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(n_display, len(axes)):
        axes[i].axis('off')
    
    # Add a colorbar
    fig.colorbar(im, ax=axes, shrink=0.8, label='Activation')
    
    # Set the title
    if layer_name:
        fig.suptitle(f'Activations for layer: {layer_name}', fontsize=16)
    else:
        fig.suptitle('Layer Activations', fontsize=16)
    
    fig.tight_layout()
    
    return fig

def _visualize_dense_activations(activations, layer_name=None, cmap='viridis'):
    """Visualize dense layer activations."""
    # Take the first example in the batch
    if len(activations.shape) > 1:
        activations = activations[0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot activations as a bar chart
    ax.bar(range(len(activations)), activations)
    ax.set_xlabel('Neuron')
    ax.set_ylabel('Activation')
    
    # Set the title
    if layer_name:
        ax.set_title(f'Activations for layer: {layer_name}')
    else:
        ax.set_title('Layer Activations')
    
    fig.tight_layout()
    
    return fig

def plot_feature_importance(feature_importances, feature_names=None, top_n=20):
    """
    Plot feature importance.
    
    Args:
        feature_importances: Array of feature importances
        feature_names: Names of features (if None, uses indices)
        top_n: Number of top features to show
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sort importances
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(feature_importances))]
    
    # Limit to top_n features
    indices = np.argsort(feature_importances)[-top_n:]
    top_importances = feature_importances[indices]
    top_names = [feature_names[i] for i in indices]
    
    # Plot as horizontal bar chart
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_importances)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()  # Highest importance at the top
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    
    fig.tight_layout()
    
    return fig

def plot_surrogate_vs_original(surrogate_preds, original_preds, title="Surrogate vs Original Model"):
    """
    Plot surrogate model predictions against original model predictions.
    
    Args:
        surrogate_preds: Predictions from the surrogate model
        original_preds: Predictions from the original model
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Flatten if needed
    surrogate_preds = surrogate_preds.flatten()
    original_preds = original_preds.flatten()
    
    # Create scatter plot
    ax.scatter(original_preds, surrogate_preds, alpha=0.5)
    
    # Add diagonal line (perfect match)
    min_val = min(np.min(surrogate_preds), np.min(original_preds))
    max_val = max(np.max(surrogate_preds), np.max(original_preds))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Labels and title
    ax.set_xlabel('Original Model Predictions')
    ax.set_ylabel('Surrogate Model Predictions')
    ax.set_title(title)
    
    # Calculate and display correlation
    correlation = np.corrcoef(original_preds, surrogate_preds)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {correlation:.4f}', 
            transform=ax.transAxes, verticalalignment='top')
    
    fig.tight_layout()
    
    return fig

def plot_drift_metrics(drift_metrics, layer_name=None):
    """
    Plot drift metrics over time.
    
    Args:
        drift_metrics: DataFrame with drift metrics
        layer_name: Name of layer to plot (if None, plots all layers)
        
    Returns:
        Matplotlib figure
    """
    # Filter for specific layer if provided
    if layer_name is not None:
        metrics = drift_metrics[drift_metrics['layer'] == layer_name]
    else:
        metrics = drift_metrics
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot mean drift
    for layer, group in metrics.groupby('layer'):
        ax1.plot(group['timestamp'], group['mean_drift'], label=layer)
    
    ax1.set_ylabel('Mean Drift')
    ax1.legend()
    ax1.grid(True)
    
    # Plot distribution drift
    for layer, group in metrics.groupby('layer'):
        ax2.plot(group['timestamp'], group['distribution_drift'], label=layer)
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Distribution Drift')
    ax2.legend()
    ax2.grid(True)
    
    # Set title
    if layer_name:
        fig.suptitle(f'Drift Metrics for Layer: {layer_name}')
    else:
        fig.suptitle('Drift Metrics Over Time')
    
    fig.tight_layout()
    
    return fig
