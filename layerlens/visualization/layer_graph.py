"""
Module for visualizing layer relationships in models.
"""

import plotly.graph_objects as go
import networkx as nx
import numpy as np

def plot_layer_graph(model, highlight_layers=None, figsize=(12, 8)):
    """
    Create a graph visualization of model layers.
    
    Args:
        model: Model to visualize
        highlight_layers: Layers to highlight in the visualization
        figsize: Size of the figure
        
    Returns:
        Plotly figure object
    """
    # Build a graph from the model
    G = build_layer_graph(model)
    
    # Create the figure
    fig = create_layer_graph_figure(G, highlighted_layer=highlight_layers)
    
    # Set the figure size
    fig.update_layout(
        width=figsize[0] * 100,
        height=figsize[1] * 100
    )
    
    return fig

def build_layer_graph(model):
    """
    Build a graph representation of a model's layers.
    
    Args:
        model: Deep learning model
        
    Returns:
        NetworkX graph representing the model
    """
    G = nx.DiGraph()
    
    # Detect framework
    framework = _detect_framework(model)
    
    if framework == 'pytorch':
        return _build_pytorch_graph(model)
    elif framework == 'tensorflow':
        return _build_tensorflow_graph(model)
    else:
        raise ValueError(f"Unsupported framework: {framework}")

def _detect_framework(model):
    """Detect which deep learning framework the model uses."""
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

def _build_pytorch_graph(model):
    """Build a graph from a PyTorch model."""
    G = nx.DiGraph()
    
    # Add nodes for each module
    for name, module in model.named_modules():
        if name == '':  # Skip the model itself
            continue
        
        # Add the node with module type information
        G.add_node(name, type=module.__class__.__name__)
    
    # Add edges based on module hierarchy
    for name, module in model.named_modules():
        if name == '':  # Skip the model itself
            continue
        
        # Find parent module
        parent_name = '.'.join(name.split('.')[:-1])
        if parent_name and parent_name in G:
            G.add_edge(parent_name, name)
    
    # If no edges were added (flat model), create a linear chain
    if len(G.edges) == 0 and len(G.nodes) > 1:
        nodes = list(G.nodes)
        for i in range(len(nodes) - 1):
            G.add_edge(nodes[i], nodes[i+1])
    
    return G

def _build_tensorflow_graph(model):
    """Build a graph from a TensorFlow/Keras model."""
    G = nx.DiGraph()
    
    # Check if model has layers attribute
    if hasattr(model, 'layers'):
        # Add nodes for each layer
        for i, layer in enumerate(model.layers):
            G.add_node(layer.name, type=layer.__class__.__name__)
        
        # Add edges based on model architecture
        if hasattr(model, 'input_layers') and hasattr(model, 'output_layers'):
            # Functional API model
            for layer in model.layers:
                for node in layer._inbound_nodes:
                    for inbound_layer in node.inbound_layers:
                        if hasattr(inbound_layer, 'name'):
                            G.add_edge(inbound_layer.name, layer.name)
        else:
            # Sequential model - create a linear chain
            for i in range(len(model.layers) - 1):
                G.add_edge(model.layers[i].name, model.layers[i+1].name)
    
    return G

def create_layer_graph_figure(explanations, highlighted_layer=None):
    """
    Create a Plotly figure for the layer graph.
    
    Args:
        explanations: Explanations object or NetworkX graph
        highlighted_layer: Layer to highlight
        
    Returns:
        Plotly figure
    """
    # Check if explanations is a graph or an explanations object
    if hasattr(explanations, 'graph'):
        G = explanations.graph
    elif isinstance(explanations, nx.Graph):
        G = explanations
    else:
        # Try to extract a graph from the explanations
        G = nx.DiGraph()
        
        # Add nodes for each layer in the explanations
        for layer_name in explanations:
            G.add_node(layer_name)
        
        # Add edges to create a chain
        layer_names = list(explanations.keys())
        for i in range(len(layer_names) - 1):
            G.add_edge(layer_names[i], layer_names[i+1])
    
    # Create a layout for the graph
    pos = nx.spring_layout(G)
    
    # Create edge trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Create node trace
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_colors = []
    for node in G.nodes():
        if node == highlighted_layer:
            node_colors.append('#f44336')  # Highlighted node
        else:
            node_colors.append('#1f77b4')  # Regular node
            
    node_sizes = []
    for node in G.nodes():
        if node == highlighted_layer:
            node_sizes.append(20)  # Larger for highlighted node
        else:
            node_sizes.append(15)  # Regular size

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line=dict(width=2)
        ),
        text=list(G.nodes()),
    )

    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Model Layer Graph',
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    
    return fig
