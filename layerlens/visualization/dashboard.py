"""
Module for creating interactive dashboards to explore explanations.
"""

import os
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


class InteractiveDashboard:
    """Interactive dashboard for visualizing model explanations."""
    
    def __init__(self, explanations, port=8050, theme='plotly_white'):
        """
        Initialize the interactive dashboard.
        
        Args:
            explanations: Explanation object or dictionary
            port (int): Port for the dashboard server
            theme (str): Plotly theme to use
        """
        self.explanations = explanations
        self.port = port
        self.theme = theme
        self.app = None
        
    def run(self, host='localhost', port=None, debug=False):
        """Run the dashboard server."""
        if port is None:
            port = self.port
        
        if self.app is None:
            self.app = create_dashboard(self.explanations, port, debug)
        
        self.app.run(host=host, port=port, debug=debug)
        
    def create(self):
        """Create the dashboard app without running it."""
        if self.app is None:
            self.app = create_dashboard(self.explanations, self.port, False)
        return self.app

def create_dashboard(explanations, port=8050, debug=False):
    """
    Create an interactive dashboard for exploring explanations.
    
    Args:
        explanations: Explanation object to visualize
        port (int): Port for the dashboard server
        debug (bool): Whether to run in debug mode
        
    Returns:
        Dashboard application
    """
    try:
        import dash
        from dash import dcc, html
        from dash.dependencies import Input, Output
    except ImportError:
        raise ImportError("Dashboard requires dash to be installed. "
                          "Install with pip install dash")
    
    # Create a Dash application
    app = dash.Dash(__name__)
    
    # Extract layer names from explanations
    if hasattr(explanations, 'layer_order'):
        layer_names = explanations.layer_order
    elif hasattr(explanations, 'layer_explanations'):
        # HierarchicalExplanation object
        layer_names = list(explanations.layer_explanations.keys())
    elif isinstance(explanations, dict):
        # Direct dictionary of layer explanations
        layer_names = list(explanations.keys())
    else:
        layer_names = []
    
    # Create the layout
    app.layout = html.Div([
        html.H1("LayerLens Dashboard"),
        
        html.Div([
            html.Div([
                html.H3("Model Structure"),
                dcc.Graph(id='layer-graph')
            ], className='six columns'),
            
            html.Div([
                html.H3("Layer Explanations"),
                dcc.Dropdown(
                    id='layer-selector',
                    options=[{'label': name, 'value': name} for name in layer_names],
                    value=layer_names[0] if layer_names else None
                ),
                dcc.Graph(id='explanation-plot')
            ], className='six columns')
        ], className='row'),
        
        html.Div([
            html.H3("Feature Flow"),
            dcc.Graph(id='feature-flow')
        ])
    ])
    
    # Define callback to update the layer graph
    @app.callback(
        Output('layer-graph', 'figure'),
        [Input('layer-selector', 'value')]
    )
    def update_layer_graph(selected_layer):
        # Create a simple text-based layer structure visualization
        import plotly.graph_objects as go
        
        # Create a text representation of the model structure
        text_lines = ["Layer Structure:"]
        for i, layer_name in enumerate(layer_names):
            marker = "➤ " if layer_name == selected_layer else "  "
            text_lines.append(f"{marker}{i+1}. {layer_name}")
        
        fig = go.Figure()
        fig.add_annotation(
            text="<br>".join(text_lines),
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, family="monospace"),
            align="left"
        )
        fig.update_layout(
            title="Model Layers",
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            height=400
        )
        
        return fig
    
    # Define callback to update the explanation plot
    @app.callback(
        Output('explanation-plot', 'figure'),
        [Input('layer-selector', 'value')]
    )
    def update_explanation_plot(selected_layer):
        if not selected_layer:
            fig = go.Figure()
            fig.add_annotation(
                text="Select a layer to view explanations",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Get the explanation for the selected layer
        layer_explanation = None
        if hasattr(explanations, 'get_layer_explanation'):
            layer_explanation = explanations.get_layer_explanation(selected_layer)
        elif hasattr(explanations, 'layer_explanations'):
            layer_explanation = explanations.layer_explanations.get(selected_layer)
        elif isinstance(explanations, dict):
            layer_explanation = explanations.get(selected_layer)
        
        if layer_explanation is None:
            fig = go.Figure()
            fig.add_annotation(
                text=f"No explanation data available for layer: {selected_layer}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Check if it's a surrogate model object with feature_importances_
        if hasattr(layer_explanation, 'feature_importances_'):
            return _create_tree_explanation_figure(layer_explanation)
        # Check if it's a dictionary with explanation data
        elif isinstance(layer_explanation, dict):
            if 'feature_importances' in layer_explanation:
                return _create_tree_explanation_figure(layer_explanation)
            elif 'coefficients' in layer_explanation:
                return _create_linear_explanation_figure(layer_explanation)
            else:
                return _create_generic_explanation_figure(layer_explanation)
        else:
            # Try generic approach
            return _create_generic_explanation_figure(layer_explanation)
    
    # Define callback to update the feature flow
    @app.callback(
        Output('feature-flow', 'figure'),
        [Input('layer-selector', 'value')]
    )
    def update_feature_flow(selected_layer):
        # Create a simple feature flow visualization
        import plotly.graph_objects as go
        import numpy as np
        
        if not selected_layer:
            fig = go.Figure()
            fig.add_annotation(
                text="Select a layer to view details",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title="Layer Information",
                height=300,
                xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                yaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
            )
            return fig
        
        # Show information about the selected layer
        layer_info = []
        layer_info.append(f"<b>Selected Layer: {selected_layer}</b>")
        layer_info.append("")
        
        # Get layer explanation if available
        layer_exp = None
        if hasattr(explanations, 'layer_explanations'):
            layer_exp = explanations.layer_explanations.get(selected_layer)
        elif isinstance(explanations, dict):
            layer_exp = explanations.get(selected_layer)
        
        if layer_exp:
            if hasattr(layer_exp, 'feature_importances_'):
                importances = layer_exp.feature_importances_
                layer_info.append(f"Total Features: {len(importances)}")
                layer_info.append(f"Non-zero Features: {np.sum(importances > 0)}")
                layer_info.append(f"Max Importance: {np.max(importances):.6f}")
                layer_info.append(f"Mean Importance: {np.mean(importances):.6f}")
                layer_info.append(f"Std Importance: {np.std(importances):.6f}")
            else:
                layer_info.append("Surrogate model trained")
                layer_info.append("Feature importance data available")
        else:
            layer_info.append("No explanation data for this layer")
        
        fig = go.Figure()
        fig.add_annotation(
            text="<br>".join(layer_info),
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, family="Arial"),
            align="left"
        )
        fig.update_layout(
            title="Layer Statistics",
            height=300,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
        )
        
        return fig
    
    # Return the application
    return app

def _create_tree_explanation_figure(explanation):
    """Create a figure for tree-based explanations."""
    # Extract data from the explanation
    if hasattr(explanation, 'feature_importances_'):
        # It's a trained model object
        feature_importances = explanation.feature_importances_
    elif isinstance(explanation, dict):
        feature_importances = explanation.get('feature_importances', [])
    else:
        feature_importances = []
    
    if len(feature_importances) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No feature importance data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Get top 50 features to avoid overcrowding
    if len(feature_importances) > 50:
        top_indices = np.argsort(feature_importances)[-50:]
        top_importances = feature_importances[top_indices]
        x_values = top_indices
    else:
        top_importances = feature_importances
        x_values = list(range(len(feature_importances)))
    
    # Create a bar chart of feature importances
    fig = px.bar(
        x=x_values,
        y=top_importances,
        labels={'x': 'Feature Index', 'y': 'Importance'},
        title=f'Feature Importances (Top {len(top_importances)} features)'
    )
    
    fig.update_layout(height=500)
    
    return fig

def _create_linear_explanation_figure(explanation):
    """Create a figure for linear model explanations."""
    # Extract data from the explanation
    coefficients = explanation.get('coefficients', [])
    
    if len(coefficients.shape) > 1:
        # For multi-output models, show the norm of the coefficients
        coef_norm = np.linalg.norm(coefficients, axis=0)
        values = coef_norm
    else:
        values = coefficients
    
    # Create a bar chart of coefficients
    fig = px.bar(
        x=list(range(len(values))),
        y=values,
        labels={'x': 'Feature', 'y': 'Coefficient'},
        title='Model Coefficients'
    )
    
    return fig

def _create_generic_explanation_figure(explanation):
    """Create a figure for generic explanations."""
    # Try to extract feature importances from various sources
    feature_importances = []
    
    if hasattr(explanation, 'feature_importances_'):
        feature_importances = explanation.feature_importances_
    elif isinstance(explanation, dict):
        feature_importances = explanation.get('feature_importances', [])
    elif isinstance(explanation, np.ndarray):
        feature_importances = explanation
    
    if len(feature_importances) > 0:
        # Get top 50 features
        if len(feature_importances) > 50:
            top_indices = np.argsort(feature_importances)[-50:]
            top_importances = feature_importances[top_indices]
            x_values = top_indices
        else:
            top_importances = feature_importances
            x_values = list(range(len(feature_importances)))
        
        # Create a bar chart of feature importances
        fig = px.bar(
            x=x_values,
            y=top_importances,
            labels={'x': 'Feature Index', 'y': 'Importance'},
            title=f'Feature Importances (Top {len(top_importances)} features)'
        )
        fig.update_layout(height=500)
    else:
        # Create an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No feature importance data available for this layer",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    return fig

def show_dashboard(explanations, port=8050, debug=False):
    """
    Create and run an interactive dashboard.
    
    Args:
        explanations: Explanation object to visualize
        port (int): Port for the dashboard server
        debug (bool): Whether to run in debug mode
    """
    app = create_dashboard(explanations, port, debug)
    app.run(port=port, debug=debug)

def export_explanation_to_html(explanations, output_file):
    """
    Export explanations to a standalone HTML file.
    
    Args:
        explanations: Explanation object to export
        output_file (str): Path to the output HTML file
    """
    try:
        import dash
        from dash import dcc, html
    except ImportError:
        raise ImportError("Export to HTML requires dash to be installed. "
                          "Install with pip install dash")
    
    # Create a dashboard
    app = create_dashboard(explanations)
    
    # Generate the HTML
    html_string = app.index_string
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_string)
