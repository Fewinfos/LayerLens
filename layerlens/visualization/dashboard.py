"""
Module for creating interactive dashboards to explore explanations.
"""

import os
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

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
    else:
        # Assume it's a dictionary of layer explanations
        layer_names = list(explanations.keys())
    
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
        from layerlens.visualization.layer_graph import create_layer_graph_figure
        
        # Create the layer graph figure
        fig = create_layer_graph_figure(explanations, highlighted_layer=selected_layer)
        
        return fig
    
    # Define callback to update the explanation plot
    @app.callback(
        Output('explanation-plot', 'figure'),
        [Input('layer-selector', 'value')]
    )
    def update_explanation_plot(selected_layer):
        if not selected_layer:
            return go.Figure()
        
        # Get the explanation for the selected layer
        if hasattr(explanations, 'get_layer_explanation'):
            layer_explanation = explanations.get_layer_explanation(selected_layer)
        else:
            # Assume it's a dictionary
            layer_explanation = explanations.get(selected_layer)
        
        if layer_explanation is None:
            return go.Figure()
        
        # Create a figure based on the explanation type
        explanation_type = getattr(layer_explanation, 'type', 'generic')
        
        if explanation_type == 'tree_path':
            return _create_tree_explanation_figure(layer_explanation)
        elif explanation_type == 'linear':
            return _create_linear_explanation_figure(layer_explanation)
        else:
            return _create_generic_explanation_figure(layer_explanation)
    
    # Define callback to update the feature flow
    @app.callback(
        Output('feature-flow', 'figure'),
        [Input('layer-selector', 'value')]
    )
    def update_feature_flow(selected_layer):
        from layerlens.visualization.feature_flow import create_feature_flow_figure
        
        # Create the feature flow figure
        fig = create_feature_flow_figure(explanations, end_layer=selected_layer)
        
        return fig
    
    # Return the application
    return app

def _create_tree_explanation_figure(explanation):
    """Create a figure for tree-based explanations."""
    # Extract data from the explanation
    feature_importances = explanation.get('feature_importances', [])
    
    # Create a bar chart of feature importances
    fig = px.bar(
        x=list(range(len(feature_importances))),
        y=feature_importances,
        labels={'x': 'Feature', 'y': 'Importance'},
        title='Feature Importances'
    )
    
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
    # Try to extract feature importances
    feature_importances = explanation.get('feature_importances', [])
    
    if len(feature_importances) > 0:
        # Create a bar chart of feature importances
        fig = px.bar(
            x=list(range(len(feature_importances))),
            y=feature_importances,
            labels={'x': 'Feature', 'y': 'Importance'},
            title='Feature Importances'
        )
    else:
        # Create an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No feature importance data available",
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
    app.run_server(port=port, debug=debug)

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
