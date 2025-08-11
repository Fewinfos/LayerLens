"""
Module for combining layer explanations into a hierarchical explanation.
"""

import numpy as np
import networkx as nx

class StitchingEngine:
    """Combines layer explanations into a hierarchical structure."""
    
    def __init__(self, fidelity_threshold=0.8):
        """
        Initialize the stitching engine.
        
        Args:
            fidelity_threshold (float): Minimum fidelity required for stitching
        """
        self.fidelity_threshold = fidelity_threshold
        self.explanation_graph = None
        self.layer_explanations = {}
    
    def stitch(self, layer_explanations):
        """
        Combine layer explanations into a hierarchical structure.
        
        Args:
            layer_explanations (dict): Dictionary of explanations by layer name
            
        Returns:
            Hierarchical explanation object
        """
        self.layer_explanations = layer_explanations
        
        # Create a graph to represent layer relationships
        self._build_explanation_graph()
        
        # Create a hierarchical explanation object
        hierarchical_explanation = HierarchicalExplanation(
            self.explanation_graph,
            self.layer_explanations
        )
        
        return hierarchical_explanation
    
    def _build_explanation_graph(self):
        """Build a graph representing layer relationships."""
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes for each layer
        for layer_name in self.layer_explanations.keys():
            G.add_node(layer_name)
        
        # Add edges between layers based on model architecture
        # This is a simplified version - in a real implementation,
        # the edges would be determined by the model architecture
        sorted_layers = sorted(self.layer_explanations.keys())
        for i in range(len(sorted_layers) - 1):
            G.add_edge(sorted_layers[i], sorted_layers[i+1])
        
        self.explanation_graph = G
    
    def optimize_path(self, start_layer, end_layer):
        """
        Find the optimal explanation path between two layers.
        
        Args:
            start_layer (str): Starting layer name
            end_layer (str): Ending layer name
            
        Returns:
            List of layer names forming the optimal path
        """
        if self.explanation_graph is None:
            raise ValueError("Explanation graph not built. Call stitch() first.")
        
        # Check if the layers exist
        if start_layer not in self.explanation_graph.nodes:
            raise ValueError(f"Start layer not found: {start_layer}")
        if end_layer not in self.explanation_graph.nodes:
            raise ValueError(f"End layer not found: {end_layer}")
        
        # Find all paths between the layers
        all_paths = list(nx.all_simple_paths(
            self.explanation_graph, start_layer, end_layer
        ))
        
        if not all_paths:
            return None
        
        # Evaluate the fidelity of each path
        path_scores = []
        for path in all_paths:
            # Calculate the compositional fidelity of the path
            path_fidelity = self._calculate_path_fidelity(path)
            path_scores.append(path_fidelity)
        
        # Return the path with the highest fidelity
        best_path_idx = np.argmax(path_scores)
        return all_paths[best_path_idx]
    
    def _calculate_path_fidelity(self, path):
        """
        Calculate the fidelity of a path through the layers.
        
        Args:
            path (list): List of layer names
            
        Returns:
            Fidelity score for the path
        """
        # This is a simplified implementation
        # In practice, we would calculate the compositional fidelity
        # by comparing the outputs of the composed surrogate models
        # with the output of the corresponding layers in the original model
        
        # For now, use the average of the individual layer fidelities
        fidelities = []
        for layer_name in path:
            if layer_name in self.layer_explanations:
                # Assume the explanation has a fidelity attribute
                fidelity = getattr(self.layer_explanations[layer_name], 'fidelity', 0.9)
                fidelities.append(fidelity)
        
        if not fidelities:
            return 0.0
        
        return np.mean(fidelities)

class HierarchicalExplanation:
    """Class to represent hierarchical explanations."""
    
    def __init__(self, graph, layer_explanations):
        """
        Initialize hierarchical explanation.
        
        Args:
            graph: NetworkX graph representing layer relationships
            layer_explanations (dict): Dictionary of explanations by layer name
        """
        self.graph = graph
        self.layer_explanations = layer_explanations
        self.layer_order = list(nx.topological_sort(graph))
    
    def get_layer_explanation(self, layer_name):
        """
        Get explanation for a specific layer.
        
        Args:
            layer_name (str): Name of the layer
            
        Returns:
            Explanation for the layer
        """
        return self.layer_explanations.get(layer_name)
    
    def get_path_explanation(self, start_layer, end_layer):
        """
        Get explanation for a path between layers.
        
        Args:
            start_layer (str): Starting layer name
            end_layer (str): Ending layer name
            
        Returns:
            List of explanations for the path
        """
        # Find the shortest path between the layers
        try:
            path = nx.shortest_path(self.graph, start_layer, end_layer)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
        
        # Get explanations for each layer in the path
        path_explanations = []
        for layer_name in path:
            explanation = self.get_layer_explanation(layer_name)
            if explanation is not None:
                path_explanations.append(explanation)
        
        return path_explanations
    
    def get_hierarchical_view(self):
        """
        Get hierarchical representation of explanations.
        
        Returns:
            Dictionary representing the hierarchical structure
        """
        hierarchy = {}
        
        # Start with the first layer
        if not self.layer_order:
            return hierarchy
        
        # Build the hierarchy recursively
        self._build_hierarchy(hierarchy, self.layer_order[0])
        
        return hierarchy
    
    def _build_hierarchy(self, hierarchy, layer_name):
        """
        Recursively build the hierarchy starting from a layer.
        
        Args:
            hierarchy (dict): Dictionary to store the hierarchy
            layer_name (str): Name of the current layer
        """
        # Add the current layer
        hierarchy[layer_name] = {
            'explanation': self.get_layer_explanation(layer_name),
            'children': {}
        }
        
        # Add child layers
        for _, child in self.graph.out_edges(layer_name):
            self._build_hierarchy(hierarchy[layer_name]['children'], child)
