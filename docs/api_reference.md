# LayerLens API Reference

## Core Module

### `layerlens.core.SurrogateBuilder`

```python
class SurrogateBuilder(surrogate_type='tree', max_samples=1000, fidelity_threshold=0.9)
```

Builds interpretable surrogate models for neural network layers.

**Parameters:**
- `surrogate_type` (str): Type of surrogate model ('tree', 'linear', 'rule')
- `max_samples` (int): Maximum number of samples to use for training
- `fidelity_threshold` (float): Minimum required fidelity score

**Methods:**
- `fit(layer, data)`: Train a surrogate model for the given layer
- `evaluate(metrics=['accuracy', 'fidelity'])`: Evaluate the surrogate model
- `explain(instance)`: Generate explanation for a specific instance

### `layerlens.core.LayerExtractor`

```python
class LayerExtractor(model, layers=None)
```

Extracts intermediate outputs from model layers.

**Parameters:**
- `model`: The model to extract layers from
- `layers`: List of layer names to extract (if None, extracts all)

**Methods:**
- `extract(data)`: Extract layer outputs for the given data
- `register_hook(layer_name, hook_fn)`: Register a custom hook function
- `remove_hooks()`: Remove all registered hooks

### `layerlens.core.StitchingEngine`

```python
class StitchingEngine(fidelity_threshold=0.8)
```

Combines layer explanations into a hierarchical structure.

**Parameters:**
- `fidelity_threshold` (float): Minimum fidelity required for stitching

**Methods:**
- `stitch(explanations)`: Combine layer explanations
- `optimize_path(start_layer, end_layer)`: Find optimal explanation path
- `get_hierarchical_view()`: Get hierarchical representation of explanations

## Visualization Module

### `layerlens.visualization.dashboard`

```python
def create_dashboard(explanations, port=8050)
```

Creates an interactive dashboard for exploring explanations.

**Parameters:**
- `explanations`: Explanation objects to visualize
- `port` (int): Port for the dashboard server

### `layerlens.visualization.layer_graph`

```python
def plot_layer_graph(model, highlight_layers=None, figsize=(12, 8))
```

Creates a graph visualization of model layers.

**Parameters:**
- `model`: Model to visualize
- `highlight_layers`: Layers to highlight in the visualization
- `figsize`: Size of the figure

## Monitoring Module

### `layerlens.monitoring.drift_detector`

```python
class DriftDetector(reference_data, sensitivity=0.05)
```

Detects drift in layer behavior over time.

**Parameters:**
- `reference_data`: Baseline data for drift comparison
- `sensitivity` (float): Sensitivity threshold for drift detection

**Methods:**
- `detect(new_data)`: Check for drift in new data
- `get_drift_metrics()`: Get detailed drift metrics
- `update_reference(new_reference)`: Update the reference data

## Utils Module

### `layerlens.utils.data_utils`

```python
def preprocess_data(data, target_model, normalize=True)
```

Preprocesses data for use with LayerLens.

**Parameters:**
- `data`: Input data to preprocess
- `target_model`: Model the data will be used with
- `normalize` (bool): Whether to normalize the data
