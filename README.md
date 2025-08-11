# LayerLens

LayerLens is a powerful library for layer-by-layer explainability of deep learning models. It provides insights into how models make decisions by analyzing the behavior of individual layers.

## Features

- **Layer-wise Explanation**: Generate surrogate models for each layer to explain its behavior
- **Hierarchical Insights**: Stitch together layer explanations to understand the full model
- **Interactive Visualization**: Explore model behavior through an intuitive dashboard
- **Production Monitoring**: Detect drift and localize failures in deployed models
- **Framework Agnostic**: Works with PyTorch, TensorFlow, and other major frameworks

## Installation

```bash
pip install layerlens
```

## Quick Start

```python
import layerlens as ll
from layerlens.core import SurrogateBuilder

# Load your model
model = load_your_model()

# Create a LayerLens explainer
explainer = ll.Explainer(model)

# Generate explanations
explanations = explainer.explain(data)

# Visualize the results
ll.visualization.dashboard.show(explanations)
```

## Documentation

For detailed documentation, see the [docs](docs/) directory or visit our [website](https://layerlens.ai).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
