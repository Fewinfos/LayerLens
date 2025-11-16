# LayerLens

LayerLens is a powerful library for layer-by-layer explainability of deep learning models. It provides insights into how models make decisions by analyzing the behavior of individual layers.

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Layer-wise Explanation**: Generate surrogate models for each layer to explain its behavior
- **Hierarchical Insights**: Stitch together layer explanations to understand the full model
- **Interactive Visualization**: Explore model behavior through an intuitive dashboard
- **Production Monitoring**: Detect drift and localize failures in deployed models
- **Framework Agnostic**: Works with PyTorch, TensorFlow, and other major frameworks
- **Comprehensive Testing**: Full test suite ensuring reliability and stability

## Installation

### From Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/LayerLens.git
cd LayerLens

# Install dependencies
pip install -r requirements.txt

# Install LayerLens
pip install -e .
```

### Via pip (when published)
```bash
pip install layerlens
```

**Note:** You'll also need to install at least one ML framework:
```bash
pip install torch  # For PyTorch support
# or
pip install tensorflow  # For TensorFlow support
```

## Quick Start

```python
from layerlens import Explainer
import numpy as np

# Load your trained model (PyTorch or TensorFlow)
# model = ... (your trained model)

# Create a LayerLens explainer
explainer = Explainer(model)

# Prepare your data
input_data = np.random.rand(10, 28, 28, 1)  # Example

# Generate explanations
explanation = explainer.explain(input_data)

# Visualize the results
explainer.visualize(explanation, output_dir='./explanations')

# Monitor for drift (production use)
explainer.monitor(input_data)
```

## Running Examples

Try the MNIST demo to see LayerLens in action:

```bash
cd examples
jupyter notebook mnist_demo.ipynb
```

## Verification

Verify your installation is working correctly:

```bash
python tests/verify_installation.py
```

Run the full test suite:

```bash
pytest tests/
```

## Documentation

For detailed documentation and usage guides:
- **[USAGE.md](USAGE.md)** - Complete usage guide with examples
- **[docs/](docs/)** - Technical documentation and theory
- **[examples/](examples/)** - Jupyter notebook demos

## Project Structure

```
LayerLens/
├── layerlens/           # Main package
│   ├── core/           # Core explainability engine
│   ├── visualization/  # Interactive dashboards and plots
│   ├── monitoring/     # Production monitoring tools
│   └── utils/          # Helper utilities
├── examples/           # Demo notebooks
├── tests/             # Test suite
└── docs/              # Documentation
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use LayerLens in your research, please cite:

```bibtex
@software{layerlens2024,
  title = {LayerLens: Layer-by-Layer Explainability for Deep Learning},
  author = {LayerLens Team},
  year = {2024},
  url = {https://github.com/your-username/LayerLens}
}
```
D:/GitHub/LayerLens/.venv/Scripts/python.exe -m jupyter notebook mnist_demo.ipynb