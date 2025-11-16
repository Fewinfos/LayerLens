# LayerLens Usage Guide

## Installation

### Option 1: Install from source (recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/LayerLens.git
   cd LayerLens
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv layerlens_env
   ```

3. **Activate the virtual environment:**
   
   **Windows (PowerShell):**
   ```powershell
   .\layerlens_env\Scripts\Activate.ps1
   ```
   
   **Windows (Command Prompt):**
   ```cmd
   layerlens_env\Scripts\activate.bat
   ```
   
   **Linux/MacOS:**
   ```bash
   source layerlens_env/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Install ML frameworks (choose one or both):**
   ```bash
   # For PyTorch support
   pip install torch torchvision
   
   # For TensorFlow support
   pip install tensorflow
   ```

6. **Install LayerLens in development mode:**
   ```bash
   pip install -e .
   ```

### Option 2: Install via pip (when published)
```bash
pip install layerlens
```

## Quick Start

### 1. Basic Usage

```python
from layerlens import Explainer
import numpy as np

# Create your model (PyTorch or TensorFlow)
# model = ... (your trained model)

# Initialize LayerLens
explainer = Explainer(model)

# Prepare your data
input_data = np.random.rand(10, 28, 28, 1)  # Example image data

# Generate explanations
explanation = explainer.explain(input_data)

# Visualize results
explainer.visualize(explanation, output_dir='./explanations')
```

### 2. Advanced Usage with Monitoring

```python
from layerlens import Explainer
from layerlens.monitoring import DriftDetector, AlertSystem

# Initialize with monitoring
explainer = Explainer(model, enable_monitoring=True)

# Set up drift detection
reference_data = np.random.rand(1000, 28, 28, 1)
drift_detector = DriftDetector(reference_data)

# Set up alerting
alert_system = AlertSystem()
alert_system.add_channel('email', {
    'type': 'email',
    'smtp_server': 'smtp.gmail.com',
    'port': 587,
    'username': 'your_email@gmail.com',
    'password': 'your_password'
})

# Monitor production data
new_data = np.random.rand(100, 28, 28, 1)
drift_results = drift_detector.detect(new_data)

if drift_results['global']['drift_detected']:
    alert_system.send_alert(
        'drift_detected',
        f"Data drift detected: {drift_results['global']['drift_score']:.3f}",
        severity=0.8
    )
```

## Running Examples

### 1. MNIST Demo (Recommended first step)

```bash
# Navigate to the examples directory
cd examples

# Install Jupyter if not already installed
pip install jupyter

# Start Jupyter notebook
jupyter notebook

# Open and run mnist_demo.ipynb
```

### 2. Other Examples

```bash
# ImageNet demo (requires larger model)
jupyter notebook imagenet_demo.ipynb

# Transformer NLP demo
jupyter notebook transformer_nlp_demo.ipynb
```

## Verification

### Quick Verification
Run the verification script to ensure everything is working:

```bash
python tests/verify_installation.py
```

### Full Test Suite
Run the complete test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pip install pytest-cov
pytest tests/ --cov=layerlens --cov-report=html

# Run specific test categories
pytest tests/ -m "unit"           # Unit tests only
pytest tests/ -m "integration"    # Integration tests only
```

## Configuration

LayerLens can be configured using environment variables or config files:

### Environment Variables
```bash
export LAYERLENS_LOG_LEVEL=INFO
export LAYERLENS_CACHE_DIR=./cache
export LAYERLENS_MAX_SAMPLES=1000
```

### Configuration File
Create a `layerlens_config.yaml` file:

```yaml
# LayerLens Configuration
logging:
  level: INFO
  file: layerlens.log

surrogate:
  type: tree
  max_samples: 1000
  max_depth: 10

visualization:
  theme: plotly_white
  width: 800
  height: 600

monitoring:
  drift_threshold: 0.05
  alert_threshold: 0.8
```

## API Reference

### Main Classes

#### `Explainer`
The main class for generating explanations.

```python
explainer = Explainer(
    model,                    # Your trained model
    surrogate_type='tree',    # 'tree' or 'linear'
    max_samples=1000,         # Max samples for surrogate training
    enable_monitoring=False   # Enable monitoring features
)
```

**Methods:**
- `explain(input_data, layer_names=None)` - Generate explanations
- `visualize(explanation, output_dir=None)` - Create visualizations
- `monitor(input_data, reference_data=None)` - Monitor for drift

#### `SurrogateBuilder`
Build interpretable surrogate models for layers.

```python
from layerlens.core import SurrogateBuilder

builder = SurrogateBuilder(surrogate_type='tree')
surrogate = builder.fit(layer_name, input_data, layer_output)
explanation = builder.explain(layer_name, test_input)
```

#### `DriftDetector`
Detect distribution drift in input data.

```python
from layerlens.monitoring import DriftDetector

detector = DriftDetector(reference_data, sensitivity=0.05)
results = detector.detect(new_data)
```

#### `InteractiveDashboard`
Create interactive web-based visualizations.

```python
from layerlens.visualization import InteractiveDashboard

dashboard = InteractiveDashboard(explanation)
dashboard.run(host='localhost', port=8050)
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   ImportError: No module named 'torch'
   ```
   **Solution:** Install the required ML framework:
   ```bash
   pip install torch  # or tensorflow
   ```

2. **Memory Issues**
   ```
   MemoryError: Unable to allocate array
   ```
   **Solution:** Reduce the `max_samples` parameter:
   ```python
   explainer = Explainer(model, max_samples=100)
   ```

3. **Visualization Issues**
   ```
   Dash app not starting
   ```
   **Solution:** Check if port 8050 is available or specify a different port:
   ```python
   dashboard.run(port=8051)
   ```

4. **Framework Detection Issues**
   ```
   Framework not detected automatically
   ```
   **Solution:** Manually specify the framework:
   ```python
   from layerlens.core import LayerExtractor
   extractor = LayerExtractor(model, framework='pytorch')  # or 'tensorflow'
   ```

### Performance Tips

1. **Use smaller sample sizes** for faster surrogate training
2. **Enable caching** to avoid recomputing explanations
3. **Use GPU acceleration** when available (automatic for PyTorch/TensorFlow)
4. **Process data in batches** for large datasets

### Getting Help

1. **Check the documentation** in the `docs/` directory
2. **Run the verification script** to diagnose issues
3. **Review example notebooks** for usage patterns
4. **Check GitHub issues** for known problems and solutions

## Advanced Features

### Custom Surrogate Models

```python
from sklearn.ensemble import RandomForestRegressor
from layerlens.core import SurrogateBuilder

# Use custom surrogate model
custom_model = RandomForestRegressor(n_estimators=50)
builder = SurrogateBuilder(custom_surrogate=custom_model)
```

### Custom Visualization Themes

```python
from layerlens.visualization import InteractiveDashboard

dashboard = InteractiveDashboard(
    explanation,
    theme='plotly_dark',  # or 'plotly_white', 'ggplot2', etc.
    custom_colors=['#1f77b4', '#ff7f0e', '#2ca02c']
)
```

### Production Monitoring Pipeline

```python
from layerlens.monitoring import ProductionMonitor

monitor = ProductionMonitor(
    model=model,
    reference_data=reference_data,
    alert_config={
        'email': 'admin@company.com',
        'slack_webhook': 'https://hooks.slack.com/...'
    }
)

# Monitor new predictions
monitor.check_prediction(new_input, prediction, ground_truth)
```

## Next Steps

1. **Try the MNIST demo** to get familiar with the basic workflow
2. **Explore the documentation** in the `docs/` directory for detailed theory
3. **Experiment with your own models** using the provided examples as templates
4. **Set up monitoring** for production models
5. **Contribute to the project** by reporting issues or submitting improvements

Happy explaining! 🔍✨