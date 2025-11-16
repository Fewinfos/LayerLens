"""
Quick start script for LayerLens - demonstrates basic functionality.
Run this after installation to verify LayerLens is working.
"""

import sys
import os

# Add layerlens to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*60)
print("LayerLens Quick Start Demo")
print("="*60)

# Step 1: Import LayerLens
print("\n[Step 1] Importing LayerLens...")
try:
    from layerlens import Explainer
    from layerlens.core import SurrogateBuilder
    from layerlens.monitoring import DriftDetector
    print("SUCCESS: LayerLens imported successfully!")
except ImportError as e:
    print(f"ERROR: Failed to import LayerLens: {e}")
    print("\nPlease install dependencies:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

# Step 2: Test basic functionality
print("\n[Step 2] Testing surrogate model building...")
try:
    import numpy as np
    
    # Create sample data
    X = np.random.rand(100, 20)  # 100 samples, 20 features
    y = np.random.rand(100, 10)   # 100 samples, 10 outputs
    
    # Build a surrogate model
    builder = SurrogateBuilder(surrogate_type='tree', max_samples=100)
    surrogate = builder.fit('demo_layer', X, y)
    
    print("SUCCESS: Surrogate model trained!")
    print(f"  - Model type: Tree-based")
    print(f"  - Input shape: {X.shape}")
    print(f"  - Output shape: {y.shape}")
    
except Exception as e:
    print(f"ERROR: Failed to build surrogate: {e}")
    sys.exit(1)

# Step 3: Test explanation generation
print("\n[Step 3] Generating explanations...")
try:
    # Generate explanation for first sample
    test_sample = X[0:1]
    explanation = builder.explain('demo_layer', test_sample)
    
    print("SUCCESS: Explanation generated!")
    print(f"  - Explanation type: {explanation.get('type', 'N/A')}")
    print(f"  - Contains prediction: {'prediction' in explanation}")
    
except Exception as e:
    print(f"ERROR: Failed to generate explanation: {e}")
    sys.exit(1)

# Step 4: Test drift detection
print("\n[Step 4] Testing drift detection...")
try:
    # Create reference data
    reference_data = np.random.normal(0, 1, (200, 20))
    
    # Initialize drift detector
    detector = DriftDetector(reference_data, sensitivity=0.05)
    
    # Test with similar data (no drift expected)
    new_data = np.random.normal(0, 1, (50, 20))
    results = detector.detect(new_data)
    
    print("SUCCESS: Drift detection working!")
    print(f"  - Drift detected: {results['global'].get('drift_detected', False)}")
    print(f"  - Features monitored: {reference_data.shape[1]}")
    
except Exception as e:
    print(f"ERROR: Failed drift detection: {e}")
    sys.exit(1)

# Step 5: Summary
print("\n" + "="*60)
print("All tests passed! LayerLens is working correctly.")
print("="*60)
print("\nNext steps:")
print("  1. Run the MNIST demo: jupyter notebook examples/mnist_demo.ipynb")
print("  2. Check USAGE.md for detailed documentation")
print("  3. Try with your own models!")
print("\nFor more examples, see the 'examples/' directory.")
print("="*60)