"""
Test script for manual verification of LayerLens functionality.
Run this script to quickly verify that LayerLens is working correctly.
"""

import os
import sys
import numpy as np
import warnings

# Add layerlens to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all LayerLens modules can be imported."""
    print("Testing imports...")
    
    try:
        from layerlens import Explainer
        print("✓ Main Explainer class imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Explainer: {e}")
        return False
    
    try:
        from layerlens.core.surrogate_builder import SurrogateBuilder
        from layerlens.core.layer_extractor import LayerExtractor
        from layerlens.core.stitching_engine import StitchingEngine
        print("✓ Core modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import core modules: {e}")
        return False
    
    try:
        from layerlens.visualization.dashboard import InteractiveDashboard
        from layerlens.visualization.layer_graph import LayerGraphVisualizer
        print("✓ Visualization modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import visualization modules: {e}")
        return False
    
    try:
        from layerlens.monitoring.drift_detector import DriftDetector
        from layerlens.monitoring.alert_system import AlertSystem
        print("✓ Monitoring modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import monitoring modules: {e}")
        return False
    
    try:
        from layerlens.utils.config import LayerLensConfig
        from layerlens.utils.data_utils import preprocess_data
        print("✓ Utility modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import utility modules: {e}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic LayerLens functionality with mock data."""
    print("\nTesting basic functionality...")
    
    try:
        from layerlens.core.surrogate_builder import SurrogateBuilder
        from layerlens.utils.data_utils import preprocess_data
        
        # Create sample data
        X = np.random.rand(50, 10)
        y = np.random.rand(50, 5)
        
        # Test data preprocessing
        X_processed = preprocess_data(X, None, normalize=True)
        print("✓ Data preprocessing works")
        
        # Test surrogate model building
        builder = SurrogateBuilder(surrogate_type='tree')
        surrogate = builder.fit('test_layer', X_processed, y)
        print("✓ Surrogate model building works")
        
        # Test explanation generation
        explanation = builder.explain('test_layer', X_processed[0:1])
        print("✓ Explanation generation works")
        
        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def test_drift_detection():
    """Test drift detection functionality."""
    print("\nTesting drift detection...")
    
    try:
        from layerlens.monitoring.drift_detector import DriftDetector
        
        # Create reference data
        reference_data = np.random.normal(0, 1, (100, 10))
        detector = DriftDetector(reference_data)
        
        # Test with similar data (should not detect drift)
        new_data = np.random.normal(0, 1, (50, 10))
        results = detector.detect(new_data)
        print("✓ Drift detection with similar data works")
        
        # Test with different data (might detect drift)
        different_data = np.random.normal(2, 1.5, (50, 10))
        results = detector.detect(different_data)
        print("✓ Drift detection with different data works")
        
        return True
    except Exception as e:
        print(f"✗ Drift detection test failed: {e}")
        return False


def test_configuration():
    """Test configuration management."""
    print("\nTesting configuration...")
    
    try:
        from layerlens.utils.config import LayerLensConfig
        
        # Test loading default configuration
        config = LayerLensConfig()
        print("✓ Default configuration loading works")
        
        # Test setting values
        config.set('test_param', 'test_value')
        assert config.get('test_param') == 'test_value'
        print("✓ Configuration get/set works")
        
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_dependencies():
    """Test that all required dependencies are available."""
    print("\nTesting dependencies...")
    
    required_packages = [
        'numpy', 'scipy', 'scikit-learn', 'pandas',
        'plotly', 'dash', 'matplotlib', 'networkx'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is available")
        except ImportError:
            print(f"✗ {package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        return False
    
    return True


def main():
    """Run all verification tests."""
    print("LayerLens Verification Test")
    print("=" * 40)
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    tests = [
        test_dependencies,
        test_imports,
        test_basic_functionality,
        test_drift_detection,
        test_configuration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"Verification Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! LayerLens is ready to use.")
        return True
    else:
        print("❌ Some tests failed. Please check the output above.")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)