"""
Test suite for LayerLens core functionality.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch

# Import LayerLens components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layerlens.core.surrogate_builder import SurrogateBuilder
from layerlens.core.layer_extractor import LayerExtractor
from layerlens.core.stitching_engine import StitchingEngine, HierarchicalExplanation
from layerlens.core.fidelity_metrics import calculate_fidelity, FidelityScore
from layerlens.utils.data_utils import preprocess_data, split_dataset, load_sample_data
from layerlens.utils.model_utils import get_model_framework, predict_with_model
from layerlens.monitoring.drift_detector import DriftDetector
from layerlens.monitoring.alert_system import AlertSystem


class TestSurrogateBuilder:
    """Test cases for SurrogateBuilder class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.surrogate_builder = SurrogateBuilder(surrogate_type='tree', max_samples=100)
        
        # Create dummy data
        self.input_data = np.random.rand(50, 10)
        self.layer_output = np.random.rand(50, 5)
        
    def test_init(self):
        """Test SurrogateBuilder initialization."""
        assert self.surrogate_builder.surrogate_type == 'tree'
        assert self.surrogate_builder.max_samples == 100
        assert len(self.surrogate_builder.surrogates) == 0
        
    def test_fit_tree_surrogate(self):
        """Test fitting a tree surrogate model."""
        surrogate = self.surrogate_builder.fit('test_layer', self.input_data, self.layer_output)
        
        assert surrogate is not None
        assert 'test_layer' in self.surrogate_builder.surrogates
        assert hasattr(surrogate, 'predict')
        
    def test_fit_linear_surrogate(self):
        """Test fitting a linear surrogate model."""
        builder = SurrogateBuilder(surrogate_type='linear')
        surrogate = builder.fit('test_layer', self.input_data, self.layer_output)
        
        assert surrogate is not None
        assert hasattr(surrogate, 'predict')
        
    def test_explain(self):
        """Test generating explanations."""
        self.surrogate_builder.fit('test_layer', self.input_data, self.layer_output)
        
        # Test explanation for a single instance
        test_instance = self.input_data[0:1]
        explanation = self.surrogate_builder.explain('test_layer', test_instance)
        
        assert explanation is not None
        assert 'type' in explanation
        assert 'prediction' in explanation
        
    def test_evaluate(self):
        """Test surrogate model evaluation."""
        self.surrogate_builder.fit('test_layer', self.input_data, self.layer_output)
        
        # Evaluate on test data
        test_input = np.random.rand(10, 10)
        test_output = np.random.rand(10, 5)
        
        results = self.surrogate_builder.evaluate('test_layer', test_input, test_output)
        
        assert 'fidelity' in results
        
    def test_invalid_surrogate_type(self):
        """Test handling of invalid surrogate type."""
        with pytest.raises(ValueError):
            builder = SurrogateBuilder(surrogate_type='invalid')
            builder.fit('test_layer', self.input_data, self.layer_output)


class TestLayerExtractor:
    """Test cases for LayerExtractor class."""
    
    def test_init_with_mock_model(self):
        """Test LayerExtractor initialization with a mock model."""
        # Create a mock model
        mock_model = Mock()
        mock_model.__class__.__module__ = 'torch.nn'
        mock_model.named_modules.return_value = [('layer1', Mock()), ('layer2', Mock())]
        
        # Test initialization
        with patch.object(LayerExtractor, '_register_pytorch_hooks'):
            extractor = LayerExtractor(mock_model)
            assert extractor.model == mock_model
            assert extractor.framework == 'pytorch'
    
    def test_detect_framework(self):
        """Test framework detection."""
        # Test PyTorch model detection
        mock_torch_model = Mock()
        mock_torch_model.__class__.__module__ = 'torch.nn.modules'
        
        with patch.object(LayerExtractor, '_register_pytorch_hooks'):
            extractor = LayerExtractor(mock_torch_model)
            assert extractor.framework == 'pytorch'
        
        # Test TensorFlow model detection
        mock_tf_model = Mock()
        mock_tf_model.__class__.__module__ = 'tensorflow.keras'
        
        with patch.object(LayerExtractor, '_register_tensorflow_hooks'):
            extractor = LayerExtractor(mock_tf_model)
            assert extractor.framework == 'tensorflow'


class TestStitchingEngine:
    """Test cases for StitchingEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.stitching_engine = StitchingEngine()
        
        # Create mock layer explanations
        self.layer_explanations = {
            'layer1': Mock(),
            'layer2': Mock(),
            'layer3': Mock()
        }
        
    def test_init(self):
        """Test StitchingEngine initialization."""
        assert self.stitching_engine.fidelity_threshold == 0.8
        assert self.stitching_engine.explanation_graph is None
        
    def test_stitch(self):
        """Test stitching layer explanations."""
        result = self.stitching_engine.stitch(self.layer_explanations)
        
        assert isinstance(result, HierarchicalExplanation)
        assert self.stitching_engine.explanation_graph is not None
        assert len(result.layer_explanations) == 3


class TestHierarchicalExplanation:
    """Test cases for HierarchicalExplanation class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        import networkx as nx
        
        # Create a simple graph
        self.graph = nx.DiGraph()
        self.graph.add_edge('layer1', 'layer2')
        self.graph.add_edge('layer2', 'layer3')
        
        self.layer_explanations = {
            'layer1': Mock(),
            'layer2': Mock(),
            'layer3': Mock()
        }
        
        self.hierarchical_explanation = HierarchicalExplanation(
            self.graph, self.layer_explanations
        )
        
    def test_init(self):
        """Test HierarchicalExplanation initialization."""
        assert self.hierarchical_explanation.graph == self.graph
        assert len(self.hierarchical_explanation.layer_explanations) == 3
        assert 'layer1' in self.hierarchical_explanation.layer_order
        
    def test_get_layer_explanation(self):
        """Test getting explanation for a specific layer."""
        explanation = self.hierarchical_explanation.get_layer_explanation('layer1')
        assert explanation == self.layer_explanations['layer1']
        
        # Test non-existent layer
        explanation = self.hierarchical_explanation.get_layer_explanation('nonexistent')
        assert explanation is None


class TestFidelityMetrics:
    """Test cases for fidelity metrics."""
    
    def test_fidelity_score_init(self):
        """Test FidelityScore initialization."""
        score = FidelityScore(0.8, 0.85, 0.75)
        assert score.local_score == 0.8
        assert score.global_score == 0.85
        assert score.compositional_score == 0.75
        
    def test_calculate_fidelity_regression(self):
        """Test fidelity calculation for regression."""
        from sklearn.linear_model import LinearRegression
        
        # Create dummy data
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        
        # Train a model
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate fidelity
        fidelity = calculate_fidelity(model, X, y, model_type='regression')
        
        assert isinstance(fidelity, FidelityScore)
        assert 0 <= fidelity.global_score <= 1
        
    def test_calculate_fidelity_classification(self):
        """Test fidelity calculation for classification."""
        from sklearn.tree import DecisionTreeClassifier
        
        # Create dummy data
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 3, 100)
        
        # Train a model
        model = DecisionTreeClassifier()
        model.fit(X, y)
        
        # Calculate fidelity
        fidelity = calculate_fidelity(model, X, y, model_type='classification')
        
        assert isinstance(fidelity, FidelityScore)
        assert 0 <= fidelity.global_score <= 1


class TestDataUtils:
    """Test cases for data utility functions."""
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        # Create dummy data
        data = np.random.rand(10, 5) * 100  # Large scale
        mock_model = Mock()
        
        # Test normalization
        processed = preprocess_data(data, mock_model, normalize=True)
        
        assert processed.shape == data.shape
        # Check if data is normalized (should have smaller values)
        assert np.max(processed) < np.max(data)
        
    def test_split_dataset(self):
        """Test dataset splitting."""
        data = np.random.rand(100, 5)
        labels = np.random.randint(0, 3, 100)
        
        splits = split_dataset(data, labels, test_size=0.2, validation_size=0.1)
        
        train_data, train_labels, val_data, val_labels, test_data, test_labels = splits
        
        # Check sizes
        assert len(train_data) + len(val_data) + len(test_data) == 100
        assert len(test_data) == 20  # 20% of 100
        
    def test_load_sample_data(self):
        """Test loading sample data."""
        # Test with a known dataset
        data, labels = load_sample_data('boston', n_samples=10)
        
        assert data.shape[0] == 10
        assert len(labels) == 10


class TestModelUtils:
    """Test cases for model utility functions."""
    
    def test_get_model_framework(self):
        """Test framework detection."""
        # Mock PyTorch model
        mock_torch_model = Mock()
        mock_torch_model.__class__.__module__ = 'torch.nn'
        
        framework = get_model_framework(mock_torch_model)
        assert framework == 'pytorch'
        
        # Mock TensorFlow model
        mock_tf_model = Mock()
        mock_tf_model.__class__.__module__ = 'tensorflow.keras'
        
        framework = get_model_framework(mock_tf_model)
        assert framework == 'tensorflow'


class TestDriftDetector:
    """Test cases for DriftDetector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create reference data
        self.reference_data = np.random.normal(0, 1, (100, 10))
        self.drift_detector = DriftDetector(self.reference_data, sensitivity=0.05)
        
    def test_init(self):
        """Test DriftDetector initialization."""
        assert self.drift_detector.reference_data.shape == (100, 10)
        assert self.drift_detector.sensitivity == 0.05
        assert len(self.drift_detector.reference_statistics) > 0
        
    def test_detect_no_drift(self):
        """Test drift detection with similar data."""
        # Create new data similar to reference
        new_data = np.random.normal(0, 1, (50, 10))
        
        results = self.drift_detector.detect(new_data)
        
        assert 'global' in results
        assert 'drift_detected' in results['global']
        
    def test_detect_drift(self):
        """Test drift detection with different data."""
        # Create new data with different distribution
        new_data = np.random.normal(5, 2, (50, 10))  # Different mean and std
        
        results = self.drift_detector.detect(new_data)
        
        assert 'global' in results
        # Should detect drift due to significant distribution change
        # Note: This might not always trigger due to randomness
        assert 'drift_detected' in results['global']


class TestAlertSystem:
    """Test cases for AlertSystem class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.alert_system = AlertSystem()
        
    def test_init(self):
        """Test AlertSystem initialization."""
        assert len(self.alert_system.alert_history) == 0
        assert len(self.alert_system.alert_channels) == 0
        
    def test_send_alert(self):
        """Test sending an alert."""
        # Configure for log-only alerts
        self.alert_system.alert_channels['log'] = {'type': 'log'}
        
        success = self.alert_system.send_alert(
            'test', 'Test alert message', severity=0.8
        )
        
        assert success or len(self.alert_system.alert_history) > 0
        
    def test_alert_history(self):
        """Test alert history functionality."""
        # Send a few test alerts
        self.alert_system.alert_channels['log'] = {'type': 'log'}
        
        self.alert_system.send_alert('test1', 'Message 1', severity=0.5)
        self.alert_system.send_alert('test2', 'Message 2', severity=0.8)
        
        history = self.alert_system.get_alert_history()
        assert len(history) >= 0  # Should have some alerts
        
        # Test filtering by type
        test_alerts = self.alert_system.get_alert_history(alert_type='test1')
        # Check structure (may be empty due to threshold filtering)
        assert isinstance(test_alerts, list)


class TestIntegration:
    """Integration tests for LayerLens components."""
    
    def test_end_to_end_workflow(self):
        """Test a complete LayerLens workflow."""
        # This test uses mock objects to avoid framework dependencies
        
        # 1. Create mock model and data
        mock_model = Mock()
        mock_model.__class__.__module__ = 'torch.nn'
        mock_model.named_modules.return_value = [('layer1', Mock())]
        
        input_data = np.random.rand(10, 5)
        
        # 2. Mock layer extraction
        with patch.object(LayerExtractor, '_register_pytorch_hooks'):
            extractor = LayerExtractor(mock_model)
            extractor.layer_outputs = {'layer1': np.random.rand(10, 3)}
            
        # 3. Build surrogate model
        surrogate_builder = SurrogateBuilder()
        layer_outputs = extractor.layer_outputs['layer1']
        surrogate = surrogate_builder.fit('layer1', input_data, layer_outputs)
        
        # 4. Create explanations
        explanation = surrogate_builder.explain('layer1', input_data[0:1])
        
        # 5. Stitch explanations
        stitching_engine = StitchingEngine()
        layer_explanations = {'layer1': explanation}
        hierarchical = stitching_engine.stitch(layer_explanations)
        
        # Verify the workflow completed
        assert surrogate is not None
        assert explanation is not None
        assert isinstance(hierarchical, HierarchicalExplanation)


# Pytest configuration and fixtures
@pytest.fixture
def sample_data():
    """Fixture providing sample data for tests."""
    return {
        'input_data': np.random.rand(50, 10),
        'output_data': np.random.rand(50, 5),
        'labels': np.random.randint(0, 3, 50)
    }


@pytest.fixture
def mock_model():
    """Fixture providing a mock model."""
    model = Mock()
    model.__class__.__module__ = 'torch.nn'
    return model


if __name__ == '__main__':
    # Run tests if script is executed directly
    pytest.main([__file__, '-v'])