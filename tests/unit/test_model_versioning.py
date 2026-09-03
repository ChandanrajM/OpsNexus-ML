"""
Unit tests for Model Versioning System
"""
import os
import sys
import tempfile
import pickle
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from model_versioning import ModelVersionManager, create_and_register_model


# Define dummy classes at module level for pickling
class DummyModel:
    def __init__(self):
        self.value = 42
        self.predictions = [1, 2, 3]


class ModelWithSave:
    def __init__(self):
        self.trained = True

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)


class GenericModel:
    def __init__(self):
        self.data = [1, 2, 3]


class TestModelVersionManager:
    """Test cases for ModelVersionManager"""

    def setup_method(self):
        """Set up test environment with temporary directory"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.registry_path = os.path.join(self.temp_dir.name, 'registry')
        self.vm = ModelVersionManager(model_registry_path=self.registry_path)

    def teardown_method(self):
        """Clean up test environment"""
        self.temp_dir.cleanup()

    def test_initialization(self):
        """Test that manager initializes correctly"""
        assert self.vm.model_registry_path == self.registry_path
        assert os.path.exists(self.registry_path)
        assert os.path.exists(self.vm.versions_path)
        # Metadata file is created on first registration, not initialization
        assert 'isolation_forest' in self.vm.metadata['model_types']
        assert 'cpu_predictor' in self.vm.metadata['model_types']

    def test_generate_version_id(self):
        """Test version ID generation"""
        version_id = self.vm._generate_version_id('isolation_forest')
        assert version_id.startswith('isolation_forest_')
        assert len(version_id) > len('isolation_forest_')

    def test_register_model(self):
        """Test model registration"""
        model = DummyModel()
        model_path = os.path.join(self.temp_dir.name, 'test_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        version_id = self.vm.register_model(
            model_type='isolation_forest',
            model_path=model_path,
            metadata={'roc_auc': 0.85},
            description='Test model'
        )

        assert version_id.startswith('isolation_forest_')
        assert self.vm.metadata['model_types']['isolation_forest']['latest_version'] == version_id
        assert len(self.vm.metadata['model_types']['isolation_forest']['versions']) == 1
        # Metadata file should now exist
        assert os.path.exists(self.vm.metadata_path)

    def test_register_invalid_model_type(self):
        """Test registration with invalid model type"""
        model_path = os.path.join(self.temp_dir.name, 'test_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({'test': 'data'}, f)

        with pytest.raises(ValueError, match="Unsupported model type"):
            self.vm.register_model('invalid_type', model_path)

    def test_register_nonexistent_model(self):
        """Test registration with nonexistent model file"""
        with pytest.raises(FileNotFoundError):
            self.vm.register_model('isolation_forest', '/nonexistent/path/model.pkl')

    def test_get_model_version(self):
        """Test getting model version info"""
        # First register a model
        model_path = os.path.join(self.temp_dir.name, 'test_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({'test': 'data'}, f)

        version_id = self.vm.register_model('isolation_forest', model_path)
        version_info = self.vm.get_model_version('isolation_forest', version_id)

        assert version_info['version_id'] == version_id
        assert version_info['model_type'] == 'isolation_forest'
        assert 'model_hash' in version_info
        assert 'created_at' in version_info

    def test_get_latest_model_version(self):
        """Test getting latest model version"""
        model_path = os.path.join(self.temp_dir.name, 'test_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({'test': 'data'}, f)

        version_id = self.vm.register_model('isolation_forest', model_path)
        latest = self.vm.get_model_version('isolation_forest')

        assert latest['version_id'] == version_id

    def test_list_model_versions(self):
        """Test listing model versions"""
        model_path = os.path.join(self.temp_dir.name, 'test_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({'test': 'data'}, f)

        # Register multiple versions with small delay for unique timestamps
        for i in range(3):
            self.vm.register_model('isolation_forest', model_path,
                                   metadata={'iteration': i},
                                   description=f'Version {i}')
            time.sleep(0.01)  # Ensure unique timestamps

        versions = self.vm.list_model_versions('isolation_forest')
        assert len(versions) == 3
        # Should be sorted newest first
        assert versions[0]['metadata']['iteration'] == 2
        assert versions[1]['metadata']['iteration'] == 1
        assert versions[2]['metadata']['iteration'] == 0

    def test_get_latest_model_path(self):
        """Test getting latest model file path"""
        model_path = os.path.join(self.temp_dir.name, 'test_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({'test': 'data'}, f)

        version_id = self.vm.register_model('isolation_forest', model_path)
        latest_path = self.vm.get_latest_model_path('isolation_forest')

        assert os.path.exists(latest_path)
        assert version_id in latest_path

    def test_load_model(self):
        """Test loading a model"""
        model = DummyModel()
        model_path = os.path.join(self.temp_dir.name, 'test_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        self.vm.register_model('isolation_forest', model_path)
        loaded = self.vm.load_model('isolation_forest')

        assert isinstance(loaded, DummyModel)
        assert loaded.value == 42
        assert loaded.predictions == [1, 2, 3]

    def test_promote_to_champion(self):
        """Test promoting a version to champion"""
        model_path = os.path.join(self.temp_dir.name, 'test_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({'test': 'data'}, f)

        v1 = self.vm.register_model('isolation_forest', model_path, description='V1')
        time.sleep(0.01)
        v2 = self.vm.register_model('isolation_forest', model_path, description='V2')

        # Latest should be v2
        assert self.vm.metadata['model_types']['isolation_forest']['latest_version'] == v2

        # Promote v1 to champion
        self.vm.promote_to_champion('isolation_forest', v1)
        assert self.vm.metadata['model_types']['isolation_forest']['latest_version'] == v1

    def test_compare_versions(self):
        """Test comparing two versions"""
        model_path = os.path.join(self.temp_dir.name, 'test_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({'test': 'data'}, f)

        v1 = self.vm.register_model('isolation_forest', model_path)
        time.sleep(0.01)
        v2 = self.vm.register_model('isolation_forest', model_path)

        comparison = self.vm.compare_versions('isolation_forest', v1, v2)

        assert comparison['model_type'] == 'isolation_forest'
        assert comparison['version_1']['id'] == v1
        assert comparison['version_2']['id'] == v2
        assert 'same_file' in comparison
        assert 'size_difference' in comparison
        assert 'time_difference_seconds' in comparison

    def test_cleanup_old_versions(self):
        """Test cleaning up old versions"""
        model_path = os.path.join(self.temp_dir.name, 'test_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({'test': 'data'}, f)

        # Register 7 versions with delays for unique timestamps
        version_ids = []
        for i in range(7):
            vid = self.vm.register_model('isolation_forest', model_path,
                                         metadata={'iteration': i},
                                         description=f'Version {i}')
            version_ids.append(vid)
            time.sleep(0.01)

        assert len(self.vm.list_model_versions('isolation_forest')) == 7

        # Cleanup keeping 5
        self.vm.cleanup_old_versions('isolation_forest', keep_count=5)
        versions = self.vm.list_model_versions('isolation_forest')
        assert len(versions) == 5
        # Should keep the 5 newest (iterations 2-6)
        assert versions[0]['metadata']['iteration'] == 6
        assert versions[-1]['metadata']['iteration'] == 2

    def test_cleanup_preserves_champion(self):
        """Test that cleanup doesn't delete the champion version"""
        model_path = os.path.join(self.temp_dir.name, 'test_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({'test': 'data'}, f)

        v1 = self.vm.register_model('isolation_forest', model_path, description='V1')
        time.sleep(0.01)
        v2 = self.vm.register_model('isolation_forest', model_path, description='V2')
        time.sleep(0.01)
        v3 = self.vm.register_model('isolation_forest', model_path, description='V3')

        # Promote v1 to champion (oldest)
        self.vm.promote_to_champion('isolation_forest', v1)

        # Cleanup keeping 2 - should keep v1 (champion) and v3 (newest)
        self.vm.cleanup_old_versions('isolation_forest', keep_count=2)
        versions = self.vm.list_model_versions('isolation_forest')
        assert len(versions) == 2
        version_ids = [v['version_id'] for v in versions]
        assert v1 in version_ids
        assert v3 in version_ids


class TestCreateAndRegisterModel:
    """Test cases for create_and_register_model convenience function"""

    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.TemporaryDirectory()

    def teardown_method(self):
        """Clean up test environment"""
        self.temp_dir.cleanup()

    def test_create_and_register_with_save_method(self):
        """Test with model that has save_model method"""
        model = ModelWithSave()
        model_path = os.path.join(self.temp_dir.name, 'model.pkl')

        version_id = create_and_register_model(
            model_type='cpu_predictor',
            model_obj=model,
            model_path=model_path,
            metadata={'mae': 1.5},
            description='Test with save_method'
        )

        assert version_id.startswith('cpu_predictor_')

    def test_create_and_register_generic(self):
        """Test with generic model (no save_model method)"""
        model = GenericModel()
        model_path = os.path.join(self.temp_dir.name, 'model.pkl')

        version_id = create_and_register_model(
            model_type='isolation_forest',
            model_obj=model,
            model_path=model_path,
            metadata={'f1': 0.8},
            description='Test generic'
        )

        assert version_id.startswith('isolation_forest_')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])