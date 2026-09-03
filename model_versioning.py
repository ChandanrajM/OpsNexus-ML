"""
Model Versioning System for OpsNexus-ML
Provides simple timestamp-based model versioning with metadata tracking
"""

import os
import json
import hashlib
import pickle
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelVersionManager:
    """
    Manages model versions with timestamp-based versioning and metadata
    """

    def __init__(self, model_registry_path: str = "/home/chandanraj-m/OpsNexus-ML/models/registry"):
        """
        Initialize the model version manager

        Args:
            model_registry_path: Path to store model registry and versions
        """
        self.model_registry_path = model_registry_path
        self.versions_path = os.path.join(model_registry_path, "versions")
        self.metadata_path = os.path.join(model_registry_path, "metadata.json")

        # Create directories if they don't exist
        os.makedirs(self.versions_path, exist_ok=True)
        os.makedirs(model_registry_path, exist_ok=True)

        # Load or create metadata
        self.metadata = self._load_metadata()

        logger.info(f"ModelVersionManager initialized at {model_registry_path}")

    def _load_metadata(self) -> Dict[str, Any]:
        """Load model metadata from JSON file"""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}. Creating new metadata.")
                return self._create_initial_metadata()
        else:
            return self._create_initial_metadata()

    def _create_initial_metadata(self) -> Dict[str, Any]:
        """Create initial metadata structure"""
        return {
            "model_types": {
                "isolation_forest": {
                    "latest_version": None,
                    "versions": []
                },
                "cpu_predictor": {
                    "latest_version": None,
                    "versions": []
                }
            },
            "created_at": datetime.utcnow().isoformat() + 'Z',
            "last_updated": datetime.utcnow().isoformat() + 'Z'
        }

    def _save_metadata(self):
        """Save metadata to JSON file"""
        self.metadata["last_updated"] = datetime.utcnow().isoformat() + 'Z'
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _generate_version_id(self, model_type: str) -> str:
        """Generate a timestamp-based version ID with microseconds for uniqueness"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        return f"{model_type}_{timestamp}"

    def _calculate_model_hash(self, model_path: str) -> str:
        """Calculate SHA256 hash of model file"""
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def register_model(self,
                      model_type: str,
                      model_path: str,
                      metadata: Optional[Dict[str, Any]] = None,
                      description: str = "") -> str:
        """
        Register a new model version

        Args:
            model_type: Type of model ('isolation_forest' or 'cpu_predictor')
            model_path: Path to the model file
            metadata: Additional metadata to store with the version
            description: Human-readable description of this version

        Returns:
            Version ID of the registered model
        """
        if model_type not in self.metadata["model_types"]:
            raise ValueError(f"Unsupported model type: {model_type}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Generate version ID
        version_id = self._generate_version_id(model_type)
        version_dir = os.path.join(self.versions_path, version_id)
        os.makedirs(version_dir, exist_ok=True)

        # Copy model file to version directory
        model_filename = os.path.basename(model_path)
        version_model_path = os.path.join(version_dir, model_filename)

        # Read and write to create a copy (or use shutil.copy2)
        with open(model_path, 'rb') as src, open(version_model_path, 'wb') as dst:
            dst.write(src.read())

        # Calculate model hash
        model_hash = self._calculate_model_hash(version_model_path)

        # Prepare version metadata
        version_metadata = {
            "version_id": version_id,
            "model_type": model_type,
            "model_path": version_model_path,
            "original_path": model_path,
            "file_size": os.path.getsize(version_model_path),
            "model_hash": model_hash,
            "created_at": datetime.utcnow().isoformat() + 'Z',
            "description": description,
            "metadata": metadata or {}
        }

        # Add to model type's versions list
        self.metadata["model_types"][model_type]["versions"].append(version_metadata)
        self.metadata["model_types"][model_type]["latest_version"] = version_id

        # Save updated metadata
        self._save_metadata()

        logger.info(f"Registered {model_type} model version: {version_id}")
        logger.info(f"Model hash: {model_hash}")
        logger.info(f"Stored at: {version_model_path}")

        return version_id

    def get_model_version(self, model_type: str, version_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a specific model version

        Args:
            model_type: Type of model
            version_id: Specific version ID (if None, returns latest)

        Returns:
            Version metadata dictionary
        """
        if model_type not in self.metadata["model_types"]:
            raise ValueError(f"Unsupported model type: {model_type}")

        if version_id is None:
            version_id = self.metadata["model_types"][model_type]["latest_version"]
            if version_id is None:
                raise ValueError(f"No versions found for model type: {model_type}")

        # Find the version
        for version in self.metadata["model_types"][model_type]["versions"]:
            if version["version_id"] == version_id:
                return version.copy()

        raise ValueError(f"Version {version_id} not found for model type: {model_type}")

    def list_model_versions(self, model_type: str) -> List[Dict[str, Any]]:
        """
        List all versions of a model type

        Args:
            model_type: Type of model

        Returns:
            List of version metadata dictionaries (sorted by creation date, newest first)
        """
        if model_type not in self.metadata["model_types"]:
            raise ValueError(f"Unsupported model type: {model_type}")

        versions = self.metadata["model_types"][model_type]["versions"].copy()
        # Sort by creation date, newest first
        versions.sort(key=lambda x: x["created_at"], reverse=True)
        return versions

    def get_latest_model_path(self, model_type: str) -> str:
        """
        Get the file path of the latest model version

        Args:
            model_type: Type of model

        Returns:
            Path to the latest model file
        """
        latest_version = self.get_model_version(model_type)
        return latest_version["model_path"]

    def load_model(self, model_type: str, version_id: Optional[str] = None) -> Any:
        """
        Load a model version using pickle

        Args:
            model_type: Type of model
            version_id: Specific version ID (if None, loads latest)

        Returns:
            Loaded model object
        """
        version_info = self.get_model_version(model_type, version_id)
        model_path = version_info["model_path"]

        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Loaded {model_type} model version {version_info['version_id']} from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            raise

    def promote_to_champion(self, model_type: str, version_id: str):
        """
        Promote a specific version to be the champion (latest) version

        Args:
            model_type: Type of model
            version_id: Version ID to promote
        """
        if model_type not in self.metadata["model_types"]:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Verify version exists
        self.get_model_version(model_type, version_id)

        # Update latest version
        self.metadata["model_types"][model_type]["latest_version"] = version_id
        self._save_metadata()

        logger.info(f"Promoted {model_type} version {version_id} to champion")

    def compare_versions(self, model_type: str, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """
        Compare two model versions

        Args:
            model_type: Type of model
            version_id1: First version ID
            version_id2: Second version ID

        Returns:
            Comparison results
        """
        v1 = self.get_model_version(model_type, version_id1)
        v2 = self.get_model_version(model_type, version_id2)

        comparison = {
            "model_type": model_type,
            "version_1": {
                "id": version_id1,
                "created_at": v1["created_at"],
                "file_size": v1["file_size"],
                "hash": v1["model_hash"]
            },
            "version_2": {
                "id": version_id2,
                "created_at": v2["created_at"],
                "file_size": v2["file_size"],
                "hash": v2["model_hash"]
            },
            "same_file": v1["model_hash"] == v2["model_hash"],
            "size_difference": v2["file_size"] - v1["file_size"],
            "time_difference_seconds": (
                datetime.fromisoformat(v2["created_at"].replace('Z', '+00:00')) -
                datetime.fromisoformat(v1["created_at"].replace('Z', '+00:00'))
            ).total_seconds()
        }

        return comparison

    def cleanup_old_versions(self, model_type: str, keep_count: int = 5):
        """
        Clean up old model versions, keeping only the most recent N versions
        Also preserves the champion (promoted) version if it exists.

        Args:
            model_type: Type of model
            keep_count: Number of versions to keep (default: 5)
        """
        if model_type not in self.metadata["model_types"]:
            raise ValueError(f"Unsupported model type: {model_type}")

        versions = self.list_model_versions(model_type)
        if len(versions) <= keep_count:
            logger.info(f"No cleanup needed for {model_type}: {len(versions)} versions <= {keep_count} to keep")
            return

        # Determine which versions to keep
        # Always keep the champion version
        champion_version_id = self.metadata["model_types"][model_type]["latest_version"]

        # Build set of version IDs to keep
        keep_ids = set()
        if champion_version_id:
            keep_ids.add(champion_version_id)

        # Add newest versions until we reach keep_count
        for version in versions:
            if len(keep_ids) >= keep_count:
                break
            keep_ids.add(version["version_id"])

        # Versions to delete are those not in keep_ids
        to_delete = [v for v in versions if v["version_id"] not in keep_ids]

        deleted_count = 0
        for version in to_delete:
            try:
                # Remove version directory
                import shutil
                version_dir = os.path.dirname(version["model_path"])
                if os.path.exists(version_dir):
                    shutil.rmtree(version_dir)
                    logger.info(f"Deleted version directory: {version_dir}")

                # Remove from metadata list
                self.metadata["model_types"][model_type]["versions"] = [
                    v for v in self.metadata["model_types"][model_type]["versions"]
                    if v["version_id"] != version["version_id"]
                ]

                deleted_count += 1
            except Exception as e:
                logger.error(f"Failed to delete version {version['version_id']}: {e}")

        # Save updated metadata
        if deleted_count > 0:
            self._save_metadata()
            logger.info(f"Cleaned up {deleted_count} old versions for {model_type}")

def create_and_register_model(model_type: str, model_obj: Any,
                            model_path: str,
                            metadata: Optional[Dict[str, Any]] = None,
                            description: str = "") -> str:
    """
    Convenience function to save and register a model

    Args:
        model_type: Type of model ('isolation_forest' or 'cpu_predictor')
        model_obj: The model object to save
        model_path: Path where to save the model initially
        metadata: Additional metadata for the version
        description: Description of this model version

    Returns:
        Version ID of the registered model
    """
    # Save the model first
    if hasattr(model_obj, 'save_model'):
        model_obj.save_model(model_path)
    else:
        # Generic pickle save
        with open(model_path, 'wb') as f:
            pickle.dump(model_obj, f)

    # Register with version manager
    version_manager = ModelVersionManager()
    version_id = version_manager.register_model(
        model_type=model_type,
        model_path=model_path,
        metadata=metadata,
        description=description
    )

    return version_id

if __name__ == "__main__":
    # Example usage
    print("Model Versioning System Example")
    print("=" * 40)

    # Initialize manager
    vm = ModelVersionManager()

    # Show current state
    print("\nCurrent model registry state:")
    for model_type in vm.metadata["model_types"]:
        info = vm.metadata["model_types"][model_type]
        print(f"  {model_type}:")
        print(f"    Latest version: {info['latest_version']}")
        print(f"    Total versions: {len(info['versions'])}")

    print("\nTo use this system:")
    print("  1. Train your model as usual")
    print("  2. Save it to a temporary path")
    print("  3. Register it with: vm.register_model('model_type', '/path/to/model.pkl')")
    print("  4. Load latest with: model = vm.load_model('model_type')")
    print("  5. Load specific version: model = vm.load_model('model_type', 'version_id')")