#!/usr/bin/env python3
"""
Test script to verify the complete hierarchical model loading workflow.
"""

import sys
import os
import torch

# Add src to path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from kamil_models.models import RegressionHierarchicalFinetuner

def test_hierarchical_model_loading():
    """Test the complete hierarchical model loading workflow."""

    # Test checkpoint paths
    local_checkpoint = "/home/mg873uh/Projects_kb/baseline-codebase/src/26_8.ckpt"  # Adjust this path as needed
    global_checkpoint = "/home/mg873uh/Projects_kb/baseline-codebase/_models/fomo/unet_b.ckpt"

    # Check if checkpoints exist
    if not os.path.exists(global_checkpoint):
        print(f"❌ Global checkpoint file not found: {global_checkpoint}")
        return False

    if not os.path.exists(local_checkpoint):
        print(f"⚠️  Local checkpoint file not found: {local_checkpoint}")
        print("   Will test only global loading...")
        local_checkpoint = None

    print(f"🔍 Testing complete hierarchical model loading")
    print(f"   Local checkpoint: {local_checkpoint}")
    print(f"   Global checkpoint: {global_checkpoint}")

    # Create global config
    global_config = {
        "model_name": "unet_b",
        "num_modalities": 1,
        "num_classes": 1
    }

    try:
        if local_checkpoint:
            # Test the class method load_from_pretrained
            model = RegressionHierarchicalFinetuner.load_from_pretrained(
                local_checkpoint=local_checkpoint,
                in_channels=1,
                target_mean=0.0,
                target_std=1.0,
                global_config=global_config,
                global_checkpoint=global_checkpoint,
                img_size=(96, 96, 96),
                feature_size=24,
                learning_rate=1e-3
            )
            print("✅ Complete hierarchical model loaded successfully via load_from_pretrained!")
        else:
            # Test individual loading
            model = RegressionHierarchicalFinetuner(
                in_channels=1,
                target_mean=0.0,
                target_std=1.0,
                global_config=global_config,
                img_size=(96, 96, 96),
                feature_size=24,
                learning_rate=1e-3
            )
            model.load_from_global_pretrained(global_checkpoint)
            print("✅ Global encoder loaded successfully!")

        return True

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing complete hierarchical model loading workflow...")
    success = test_hierarchical_model_loading()

    if success:
        print("\n🎉 All tests completed successfully!")
    else:
        print("\n💥 Tests failed!")
        sys.exit(1)
