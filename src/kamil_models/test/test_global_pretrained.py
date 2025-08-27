#!/usr/bin/env python3
"""
Test script to verify the updated load_from_global_pretrained() method works correctly.
"""

import sys
import os
import torch

# Add src to path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from kamil_models.models import RegressionHierarchicalFinetuner

def test_load_from_global_pretrained():
    """Test the updated load_from_global_pretrained method."""

    # Test checkpoint path
    global_checkpoint = "/home/mg873uh/Projects_kb/baseline-codebase/_models/fomo/unet_b.ckpt"

    # Check if checkpoint exists
    if not os.path.exists(global_checkpoint):
        print(f"❌ Checkpoint file not found: {global_checkpoint}")
        return False

    print(f"🔍 Testing load_from_global_pretrained with: {global_checkpoint}")

    # Create a simple global config for testing
    global_config = {
        "model_name": "unet_b",  # This is available in models.networks
        "num_modalities": 1,
        "num_classes": 1
    }

    try:
        # Create a minimal hierarchical finetuner instance
        model = RegressionHierarchicalFinetuner(
            in_channels=1,
            target_mean=0.0,
            target_std=1.0,
            global_config=global_config,
            img_size=(96, 96, 96),
            feature_size=24,
            learning_rate=1e-3
        )

        print("✅ Model created successfully")

        # Test the load_from_global_pretrained method
        model.load_from_global_pretrained(global_checkpoint)

        print("✅ load_from_global_pretrained completed without errors!")
        return True

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing updated load_from_global_pretrained method...")
    success = test_load_from_global_pretrained()

    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n💥 Test failed!")
        sys.exit(1)
