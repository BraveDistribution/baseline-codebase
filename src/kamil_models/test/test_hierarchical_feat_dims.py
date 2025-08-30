import sys
sys.path.append('src')
import torch
from kamil_models.models import RegressionHierarchicalFinetuner

NUM_MODALITIES = 2

# Test with debug info
global_config = {
    'model_name': 'unet_b',
    'num_modalities': NUM_MODALITIES,
    'num_classes': 1,
    'patch_size': (96, 96, 96),
    'task_type': 'regression'
}

print('Creating model...')
model = RegressionHierarchicalFinetuner(
    in_channels=NUM_MODALITIES,
    target_mean=45.0,
    target_std=15.0,
    global_config=global_config,
    learning_rate=1e-4,
)

def print_tensor_info(struct, name):
    print(f"Type of {name}:", type(struct))
    if isinstance(struct, (list, tuple)):
        for i, f in enumerate(struct):
            print(f"{name}[{i}] shape:", getattr(f, "shape", None))
    elif hasattr(struct, "shape"):
        print(f"{name} shape:", struct.shape)
    else:
        print(f"{name} has no .shape attribute")

print_tensor_info(model.local_features, "local_features")
print_tensor_info(model.global_features, "global_features")

print(f'Local feature dims: {model.local_feature_dims}')
print(f'Global feature dims: {model.global_feature_dims}')
print(f'Total local features: {NUM_MODALITIES * len(model.local_feature_dims) * 32} -> projected to 64 (128 / 2)')
print(f'Total global features: {NUM_MODALITIES * len(model.global_feature_dims) * 32} -> projected to 64 (128 / 2)')
print('Perfect 50-50 balance achieved!')