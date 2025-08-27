# Test training step
import sys
sys.path.append('src')
import torch

try:
    from kamil_models.models import RegressionHierarchicalFinetuner

    NUM_MODALITIES = 2

    # Create model
    global_config = {
        'model_name': 'unet_b',
        'num_modalities': NUM_MODALITIES,
        'num_classes': 1,
        'patch_size': (96, 96, 96),
        'task_type': 'regression'
    }

    model = RegressionHierarchicalFinetuner(
        in_channels=NUM_MODALITIES,
        target_mean=45.0,
        target_std=15.0,
        global_config=global_config,
        learning_rate=1e-4,
    )

    print('✓ Model created')

    # Test training step simulation
    batch = {
        'local': torch.randn(2, NUM_MODALITIES, 96, 96, 96),
        'global': torch.randn(2, NUM_MODALITIES, 96, 96, 96),
        'label': torch.tensor([45.5, 67.2])  # Age targets
    }

    print('✓ Testing loss computation...')
    targets = batch['label'].float().view(-1)
    preds = model(batch)

    # Test normalization
    targets_normalized = model._normalize(targets)
    loss = model.compute_loss(preds, targets_normalized)

    print(f'✓ Loss computation successful!')
    print(f'  Raw targets: {targets.tolist()}')
    print(f'  Normalized targets: {targets_normalized.tolist()}')
    print(f'  Predictions: {preds.tolist()}')
    print(f'  Loss: {loss.item():.4f}')

    # Test unnormalization
    unnorm_preds = model._unnormalize(preds)
    print(f'  Unnormalized predictions: {unnorm_preds.tolist()}')

    print('\n✓ All tests passed! Model is ready for training.')

except Exception as e:
    print(f'✗ Error: {e}')
    import traceback
    traceback.print_exc()