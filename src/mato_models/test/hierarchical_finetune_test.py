# Test model with fixed feature dimensions
import sys
sys.path.append('src')
import torch

try:
    from finetune_hierarchical import HierarchicalConfig, create_hierarchical_model, create_data_module

    # Create mock args for regression task
    class MockArgs:
        task_id = 3
        model_type = 'regression'
        global_encoder = 'unet_b'
        patch_size = 96
        feature_size = 24
        freeze_global_encoder = True
        lora_r = 128
        lora_alpha = 16
        target_mean = 45.0
        target_std = 15.0
        predict_uncertainty = False
        mixup_alpha = 0.4
        mixup_prob = 0.5
        learning_rate = 1e-4
        weight_decay = 0.01
        dropout_rate = 0.1
        batch_size = 2  # Small for testing
        epochs = 5      # Small for testing
        train_batches_per_epoch = 10  # Small for testing
        data_dir = './data/preprocessed'
        local_data_dir = '/home/mg873uh/Projects_kb/data/finetuning_preproc/'
        global_data_dir = '/home/mg873uh/Projects_kb/data/finetuning_preproc/Unified_2.6667mm_float16'
        augmentation_preset = 'none'  # No augmentation for testing
        save_dir = './data/models'
        experiment_name = 'test_experiment'
        pretrained_checkpoint = None
        continue_training = False
        precision = 'bf16-mixed'
        num_devices = 1
        num_workers = 4  # Reduced for testing
        accelerator = 'gpu'
        split_method = 'simple_train_val_split'
        split_param = '0.8'  # Small validation set
        split_idx = 0

    config = HierarchicalConfig(MockArgs())
    print('✓ Configuration created')

    # Test model creation
    print('\n🏗️  Testing model creation...')
    model = create_hierarchical_model(config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'✓ Model created successfully!')
    print(f'  Total parameters: {total_params:,}')
    print(f'  Trainable parameters: {trainable_params:,}')
    print(f'  Efficiency: {100*trainable_params/total_params:.1f}% trainable')

    # Test forward pass
    print('\n🧪 Testing forward pass...')
    with torch.no_grad():
        batch = {
            'local': torch.randn(1, 1, 96, 96, 96),   # 1 channel for SwinUNETR
            'global': torch.randn(1, 2, 96, 96, 96),  # 2 channels for unet_b (T1+T2)
            'label': torch.tensor([50.0])
        }

        output = model(batch)
        print(f'✓ Forward pass successful!')
        print(f'  Input shapes: local={list(batch["local"].shape)}, global={list(batch["global"].shape)}')
        print(f'  Output shape: {list(output.shape)}')
        print(f'  Output value: {output.item():.4f}')

        # Test loss computation
        targets_norm = model._normalize(batch['label'])
        loss = model.compute_loss(output, targets_norm)
        print(f'  Loss: {loss.item():.4f}')

    # Test training step
    print('\n🏃 Testing training step...')
    model.train()
    loss = model.training_step(batch, 0)
    print(f'✓ Training step successful! Loss: {loss.item():.4f}')

    # Test validation step
    print('\n🔍 Testing validation step...')
    model.eval()
    val_output = model.validation_step(batch, 0)
    print(f'✓ Validation step successful!')

    # Test batch processing
    print('\n🔄 Testing batch processing...')
    batch_2 = {
        'local': torch.randn(2, 1, 96, 96, 96),
        'global': torch.randn(2, 2, 96, 96, 96),
        'label': torch.tensor([45.0, 55.0])
    }

    with torch.no_grad():
        output_2 = model(batch_2)
        print(f'✓ Batch processing successful!')
        print(f'  Batch size 2 output: {output_2.shape} = {output_2.flatten().tolist()}')

    print('\n✅ ALL TESTS PASSED!')
    print('\n🎯 RegressionHierarchicalFinetuner is ready for production!')
    print('\n📊 Architecture Summary:')
    print('   • Local Encoder: SwinUNETR (1 channel → multi-scale features)')
    print('   • Global Encoder: UNet-B (2 channels → global context)')
    print('   • Feature Fusion: 5 local scales + 1 global → regression')
    print('   • Training: Parameter-efficient LoRA + MixUp augmentation')
    print(f'   • Efficiency: {100*trainable_params/total_params:.1f}% trainable parameters')

except Exception as e:
    print(f'✗ Error: {e}')
    import traceback
    traceback.print_exc()