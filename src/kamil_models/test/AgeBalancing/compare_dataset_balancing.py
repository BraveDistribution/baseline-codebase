#!/usr/bin/env python3
"""
Script to compare HierarchicalDataset and HierarchicalAgeBalancedDataset
for age regression balancing improvements.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
import pandas as pd
from scipy import stats

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.dataset import HierarchicalDataset, HierarchicalAgeBalancedDataset


def create_mock_samples(n_samples: int = 100, data_dir: str = "/tmp/mock_data") -> List[str]:
    """
    Create mock sample paths and labels for testing.
    Simulates age distribution typical in medical datasets (skewed towards older ages).
    """
    os.makedirs(data_dir, exist_ok=True)
    task_dir = os.path.join(data_dir, "Task001_FOMO1")
    os.makedirs(task_dir, exist_ok=True)

    samples = []

    # Create age distribution that mimics real medical datasets
    # Skewed towards older ages (more common in medical imaging)
    ages = []

    # Generate realistic age distribution
    for i in range(n_samples):
        # Bimodal distribution: some young adults, mostly older adults
        if np.random.random() < 0.3:  # 30% young adults
            age = np.random.normal(35, 8)  # Young adults around 35
        else:  # 70% older adults
            age = np.random.normal(65, 12)  # Older adults around 65

        # Clip to reasonable range
        age = np.clip(age, 20, 100)
        ages.append(age)

    ages = np.array(ages)

    # Create mock files
    for i, age in enumerate(ages):
        sample_name = f"FOMO1_sub_{i+1:03d}"
        sample_path = os.path.join(task_dir, sample_name)
        samples.append(sample_path)

        # Create mock .npy files (empty, just for path validation)
        np_file = sample_path + ".npy"
        txt_file = sample_path + ".txt"

        # Create minimal numpy array
        np.save(np_file, np.random.rand(1, 64, 64, 64).astype(np.float32))

        # Save age label
        np.savetxt(txt_file, [age])

    print(f"Created {len(samples)} mock samples with ages {ages.min():.1f}-{ages.max():.1f}")
    print(f"Age distribution: mean={ages.mean():.1f}, std={ages.std():.1f}")

    return samples


def analyze_dataset_distribution(dataset, title: str) -> dict:
    """Analyze the age distribution of a dataset"""
    print(f"\n{'='*50}")
    print(f"Analyzing: {title}")
    print(f"{'='*50}")

    # Extract all labels
    labels = []
    sample_weights = []

    for i in range(len(dataset)):
        try:
            item = dataset[i]
            label = item['label']
            if isinstance(label, np.ndarray):
                label = label.item()
            labels.append(label)

            # Get sample weight if available
            weight = item.get('sample_weight', 1.0)
            sample_weights.append(weight)

        except Exception as e:
            print(f"Error loading sample {i}: {e}")
            continue

    labels = np.array(labels)
    sample_weights = np.array(sample_weights)

    # Basic statistics
    stats_dict = {
        'title': title,
        'n_samples': len(labels),
        'mean_age': labels.mean(),
        'std_age': labels.std(),
        'min_age': labels.min(),
        'max_age': labels.max(),
        'median_age': np.median(labels),
        'labels': labels,
        'sample_weights': sample_weights
    }

    print(f"Number of samples: {stats_dict['n_samples']}")
    print(f"Age range: {stats_dict['min_age']:.1f} - {stats_dict['max_age']:.1f}")
    print(f"Mean age: {stats_dict['mean_age']:.1f} ± {stats_dict['std_age']:.1f}")
    print(f"Median age: {stats_dict['median_age']:.1f}")

    # Age bin analysis
    n_bins = 8
    bin_edges = np.linspace(20, 100, n_bins + 1)
    bin_indices = np.digitize(labels, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    bin_counts = np.bincount(bin_indices, minlength=n_bins)

    print(f"\nAge distribution across {n_bins} bins:")
    for i, (count, edge_low, edge_high) in enumerate(zip(bin_counts, bin_edges[:-1], bin_edges[1:])):
        percentage = (count / len(labels)) * 100
        print(f"  Bin {i+1} [{edge_low:.0f}-{edge_high:.0f}): {count:3d} samples ({percentage:5.1f}%)")

    # Calculate distribution balance metrics
    stats_dict['bin_counts'] = bin_counts
    stats_dict['bin_edges'] = bin_edges

    # Gini coefficient (measure of inequality)
    sorted_counts = np.sort(bin_counts)
    n = len(sorted_counts)
    cumsum = np.cumsum(sorted_counts)
    gini = (2 * np.sum((np.arange(1, n+1) * sorted_counts))) / (n * cumsum[-1]) - (n + 1) / n
    stats_dict['gini_coefficient'] = gini

    # Coefficient of variation
    cv = np.std(bin_counts) / np.mean(bin_counts) if np.mean(bin_counts) > 0 else float('inf')
    stats_dict['cv_bin_counts'] = cv

    # Entropy (higher = more balanced)
    probs = bin_counts / bin_counts.sum()
    probs = probs[probs > 0]  # Remove zero probabilities
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(n_bins)  # Maximum possible entropy
    normalized_entropy = entropy / max_entropy
    stats_dict['entropy'] = entropy
    stats_dict['normalized_entropy'] = normalized_entropy

    print(f"\nBalance Metrics:")
    print(f"  Gini coefficient: {gini:.3f} (0=perfect equality, 1=perfect inequality)")
    print(f"  CV of bin counts: {cv:.3f} (lower=more balanced)")
    print(f"  Normalized entropy: {normalized_entropy:.3f} (1=perfectly balanced)")

    return stats_dict


def plot_comparison(original_stats: dict, balanced_stats: dict, save_path: str = None):
    """Create comprehensive comparison plots"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Dataset Balancing Comparison: Original vs Balanced', fontsize=16, fontweight='bold')

    # 1. Age histograms
    ax1 = axes[0, 0]
    bins = np.linspace(20, 100, 21)
    ax1.hist(original_stats['labels'], bins=bins, alpha=0.7, label='Original',
             color='skyblue', edgecolor='black')
    ax1.hist(balanced_stats['labels'], bins=bins, alpha=0.7, label='Balanced',
             color='lightcoral', edgecolor='black')
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Age Distribution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Bin counts comparison
    ax2 = axes[0, 1]
    x_pos = np.arange(len(original_stats['bin_counts']))
    width = 0.35

    ax2.bar(x_pos - width/2, original_stats['bin_counts'], width,
            label='Original', color='skyblue', alpha=0.7)
    ax2.bar(x_pos + width/2, balanced_stats['bin_counts'], width,
            label='Balanced', color='lightcoral', alpha=0.7)

    ax2.set_xlabel('Age Bin')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Samples per Age Bin')
    ax2.set_xticks(x_pos)
    bin_labels = [f"{int(edge)}-{int(balanced_stats['bin_edges'][i+1])}"
                  for i, edge in enumerate(balanced_stats['bin_edges'][:-1])]
    ax2.set_xticklabels(bin_labels, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Box plots
    ax3 = axes[0, 2]
    data_to_plot = [original_stats['labels'], balanced_stats['labels']]
    box_plot = ax3.boxplot(data_to_plot, labels=['Original', 'Balanced'], patch_artist=True)
    box_plot['boxes'][0].set_facecolor('skyblue')
    box_plot['boxes'][1].set_facecolor('lightcoral')
    ax3.set_ylabel('Age')
    ax3.set_title('Age Distribution Box Plots')
    ax3.grid(True, alpha=0.3)

    # 4. Cumulative distribution
    ax4 = axes[1, 0]
    sorted_orig = np.sort(original_stats['labels'])
    sorted_bal = np.sort(balanced_stats['labels'])

    ax4.plot(sorted_orig, np.linspace(0, 1, len(sorted_orig)),
             label='Original', color='blue', linewidth=2)
    ax4.plot(sorted_bal, np.linspace(0, 1, len(sorted_bal)),
             label='Balanced', color='red', linewidth=2)

    ax4.set_xlabel('Age')
    ax4.set_ylabel('Cumulative Probability')
    ax4.set_title('Cumulative Distribution Functions')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Balance metrics comparison
    ax5 = axes[1, 1]
    metrics = ['Gini\nCoefficient', 'CV Bin\nCounts', 'Normalized\nEntropy']
    orig_values = [original_stats['gini_coefficient'],
                   original_stats['cv_bin_counts'],
                   original_stats['normalized_entropy']]
    bal_values = [balanced_stats['gini_coefficient'],
                  balanced_stats['cv_bin_counts'],
                  balanced_stats['normalized_entropy']]

    x_pos = np.arange(len(metrics))
    width = 0.35

    ax5.bar(x_pos - width/2, orig_values, width, label='Original',
            color='skyblue', alpha=0.7)
    ax5.bar(x_pos + width/2, bal_values, width, label='Balanced',
            color='lightcoral', alpha=0.7)

    ax5.set_ylabel('Metric Value')
    ax5.set_title('Balance Metrics Comparison')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(metrics)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (orig, bal) in enumerate(zip(orig_values, bal_values)):
        ax5.text(i - width/2, orig + 0.01, f'{orig:.3f}', ha='center', va='bottom', fontsize=9)
        ax5.text(i + width/2, bal + 0.01, f'{bal:.3f}', ha='center', va='bottom', fontsize=9)

    # 6. Sample weights distribution (for balanced dataset)
    ax6 = axes[1, 2]
    if 'sample_weights' in balanced_stats and len(balanced_stats['sample_weights']) > 0:
        weights = balanced_stats['sample_weights']
        ax6.hist(weights, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax6.set_xlabel('Sample Weight')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Sample Weights Distribution\n(Balanced Dataset)')
        ax6.grid(True, alpha=0.3)

        # Add stats
        ax6.text(0.7, 0.9, f'Mean: {weights.mean():.3f}\nStd: {weights.std():.3f}',
                transform=ax6.transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
    else:
        ax6.text(0.5, 0.5, 'No sample weights\navailable', ha='center', va='center',
                transform=ax6.transAxes, fontsize=12)
        ax6.set_title('Sample Weights Distribution')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()


def statistical_comparison(original_stats: dict, balanced_stats: dict):
    """Perform statistical tests to compare distributions"""
    print(f"\n{'='*50}")
    print("Statistical Comparison")
    print(f"{'='*50}")

    # Kolmogorov-Smirnov test
    ks_stat, ks_p = stats.ks_2samp(original_stats['labels'], balanced_stats['labels'])
    print(f"Kolmogorov-Smirnov test:")
    print(f"  Statistic: {ks_stat:.4f}")
    print(f"  P-value: {ks_p:.4e}")
    print(f"  Interpretation: {'Significantly different' if ks_p < 0.05 else 'Not significantly different'} distributions")

    # Mann-Whitney U test
    mw_stat, mw_p = stats.mannwhitneyu(original_stats['labels'], balanced_stats['labels'])
    print(f"\nMann-Whitney U test:")
    print(f"  Statistic: {mw_stat:.2f}")
    print(f"  P-value: {mw_p:.4e}")
    print(f"  Interpretation: {'Significantly different' if mw_p < 0.05 else 'Not significantly different'} medians")

    # Chi-square test for bin distributions
    contingency_table = np.array([original_stats['bin_counts'], balanced_stats['bin_counts']])
    chi2_stat, chi2_p, _, _ = stats.chi2_contingency(contingency_table)
    print(f"\nChi-square test for bin distributions:")
    print(f"  Statistic: {chi2_stat:.4f}")
    print(f"  P-value: {chi2_p:.4e}")
    print(f"  Interpretation: {'Significantly different' if chi2_p < 0.05 else 'Not significantly different'} bin distributions")

    # Balance improvement metrics
    print(f"\n{'='*30}")
    print("Balance Improvement Summary")
    print(f"{'='*30}")

    # Gini coefficient improvement (lower is better)
    gini_improvement = (original_stats['gini_coefficient'] - balanced_stats['gini_coefficient']) / original_stats['gini_coefficient'] * 100
    print(f"Gini coefficient improvement: {gini_improvement:+.1f}%")

    # CV improvement (lower is better)
    cv_improvement = (original_stats['cv_bin_counts'] - balanced_stats['cv_bin_counts']) / original_stats['cv_bin_counts'] * 100
    print(f"CV bin counts improvement: {cv_improvement:+.1f}%")

    # Entropy improvement (higher is better)
    entropy_improvement = (balanced_stats['normalized_entropy'] - original_stats['normalized_entropy']) / original_stats['normalized_entropy'] * 100
    print(f"Normalized entropy improvement: {entropy_improvement:+.1f}%")

    # Overall balance score (combination of metrics)
    balance_score_orig = (1 - original_stats['gini_coefficient']) * original_stats['normalized_entropy'] / (1 + original_stats['cv_bin_counts'])
    balance_score_bal = (1 - balanced_stats['gini_coefficient']) * balanced_stats['normalized_entropy'] / (1 + balanced_stats['cv_bin_counts'])
    overall_improvement = (balance_score_bal - balance_score_orig) / balance_score_orig * 100

    print(f"Overall balance score improvement: {overall_improvement:+.1f}%")

    return {
        'ks_test': {'statistic': ks_stat, 'p_value': ks_p},
        'mw_test': {'statistic': mw_stat, 'p_value': mw_p},
        'chi2_test': {'statistic': chi2_stat, 'p_value': chi2_p},
        'improvements': {
            'gini': gini_improvement,
            'cv': cv_improvement,
            'entropy': entropy_improvement,
            'overall': overall_improvement
        }
    }


def main():
    """Main comparison function"""
    print("Dataset Balancing Comparison Script")
    print("="*50)

    # Configuration
    n_samples = 200  # Number of mock samples
    patch_size = (64, 64, 64)
    mock_data_dir = "/tmp/mock_hierarchical_data"

    # Create mock data
    print("Creating mock dataset...")
    mock_samples = create_mock_samples(n_samples, mock_data_dir)

    # Create datasets
    print("\nInitializing datasets...")

    try:
        # Original dataset
        original_dataset = HierarchicalDataset(
            samples=mock_samples,
            patch_size=patch_size,
            local_data_dir=mock_data_dir,
            global_data_dir=mock_data_dir,  # Use same for simplicity
            task_type="regression"
        )

        # Balanced dataset
        balanced_dataset = HierarchicalAgeBalancedDataset(
            samples=mock_samples,
            patch_size=patch_size,
            local_data_dir=mock_data_dir,
            global_data_dir=mock_data_dir,  # Use same for simplicity
            task_type="regression",
            n_bins=8,
            oversample_factor=1.2,
            balancing_strategy="oversample",
            verbose=True
        )

        # Analyze distributions
        original_stats = analyze_dataset_distribution(original_dataset, "Original HierarchicalDataset")
        balanced_stats = analyze_dataset_distribution(balanced_dataset, "Balanced HierarchicalAgeBalancedDataset")

        # Statistical comparison
        statistical_results = statistical_comparison(original_stats, balanced_stats)

        # Create comparison plot
        plot_save_path = os.path.join(os.path.dirname(__file__), "dataset_balancing_comparison.png")
        plot_comparison(original_stats, balanced_stats, plot_save_path)

        # Summary report
        print(f"\n{'='*50}")
        print("SUMMARY REPORT")
        print(f"{'='*50}")

        print(f"Original dataset: {len(original_dataset)} samples")
        print(f"Balanced dataset: {len(balanced_dataset)} samples")
        print(f"Size change: {(len(balanced_dataset) / len(original_dataset) - 1) * 100:+.1f}%")

        print(f"\nBalance improvements:")
        for metric, improvement in statistical_results['improvements'].items():
            print(f"  {metric.capitalize()}: {improvement:+.1f}%")

        print(f"\nFiles created:")
        print(f"  - Comparison plot: {plot_save_path}")
        print(f"  - Mock data directory: {mock_data_dir}")

    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up mock data
        import shutil
        if os.path.exists(mock_data_dir):
            shutil.rmtree(mock_data_dir)
            print(f"\nCleaned up mock data directory: {mock_data_dir}")


if __name__ == "__main__":
    main()
