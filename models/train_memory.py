import os
import torch
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import argparse
import json
import time
import pickle
import warnings

from bulider import EnhancedTrajectoryBuilder, merge_trajectory_storages_enhanced
from progressive_model import TrajectoryGuidedProgressiveModel
from trajectory_dataset import create_enhanced_dataloader, L1000EnhancedDataset

warnings.filterwarnings('ignore')


def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def group_batch_by_time_type(batch):
    """Group batch data by time type"""
    type_groups = {
        'complete': {'indices': [], 'data': {}},
        'partial_6h': {'indices': [], 'data': {}},
        'partial_24h': {'indices': [], 'data': {}}
    }

    time_types = batch.get('time_type', [])

    for i, time_type in enumerate(time_types):
        if time_type in type_groups:
            type_groups[time_type]['indices'].append(i)

    for time_type, group_info in type_groups.items():
        indices = group_info['indices']
        if not indices:
            group_info['data'] = {
                'x0': torch.empty(0),
                'x6': None,
                'x24': None,
                'smiles': torch.empty(0),
                'dose': [],
                'data_idx': []
            }
            continue

        group_data = {}
        for key, value in batch.items():
            if key == 'time_type':
                continue

            if isinstance(value, torch.Tensor):
                group_data[key] = value[indices]
            elif isinstance(value, list):
                group_data[key] = [value[i] for i in indices]
            else:
                group_data[key] = value

        type_groups[time_type]['data'] = group_data

    result = {}
    for time_type, group_info in type_groups.items():
        result[time_type] = group_info['data']

    return result


def move_batch_to_device(batch, device):
    """Move batch data to specified device"""
    moved_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved_batch[key] = value.to(device)
        elif isinstance(value, list) and key == 'dose':
            moved_batch[key] = torch.tensor(value, dtype=torch.float32, device=device)
        else:
            moved_batch[key] = value
    return moved_batch


def _generate_composite_index_for_sample(sample_data):
    """Generate composite index for a sample"""
    smiles_str = str(sample_data.get('smiles_str', 'unknown')).strip()
    time_type = str(sample_data.get('time_type', 'unknown')).strip()
    dose = float(sample_data.get('dose', 0.0))
    cell_id = str(sample_data.get('cell_id', 'unknown')).strip()
    data_idx = int(sample_data.get('data_idx', -1))

    composite_key = f"{smiles_str}_{time_type}_{dose:.6f}_{cell_id}_{data_idx}"
    return composite_key


def get_trajectories_for_batch(batch, trajectory_builder, time_type):
    """Get trajectories for a batch"""
    expected_lengths = {
        'complete': 3,
        'partial_6h': 2,
        'partial_24h': 2
    }
    expected_length = expected_lengths.get(time_type, 3)

    batch_size = batch['x0'].size(0)
    target_device = batch['x0'].device

    trajectories = []
    valid_indices = []

    for i in range(batch_size):
        try:
            sample_data = {
                'smiles_str': batch.get('smiles_str', ['unknown'] * batch_size)[i],
                'time_type': time_type,
                'dose': batch.get('dose', [0.0] * batch_size)[i],
                'cell_id': batch.get('cell_id', ['unknown'] * batch_size)[i],
                'data_idx': batch.get('data_idx', list(range(batch_size)))[i]
            }

            composite_index = _generate_composite_index_for_sample(sample_data)
            trajectory = trajectory_builder.get_trajectory_by_index(composite_index)

            if trajectory is not None and trajectory.shape[0] == expected_length:
                trajectory = trajectory.to(target_device)
                trajectories.append(trajectory)
                valid_indices.append(i)

        except Exception as e:
            print(f"Warning: Error processing sample {i}: {e}")
            continue

    if not trajectories:
        print(f"Warning: No valid {time_type} trajectories found in batch")
        return None

    traj_lengths = [traj.shape[0] for traj in trajectories]
    unique_lengths = set(traj_lengths)

    if len(unique_lengths) > 1:
        print(f"Warning: Inconsistent {time_type} trajectory lengths: {unique_lengths}")
        trajectories = [traj for traj in trajectories if traj.shape[0] == expected_length]
        if not trajectories:
            return None

    try:
        trajectories = [traj.to(target_device) for traj in trajectories]
        combined_trajectories = torch.stack(trajectories, dim=1)

        if combined_trajectories.shape[0] != expected_length:
            print(f"Error: Final trajectory length mismatch: {combined_trajectories.shape[0]} vs {expected_length}")
            return None

        return combined_trajectories

    except Exception as e:
        print(f"Error: Trajectory stacking failed: {e}")
        return None


def validate_model(model, valid_loader, trajectory_builder, type_weights, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_samples = 0
    type_losses = {'complete': [], 'partial_6h': [], 'partial_24h': []}

    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Validating", leave=False):
            type_groups = group_batch_by_time_type(batch)

            batch_loss = 0
            batch_samples = 0
            valid_groups = 0

            for time_type, type_batch in type_groups.items():
                if type_batch['x0'].size(0) == 0:
                    continue

                trajectories = get_trajectories_for_batch(type_batch, trajectory_builder, time_type)

                if trajectories is None:
                    continue

                type_batch = move_batch_to_device(type_batch, device)
                trajectories = trajectories.to(device)

                try:
                    loss = model.forward(
                        x0=type_batch['x0'],
                        drug_features=type_batch['smiles'],
                        x6=type_batch.get('x6'),
                        x24=type_batch.get('x24'),
                        dose=type_batch.get('dose'),
                        time_type=time_type,
                        train_mode=True,
                        guided_trajectory=trajectories,
                        data_indices=type_batch.get('data_idx')
                    )

                    weighted_loss = loss * type_weights[time_type]
                    batch_loss += weighted_loss
                    batch_samples += type_batch['x0'].size(0)
                    valid_groups += 1

                    type_losses[time_type].append(loss.item())

                except Exception as e:
                    print(f"Warning: Validation error ({time_type}): {e}")
                    continue

            if valid_groups > 0:
                total_loss += batch_loss / valid_groups
                total_samples += batch_samples

    avg_loss = total_loss / max(1, len(valid_loader))
    return avg_loss


def check_trajectory_completeness(adata, timeseries_data, trajectory_builder, args):
    """Check trajectory library completeness"""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("\nChecking trajectory library completeness...")
    print("=" * 80)

    train_loader, valid_loader, test_loader, train_dataset, valid_dataset, test_dataset = create_enhanced_dataloader(
        adata, timeseries_data,
        trajectory_builder=None,
        split_key=args.split_key,
        batch_size=1,
        extract_additional=True,
        data_filter='all'
    )

    datasets = {
        'train': train_dataset,
        'valid': valid_dataset,
        'test': test_dataset
    }

    total_stats = {
        'complete': {'total': 0, 'found': 0, 'missing': 0},
        'partial_6h': {'total': 0, 'found': 0, 'missing': 0},
        'partial_24h': {'total': 0, 'found': 0, 'missing': 0}
    }

    print("Checking trajectory completeness for each dataset...")
    print("=" * 80)

    for dataset_name, dataset in datasets.items():
        print(f"\nChecking {dataset_name.upper()} dataset")
        print("-" * 50)

        dataset_stats = {
            'complete': {'total': 0, 'found': 0, 'missing': 0},
            'partial_6h': {'total': 0, 'found': 0, 'missing': 0},
            'partial_24h': {'total': 0, 'found': 0, 'missing': 0}
        }

        missing_samples = {
            'complete': [],
            'partial_6h': [],
            'partial_24h': []
        }

        for idx in tqdm(range(len(dataset)), desc=f"Checking {dataset_name}", ncols=100):
            sample = dataset[idx]
            time_type = sample['time_type']
            data_idx = sample.get('data_idx', idx)

            dataset_stats[time_type]['total'] += 1
            total_stats[time_type]['total'] += 1

            sample_data = {
                'smiles_str': sample.get('smiles_str', 'unknown'),
                'time_type': time_type,
                'dose': sample.get('dose', 0.0),
                'cell_id': sample.get('cell_id', 'unknown'),
                'data_idx': data_idx
            }

            composite_index = _generate_composite_index_for_sample(sample_data)
            trajectory = trajectory_builder.get_trajectory_by_index(composite_index)

            if trajectory is not None:
                dataset_stats[time_type]['found'] += 1
                total_stats[time_type]['found'] += 1
            else:
                dataset_stats[time_type]['missing'] += 1
                total_stats[time_type]['missing'] += 1
                missing_samples[time_type].append({
                    'data_idx': data_idx,
                    'composite_index': composite_index,
                    'drug_id': sample.get('drug_id', 'unknown'),
                    'cell_id': sample.get('cell_id', 'unknown'),
                    'dose': sample.get('dose', 0.0)
                })

        print(f"\n{dataset_name.upper()} dataset results:")
        print("-" * 40)

        for time_type in ['complete', 'partial_6h', 'partial_24h']:
            stats = dataset_stats[time_type]
            total = stats['total']

            if total == 0:
                print(f"{time_type:>12}: No samples")
                continue

            found_rate = (stats['found'] / total) * 100
            status = "✅" if found_rate >= 95 else "⚠️" if found_rate >= 80 else "❌"

            print(f"{time_type:>12}: {status}")
            print(f"{'':>14} Total: {total:>4}")
            print(f"{'':>14} Found: {stats['found']:>4} ({found_rate:>5.1f}%)")
            print(f"{'':>14} Missing: {stats['missing']:>4}")

    print(f"\n" + "=" * 60)
    print("Overall Statistics")
    print("=" * 60)

    total_samples = sum(total_stats[t]['total'] for t in total_stats)
    total_found = sum(total_stats[t]['found'] for t in total_stats)
    total_missing = sum(total_stats[t]['missing'] for t in total_stats)

    if total_samples > 0:
        overall_coverage = (total_found / total_samples) * 100
        print(f"Total samples: {total_samples:,}")
        print(f"Found trajectories: {total_found:,} ({overall_coverage:.2f}%)")
        print(f"Missing trajectories: {total_missing:,} ({100 - overall_coverage:.2f}%)")
    else:
        print("No samples found")
        return None

    print(f"\nBy trajectory type:")
    for time_type in ['complete', 'partial_6h', 'partial_24h']:
        total = total_stats[time_type]['total']
        found = total_stats[time_type]['found']
        missing = total_stats[time_type]['missing']

        if total > 0:
            coverage = (found / total) * 100
            status = "✅" if coverage >= 95 else "⚠️" if coverage >= 80 else "❌"
            print(f"  {time_type:>12}: {found:>6}/{total:<6} ({coverage:>6.2f}%) {status} | Missing: {missing}")

    print(f"\nTrajectory storage statistics")
    print("-" * 40)
    print(f"Trajectories in memory: {len(trajectory_builder.trajectory_storage):,}")
    print(f"Saved trajectories: {len(trajectory_builder.saved_to_disk):,}")
    print(f"Metadata rows: {len(trajectory_builder.metadata_df):,}")

    memory_usage = sum(
        tensor.element_size() * tensor.nelement()
        for tensor in trajectory_builder.trajectory_storage.values()
    ) / (1024 * 1024)
    print(f"Memory usage: {memory_usage:.2f} MB")

    completeness_score = overall_coverage if total_samples > 0 else 0

    print(f"\n" + "=" * 60)
    print("Completeness Assessment")
    print("=" * 60)

    if completeness_score >= 99.5:
        status = "🏆 Excellent"
        recommendation = "Trajectory library is very complete, ready for training!"
    elif completeness_score >= 95.0:
        status = "✅ Good"
        recommendation = "Trajectory library is mostly complete, minor补充 recommended."
    elif completeness_score >= 90.0:
        status = "⚠️ Fair"
        recommendation = "Trajectory library has some gaps,补充 recommended before training."
    elif completeness_score >= 80.0:
        status = "⚠️ Poor"
        recommendation = "Trajectory library has significant gaps, re-optimization recommended."
    else:
        status = "❌ Bad"
        recommendation = "Trajectory library is severely incomplete, full re-optimization needed."

    print(f"Completeness score: {completeness_score:.2f}% {status}")
    print(f"Recommendation: {recommendation}")

    report = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'total_samples': total_samples,
        'total_found': total_found,
        'total_missing': total_missing,
        'completeness_score': completeness_score,
        'stats_by_type': total_stats,
        'storage_stats': {
            'memory_trajectories': len(trajectory_builder.trajectory_storage),
            'saved_trajectories': len(trajectory_builder.saved_to_disk),
            'metadata_rows': len(trajectory_builder.metadata_df),
            'memory_usage_mb': memory_usage
        },
        'recommendation': recommendation
    }

    report_path = os.path.join(args.memory_dir, "completeness_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    print(f"\nCompleteness report saved to: {report_path}")

    return report


def build_and_optimize_trajectories(adata, timeseries_data, model_config, args):
    """Build and optimize trajectory library"""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.shard_index >= 0:
        shard_memory_dir = os.path.join(args.memory_dir, f"shard_{args.shard_index}")
        os.makedirs(shard_memory_dir, exist_ok=True)
        trajectory_dir = shard_memory_dir
    else:
        trajectory_dir = args.memory_dir

    if args.merge_only:
        print("Merge only mode: merging all shard trajectory libraries")
        return merge_trajectory_storages_enhanced(args.memory_dir, args.total_shards)

    trajectory_builder = EnhancedTrajectoryBuilder(
        transform_net=None,
        condition_fusion=None,
        timesteps=args.timesteps,
        n_latent=adata.shape[1],
        memory_dir=trajectory_dir
    )

    if os.path.exists(os.path.join(trajectory_dir, "trajectory_storage.pkl")):
        print("Loading existing trajectory library...")
        trajectory_builder.load_trajectories_and_metadata()

    train_loader, valid_loader, test_loader, train_dataset, valid_dataset, test_dataset = create_enhanced_dataloader(
        adata, timeseries_data,
        trajectory_builder=None,
        split_key=args.split_key,
        batch_size=args.batch_size,
        extract_additional=True,
        data_filter='all'
    )

    if args.shard_index >= 0:
        print(f"Shard mode: processing shard {args.shard_index + 1}/{args.total_shards}")

        train_samples = len(train_dataset)
        samples_per_shard = (train_samples + args.total_shards - 1) // args.total_shards
        start_idx = args.shard_index * samples_per_shard
        end_idx = min(start_idx + samples_per_shard, train_samples)
        train_indices = list(range(start_idx, end_idx))

        valid_samples = len(valid_dataset)
        valid_per_shard = (valid_samples + args.total_shards - 1) // args.total_shards
        valid_start = args.shard_index * valid_per_shard
        valid_end = min(valid_start + valid_per_shard, valid_samples)
        valid_indices = list(range(valid_start, valid_end))

        test_samples = len(test_dataset)
        test_per_shard = (test_samples + args.total_shards - 1) // args.total_shards
        test_start = args.shard_index * test_per_shard
        test_end = min(test_start + test_per_shard, test_samples)
        test_indices = list(range(test_start, test_end))

        print(f"Processing train indices: {start_idx} - {end_idx - 1}, total {len(train_indices)} samples")
        print(f"Processing valid indices: {valid_start} - {valid_end - 1}, total {len(valid_indices)} samples")
        print(f"Processing test indices: {test_start} - {test_end - 1}, total {len(test_indices)} samples")
    else:
        train_indices = list(range(len(train_dataset)))
        valid_indices = list(range(len(valid_dataset)))
        test_indices = list(range(len(test_dataset)))

    datasets_info = [
        (train_dataset, train_indices, "Training"),
        (valid_dataset, valid_indices, "Validation"),
        (test_dataset, test_indices, "Test")
    ]

    for dataset, indices, dataset_name in datasets_info:
        print(f"\n=== Optimizing {dataset_name} trajectories ===")
        _optimize_trajectories_for_dataset(
            dataset, trajectory_builder, indices, args.optimize_steps, device
        )

    trajectory_builder.finalize_storage()

    return trajectory_builder


def _optimize_trajectories_for_dataset(dataset, trajectory_builder, indices, optimize_steps, device):
    """Optimize trajectories for dataset samples"""
    if not indices:
        print("No samples to process")
        return

    print(f"Optimizing trajectories for {len(indices)} samples...")

    for idx in tqdm(indices, desc="Optimizing"):
        sample = dataset[idx]

        endpoint_6h_index = None
        endpoint_24h_index = None

        if hasattr(dataset, 'timeseries_samples') and idx < len(dataset.timeseries_samples):
            original_sample = dataset.timeseries_samples[idx]
            endpoint_6h_index = original_sample.get('h6_idx')
            endpoint_24h_index = original_sample.get('h24_idx')
        else:
            endpoint_6h_index = sample.get('h6_idx')
            endpoint_24h_index = sample.get('h24_idx')

        sample_data = {
            'x0': sample['x0'].to(device).unsqueeze(0),
            'x6': sample['x6'].to(device).unsqueeze(0) if sample['x6'] is not None else None,
            'x24': sample['x24'].to(device).unsqueeze(0) if sample['x24'] is not None else None,
            'smiles': sample['smiles'].to(device),
            'smiles_str': sample.get('smiles_str', ''),
            'time_type': sample['time_type'],
            'drug_id': sample.get('drug_id', 'unknown'),
            'cell_id': sample.get('cell_id', 'unknown'),
            'dose': sample.get('dose', 0.0),
            'data_idx': sample.get('data_idx', idx),
            'source': sample.get('source', 'dataset'),
            'endpoint_6h_index': endpoint_6h_index,
            'endpoint_24h_index': endpoint_24h_index
        }

        composite_index = trajectory_builder.optimize_single_trajectory_with_metadata(
            sample_data, optimize_steps
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def infer_and_add_split_labels(trajectory_builder, adata, timeseries_data, split_key='drug_splits_4'):
    """Infer and add split labels to trajectory metadata"""
    print("Inferring split labels for trajectories...")

    if trajectory_builder.metadata_df.empty:
        print("Error: Metadata DataFrame is empty")
        return trajectory_builder

    if 'split' in trajectory_builder.metadata_df.columns:
        print("Split labels already exist in metadata")
        split_counts = trajectory_builder.metadata_df['split'].value_counts()
        print("Current split distribution:")
        for split_type, count in split_counts.items():
            print(f"  {split_type}: {count}")
        return trajectory_builder

    print("Creating temporary datasets to get split information...")
    temp_datasets = {}

    try:
        for split_type in ['train', 'valid', 'test']:
            temp_datasets[split_type] = L1000EnhancedDataset(
                adata=adata,
                timeseries_data=timeseries_data,
                trajectory_builder=None,
                dtype=split_type,
                split_key=split_key,
                extract_additional=True,
                data_filter='all'
            )
            print(f"  {split_type} set: {len(temp_datasets[split_type])} samples")
    except Exception as e:
        print(f"Error creating temporary datasets: {e}")
        return trajectory_builder

    print("Building sample feature mappings...")

    full_feature_to_split = {}
    partial_feature_to_split = {}
    composite_to_split = {}

    for split_type, dataset in temp_datasets.items():
        for idx in range(len(dataset)):
            try:
                sample = dataset[idx]

                full_feature_key = (
                    str(sample.get('smiles_str', 'unknown')).strip(),
                    str(sample.get('time_type', 'unknown')).strip(),
                    float(sample.get('dose', 0.0)),
                    str(sample.get('cell_id', 'unknown')).strip(),
                    str(sample.get('drug_id', 'unknown')).strip(),
                    int(sample.get('data_idx', -1))
                )
                full_feature_to_split[full_feature_key] = split_type

                partial_feature_key = (
                    str(sample.get('smiles_str', 'unknown')).strip(),
                    str(sample.get('time_type', 'unknown')).strip(),
                    float(sample.get('dose', 0.0)),
                    str(sample.get('cell_id', 'unknown')).strip()
                )
                if partial_feature_key not in partial_feature_to_split:
                    partial_feature_to_split[partial_feature_key] = split_type

                if sample.get('composite_index'):
                    composite_to_split[sample['composite_index']] = split_type

            except Exception as e:
                print(f"Warning: Error processing sample {idx}: {e}")
                continue

    print(f"  Full feature mappings: {len(full_feature_to_split)}")
    print(f"  Partial feature mappings: {len(partial_feature_to_split)}")
    print(f"  Composite index mappings: {len(composite_to_split)}")

    print("Inferring labels for trajectories...")
    split_labels = []
    match_methods = []
    matched_count = 0

    for idx, row in tqdm(trajectory_builder.metadata_df.iterrows(),
                         total=len(trajectory_builder.metadata_df),
                         desc="Inferring labels"):
        matched_split = None
        match_method = 'none'

        try:
            if row.get('composite_index') and row['composite_index'] in composite_to_split:
                matched_split = composite_to_split[row['composite_index']]
                match_method = 'composite_index'

            elif matched_split is None:
                full_feature_key = (
                    str(row.get('smiles_str', 'unknown')).strip(),
                    str(row.get('time_type', 'unknown')).strip(),
                    float(row.get('dose', 0.0)),
                    str(row.get('cell_id', 'unknown')).strip(),
                    str(row.get('drug_id', 'unknown')).strip(),
                    int(row.get('data_idx', -1))
                )

                if full_feature_key in full_feature_to_split:
                    matched_split = full_feature_to_split[full_feature_key]
                    match_method = 'full_feature'

            if matched_split is None:
                partial_feature_key = (
                    str(row.get('smiles_str', 'unknown')).strip(),
                    str(row.get('time_type', 'unknown')).strip(),
                    float(row.get('dose', 0.0)),
                    str(row.get('cell_id', 'unknown')).strip()
                )

                if partial_feature_key in partial_feature_to_split:
                    matched_split = partial_feature_to_split[partial_feature_key]
                    match_method = 'partial_feature'

            if matched_split:
                split_labels.append(matched_split)
                match_methods.append(match_method)
                matched_count += 1
            else:
                split_labels.append('unknown')
                match_methods.append('none')

        except Exception as e:
            print(f"Warning: Error processing trajectory {idx}: {e}")
            split_labels.append('unknown')
            match_methods.append('error')

    trajectory_builder.metadata_df['split'] = split_labels
    trajectory_builder.metadata_df['match_method'] = match_methods

    split_counts = trajectory_builder.metadata_df['split'].value_counts()
    method_counts = trajectory_builder.metadata_df['match_method'].value_counts()

    print(f"\nLabel inference complete!")
    print(f"Matched: {matched_count}/{len(trajectory_builder.metadata_df)} ({matched_count / len(trajectory_builder.metadata_df) * 100:.1f}%)")

    print(f"\nSplit distribution:")
    for split_type, count in split_counts.items():
        percentage = count / len(trajectory_builder.metadata_df) * 100
        print(f"  {split_type}: {count} ({percentage:.1f}%)")

    print(f"\nMatch method distribution:")
    for method, count in method_counts.items():
        percentage = count / len(trajectory_builder.metadata_df) * 100
        print(f"  {method}: {count} ({percentage:.1f}%)")

    try:
        metadata_path = os.path.join(trajectory_builder.memory_dir, "trajectory_metadata.csv")
        trajectory_builder.metadata_df.to_csv(metadata_path, index=False)
        print(f"Updated metadata saved to: {metadata_path}")
    except Exception as e:
        print(f"Warning: Failed to save metadata: {e}")

    return trajectory_builder


def main():
    parser = argparse.ArgumentParser(description="Trajectory optimization and progressive model training")

    parser.add_argument("--adata-path", type=str, default="../../dataset/Lincs_L1000_with_pairs_splits.h5ad",
                        help="Path to AnnData file")
    parser.add_argument("--timeseries-path", type=str, default="../../dataset/L1000_0_6_24.csv",
                        help="Path to timeseries metadata CSV")
    parser.add_argument("--split-key", type=str, default="drug_splits_4",
                        help="Column name for data split")
    parser.add_argument("--normalize", action="store_true",
                        help="Apply normalization to gene expression data")

    parser.add_argument("--latent-dim", type=int, default=64,
                        help="Latent dimension size")
    parser.add_argument("--timesteps", type=int, default=50,
                        help="Number of timesteps in trajectory")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")

    parser.add_argument("--complete-trajectories", action="store_true",
                        help="Optimize complete trajectories (0-6-24h)")
    parser.add_argument("--partial-6h-trajectories", action="store_true",
                        help="Optimize partial trajectories (0-6h)")
    parser.add_argument("--partial-24h-trajectories", action="store_true",
                        help="Optimize partial trajectories (0-24h)")
    parser.add_argument("--only-optimize", action="store_true", default=True,
                        help="Only optimize trajectories, skip progressive model training")
    parser.add_argument("--optimize-steps", type=int, default=30,
                        help="Number of trajectory optimization steps")

    parser.add_argument("--shard-index", type=int, default=-1,
                        help="Current shard index (0-79), -1 for all data")
    parser.add_argument("--total-shards", type=int, default=80,
                        help="Total number of shards")
    parser.add_argument("--merge-only", action="store_true",
                        help="Only merge shard trajectory libraries")
    parser.add_argument("--memory-dir", type=str, default="./trajectory_memory",
                        help="Trajectory memory directory")

    parser.add_argument("--check-completeness", action="store_true",
                        help="Check trajectory library completeness")

    parser.add_argument("--skip-trajectory-optimization", action="store_true",
                        help="Skip trajectory optimization, directly load existing library for training")

    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Initial learning rate")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping value")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="Model save interval")
    parser.add_argument("--early-stop-patience", type=int, default=15,
                        help="Early stopping patience")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--output-dir", type=str, default="./checkpoint/",
                        help="Output directory")

    args = parser.parse_args()

    if args.check_completeness:
        print("\nTrajectory library completeness check mode")
        print("=" * 60)

        print("Loading data...")
        adata = sc.read(args.adata_path)
        if args.normalize:
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)
        timeseries_data = pd.read_csv(args.timeseries_path)

        trajectory_builder = EnhancedTrajectoryBuilder(
            transform_net=None,
            condition_fusion=None,
            timesteps=args.timesteps,
            n_latent=adata.shape[1],
            memory_dir=args.memory_dir
        )

        storage_file = os.path.join(args.memory_dir, "trajectory_storage.pkl")
        metadata_file = os.path.join(args.memory_dir, "trajectory_metadata.csv")

        if os.path.exists(storage_file) and os.path.exists(metadata_file):
            print("Loading trajectory storage and metadata...")
            trajectory_builder.load_trajectories_and_metadata()
        else:
            print("Error: Trajectory storage or metadata file not found")
            return

        report = check_trajectory_completeness(adata, timeseries_data, trajectory_builder, args)

        print(f"\nCompleteness check complete!")
        return

    if args.shard_index >= args.total_shards:
        print(f"Error: Shard index {args.shard_index} exceeds total shards {args.total_shards}")
        return

    set_random_seeds(args.seed)

    print("Loading data...")
    adata = sc.read(args.adata_path)
    args.normalize = True
    if args.normalize:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
    timeseries_data = pd.read_csv(args.timeseries_path)

    model_config = {
        "n_genes": adata.shape[1],
        "n_latent": args.latent_dim,
        "features_dim": 2304,
        "timesteps": args.timesteps,
        "dropout": args.dropout
    }

    if args.skip_trajectory_optimization:
        print("\nSkip trajectory optimization mode: directly loading existing library for training")
        print("=" * 60)
    else:
        if not (args.complete_trajectories or args.partial_6h_trajectories or args.partial_24h_trajectories):
            args.complete_trajectories = True
            args.partial_6h_trajectories = True
            args.partial_24h_trajectories = True

        trajectory_builder = build_and_optimize_trajectories(adata, timeseries_data, model_config, args)

        if args.only_optimize:
            print("Trajectory optimization complete, exiting")
            return


if __name__ == "__main__":
    main()