import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import json
import time
from pathlib import Path
import pickle
import warnings
from collections import defaultdict
import gc
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

from bulider import EnhancedTrajectoryBuilder
from progressive_model import TrajectoryGuidedProgressiveModel
from trajectory_dataset import L1000EnhancedDataset, custom_collate

warnings.filterwarnings('ignore')

CACHE_DIR = "./cache"
TRAJECTORIES_CACHE_FILE = os.path.join(CACHE_DIR, "unified_trajectories.pkl")
METADATA_CACHE_FILE = os.path.join(CACHE_DIR, "unified_metadata.csv")


def safe_tensor_to_scalar(tensor, name="tensor"):
    if isinstance(tensor, torch.Tensor):
        if tensor.numel() == 0:
            return 0.0
        elif tensor.numel() == 1:
            value = tensor.item()
            if np.isnan(value) or np.isinf(value):
                return 0.0
            return value
        elif tensor.numel() > 1:
            mean_val = tensor.mean().item()
            if np.isnan(mean_val) or np.isinf(mean_val):
                return 0.0
            return mean_val
    elif isinstance(tensor, (int, float)):
        if np.isnan(tensor) or np.isinf(tensor):
            return 0.0
        return float(tensor)
    else:
        return 0.0


def compute_evaluation_metrics(y_true, y_pred):
    metrics = {}
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) == 0:
        return {'pcc': 0.0, 'spearman': 0.0, 'mse': float('inf'), 'mae': float('inf')}

    try:
        pcc, _ = pearsonr(y_true_clean, y_pred_clean)
        metrics['pcc'] = pcc if not np.isnan(pcc) else 0.0
        spearman, _ = spearmanr(y_true_clean, y_pred_clean)
        metrics['spearman'] = spearman if not np.isnan(spearman) else 0.0
        metrics['mse'] = mean_squared_error(y_true_clean, y_pred_clean)
        metrics['mae'] = mean_absolute_error(y_true_clean, y_pred_clean)
    except Exception:
        metrics = {'pcc': 0.0, 'spearman': 0.0, 'mse': float('inf'), 'mae': float('inf')}
    return metrics


class UnifiedTrajectoryManager:
    def __init__(self, memory_dir, adata, timeseries_data, split_key='drug_splits_4', verbose=False):
        self.memory_dir = memory_dir
        self.adata = adata
        self.timeseries_data = timeseries_data
        self.split_key = split_key
        self.verbose = verbose

        self.unified_trajectories = {}
        self.unified_metadata = pd.DataFrame()
        self.trajectory_splits = {}

        self.shard_stats = {}
        self.split_stats = {'train': 0, 'valid': 0, 'test': 0, 'unknown': 0}

    def _move_trajectory_to_cpu(self, trajectory):
        if isinstance(trajectory, torch.Tensor):
            return trajectory.detach().cpu()
        return trajectory

    def _move_trajectory_to_device(self, trajectory, device):
        if isinstance(trajectory, torch.Tensor):
            return trajectory.to(device)
        return trajectory

    def _force_cpu_conversion(self, trajectory):
        if isinstance(trajectory, torch.Tensor):
            if trajectory.is_cuda:
                cpu_tensor = trajectory.detach().cpu()
                independent_tensor = cpu_tensor.clone()
                del cpu_tensor, trajectory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return independent_tensor
            else:
                return trajectory.detach().clone()
        return trajectory

    def _aggressive_memory_cleanup(self):
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            if torch.cuda.memory_allocated() > 0:
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
        gc.collect()

    def _verify_cpu_storage(self):
        gpu_count = 0
        cpu_count = 0
        for composite_index, trajectory in self.unified_trajectories.items():
            if isinstance(trajectory, torch.Tensor):
                if trajectory.is_cuda:
                    gpu_count += 1
                else:
                    cpu_count += 1
        if gpu_count > 0:
            for composite_index, trajectory in list(self.unified_trajectories.items()):
                if isinstance(trajectory, torch.Tensor) and trajectory.is_cuda:
                    self.unified_trajectories[composite_index] = self._force_cpu_conversion(trajectory)

    def load_from_cache(self):
        if os.path.exists(TRAJECTORIES_CACHE_FILE) and os.path.exists(METADATA_CACHE_FILE):
            try:
                with open(TRAJECTORIES_CACHE_FILE, 'rb') as f:
                    self.unified_trajectories = pickle.load(f)
                self.unified_metadata = pd.read_csv(METADATA_CACHE_FILE)
                return True
            except Exception as e:
                return False
        return False

    def save_to_cache(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        try:
            with open(TRAJECTORIES_CACHE_FILE, 'wb') as f:
                pickle.dump(self.unified_trajectories, f)
            self.unified_metadata.to_csv(METADATA_CACHE_FILE, index=False)
        except Exception as e:
            pass

    def load_all_shards(self, max_shards=100):
        all_metadata_dfs = []
        loaded_shards = 0

        for shard_idx in range(max_shards):
            shard_dir = os.path.join(self.memory_dir, f"shard_{shard_idx}")
            shard_file = os.path.join(shard_dir, "trajectory_storage.pkl")
            metadata_file = os.path.join(shard_dir, "trajectory_metadata.csv")

            if not os.path.exists(shard_file) or not os.path.exists(metadata_file):
                continue

            try:
                self._aggressive_memory_cleanup()

                with open(shard_file, 'rb') as f:
                    temp_data = pickle.load(f)
                    shard_trajectories = {k: self._move_trajectory_to_cpu(v) for k, v in
                                          temp_data.get('trajectories', {}).items()}
                    del temp_data

                shard_metadata = pd.read_csv(metadata_file)
                new_trajectories = 0
                duplicate_trajectories = 0

                for composite_index, trajectory in shard_trajectories.items():
                    if composite_index not in self.unified_trajectories:
                        cpu_trajectory = self._force_cpu_conversion(trajectory)
                        self.unified_trajectories[composite_index] = cpu_trajectory
                        new_trajectories += 1
                    else:
                        duplicate_trajectories += 1

                del shard_trajectories
                all_metadata_dfs.append(shard_metadata)

                self.shard_stats[shard_idx] = {
                    'new_trajectories': new_trajectories,
                    'duplicate_trajectories': duplicate_trajectories,
                    'metadata_rows': len(shard_metadata)
                }
                loaded_shards += 1

                if shard_idx % 2 == 0:
                    self._aggressive_memory_cleanup()

            except Exception as e:
                self._aggressive_memory_cleanup()
                print(e)
                continue

        if all_metadata_dfs:
            self.unified_metadata = pd.concat(all_metadata_dfs, ignore_index=True)
            self.unified_metadata = self.unified_metadata.drop_duplicates(subset=['composite_index'],
                                                                          keep='last').reset_index(drop=True)

        self._aggressive_memory_cleanup()
        self._verify_cpu_storage()

    def assign_trajectory_splits_from_endpoints(self):
        if self.unified_metadata.empty:
            return

        if self.split_key not in self.adata.obs.columns:
            return

        adata_split_df = pd.DataFrame({
            'endpoint_adata_index': self.adata.obs.index,
            'split_value': self.adata.obs[self.split_key]
        })

        merged_df = self.unified_metadata.merge(
            adata_split_df,
            on='endpoint_adata_index',
            how='left'
        )

        merged_df['split_value'] = merged_df['split_value'].fillna('unknown')

        self.trajectory_splits = dict(zip(
            merged_df['composite_index'],
            merged_df['split_value']
        ))

        split_counts = merged_df['split_value'].value_counts()
        self.split_stats = {
            'train': int(split_counts.get('train', 0)),
            'valid': int(split_counts.get('valid', 0)),
            'test': int(split_counts.get('test', 0)),
            'unknown': int(split_counts.get('unknown', 0))
        }

        self.unified_metadata['split'] = merged_df['split_value'].values

        del adata_split_df, merged_df

    def get_trajectory_by_index(self, composite_index, device=None):
        if composite_index in self.unified_trajectories:
            trajectory = self.unified_trajectories[composite_index]
            if device is not None:
                trajectory = self._move_trajectory_to_device(trajectory, device)
            return trajectory
        return None


class UnifiedTrajectoryDataset(torch.utils.data.Dataset):
    _smiles_cache = None
    _smiles_cache_loaded = False

    def __init__(self, trajectory_manager, adata, split_type='train', time_type_filter=None,
                 split_key='drug_splits_4', smiles_embedding_path='../../dataset/KPGT_prnet_2304.pkl'):
        self.trajectory_manager = trajectory_manager
        self.split_type = split_type
        self.time_type_filter = time_type_filter
        self.split_key = split_key
        self.adata = adata

        if split_key not in adata.obs.columns:
            raise ValueError(f"Split key '{split_key}' not found in adata.obs.columns")

        total_metadata_rows = len(trajectory_manager.unified_metadata)

        adata_split_df = pd.DataFrame({
            'endpoint_adata_index': adata.obs.index,
            'split_value': adata.obs[split_key]
        })

        merged_df = trajectory_manager.unified_metadata.merge(
            adata_split_df,
            on='endpoint_adata_index',
            how='inner'
        )

        filtered_metadata = merged_df[merged_df['split_value'] == split_type].copy()

        valid_endpoint_count = len(merged_df)
        matched_split_count = len(filtered_metadata)

        del adata_split_df, merged_df

        if time_type_filter is not None and not filtered_metadata.empty:
            before_time_filter = len(filtered_metadata)
            filtered_metadata = filtered_metadata[
                filtered_metadata['time_type'] == time_type_filter].copy()
            after_time_filter = len(filtered_metadata)

        if not UnifiedTrajectoryDataset._smiles_cache_loaded:
            UnifiedTrajectoryDataset._load_smiles_embedding(smiles_embedding_path)

        self.smi2emb = UnifiedTrajectoryDataset._smiles_cache or {}

        self.samples = []
        self._precompute_samples(filtered_metadata)

        del filtered_metadata

    @classmethod
    def _load_smiles_embedding(cls, smiles_embedding_path):
        if cls._smiles_cache_loaded:
            return

        cls._smiles_cache = {}
        if os.path.exists(smiles_embedding_path):
            try:
                print(f"Loading SMILES embeddings from {smiles_embedding_path}...")
                with open(smiles_embedding_path, 'rb') as f:
                    smi2emb_raw = pickle.load(f)
                    cls._smiles_cache = {key[0]: value for key, value in smi2emb_raw.items()}
                    del smi2emb_raw
                print(f"Loaded {len(cls._smiles_cache)} SMILES embeddings")
            except Exception as e:
                print(f"Failed to load SMILES embeddings: {e}")
                cls._smiles_cache = {}

        cls._smiles_cache_loaded = True

    def _precompute_samples(self, filtered_metadata):
        if filtered_metadata.empty:
            return

        available_trajectories = set(self.trajectory_manager.unified_trajectories.keys())
        valid_mask = filtered_metadata['composite_index'].isin(available_trajectories)
        valid_metadata = filtered_metadata[valid_mask].copy()

        if valid_metadata.empty:
            return

        processed_metadata = valid_metadata.copy()
        processed_metadata['time_type'] = processed_metadata.get('time_type',
                                                                 pd.Series(['unknown'] * len(valid_metadata))).fillna(
            'unknown')
        processed_metadata['drug_id'] = processed_metadata.get('drug_id',
                                                               pd.Series(['unknown'] * len(valid_metadata))).fillna(
            'unknown')
        processed_metadata['cell_id'] = processed_metadata.get('cell_id',
                                                               pd.Series(['unknown'] * len(valid_metadata))).fillna(
            'unknown')
        processed_metadata['dose'] = pd.to_numeric(processed_metadata.get('dose', 0.0), errors='coerce').fillna(0.0)
        processed_metadata['smiles_str'] = processed_metadata.get('smiles_str',
                                                                  pd.Series([''] * len(valid_metadata))).fillna('')
        processed_metadata['data_idx'] = pd.to_numeric(processed_metadata.get('data_idx', -1), errors='coerce').fillna(
            -1).astype(int)
        processed_metadata['endpoint_adata_index'] = processed_metadata.get('endpoint_adata_index', pd.Series(
            [''] * len(valid_metadata))).fillna('')
        processed_metadata['split'] = self.split_type

        columns_needed = ['composite_index', 'time_type', 'drug_id', 'cell_id', 'dose',
                          'smiles_str', 'data_idx', 'split', 'endpoint_adata_index']
        final_metadata = processed_metadata[columns_needed]

        self.samples = final_metadata.to_dict('records')

        del valid_metadata, processed_metadata, final_metadata

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        composite_index = sample['composite_index']

        trajectory = self.trajectory_manager.get_trajectory_by_index(composite_index, device=None)
        if trajectory is None:
            trajectory = torch.zeros(50, 64)

        x0 = trajectory[0]
        x6 = None
        x24 = None

        time_type = sample['time_type']
        if time_type == 'complete':
            mid_point = len(trajectory) // 4
            if mid_point < len(trajectory):
                x6 = trajectory[mid_point]
            x24 = trajectory[-1]
        elif time_type == 'partial_6h':
            x6 = trajectory[-1]
        elif time_type == 'partial_24h':
            x24 = trajectory[-1]

        smiles_str = sample['smiles_str']
        if smiles_str in self.smi2emb:
            smiles_encoding = torch.tensor(self.smi2emb[smiles_str], dtype=torch.float32)
        else:
            smiles_encoding = torch.zeros(2304, dtype=torch.float32)

        result = {
            'x0': x0,
            'x6': x6,
            'x24': x24,
            'smiles': smiles_encoding,
            'smiles_str': smiles_str,
            'cell_id': sample['cell_id'],
            'dose': sample['dose'],
            'drug_id': sample['drug_id'],
            'time_type': time_type,
            'data_idx': sample['data_idx'],
            'trajectory': trajectory,
            'composite_index': composite_index
        }
        return result


def robust_collate_fn(batch):
    if not batch:
        return None

    batch_size = len(batch)
    result = {}

    try:
        x0_items = [item['x0'] for item in batch if item.get('x0') is not None]
        if x0_items:
            result['x0'] = torch.stack(x0_items)
        else:
            result['x0'] = None

        smiles_items = [item['smiles'] for item in batch if item.get('smiles') is not None]
        if smiles_items:
            result['smiles'] = torch.stack(smiles_items)
        else:
            result['smiles'] = None

        x6_items = [item['x6'] for item in batch if item.get('x6') is not None]
        result['x6'] = torch.stack(x6_items) if x6_items else None

        x24_items = [item['x24'] for item in batch if item.get('x24') is not None]
        result['x24'] = torch.stack(x24_items) if x24_items else None

        dose_values = []
        for item in batch:
            dose_val = item.get('dose', 0.0)
            if isinstance(dose_val, (int, float)) and not (np.isnan(dose_val) or np.isinf(dose_val)):
                dose_values.append(float(dose_val))
            else:
                dose_values.append(0.0)
        result['dose'] = torch.tensor(dose_values, dtype=torch.float32)

        trajectory_items = [item['trajectory'] for item in batch if item.get('trajectory') is not None]
        if trajectory_items:
            try:
                shapes = [traj.shape for traj in trajectory_items]
                if len(set(shapes)) == 1:
                    result['trajectory'] = torch.stack(trajectory_items, dim=0)
                else:
                    min_steps = min(traj.shape[0] for traj in trajectory_items)
                    min_features = min(traj.shape[1] for traj in trajectory_items)
                    truncated = [traj[:min_steps, :min_features] for traj in trajectory_items]
                    result['trajectory'] = torch.stack(truncated, dim=0)
            except Exception:
                result['trajectory'] = None
        else:
            result['trajectory'] = None

        result['time_type'] = [item.get('time_type', 'unknown') for item in batch]
        result['composite_index'] = [item.get('composite_index', '') for item in batch]
        result['data_idx'] = [item.get('data_idx', -1) for item in batch]
        result['smiles_str'] = [item.get('smiles_str', '') for item in batch]
        result['cell_id'] = [item.get('cell_id', 'unknown') for item in batch]
        result['drug_id'] = [item.get('drug_id', 'unknown') for item in batch]

    except Exception as e:
        print(f"Collate function error: {e}")
        return None

    return result


def create_unified_dataloaders(trajectory_manager, adata, batch_size=512, num_workers=16,
                               distributed=False, world_size=1, rank=0, split_key='drug_splits_4'):
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        optimized_batch_size = min(batch_size, 1024 * num_gpus)
        per_gpu_batch_size = optimized_batch_size // num_gpus
    else:
        optimized_batch_size = min(batch_size, 2048)
        per_gpu_batch_size = optimized_batch_size

    print(f"Optimized batch size: {optimized_batch_size} (per GPU: {per_gpu_batch_size})")
    print(f"Using split key: {split_key}")

    time_types = ['complete', 'partial_6h', 'partial_24h']
    dataloaders = {}
    datasets = {}

    for time_type in time_types:
        print(f"\nCreating {time_type} dataset...")

        train_dataset = UnifiedTrajectoryDataset(
            trajectory_manager, adata, 'train', time_type, split_key
        )
        valid_dataset = UnifiedTrajectoryDataset(
            trajectory_manager, adata, 'valid', time_type, split_key
        )
        test_dataset = UnifiedTrajectoryDataset(
            trajectory_manager, adata, 'test', time_type, split_key
        )

        datasets[time_type] = {
            'train': train_dataset,
            'valid': valid_dataset,
            'test': test_dataset
        }

        train_size = len(train_dataset)
        if train_size > 800:
            optimized_num_workers = min(2, num_workers)
        elif train_size > 100:
            optimized_num_workers = min(2, num_workers)
        else:
            optimized_num_workers = 0

        print(f"{time_type} - Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")

        train_sampler = None
        valid_sampler = None
        test_sampler = None

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=world_size, rank=rank, shuffle=True
            )
            valid_sampler = torch.utils.data.distributed.DistributedSampler(
                valid_dataset, num_replicas=world_size, rank=rank, shuffle=False
            )
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset, num_replicas=world_size, rank=rank, shuffle=False
            )

        dataloader_kwargs = {
            'batch_size': per_gpu_batch_size,
            'num_workers': optimized_num_workers,
            'pin_memory': True if optimized_num_workers > 0 else False,
            'collate_fn': robust_collate_fn,
            'persistent_workers': True if optimized_num_workers > 0 else False,
            'prefetch_factor': 2 if optimized_num_workers > 0 else 2,
            'timeout': 300 if optimized_num_workers > 0 else 0,
            'drop_last': True
        }

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            **dataloader_kwargs
        )

        valid_test_kwargs = dataloader_kwargs.copy()
        valid_test_kwargs['drop_last'] = False

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            shuffle=False,
            sampler=valid_sampler,
            **valid_test_kwargs
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            sampler=test_sampler,
            **valid_test_kwargs
        )

        dataloaders[time_type] = {
            'train': train_loader,
            'valid': valid_loader,
            'test': test_loader
        }

    return dataloaders, datasets


def move_batch_to_device_optimized(batch, device):
    moved_batch = {}

    for key, value in batch.items():
        try:
            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(device, non_blocking=True)
            elif key == 'dose':
                if isinstance(value, torch.Tensor):
                    moved_batch[key] = value.to(device, non_blocking=True)
                elif isinstance(value, list):
                    cleaned_values = []
                    for v in value:
                        if isinstance(v, (int, float)) and not (np.isnan(v) or np.isinf(v)):
                            cleaned_values.append(float(v))
                        else:
                            cleaned_values.append(0.0)
                    moved_batch[key] = torch.tensor(cleaned_values, dtype=torch.float32, device=device)
                else:
                    moved_batch[key] = torch.zeros(1, dtype=torch.float32, device=device)
            else:
                moved_batch[key] = value
        except Exception:
            if key in ['x0', 'smiles', 'trajectory']:
                moved_batch[key] = None
            elif key == 'dose':
                moved_batch[key] = torch.zeros(1, dtype=torch.float32, device=device)
            else:
                moved_batch[key] = value

    return moved_batch


def evaluate_train_set_endpoints(model, dataloaders_dict, device):
    model.eval()

    all_endpoint_pccs = []
    time_types = ['complete', 'partial_6h', 'partial_24h']

    print("  Evaluating training set endpoint PCC...")

    with torch.no_grad():
        for time_type in time_types:
            train_loader = dataloaders_dict[time_type]['valid']

            if len(train_loader) == 0:
                continue

            type_endpoint_pccs = []
            valid_samples = 0
            total_samples = 0

            for batch_idx, batch in enumerate(train_loader):
                try:
                    if batch is None or batch.get('x0') is None or batch['x0'].size(0) == 0:
                        continue

                    batch = move_batch_to_device_optimized(batch, device)
                    batch_size = batch['x0'].size(0)
                    total_samples += batch_size

                    batch_time_types = batch.get('time_type', [])
                    if not batch_time_types:
                        continue

                    if not all(tt == time_type for tt in batch_time_types):
                        print(f"    Warning: batch contains mismatched time_type")
                        continue

                    pred_trajectory_gene = model.forward(
                        x0=batch['x0'],
                        drug_features=batch['smiles'],
                        x6=batch.get('x6'),
                        x24=batch.get('x24'),
                        dose=batch.get('dose'),
                        time_type=time_type,
                        train_mode=False
                    )

                    if pred_trajectory_gene is None:
                        continue

                    if pred_trajectory_gene.dim() != 3:
                        print(f"    Warning: prediction trajectory dimension incorrect: {pred_trajectory_gene.shape}")
                        continue

                    current_batch_size, traj_length, n_genes = pred_trajectory_gene.shape

                    if current_batch_size != batch_size:
                        print(f"    Warning: prediction trajectory batch_size mismatch, expected: {batch_size}, actual: {current_batch_size}")
                        actual_batch_size = min(current_batch_size, batch_size)
                    else:
                        actual_batch_size = batch_size

                    for sample_idx in range(actual_batch_size):
                        sample_pccs = []

                        if time_type == 'complete':
                            if batch.get('x6') is not None and sample_idx < batch['x6'].size(0):
                                h6_idx = max(1, traj_length // 4)
                                if h6_idx < traj_length:
                                    pred_6h_gene = pred_trajectory_gene[sample_idx, h6_idx].cpu().numpy()
                                    true_6h_gene = batch['x6'][sample_idx].cpu().numpy()

                                    pcc_6h = compute_evaluation_metrics(true_6h_gene, pred_6h_gene)['pcc']
                                    if not np.isnan(pcc_6h):
                                        sample_pccs.append(pcc_6h)

                            if batch.get('x24') is not None and sample_idx < batch['x24'].size(0):
                                pred_24h_gene = pred_trajectory_gene[sample_idx, -1].cpu().numpy()
                                true_24h_gene = batch['x24'][sample_idx].cpu().numpy()

                                pcc_24h = compute_evaluation_metrics(true_24h_gene, pred_24h_gene)['pcc']
                                if not np.isnan(pcc_24h):
                                    sample_pccs.append(pcc_24h)

                        elif time_type == 'partial_6h':
                            if batch.get('x6') is not None and sample_idx < batch['x6'].size(0):
                                pred_6h_gene = pred_trajectory_gene[sample_idx, -1].cpu().numpy()
                                true_6h_gene = batch['x6'][sample_idx].cpu().numpy()

                                pcc_6h = compute_evaluation_metrics(true_6h_gene, pred_6h_gene)['pcc']
                                if not np.isnan(pcc_6h):
                                    sample_pccs.append(pcc_6h)

                        elif time_type == 'partial_24h':
                            if batch.get('x24') is not None and sample_idx < batch['x24'].size(0):
                                pred_24h_gene = pred_trajectory_gene[sample_idx, -1].cpu().numpy()
                                true_24h_gene = batch['x24'][sample_idx].cpu().numpy()

                                pcc_24h = compute_evaluation_metrics(true_24h_gene, pred_24h_gene)['pcc']
                                if not np.isnan(pcc_24h):
                                    sample_pccs.append(pcc_24h)

                        if sample_pccs:
                            type_endpoint_pccs.extend(sample_pccs)
                            valid_samples += 1

                except Exception as e:
                    print(f"    Error processing batch {batch_idx}: {e}")
                    continue

            all_endpoint_pccs.extend(type_endpoint_pccs)

            if type_endpoint_pccs:
                avg_pcc = np.mean(type_endpoint_pccs)
                print(f"    {time_type}: {len(type_endpoint_pccs)} endpoints, avg PCC: {avg_pcc:.4f} "
                      f"(valid samples: {valid_samples}/{total_samples})")
            else:
                print(f"    {time_type}: no valid endpoints (total samples: {total_samples})")

    if all_endpoint_pccs:
        overall_avg_pcc = np.mean(all_endpoint_pccs)
        print(f"    Overall training set endpoint avg PCC: {overall_avg_pcc:.4f} (based on {len(all_endpoint_pccs)} endpoints)")
        return overall_avg_pcc
    else:
        print(f"    Training set has no valid endpoints")
        return 0.0


def aggressive_memory_cleanup():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()
    gc.collect()
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except:
        pass


def train_unified_model_optimized(model, dataloaders_dict, args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model = model.to(device)
        model = torch.nn.DataParallel(model)
        print(f"Using {num_gpus} GPUs for training")
    else:
        model = model.to(device)
        print(f"Using single GPU: {device}")

    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=2,
        verbose=True,
        threshold=0.001,
        threshold_mode='abs'
    )
    type_weights = {'complete': 3.0, 'partial_6h': 1.2, 'partial_24h': 2.0}
    time_types = ['complete', 'partial_6h', 'partial_24h']

    history = {
        'train_loss': [],
        'train_endpoint_pcc': [],
        'learning_rate': [],
        'complete_loss': [],
        'partial_6h_loss': [],
        'partial_24h_loss': []
    }

    best_train_endpoint_pcc = -1.0
    patience_counter = 0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = {'complete': [], 'partial_6h': [], 'partial_24h': []}

        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")

        for time_type in time_types:
            if hasattr(dataloaders_dict[time_type]['train'].sampler, 'set_epoch'):
                dataloaders_dict[time_type]['train'].sampler.set_epoch(epoch)

        for time_type in time_types:
            train_loader = dataloaders_dict[time_type]['train']

            print(f"Training {time_type}: {len(train_loader)} batches")

            aggressive_memory_cleanup()

            batch_errors = 0
            successful_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                try:
                    if (batch is None or not isinstance(batch.get('x0'), torch.Tensor) or
                            batch['x0'].numel() == 0):
                        batch_errors += 1
                        continue

                    trajectories = batch.get('trajectory')
                    if trajectories is None or (isinstance(trajectories, torch.Tensor) and trajectories.numel() == 0):
                        batch_errors += 1
                        continue

                    try:
                        batch = move_batch_to_device_optimized(batch, device)
                        if isinstance(trajectories, torch.Tensor):
                            trajectories = trajectories.to(device, non_blocking=True)
                    except Exception as e:
                        print(f"  Device movement error at batch {batch_idx}: {e}")
                        batch_errors += 1
                        continue

                    try:
                        loss = model(
                            x0=batch['x0'],
                            drug_features=batch['smiles'],
                            x6=batch.get('x6'),
                            x24=batch.get('x24'),
                            dose=batch.get('dose'),
                            time_type=time_type,
                            train_mode=True,
                            guided_trajectory=trajectories
                        )

                        if isinstance(loss, torch.Tensor):
                            if loss.numel() == 0:
                                batch_errors += 1
                                continue
                            elif loss.numel() > 1:
                                loss = loss.mean()
                            if torch.isnan(loss) or torch.isinf(loss):
                                batch_errors += 1
                                continue
                        else:
                            batch_errors += 1
                            continue

                        weighted_loss = loss * type_weights[time_type]

                        optimizer.zero_grad()
                        weighted_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                        optimizer.step()

                        loss_scalar = safe_tensor_to_scalar(loss, f"{time_type}_loss")
                        epoch_losses[time_type].append(loss_scalar)
                        successful_batches += 1

                        if batch_idx % 50 == 0:
                            print(f"  Batch {batch_idx}/{len(train_loader)}: loss={loss_scalar:.4f}")

                        if batch_idx % 100 == 0:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                    except Exception as e:
                        print(f"  Training error at batch {batch_idx}: {e}")
                        batch_errors += 1
                        continue
                    finally:
                        if isinstance(trajectories, torch.Tensor):
                            del trajectories
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"  GPU memory error at batch {batch_idx}, cleaning up...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        batch_errors += 1
                        continue
                    else:
                        print(f"  Runtime error at batch {batch_idx}: {e}")
                        batch_errors += 1
                        continue
                except Exception as e:
                    print(f"  Unexpected error at batch {batch_idx}: {e}")
                    batch_errors += 1
                    continue

            if batch_errors > 0:
                print(f"  {time_type}: {batch_errors} batch errors, {successful_batches} successful batches")

            aggressive_memory_cleanup()

            avg_loss = np.mean(epoch_losses[time_type]) if epoch_losses[time_type] else 0
            print(f"  {time_type} average loss: {avg_loss:.4f}")

        print("Validating training set endpoint PCC...")
        try:
            train_endpoint_pcc = evaluate_train_set_endpoints(model, dataloaders_dict, device)
        except Exception as e:
            print(f"Training set endpoint evaluation error: {e}")
            train_endpoint_pcc = 0.0

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(train_endpoint_pcc)
        new_lr = optimizer.param_groups[0]['lr']

        train_loss = np.mean([np.mean(losses) if losses else 0 for losses in epoch_losses.values()])
        history['train_loss'].append(train_loss)
        history['train_endpoint_pcc'].append(train_endpoint_pcc)
        history['learning_rate'].append(new_lr)
        history['complete_loss'].append(np.mean(epoch_losses['complete']) if epoch_losses['complete'] else 0)
        history['partial_6h_loss'].append(np.mean(epoch_losses['partial_6h']) if epoch_losses['partial_6h'] else 0)
        history['partial_24h_loss'].append(np.mean(epoch_losses['partial_24h']) if epoch_losses['partial_24h'] else 0)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Training set endpoint avg PCC: {train_endpoint_pcc:.4f}")
        print(f"Learning Rate: {old_lr:.2e} -> {new_lr:.2e}")
        print("=" * 60)

        if train_endpoint_pcc > best_train_endpoint_pcc:
            best_train_endpoint_pcc = train_endpoint_pcc
            patience_counter = 0
            best_model_path = os.path.join(args.output_dir, 'best_unified_model_optimized.pth')
            try:
                if isinstance(model, torch.nn.DataParallel):
                    model.module.save_model(best_model_path, optimizer, scheduler, epoch, train_loss, args)
                else:
                    model.save_model(best_model_path, optimizer, scheduler, epoch, train_loss, args)
                print(f"✓ New best model saved! Training set endpoint avg PCC: {best_train_endpoint_pcc:.4f}")
            except Exception as e:
                print(f"Model save error: {e}")
        else:
            patience_counter += 1

        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f'unified_checkpoint_epoch_{epoch + 1}_optimized.pth')
            try:
                if isinstance(model, torch.nn.DataParallel):
                    model.module.save_model(checkpoint_path, optimizer, scheduler, epoch, train_loss, args)
                else:
                    model.save_model(checkpoint_path, optimizer, scheduler, epoch, train_loss, args)
                print(f"✓ Checkpoint saved: {checkpoint_path}")
            except Exception as e:
                print(f"Checkpoint save error: {e}")

        if patience_counter >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        aggressive_memory_cleanup()

    history_path = os.path.join(args.output_dir, 'unified_training_history_optimized.json')
    try:
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
    except Exception as e:
        print(f"History save error: {e}")

    print(f"\nTraining completed! Best training set endpoint avg PCC: {best_train_endpoint_pcc:.4f}")
    return model, best_train_endpoint_pcc, history


def compute_single_pcc(args):
    y_true, y_pred = args

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) == 0:
        return 0.0

    try:
        pcc, _ = pearsonr(y_true_clean, y_pred_clean)
        return pcc if not np.isnan(pcc) else 0.0
    except Exception:
        return 0.0


def compute_batch_pcc_serial(y_true_batch, y_pred_batch):
    if torch.is_tensor(y_true_batch):
        y_true_batch = y_true_batch.cpu().numpy()
    if torch.is_tensor(y_pred_batch):
        y_pred_batch = y_pred_batch.cpu().numpy()

    batch_size = y_true_batch.shape[0]
    pcc_list = []

    for i in range(batch_size):
        y_true = y_true_batch[i].flatten()
        y_pred = y_pred_batch[i].flatten()

        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]

        if len(y_true_clean) == 0:
            pcc_list.append(0.0)
        else:
            try:
                pcc, _ = pearsonr(y_true_clean, y_pred_clean)
                pcc_list.append(pcc if not np.isnan(pcc) else 0.0)
            except Exception:
                pcc_list.append(0.0)

    return pcc_list


def compute_parallel_pcc(y_true_batch, y_pred_batch, max_workers=None):
    if torch.is_tensor(y_true_batch):
        y_true_batch = y_true_batch.cpu().numpy()
    if torch.is_tensor(y_pred_batch):
        y_pred_batch = y_pred_batch.cpu().numpy()

    batch_size = y_true_batch.shape[0]

    if batch_size <= 8:
        return compute_batch_pcc_serial(y_true_batch, y_pred_batch)

    if max_workers is None:
        max_workers = min(batch_size, mp.cpu_count())

    args_list = [(y_true_batch[i], y_pred_batch[i]) for i in range(batch_size)]

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            pcc_list = list(executor.map(compute_single_pcc, args_list))
        return pcc_list
    except Exception:
        return compute_batch_pcc_serial(y_true_batch, y_pred_batch)


def final_test_evaluation(model, dataloaders_dict, device, output_dir):
    def safe_mean(values, default_value=0.0):
        if not values:
            return default_value

        filtered_values = [x for x in values if np.isfinite(x) and x != float('inf') and x != float('-inf')]
        if not filtered_values:
            return default_value

        return np.mean(filtered_values)

    def safe_std(values, default_value=0.0):
        if not values:
            return default_value

        filtered_values = [x for x in values if np.isfinite(x) and x != float('inf') and x != float('-inf')]
        if len(filtered_values) < 2:
            return default_value

        return np.std(filtered_values)

    def compute_single_metrics(args):
        y_true, y_pred = args

        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]

        if len(y_true_clean) == 0:
            return {'pcc': 0.0, 'rmse': float('inf'), 'mse': float('inf')}

        try:
            from scipy.stats import pearsonr
            pcc, _ = pearsonr(y_true_clean, y_pred_clean)
            pcc = pcc if not np.isnan(pcc) else 0.0

            from sklearn.metrics import mean_squared_error
            mse = mean_squared_error(y_true_clean, y_pred_clean)
            rmse = np.sqrt(mse)

            return {'pcc': pcc, 'rmse': rmse, 'mse': mse}
        except Exception:
            return {'pcc': 0.0, 'rmse': float('inf'), 'mse': float('inf')}

    def compute_batch_metrics_serial(y_true_batch, y_pred_batch):
        if torch.is_tensor(y_true_batch):
            y_true_batch = y_true_batch.cpu().numpy()
        if torch.is_tensor(y_pred_batch):
            y_pred_batch = y_pred_batch.cpu().numpy()

        batch_size = y_true_batch.shape[0]
        metrics_list = []

        for i in range(batch_size):
            y_true = y_true_batch[i].flatten()
            y_pred = y_pred_batch[i].flatten()

            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]

            if len(y_true_clean) == 0:
                metrics_list.append({'pcc': 0.0, 'rmse': float('inf'), 'mse': float('inf')})
            else:
                try:
                    from scipy.stats import pearsonr
                    from sklearn.metrics import mean_squared_error

                    pcc, _ = pearsonr(y_true_clean, y_pred_clean)
                    pcc = pcc if not np.isnan(pcc) else 0.0

                    mse = mean_squared_error(y_true_clean, y_pred_clean)
                    rmse = np.sqrt(mse)

                    metrics_list.append({'pcc': pcc, 'rmse': rmse, 'mse': mse})
                except Exception:
                    metrics_list.append({'pcc': 0.0, 'rmse': float('inf'), 'mse': float('inf')})

        return metrics_list

    def compute_parallel_metrics(y_true_batch, y_pred_batch, max_workers=None):
        if torch.is_tensor(y_true_batch):
            y_true_batch = y_true_batch.cpu().numpy()
        if torch.is_tensor(y_pred_batch):
            y_pred_batch = y_pred_batch.cpu().numpy()

        batch_size = y_true_batch.shape[0]

        if batch_size <= 8:
            return compute_batch_metrics_serial(y_true_batch, y_pred_batch)

        if max_workers is None:
            import multiprocessing as mp
            max_workers = min(batch_size, mp.cpu_count())

        args_list = [(y_true_batch[i], y_pred_batch[i]) for i in range(batch_size)]

        try:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                metrics_list = list(executor.map(compute_single_metrics, args_list))
            return metrics_list
        except Exception:
            return compute_batch_metrics_serial(y_true_batch, y_pred_batch)

    def move_batch_to_device_optimized(batch, device):
        moved_batch = {}

        for key, value in batch.items():
            try:
                if isinstance(value, torch.Tensor):
                    moved_batch[key] = value.to(device, non_blocking=True)
                elif key == 'dose':
                    if isinstance(value, torch.Tensor):
                        moved_batch[key] = value.to(device, non_blocking=True)
                    elif isinstance(value, list):
                        cleaned_values = []
                        for v in value:
                            if isinstance(v, (int, float)) and not (np.isnan(v) or np.isinf(v)):
                                cleaned_values.append(float(v))
                            else:
                                cleaned_values.append(0.0)
                        moved_batch[key] = torch.tensor(cleaned_values, dtype=torch.float32, device=device)
                    else:
                        moved_batch[key] = torch.zeros(1, dtype=torch.float32, device=device)
                else:
                    moved_batch[key] = value
            except Exception:
                if key in ['x0', 'smiles', 'trajectory']:
                    moved_batch[key] = None
                elif key == 'dose':
                    moved_batch[key] = torch.zeros(1, dtype=torch.float32, device=device)
                else:
                    moved_batch[key] = value

        return moved_batch

    print("Performing final detailed test set evaluation...")
    model.eval()
    model.to(device)

    import numpy as np
    import torch
    import json
    import os

    all_endpoints_pcc = []
    all_endpoints_rmse = []
    all_endpoints_mse = []

    h6_endpoints_pcc = []
    h6_endpoints_rmse = []
    h6_endpoints_mse = []

    h24_endpoints_pcc = []
    h24_endpoints_rmse = []
    h24_endpoints_mse = []

    all_diff_genes_pcc = []
    all_diff_genes_rmse = []
    all_diff_genes_mse = []

    h6_diff_genes_pcc = []
    h6_diff_genes_rmse = []
    h6_diff_genes_mse = []

    h24_diff_genes_pcc = []
    h24_diff_genes_rmse = []
    h24_diff_genes_mse = []

    dose_10_endpoints_pcc = []
    dose_10_endpoints_rmse = []
    dose_10_endpoints_mse = []

    dose_10_diff_genes_pcc = []
    dose_10_diff_genes_rmse = []
    dose_10_diff_genes_mse = []

    dose_10h24_endpoints_pcc = []
    dose_10h24_endpoints_rmse = []
    dose_10h24_endpoints_mse = []

    dose_10h24_diff_genes_pcc = []
    dose_10h24_diff_genes_rmse = []
    dose_10h24_diff_genes_mse = []

    time_types = ['complete', 'partial_6h', 'partial_24h']

    type_stats = {
        'complete': {'samples': 0, '6h_endpoints': 0, '24h_endpoints': 0},
        'partial_6h': {'samples': 0, '6h_endpoints': 0, '24h_endpoints': 0},
        'partial_24h': {'samples': 0, '6h_endpoints': 0, '24h_endpoints': 0}
    }

    dose_10_count = 0
    dose_10_h24_count = 0
    total_samples_count = 0

    print("Processing test data...")

    with torch.no_grad():
        for time_type in time_types:
            test_loader = dataloaders_dict[time_type]['test']

            if len(test_loader) == 0:
                print(f"  {time_type}: No test data")
                continue

            print(f"  Processing {time_type} dataset...")

            batch_true_6h_list = []
            batch_pred_6h_list = []
            batch_true_24h_list = []
            batch_pred_24h_list = []
            batch_true_diff_6h_list = []
            batch_pred_diff_6h_list = []
            batch_true_diff_24h_list = []
            batch_pred_diff_24h_list = []

            batch_dose_6h_list = []
            batch_dose_24h_list = []
            batch_dose_diff_6h_list = []
            batch_dose_diff_24h_list = []

            processed_batches = 0
            error_batches = 0

            for batch_idx, batch in enumerate(test_loader):
                try:
                    if batch is None or batch.get('x0') is None or batch['x0'].size(0) == 0:
                        error_batches += 1
                        continue

                    batch = move_batch_to_device_optimized(batch, device)
                    batch_size = batch['x0'].size(0)

                    batch_time_types = batch.get('time_type', [])
                    if not batch_time_types or not all(tt == time_type for tt in batch_time_types):
                        error_batches += 1
                        continue

                    batch_doses = batch.get('dose', torch.zeros(batch_size))
                    if torch.is_tensor(batch_doses):
                        batch_doses = batch_doses.cpu().numpy()
                    elif isinstance(batch_doses, (list, tuple)):
                        batch_doses = np.array(batch_doses, dtype=np.float32)
                    else:
                        batch_doses = np.zeros(batch_size, dtype=np.float32)

                    pred_trajectory_gene = model.forward(
                        x0=batch['x0'],
                        drug_features=batch['smiles'],
                        x6=batch.get('x6'),
                        x24=batch.get('x24'),
                        dose=batch.get('dose'),
                        time_type=time_type,
                        train_mode=False
                    )

                    if pred_trajectory_gene is None:
                        error_batches += 1
                        continue

                    if pred_trajectory_gene.dim() != 3:
                        print(f"    Warning: prediction trajectory dimension incorrect: {pred_trajectory_gene.shape}")
                        error_batches += 1
                        continue

                    current_batch_size, traj_length, n_genes = pred_trajectory_gene.shape

                    if current_batch_size != batch_size:
                        print(f"    Warning: prediction trajectory batch_size mismatch, expected: {batch_size}, actual: {current_batch_size}")
                        actual_batch_size = min(current_batch_size, batch_size)
                    else:
                        actual_batch_size = batch_size

                    true_x0_batch = batch['x0'][:actual_batch_size].cpu().numpy()
                    total_samples_count += actual_batch_size

                    if time_type == 'complete':
                        if batch.get('x6') is not None:
                            h6_idx = max(1, traj_length // 4)
                            if h6_idx < traj_length:
                                true_x6_batch = batch['x6'][:actual_batch_size].cpu().numpy()
                                pred_x6_batch = pred_trajectory_gene[:actual_batch_size, h6_idx].cpu().numpy()

                                true_diff_6h_batch = true_x6_batch - true_x0_batch
                                pred_diff_6h_batch = pred_x6_batch - true_x0_batch

                                batch_true_6h_list.append(true_x6_batch)
                                batch_pred_6h_list.append(pred_x6_batch)
                                batch_true_diff_6h_list.append(true_diff_6h_batch)
                                batch_pred_diff_6h_list.append(pred_diff_6h_batch)
                                batch_dose_6h_list.append(batch_doses)
                                batch_dose_diff_6h_list.append(batch_doses)

                                type_stats[time_type]['6h_endpoints'] += actual_batch_size

                        if batch.get('x24') is not None:
                            true_x24_batch = batch['x24'][:actual_batch_size].cpu().numpy()
                            pred_x24_batch = pred_trajectory_gene[:actual_batch_size, -1].cpu().numpy()

                            true_diff_24h_batch = true_x24_batch - true_x0_batch
                            pred_diff_24h_batch = pred_x24_batch - true_x0_batch

                            batch_true_24h_list.append(true_x24_batch)
                            batch_pred_24h_list.append(pred_x24_batch)
                            batch_true_diff_24h_list.append(true_diff_24h_batch)
                            batch_pred_diff_24h_list.append(pred_diff_24h_batch)
                            batch_dose_24h_list.append(batch_doses)
                            batch_dose_diff_24h_list.append(batch_doses)

                            type_stats[time_type]['24h_endpoints'] += actual_batch_size

                    elif time_type == 'partial_6h':
                        if batch.get('x6') is not None:
                            true_x6_batch = batch['x6'][:actual_batch_size].cpu().numpy()
                            pred_x6_batch = pred_trajectory_gene[:actual_batch_size, -1].cpu().numpy()

                            true_diff_6h_batch = true_x6_batch - true_x0_batch
                            pred_diff_6h_batch = pred_x6_batch - true_x0_batch

                            batch_true_6h_list.append(true_x6_batch)
                            batch_pred_6h_list.append(pred_x6_batch)
                            batch_true_diff_6h_list.append(true_diff_6h_batch)
                            batch_pred_diff_6h_list.append(pred_diff_6h_batch)
                            batch_dose_6h_list.append(batch_doses)
                            batch_dose_diff_6h_list.append(batch_doses)

                            type_stats[time_type]['6h_endpoints'] += actual_batch_size

                    elif time_type == 'partial_24h':
                        if batch.get('x24') is not None:
                            true_x24_batch = batch['x24'][:actual_batch_size].cpu().numpy()
                            pred_x24_batch = pred_trajectory_gene[:actual_batch_size, -1].cpu().numpy()

                            true_diff_24h_batch = true_x24_batch - true_x0_batch
                            pred_diff_24h_batch = pred_x24_batch - true_x0_batch

                            batch_true_24h_list.append(true_x24_batch)
                            batch_pred_24h_list.append(pred_x24_batch)
                            batch_true_diff_24h_list.append(true_diff_24h_batch)
                            batch_pred_diff_24h_list.append(pred_diff_24h_batch)
                            batch_dose_24h_list.append(batch_doses)
                            batch_dose_diff_24h_list.append(batch_doses)

                            type_stats[time_type]['24h_endpoints'] += actual_batch_size

                    type_stats[time_type]['samples'] += actual_batch_size
                    processed_batches += 1

                except Exception as e:
                    print(f"    Error processing batch {batch_idx}: {e}")
                    error_batches += 1
                    continue

            print(f"    {time_type}: processed {processed_batches} batches, {error_batches} batch errors")
            print(f"    Computing PCC, RMSE, MSE metrics in parallel...")

            if batch_true_6h_list:
                all_true_6h = np.concatenate(batch_true_6h_list, axis=0)
                all_pred_6h = np.concatenate(batch_pred_6h_list, axis=0)
                all_doses_6h = np.concatenate(batch_dose_6h_list, axis=0)
                print(f"      Computing 6h endpoint metrics: {all_true_6h.shape[0]} samples")

                metrics_6h_results = compute_parallel_metrics(all_true_6h, all_pred_6h)

                pcc_6h_results = [m['pcc'] for m in metrics_6h_results]
                rmse_6h_results = [m['rmse'] for m in metrics_6h_results]
                mse_6h_results = [m['mse'] for m in metrics_6h_results]

                h6_endpoints_pcc.extend(pcc_6h_results)
                h6_endpoints_rmse.extend(rmse_6h_results)
                h6_endpoints_mse.extend(mse_6h_results)

                all_endpoints_pcc.extend(pcc_6h_results)
                all_endpoints_rmse.extend(rmse_6h_results)
                all_endpoints_mse.extend(mse_6h_results)

                for i, (pcc_val, rmse_val, mse_val) in enumerate(zip(pcc_6h_results, rmse_6h_results, mse_6h_results)):
                    if abs(all_doses_6h[i] - 10.0) < 0.1:
                        dose_10_endpoints_pcc.append(pcc_val)
                        dose_10_endpoints_rmse.append(rmse_val)
                        dose_10_endpoints_mse.append(mse_val)
                        dose_10_count += 1

            if batch_true_24h_list:
                all_true_24h = np.concatenate(batch_true_24h_list, axis=0)
                all_pred_24h = np.concatenate(batch_pred_24h_list, axis=0)
                all_doses_24h = np.concatenate(batch_dose_24h_list, axis=0)
                print(f"      Computing 24h endpoint metrics: {all_true_24h.shape[0]} samples")

                metrics_24h_results = compute_parallel_metrics(all_true_24h, all_pred_24h)

                pcc_24h_results = [m['pcc'] for m in metrics_24h_results]
                rmse_24h_results = [m['rmse'] for m in metrics_24h_results]
                mse_24h_results = [m['mse'] for m in metrics_24h_results]

                h24_endpoints_pcc.extend(pcc_24h_results)
                h24_endpoints_rmse.extend(rmse_24h_results)
                h24_endpoints_mse.extend(mse_24h_results)

                all_endpoints_pcc.extend(pcc_24h_results)
                all_endpoints_rmse.extend(rmse_24h_results)
                all_endpoints_mse.extend(mse_24h_results)

                for i, (pcc_val, rmse_val, mse_val) in enumerate(
                        zip(pcc_24h_results, rmse_24h_results, mse_24h_results)):
                    if abs(all_doses_24h[i] - 10.0) < 0.000001:
                        dose_10_endpoints_pcc.append(pcc_val)
                        dose_10h24_endpoints_pcc.append(pcc_val)
                        dose_10_endpoints_rmse.append(rmse_val)
                        dose_10h24_endpoints_rmse.append(rmse_val)
                        dose_10_endpoints_mse.append(mse_val)
                        dose_10h24_endpoints_mse.append(mse_val)
                        dose_10_count += 1
                        dose_10_h24_count+=1

            if batch_true_diff_6h_list:
                all_true_diff_6h = np.concatenate(batch_true_diff_6h_list, axis=0)
                all_pred_diff_6h = np.concatenate(batch_pred_diff_6h_list, axis=0)
                all_doses_diff_6h = np.concatenate(batch_dose_diff_6h_list, axis=0)
                print(f"      Computing 6h differential gene metrics: {all_true_diff_6h.shape[0]} samples")

                diff_metrics_6h_results = compute_parallel_metrics(all_true_diff_6h, all_pred_diff_6h)

                diff_pcc_6h_results = [m['pcc'] for m in diff_metrics_6h_results]
                diff_rmse_6h_results = [m['rmse'] for m in diff_metrics_6h_results]
                diff_mse_6h_results = [m['mse'] for m in diff_metrics_6h_results]

                h6_diff_genes_pcc.extend(diff_pcc_6h_results)
                h6_diff_genes_rmse.extend(diff_rmse_6h_results)
                h6_diff_genes_mse.extend(diff_mse_6h_results)

                all_diff_genes_pcc.extend(diff_pcc_6h_results)
                all_diff_genes_rmse.extend(diff_rmse_6h_results)
                all_diff_genes_mse.extend(diff_mse_6h_results)

                for i, (pcc_val, rmse_val, mse_val) in enumerate(
                        zip(diff_pcc_6h_results, diff_rmse_6h_results, diff_mse_6h_results)):
                    if abs(all_doses_diff_6h[i] - 10.0) < 0.1:
                        dose_10_diff_genes_pcc.append(pcc_val)
                        dose_10_diff_genes_rmse.append(rmse_val)
                        dose_10_diff_genes_mse.append(mse_val)

            if batch_true_diff_24h_list:
                all_true_diff_24h = np.concatenate(batch_true_diff_24h_list, axis=0)
                all_pred_diff_24h = np.concatenate(batch_pred_diff_24h_list, axis=0)
                all_doses_diff_24h = np.concatenate(batch_dose_diff_24h_list, axis=0)
                print(f"      Computing 24h differential gene metrics: {all_true_diff_24h.shape[0]} samples")

                diff_metrics_24h_results = compute_parallel_metrics(all_true_diff_24h, all_pred_diff_24h)

                diff_pcc_24h_results = [m['pcc'] for m in diff_metrics_24h_results]
                diff_rmse_24h_results = [m['rmse'] for m in diff_metrics_24h_results]
                diff_mse_24h_results = [m['mse'] for m in diff_metrics_24h_results]

                h24_diff_genes_pcc.extend(diff_pcc_24h_results)
                h24_diff_genes_rmse.extend(diff_rmse_24h_results)
                h24_diff_genes_mse.extend(diff_mse_24h_results)

                all_diff_genes_pcc.extend(diff_pcc_24h_results)
                all_diff_genes_rmse.extend(diff_rmse_24h_results)
                all_diff_genes_mse.extend(diff_mse_24h_results)

                for i, (pcc_val, rmse_val, mse_val) in enumerate(
                        zip(diff_pcc_24h_results, diff_rmse_24h_results, diff_mse_24h_results)):
                    if abs(all_doses_diff_24h[i] - 10.0) < 0.00000001:
                        dose_10_diff_genes_pcc.append(pcc_val)
                        dose_10_diff_genes_rmse.append(rmse_val)
                        dose_10_diff_genes_mse.append(mse_val)
                        dose_10h24_diff_genes_pcc.append(pcc_val)
                        dose_10h24_diff_genes_rmse.append(rmse_val)
                        dose_10h24_diff_genes_mse.append(mse_val)

    print("\n" + "=" * 80)
    print("Final Test Set Evaluation Results")
    print("=" * 80)

    if all_endpoints_pcc:
        overall_endpoint_pcc = safe_mean(all_endpoints_pcc)
        overall_endpoint_rmse = safe_mean(all_endpoints_rmse)
        overall_endpoint_mse = safe_mean(all_endpoints_mse)
        print(f"\n1. All concentration endpoint average metrics:")
        print(f"   Average PCC: {overall_endpoint_pcc:.6f}")
        print(f"   Average RMSE: {overall_endpoint_rmse:.6f}")
        print(f"   Average MSE: {overall_endpoint_mse:.6f}")
        print(f"   Total endpoints: {len(all_endpoints_pcc)}")
        print(f"   PCC std: {safe_std(all_endpoints_pcc):.6f}")
    else:
        overall_endpoint_pcc = 0.0
        overall_endpoint_rmse = float('inf')
        overall_endpoint_mse = float('inf')
        print(f"\n1. All concentration endpoint average metrics: No valid endpoints")

    if dose_10_endpoints_pcc:
        dose_10_avg_pcc = safe_mean(dose_10_endpoints_pcc)
        dose_10_avg_rmse = safe_mean(dose_10_endpoints_rmse)
        dose_10_avg_mse = safe_mean(dose_10_endpoints_mse)
        print(f"\n2. Dose 10 endpoint average metrics:")
        print(f"   Average PCC: {dose_10_avg_pcc:.6f}")
        print(f"   Average RMSE: {dose_10_avg_rmse:.6f}")
        print(f"   Average MSE: {dose_10_avg_mse:.6f}")
        print(f"   Dose 10 endpoints: {len(dose_10_endpoints_pcc)}")
        print(f"   PCC std: {safe_std(dose_10_endpoints_pcc):.6f}")
    else:
        dose_10_avg_pcc = 0.0
        dose_10_avg_rmse = float('inf')
        dose_10_avg_mse = float('inf')
        print(f"\n2. Dose 10 endpoint average metrics: No valid endpoints")

    if dose_10_endpoints_pcc:
        dose_10h24_avg_pcc = safe_mean(dose_10h24_endpoints_pcc)
        dose_10h24_avg_rmse = safe_mean(dose_10h24_endpoints_rmse)
        dose_10h24_avg_mse = safe_mean(dose_10h24_endpoints_mse)
        print(f"\n2. Dose 10, 24h endpoint average metrics:")
        print(f"   Average PCC: {dose_10h24_avg_pcc:.6f}")
        print(f"   Average RMSE: {dose_10h24_avg_rmse:.6f}")
        print(f"   Average MSE: {dose_10h24_avg_mse:.6f}")
        print(f"   Dose 10 endpoints: {len(dose_10h24_endpoints_pcc)}")
        print(f"   PCC std: {safe_std(dose_10h24_endpoints_pcc):.6f}")
    else:
        dose_10h24_avg_pcc = 0.0
        dose_10h24_avg_rmse = float('inf')
        dose_10h24_avg_mse = float('inf')
        print(f"\n2. Dose 10, 24h endpoint average metrics: No valid endpoints")

    if h24_endpoints_pcc:
        h24_avg_pcc = safe_mean(h24_endpoints_pcc)
        h24_avg_rmse = safe_mean(h24_endpoints_rmse)
        h24_avg_mse = safe_mean(h24_endpoints_mse)
        print(f"\n3. 24h endpoint average metrics:")
        print(f"   Average PCC: {h24_avg_pcc:.6f}")
        print(f"   Average RMSE: {h24_avg_rmse:.6f}")
        print(f"   Average MSE: {h24_avg_mse:.6f}")
        print(f"   24h endpoints: {len(h24_endpoints_pcc)}")
        print(f"   PCC std: {safe_std(h24_endpoints_pcc):.6f}")
    else:
        h24_avg_pcc = 0.0
        h24_avg_rmse = float('inf')
        h24_avg_mse = float('inf')
        print(f"\n3. 24h endpoint average metrics: No valid 24h endpoints")

    if h6_endpoints_pcc:
        h6_avg_pcc = safe_mean(h6_endpoints_pcc)
        h6_avg_rmse = safe_mean(h6_endpoints_rmse)
        h6_avg_mse = safe_mean(h6_endpoints_mse)
        print(f"\n4. 6h endpoint average metrics:")
        print(f"   Average PCC: {h6_avg_pcc:.6f}")
        print(f"   Average RMSE: {h6_avg_rmse:.6f}")
        print(f"   Average MSE: {h6_avg_mse:.6f}")
        print(f"   6h endpoints: {len(h6_endpoints_pcc)}")
        print(f"   PCC std: {safe_std(h6_endpoints_pcc):.6f}")
    else:
        h6_avg_pcc = 0.0
        h6_avg_rmse = float('inf')
        h6_avg_mse = float('inf')
        print(f"\n4. 6h endpoint average metrics: No valid 6h endpoints")

    print(f"\n5. Differential gene metrics results (endpoint-start vs true endpoint-true start):")

    if all_diff_genes_pcc:
        overall_diff_pcc = safe_mean(all_diff_genes_pcc)
        overall_diff_rmse = safe_mean(all_diff_genes_rmse)
        overall_diff_mse = safe_mean(all_diff_genes_mse)
        print(f"   All differential genes average PCC: {overall_diff_pcc:.6f}")
        print(f"   All differential genes average RMSE: {overall_diff_rmse:.6f}")
        print(f"   All differential genes average MSE: {overall_diff_mse:.6f}")
        print(f"   Total differential genes: {len(all_diff_genes_pcc)}")
        print(f"   PCC std: {safe_std(all_diff_genes_pcc):.6f}")
    else:
        overall_diff_pcc = 0.0
        overall_diff_rmse = float('inf')
        overall_diff_mse = float('inf')
        print(f"   All differential genes average metrics: No valid data")

    if dose_10_diff_genes_pcc:
        dose_10_diff_avg_pcc = safe_mean(dose_10_diff_genes_pcc)
        dose_10_diff_avg_rmse = safe_mean(dose_10_diff_genes_rmse)
        dose_10_diff_avg_mse = safe_mean(dose_10_diff_genes_mse)
        print(f"   Dose 10 differential genes average PCC: {dose_10_diff_avg_pcc:.6f}")
        print(f"   Dose 10 differential genes average RMSE: {dose_10_diff_avg_rmse:.6f}")
        print(f"   Dose 10 differential genes average MSE: {dose_10_diff_avg_mse:.6f}")
        print(f"   Dose 10 differential genes: {len(dose_10_diff_genes_pcc)}")
        print(f"   PCC std: {safe_std(dose_10_diff_genes_pcc):.6f}")
    else:
        dose_10_diff_avg_pcc = 0.0
        dose_10_diff_avg_rmse = float('inf')
        dose_10_diff_avg_mse = float('inf')
        print(f"   Dose 10 differential genes average metrics: No valid data")

    if dose_10h24_diff_genes_pcc:
        dose_10h24_diff_avg_pcc = safe_mean(dose_10h24_diff_genes_pcc)
        dose_10h24_diff_avg_rmse = safe_mean(dose_10h24_diff_genes_rmse)
        dose_10h24_diff_avg_mse = safe_mean(dose_10h24_diff_genes_mse)
        print(f"   Dose 10, 24h differential genes average PCC: {dose_10h24_diff_avg_pcc:.6f}")
        print(f"   Dose 10, 24h differential genes average RMSE: {dose_10h24_diff_avg_rmse:.6f}")
        print(f"   Dose 10, 24h differential genes average MSE: {dose_10h24_diff_avg_mse:.6f}")
        print(f"   Dose 10, 24h differential genes: {len(dose_10h24_diff_genes_pcc)}")
        print(f"   PCC std: {safe_std(dose_10h24_diff_genes_pcc):.6f}")
    else:
        dose_10h24_diff_avg_pcc = 0.0
        dose_10h24_diff_avg_rmse = float('inf')
        dose_10h24_diff_avg_mse = float('inf')
        print(f"   Dose 10, 24h differential genes average metrics: No valid data")

    print(f"\n6. Dose statistics:")
    print(f"   Total samples: {total_samples_count}")
    print(f"   Dose 10 endpoints: {dose_10_count}")
    print(f"   Dose 10 ratio: {dose_10_count / max(total_samples_count, 1) * 100:.2f}%")

    print(f"\n7. Detailed statistics by time type:")
    for time_type, stats in type_stats.items():
        print(f"   {time_type}:")
        print(f"     Processed samples: {stats['samples']}")
        print(f"     6h endpoints: {stats['6h_endpoints']}")
        print(f"     24h endpoints: {stats['24h_endpoints']}")

    print(f"\n8. Hardware info:")
    print(f"   GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")

    print("=" * 80)

    print(safe_mean(h24_diff_genes_pcc))
    print('h24_diff_genes_avg_pcc:',safe_mean(h24_diff_genes_pcc))
    print('h6_diff_genes_avg_pcc:', safe_mean(h6_diff_genes_pcc),)
    print('h24_diff_genes_avg_rmse: ',safe_mean(h24_diff_genes_rmse))
    print('h6_diff_genes_avg_rmse:' ,safe_mean(h6_diff_genes_rmse))

    test_results = {
        'all_endpoints_avg_pcc': overall_endpoint_pcc,
        'dose_10_endpoints_avg_pcc': dose_10_avg_pcc,
        'h24_endpoints_avg_pcc': h24_avg_pcc,
        'h6_endpoints_avg_pcc': h6_avg_pcc,

        'all_endpoints_avg_rmse': overall_endpoint_rmse,
        'dose_10_endpoints_avg_rmse': dose_10_avg_rmse,
        'h24_endpoints_avg_rmse': h24_avg_rmse,
        'h6_endpoints_avg_rmse': h6_avg_rmse,

        'all_endpoints_avg_mse': overall_endpoint_mse,
        'dose_10_endpoints_avg_mse': dose_10_avg_mse,
        'h24_endpoints_avg_mse': h24_avg_mse,
        'h6_endpoints_avg_mse': h6_avg_mse,

        'all_diff_genes_avg_pcc': overall_diff_pcc,
        'dose_10_diff_genes_avg_pcc': dose_10_diff_avg_pcc,
        'h24_diff_genes_avg_pcc': safe_mean(h24_diff_genes_pcc),
        'h6_diff_genes_avg_pcc': safe_mean(h6_diff_genes_pcc),

        'all_diff_genes_avg_rmse': overall_diff_rmse,
        'dose_10_diff_genes_avg_rmse': dose_10_diff_avg_rmse,
        'h24_diff_genes_avg_rmse': safe_mean(h24_diff_genes_rmse),
        'h6_diff_genes_avg_rmse': safe_mean(h6_diff_genes_rmse),

        'all_diff_genes_avg_mse': overall_diff_mse,
        'dose_10_diff_genes_avg_mse': dose_10_diff_avg_mse,
        'h24_diff_genes_avg_mse': safe_mean(h24_diff_genes_mse),
        'h6_diff_genes_avg_mse': safe_mean(h6_diff_genes_mse),

        'total_endpoints': len(all_endpoints_pcc),
        'dose_10_endpoints_count': len(dose_10_endpoints_pcc),
        'total_h6_endpoints': len(h6_endpoints_pcc),
        'total_h24_endpoints': len(h24_endpoints_pcc),
        'total_diff_genes': len(all_diff_genes_pcc),
        'dose_10_diff_genes_count': len(dose_10_diff_genes_pcc),

        'all_endpoints_pcc_std': safe_std(all_endpoints_pcc),
        'dose_10_endpoints_pcc_std': safe_std(dose_10_endpoints_pcc),
        'h24_endpoints_pcc_std': safe_std(h24_endpoints_pcc),
        'h6_endpoints_pcc_std': safe_std(h6_endpoints_pcc),
        'all_diff_genes_pcc_std': safe_std(all_diff_genes_pcc),
        'dose_10_diff_genes_pcc_std': safe_std(dose_10_diff_genes_pcc),

        'all_endpoints_rmse_std': safe_std(all_endpoints_rmse),
        'dose_10_endpoints_rmse_std': safe_std(dose_10_endpoints_rmse),
        'h24_endpoints_rmse_std': safe_std(h24_endpoints_rmse),
        'h6_endpoints_rmse_std': safe_std(h6_endpoints_rmse),

        'all_endpoints_mse_std': safe_std(all_endpoints_mse),
        'dose_10_endpoints_mse_std': safe_std(dose_10_endpoints_mse),
        'h24_endpoints_mse_std': safe_std(h24_endpoints_mse),
        'h6_endpoints_mse_std': safe_std(h6_endpoints_mse),

        'total_samples_count': total_samples_count,
        'dose_10_samples_count': dose_10_count,
        'dose_10_ratio': dose_10_count / max(total_samples_count, 1),

        'type_statistics': type_stats,

        'hardware_info': {
            'num_gpus_used': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available() else []
        }
    }

    test_report_path = os.path.join(output_dir, 'comprehensive_test_evaluation_report.json')
    try:
        with open(test_report_path, 'w') as f:
            json.dump(test_results, f, indent=4)
        print(f"\nComprehensive test report (with PCC, RMSE, MSE metrics) saved to: {test_report_path}")
    except Exception as e:
        print(f"Error saving test report: {e}")

    return test_results


def main():
    parser = argparse.ArgumentParser(description="Optimized Unified Trajectory Training")
    parser.add_argument("--adata-path", type=str, default="../../dataset/Lincs_L1000_with_pairs_splits.h5ad")
    parser.add_argument("--timeseries-path", type=str, default="../../dataset/L1000_0_6_24.csv")
    parser.add_argument("--split-key", type=str, default="drug_splits_4")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--memory-dir", type=str, default="./trajectory_memory")
    parser.add_argument("--max-shards", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="./checkpoint_unified_optimized")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--multi-gpu", action="store_true", help="Force multi-GPU training", default=True)
    parser.add_argument("--gpu-ids", type=str, default="", help="Comma-separated GPU IDs to use")
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.split_key)

    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    print(f"GPU Setup:")
    print(f"  Available GPUs: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({gpu_memory:.1f}GB)")

    print("Loading data...")
    adata = sc.read(args.adata_path)
    args.normalize = True
    if args.normalize:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
    timeseries_data = pd.read_csv(args.timeseries_path)
    print(f"  AnnData shape: {adata.shape}")
    print(f"  Timeseries data: {timeseries_data.shape}")

    print("Creating trajectory manager...")
    trajectory_manager = UnifiedTrajectoryManager(
        memory_dir=args.memory_dir,
        adata=adata,
        timeseries_data=timeseries_data,
        split_key=args.split_key,
        verbose=False
    )

    if trajectory_manager.load_from_cache():
        print("Using cached trajectories")
    else:
        print("Loading and processing trajectories...")
        trajectory_manager.load_all_shards(max_shards=args.max_shards)

    print(f"Smart multiprocessing: adapting workers based on dataset size")
    print(f"Batch size: {args.batch_size}")

    print("Creating intelligent dataloaders...")
    dataloaders_dict, datasets_dict = create_unified_dataloaders(
        trajectory_manager, adata, args.batch_size, args.num_workers,
        split_key=args.split_key
    )

    total_train = sum(len(datasets_dict[tt]['train']) for tt in ['complete', 'partial_6h', 'partial_24h'])
    total_valid = sum(len(datasets_dict[tt]['valid']) for tt in ['complete', 'partial_6h', 'partial_24h'])
    total_test = sum(len(datasets_dict[tt]['test']) for tt in ['complete', 'partial_6h', 'partial_24h'])
    print(f"Dataset Summary:")
    print(f"  Total - Train: {total_train}, Valid: {total_valid}, Test: {total_test}")

    print("Creating model...")
    model_config = {
        "n_genes": adata.shape[1],
        "n_latent": args.latent_dim,
        "features_dim": 2304,
        "timesteps": args.timesteps,
        "dropout": args.dropout
    }
    model = TrajectoryGuidedProgressiveModel(
        n_genes=model_config["n_genes"],
        n_latent=model_config["n_latent"],
        features_dim=model_config["features_dim"],
        timesteps=model_config["timesteps"],
        dropout=model_config["dropout"],
        device=args.device
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    print("Starting training with train set endpoint validation...")
    model, best_train_endpoint_pcc, history = train_unified_model_optimized(model, dataloaders_dict, args)

    print("Loading best model for final evaluation...")

    best_model_path = os.path.join(args.output_dir, 'best_unified_model_optimized.pth')
    if best_model_path:
        try:
            print(f"  Loading model from: {best_model_path}")
            loaded_info = model.load_model(best_model_path)
            print(f"  Best model loaded successfully!")

            if 'epoch' in loaded_info:
                print(f"    Model epoch: {loaded_info['epoch']}")
            if 'valid_loss' in loaded_info:
                print(f"    Model validation loss: {loaded_info['valid_loss']:.6f}")

        except Exception as e:
            print(f"  Failed to load best model: {e}")
            print(f"  Will use current model state for evaluation")
    else:
        print(f"  No best model found, using current model state for evaluation")

    print("Performing final test evaluation...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    final_test_results = final_test_evaluation(model, dataloaders_dict, device, args.output_dir)

    print(f"\nTraining completed!")
    print(f"  Final test set endpoint avg PCC: {final_test_results['all_endpoints_avg_pcc']:.6f}")
    if 'endpoint_avg_rmse' in final_test_results:
        print(f"  Final test set endpoint avg RMSE: {final_test_results['endpoint_avg_rmse']:.6f}")

    return model, trajectory_manager, final_test_results


if __name__ == "__main__":
    main()