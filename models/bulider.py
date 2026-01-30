import os
import torch
import torch.nn.functional as F
import numpy as np
import pickle
import time
from tqdm import tqdm
import pandas as pd
import gc


class EnhancedTrajectoryBuilder:

    def __init__(self, transform_net=None, condition_fusion=None,
                 timesteps=50, n_latent=64, memory_dir="./trajectory_memory"):
        self.transform_net = transform_net
        self.condition_fusion = condition_fusion
        self.timesteps = timesteps
        self.n_latent = n_latent
        self.memory_dir = memory_dir
        self.verbose = True

        self.trajectory_storage = {}
        self.metadata_df = pd.DataFrame()
        self.current_numeric_index = 0
        self.save_counter = 0
        self.save_interval = 10000

        self.temp_storage = {}
        self.saved_to_disk = set()

        self.total_optimized = 0
        self.optimization_stats = {
            'complete': 0,
            'partial_6h': 0,
            'partial_24h': 0
        }

        os.makedirs(memory_dir, exist_ok=True)

    def _generate_composite_index(self, sample_data):
        smiles_str = sample_data.get('smiles_str', 'unknown')
        time_type = sample_data.get('time_type', 'unknown')
        dose = sample_data.get('dose', 0.0)
        cell_id = sample_data.get('cell_id', 'unknown')
        ctrl_index = sample_data.get('data_idx', 'unknown')

        clean_smiles = str(smiles_str).replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace(
            '?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
        clean_smiles = clean_smiles[:50]

        dose_str = f"{float(dose):.3f}"
        clean_cell = str(cell_id).replace('/', '_').replace('\\', '_').replace(':', '_')

        composite_index = f"{clean_smiles}_{time_type}_{dose_str}_{clean_cell}_{ctrl_index}"

        return composite_index

    def _determine_endpoint_index(self, sample_data):
        time_type = sample_data.get('time_type', 'unknown')

        if time_type == "complete":
            endpoint_index = sample_data.get('endpoint_24h_index')
        elif time_type == "partial_6h":
            endpoint_index = sample_data.get('endpoint_6h_index')
        elif time_type == "partial_24h":
            endpoint_index = sample_data.get('endpoint_24h_index')
        else:
            endpoint_index = None

        return endpoint_index

    def init_trajectory(self, x1, x2=None, x6=None, time_type="complete"):
        batch_size = x1.shape[0]
        device = x1.device
        n_latent = x1.shape[1]

        if time_type == "partial_6h":
            traj_length = self.timesteps // 2 + 1
        else:
            traj_length = self.timesteps

        trajectory = torch.zeros((traj_length, batch_size, n_latent), device=device)
        trajectory[0] = x1

        if time_type == "complete" and x6 is not None and x2 is not None:
            mid_point = self.timesteps // 2
            trajectory[mid_point] = x6
            trajectory[-1] = x2

            for t in range(1, mid_point):
                alpha = t / mid_point
                alpha_smooth = alpha * alpha * (3 - 2 * alpha)
                trajectory[t] = (1 - alpha_smooth) * x1 + alpha_smooth * x6

            for t in range(mid_point + 1, traj_length):
                alpha = (t - mid_point) / (traj_length - 1 - mid_point)
                h00 = 2 * alpha ** 3 - 3 * alpha ** 2 + 1
                h10 = alpha ** 3 - 2 * alpha ** 2 + alpha
                h01 = -2 * alpha ** 3 + 3 * alpha ** 2
                h11 = alpha ** 3 - alpha ** 2

                if mid_point > 0:
                    p0_deriv = (x6 - x1) * (traj_length - mid_point) / traj_length
                else:
                    p0_deriv = (x6 - x1)

                p1_deriv = (x2 - x6) * mid_point / traj_length
                trajectory[t] = h00 * x6 + h10 * p0_deriv + h01 * x2 + h11 * p1_deriv

        else:
            if x2 is not None:
                trajectory[-1] = x2
                for t in range(1, traj_length - 1):
                    alpha = t / (traj_length - 1)
                    alpha_smooth = alpha * alpha * (3 - 2 * alpha)
                    trajectory[t] = (1 - alpha_smooth) * x1 + alpha_smooth * x2
            else:
                trajectory[1:] = x1.unsqueeze(0) + torch.zeros(traj_length - 1, batch_size, n_latent, device=device)

        noise = torch.randn_like(trajectory) * 0.001
        noise[0] = 0
        if time_type == "complete" and x6 is not None:
            mid_point = self.timesteps // 2
            if mid_point < traj_length:
                noise[mid_point] = 0
        if x2 is not None:
            noise[-1] = 0
        trajectory = trajectory + noise

        return trajectory

    def _compute_single_sample_loss(self, trajectory, x1, x2, time_type):
        x1 = x1.squeeze(0) if x1.dim() > 1 else x1
        if x2 is not None:
            x2 = x2.squeeze(0) if x2.dim() > 1 else x2

        traj_length = trajectory.shape[0]

        if traj_length > 2:
            second_derivatives = trajectory[2:] - 2 * trajectory[1:-1] + trajectory[:-2]
            smoothness_loss = torch.mean(torch.sum(second_derivatives ** 2, dim=-1))
        else:
            smoothness_loss = torch.tensor(0.0, device=trajectory.device)

        if traj_length > 3:
            first_derivatives = trajectory[1:] - trajectory[:-1]
            derivative_changes = first_derivatives[1:] - first_derivatives[:-1]
            continuity_loss = torch.mean(torch.sum(derivative_changes ** 2, dim=-1))
        else:
            continuity_loss = torch.tensor(0.0, device=trajectory.device)

        midpoint_loss = torch.tensor(0.0, device=trajectory.device)
        if x2 is not None and traj_length > 2:
            num_samples = min(5, traj_length - 2)
            if num_samples > 0:
                sample_indices = torch.randperm(traj_length - 2)[:num_samples] + 1

                for idx in sample_indices:
                    progress = idx / (traj_length - 1)
                    weight = 4 * progress * (1 - progress)

                    linear_interp = (1 - progress) * x1 + progress * x2
                    pos_loss = F.mse_loss(trajectory[idx], linear_interp)
                    midpoint_loss = midpoint_loss + pos_loss * weight

                midpoint_loss = midpoint_loss / num_samples

        total_loss = (
                0.55 * smoothness_loss +
                0.35 * continuity_loss +
                0.1 * midpoint_loss
        )

        return total_loss

    def _create_single_sample_mask(self, traj_length, n_latent, x2, x6, time_type, device):
        mask = torch.ones((traj_length, n_latent), dtype=torch.bool, device=device)
        mask[0, :] = False

        if time_type == "complete" and x6 is not None:
            mid_point = self.timesteps // 2
            if mid_point < traj_length:
                mask[mid_point, :] = False

        if x2 is not None:
            mask[-1, :] = False

        return mask

    def _cleanup_gpu_memory(self):
        if torch.cuda.is_available():
            keys_to_remove = []
            for composite_index in list(self.trajectory_storage.keys()):
                if composite_index in self.saved_to_disk:
                    keys_to_remove.append(composite_index)

            for key in keys_to_remove:
                del self.trajectory_storage[key]

            torch.cuda.empty_cache()

            gc.collect()

            if self.verbose:
                print(f"Memory cleaned, removed {len(keys_to_remove)} saved trajectories")

    def optimize_single_trajectory_with_metadata(self, sample_data, optimize_steps=30):
        device = sample_data['x0'].device

        composite_index = self._generate_composite_index(sample_data)

        if composite_index in self.trajectory_storage or composite_index in self.saved_to_disk:
            if self.verbose:
                print(f"Trajectory {composite_index} already exists, skipping")
            return composite_index

        x0 = sample_data['x0']
        x6 = sample_data.get('x6')
        x24 = sample_data.get('x24')
        time_type = sample_data['time_type']

        if time_type == "complete":
            x2 = x24
        elif time_type == "partial_6h":
            x2 = x6
        elif time_type == "partial_24h":
            x2 = x24
        else:
            x2 = None

        initial_trajectory = self.init_trajectory(x0, x2, x6, time_type)
        initial_trajectory = initial_trajectory.squeeze(1)
        n_latent = initial_trajectory.shape[1]
        traj_length = initial_trajectory.shape[0]

        trajectory_offset = torch.zeros_like(initial_trajectory, requires_grad=True)

        mask = self._create_single_sample_mask(traj_length, n_latent, x2, x6, time_type, device)

        optimizer = torch.optim.AdamW([trajectory_offset], lr=2e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=optimize_steps, eta_min=5e-6
        )

        best_loss = float('inf')
        best_trajectory = None
        patience_counter = 0
        patience = 5

        for step in range(optimize_steps):
            optimizer.zero_grad()

            trajectory = initial_trajectory + trajectory_offset * mask.float()

            total_loss = self._compute_single_sample_loss(trajectory, x0, x2, time_type)

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_trajectory = trajectory.detach().clone()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        final_trajectory = best_trajectory if best_trajectory is not None else trajectory.detach().clone()

        self.trajectory_storage[composite_index] = final_trajectory.detach().clone()

        endpoint_index = self._determine_endpoint_index(sample_data)

        metadata_row = {
            'composite_index': composite_index,
            'numeric_index': self.current_numeric_index,
            'time_type': time_type,
            'drug_id': sample_data.get('drug_id', 'unknown'),
            'cell_id': sample_data.get('cell_id', 'unknown'),
            'dose': sample_data.get('dose', 0.0),
            'smiles_str': sample_data.get('smiles_str', ''),
            'data_idx': sample_data.get('data_idx', -1),
            'source': sample_data.get('source', 'unknown'),
            'trajectory_length': traj_length,
            'optimization_loss': best_loss,
            'optimization_steps': step + 1,
            'endpoint_adata_index': endpoint_index
        }

        if sample_data.get('x0') is not None:
            metadata_row['x0_shape'] = list(sample_data['x0'].shape)
        if sample_data.get('x6') is not None:
            metadata_row['x6_shape'] = list(sample_data['x6'].shape)
        if sample_data.get('x24') is not None:
            metadata_row['x24_shape'] = list(sample_data['x24'].shape)
        if sample_data.get('smiles') is not None:
            metadata_row['smiles_encoding_shape'] = list(sample_data['smiles'].shape)

        new_row_df = pd.DataFrame([metadata_row])
        self.metadata_df = pd.concat([self.metadata_df, new_row_df], ignore_index=True)

        self.current_numeric_index += 1
        self.total_optimized += 1
        self.optimization_stats[time_type] += 1
        self.save_counter += 1

        if self.save_counter >= self.save_interval:
            self.save_trajectories_and_metadata()
            self._cleanup_gpu_memory()
            self.save_counter = 0

        return composite_index

    def save_trajectories_and_metadata(self):
        try:
            trajectory_path = os.path.join(self.memory_dir, "trajectory_storage.pkl")

            existing_trajectories = {}
            if os.path.exists(trajectory_path):
                with open(trajectory_path, 'rb') as f:
                    existing_data = pickle.load(f)
                    existing_trajectories = existing_data.get('trajectories', {})

            all_trajectories = {**existing_trajectories, **self.trajectory_storage}

            save_data = {
                'trajectories': all_trajectories,
                'current_numeric_index': self.current_numeric_index,
                'total_optimized': self.total_optimized,
                'optimization_stats': self.optimization_stats,
                'timesteps': self.timesteps,
                'n_latent': self.n_latent,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }

            with open(trajectory_path, 'wb') as f:
                pickle.dump(save_data, f)

            metadata_path = os.path.join(self.memory_dir, "trajectory_metadata.csv")

            if os.path.exists(metadata_path):
                existing_metadata = pd.read_csv(metadata_path)
                combined_metadata = pd.concat([existing_metadata, self.metadata_df], ignore_index=True)
                combined_metadata = combined_metadata.drop_duplicates(subset=['composite_index'], keep='last')
                combined_metadata.to_csv(metadata_path, index=False)
            else:
                self.metadata_df.to_csv(metadata_path, index=False)

            for composite_index in self.trajectory_storage.keys():
                self.saved_to_disk.add(composite_index)

            stats_path = os.path.join(self.memory_dir, "trajectory_stats.json")
            import json
            stats = {
                'total_trajectories': len(all_trajectories),
                'current_numeric_index': self.current_numeric_index,
                'total_optimized': self.total_optimized,
                'optimization_stats': self.optimization_stats,
                'timesteps': self.timesteps,
                'n_latent': self.n_latent,
                'memory_usage_mb': sum(
                    tensor.element_size() * tensor.nelement()
                    for tensor in all_trajectories.values()
                ) / (1024 * 1024),
                'last_save_time': time.strftime("%Y-%m-%d %H:%M:%S")
            }

            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=4)

            if self.verbose:
                print(f"Saved {len(all_trajectories)} trajectories and metadata to {self.memory_dir}")

            return True

        except Exception as e:
            print(f"Failed to save trajectory storage: {e}")
            return False

    def load_trajectories_and_metadata(self):
        try:
            trajectory_path = os.path.join(self.memory_dir, "trajectory_storage.pkl")
            if os.path.exists(trajectory_path):
                with open(trajectory_path, 'rb') as f:
                    data = pickle.load(f)
                    self.trajectory_storage = data.get('trajectories', {})
                    self.current_numeric_index = data.get('current_numeric_index', 0)
                    self.total_optimized = data.get('total_optimized', 0)
                    self.optimization_stats = data.get('optimization_stats', {
                        'complete': 0, 'partial_6h': 0, 'partial_24h': 0
                    })
                    self.timesteps = data.get('timesteps', self.timesteps)
                    self.n_latent = data.get('n_latent', self.n_latent)

                print(f"Loaded {len(self.trajectory_storage)} trajectories")

                for composite_index in self.trajectory_storage.keys():
                    self.saved_to_disk.add(composite_index)

            metadata_path = os.path.join(self.memory_dir, "trajectory_metadata.csv")
            if os.path.exists(metadata_path):
                self.metadata_df = pd.read_csv(metadata_path)
                print(f"Loaded {len(self.metadata_df)} metadata rows")
            else:
                self.metadata_df = pd.DataFrame()

            return True

        except Exception as e:
            print(f"Failed to load trajectory storage: {e}")
            return False

    def get_trajectory_by_index(self, composite_index):
        if composite_index in self.trajectory_storage:
            return self.trajectory_storage[composite_index]

        if composite_index in self.saved_to_disk:
            try:
                trajectory_path = os.path.join(self.memory_dir, "trajectory_storage.pkl")
                if os.path.exists(trajectory_path):
                    with open(trajectory_path, 'rb') as f:
                        data = pickle.load(f)
                        all_trajectories = data.get('trajectories', {})
                        if composite_index in all_trajectories:
                            trajectory = all_trajectories[composite_index]
                            self.trajectory_storage[composite_index] = trajectory
                            return trajectory
            except Exception as e:
                print(f"Failed to load trajectory from disk: {e}")

        return None

    def get_metadata_by_index(self, composite_index):
        mask = self.metadata_df['composite_index'] == composite_index
        if mask.any():
            return self.metadata_df[mask].iloc[0].to_dict()
        return None

    def get_trajectories_by_time_type(self, time_type):
        mask = self.metadata_df['time_type'] == time_type
        return self.metadata_df[mask]['composite_index'].tolist()

    def print_storage_stats(self):
        print("\n===== Trajectory Storage Statistics =====")
        print(f"Trajectories in memory: {len(self.trajectory_storage)}")
        print(f"Trajectories saved: {len(self.saved_to_disk)}")
        print(f"Metadata rows: {len(self.metadata_df)}")
        print(f"Current numeric index: {self.current_numeric_index}")
        print(f"Total optimized: {self.total_optimized}")

        print(f"Trajectory type distribution:")
        for time_type, count in self.optimization_stats.items():
            print(f"  - {time_type}: {count}")

        memory_usage = sum(
            tensor.element_size() * tensor.nelement()
            for tensor in self.trajectory_storage.values()
        )
        memory_usage_mb = memory_usage / (1024 * 1024)
        print(f"Memory usage: {memory_usage_mb:.2f} MB")

        if not self.metadata_df.empty and 'endpoint_adata_index' in self.metadata_df.columns:
            endpoint_count = self.metadata_df['endpoint_adata_index'].notna().sum()
            print(f"Trajectories with endpoint index: {endpoint_count}")

        print("==========================================\n")

    def finalize_storage(self):
        if self.save_counter > 0:
            self.save_trajectories_and_metadata()
            self.save_counter = 0
        self.print_storage_stats()


def build_and_optimize_trajectories_enhanced(adata, timeseries_data, model_config, args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.shard_index >= 0:
        shard_memory_dir = os.path.join(args.memory_dir, f"shard_{args.shard_index}")
        os.makedirs(shard_memory_dir, exist_ok=True)
        trajectory_dir = shard_memory_dir
    else:
        trajectory_dir = args.memory_dir

    if args.merge_only:
        print("Merge-only mode: merging all shard trajectory storages")
        return merge_trajectory_storages_enhanced(args.memory_dir, args.total_shards)

    trajectory_builder = EnhancedTrajectoryBuilder(
        transform_net=None,
        condition_fusion=None,
        timesteps=args.timesteps,
        n_latent=adata.shape[1],
        memory_dir=trajectory_dir
    )

    if os.path.exists(os.path.join(trajectory_dir, "trajectory_storage.pkl")):
        print("Loading existing trajectory storage...")
        trajectory_builder.load_trajectories_and_metadata()

    from trajectory_dataset import create_enhanced_dataloader
    train_loader, valid_loader, test_loader, train_dataset, valid_dataset, test_dataset = create_enhanced_dataloader(
        adata, timeseries_data,
        trajectory_builder=None,
        split_key=args.split_key,
        batch_size=args.batch_size,
        extract_additional=True,
        data_filter='all'
    )

    if args.shard_index >= 0:
        print(f"Shard processing mode: processing shard {args.shard_index + 1}/{args.total_shards}")

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
        (train_dataset, train_indices, "train"),
        (valid_dataset, valid_indices, "valid"),
        (test_dataset, test_indices, "test")
    ]

    for dataset, indices, dataset_name in datasets_info:
        print(f"\n=== Optimizing {dataset_name} trajectories ===")
        _optimize_trajectories_for_dataset_enhanced(
            dataset, trajectory_builder, indices, args.optimize_steps, device
        )

    trajectory_builder.finalize_storage()

    return trajectory_builder


def _optimize_trajectories_for_dataset_enhanced(dataset, trajectory_builder, indices, optimize_steps, device):
    if not indices:
        print("No samples to process")
        return

    print(f"Optimizing trajectories for {len(indices)} samples...")

    for idx in tqdm(indices, desc="Optimizing trajectories"):
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


def merge_trajectory_storages_enhanced(memory_dir, total_shards):
    print(f"Starting trajectory storage merge, total {total_shards} shards...")

    main_builder = EnhancedTrajectoryBuilder(
        transform_net=None,
        condition_fusion=None,
        timesteps=50,
        n_latent=64,
        memory_dir=memory_dir
    )

    main_trajectory_path = os.path.join(memory_dir, "trajectory_storage.pkl")
    main_metadata_path = os.path.join(memory_dir, "trajectory_metadata.csv")

    if os.path.exists(main_trajectory_path):
        main_builder.load_trajectories_and_metadata()
        print(f"Loaded main storage, initial size: {len(main_builder.trajectory_storage)} records")

    all_metadata_dfs = [main_builder.metadata_df] if not main_builder.metadata_df.empty else []
    total_merged = 0
    all_merged_trajectories = {}

    if main_builder.trajectory_storage:
        all_merged_trajectories.update(main_builder.trajectory_storage)

    for shard_idx in range(total_shards):
        shard_dir = os.path.join(memory_dir, f"shard_{shard_idx}")
        shard_file = os.path.join(shard_dir, "trajectory_storage.pkl")
        shard_metadata_file = os.path.join(shard_dir, "trajectory_metadata.csv")

        if not os.path.exists(shard_file):
            print(f"Shard {shard_idx} storage file does not exist, skipping")
            continue

        shard_builder = EnhancedTrajectoryBuilder(
            transform_net=None,
            condition_fusion=None,
            timesteps=50,
            n_latent=64,
            memory_dir=shard_dir
        )

        if shard_builder.load_trajectories_and_metadata():
            before_size = len(all_merged_trajectories)

            new_trajectories = 0
            for composite_index, trajectory in shard_builder.trajectory_storage.items():
                if composite_index not in all_merged_trajectories:
                    all_merged_trajectories[composite_index] = trajectory
                    new_trajectories += 1
                else:
                    print(f"Duplicate index found: {composite_index}, skipping")

            if not shard_builder.metadata_df.empty:
                all_metadata_dfs.append(shard_builder.metadata_df)

            for time_type, count in shard_builder.optimization_stats.items():
                main_builder.optimization_stats[time_type] += count

            main_builder.total_optimized += shard_builder.total_optimized

            total_merged += new_trajectories
            print(f"Merged shard {shard_idx}: added {new_trajectories} records")

    main_builder.trajectory_storage = all_merged_trajectories

    if all_metadata_dfs:
        main_builder.metadata_df = pd.concat(all_metadata_dfs, ignore_index=True)
        main_builder.metadata_df = main_builder.metadata_df.drop_duplicates(
            subset=['composite_index'], keep='last'
        ).reset_index(drop=True)

    if not main_builder.metadata_df.empty:
        main_builder.current_numeric_index = main_builder.metadata_df['numeric_index'].max() + 1

    for composite_index in main_builder.trajectory_storage.keys():
        main_builder.saved_to_disk.add(composite_index)

    main_builder.save_trajectories_and_metadata()
    main_builder.print_storage_stats()

    print(
        f"Trajectory storage merge completed: total {len(main_builder.trajectory_storage)} records, merged {total_merged} new records")

    return main_builder