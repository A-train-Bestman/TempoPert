import os
import torch
import numpy as np
import pandas as pd
import scanpy as sc
from torch.utils.data import Dataset, DataLoader
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem


class L1000EnhancedDataset(Dataset):
    def __init__(self,
                 adata,
                 timeseries_data,
                 trajectory_builder=None,
                 dtype='train',
                 split_key='random_split_0',
                 use_smile_embedding=True,
                 smiles_embedding_path='../../dataset/KPGT_prnet_2304.pkl',
                 extract_additional=True,
                 data_filter='all'
                 ):
        self.dtype = dtype
        self.split_key = split_key
        self.adata = adata
        self.timeseries_data = timeseries_data
        self.data_filter = data_filter
        self.trajectory_builder = trajectory_builder

        if 'pert_id' not in self.timeseries_data.columns:
            self.timeseries_data.columns = ['pert_id', 'ctrl_idx', 'x_6_idx', 'x_24_idx', 'dose']

        self._detect_column_names()

        if hasattr(adata.X, 'toarray'):
            self.X = adata.X.toarray()
        else:
            self.X = adata.X

        self.adata_idx_dict = {idx: i for i, idx in enumerate(adata.obs.index)}

        self.timeseries_samples = []
        self.processed_indices = set()

        self._process_timeseries_samples()

        if extract_additional:
            self._extract_additional_samples()

        self._filter_samples()

        print(f"Loaded {len(self.timeseries_samples)} {dtype} samples, filter mode: {data_filter}")

        self.use_smile_embedding = use_smile_embedding
        self.smi2emb = {}

        if use_smile_embedding and os.path.exists(smiles_embedding_path):
            with open(smiles_embedding_path, 'rb') as f:
                self.smi2emb = pickle.load(f)
                self.smi2emb = {key[0]: value for key, value in self.smi2emb.items()}
            print(f"Loaded SMILES embeddings, {len(self.smi2emb)} compounds")

    def _detect_column_names(self):
        try:
            drug_col = [col for col in self.timeseries_data.columns if
                        'id' in col.lower() and ('drug' in col.lower() or 'pert' in col.lower())][0]
            ctrl_col = \
                [col for col in self.timeseries_data.columns if 'ctrl' in col.lower() or 'control' in col.lower()][0]
            h6_col = [col for col in self.timeseries_data.columns if '6' in col or 'x_6' in col][0]
            h24_col = [col for col in self.timeseries_data.columns if '24' in col or 'x_24' in col][0]
            dose_col = [col for col in self.timeseries_data.columns if 'dose' in col.lower()][0]

            print(f"Detected columns: drug_id={drug_col}, control={ctrl_col}, 6h={h6_col}, 24h={h24_col}, dose={dose_col}")
        except:
            print("Auto-detection failed, using default column names")
            drug_col = 'pert_id'
            ctrl_col = 'ctrl_idx'
            h6_col = 'x_6_idx'
            h24_col = 'x_24_idx'
            dose_col = 'dose'

        self.drug_col = drug_col
        self.ctrl_col = ctrl_col
        self.h6_col = h6_col
        self.h24_col = h24_col
        self.dose_col = dose_col

    def _process_timeseries_samples(self):
        for idx, row in self.timeseries_data.iterrows():
            ctrl_idx = str(row[self.ctrl_col])
            h6_idx = str(row[self.h6_col]) if pd.notna(row[self.h6_col]) else None
            h24_idx = str(row[self.h24_col]) if pd.notna(row[self.h24_col]) else None

            if ctrl_idx in self.adata_idx_dict:
                if h6_idx and h6_idx in self.adata.obs.index:
                    h6_split = self.adata.obs.loc[
                        h6_idx, self.split_key] if self.split_key in self.adata.obs.columns else None
                else:
                    h6_split = None

                if h24_idx and h24_idx in self.adata.obs.index:
                    h24_split = self.adata.obs.loc[
                        h24_idx, self.split_key] if self.split_key in self.adata.obs.columns else None
                else:
                    h24_split = None

                is_matching_split = (not h6_split and not h24_split) or \
                                    (h6_split == self.dtype and h24_split == self.dtype) or \
                                    (h6_split == self.dtype and not h24_split) or \
                                    (not h6_split and h24_split == self.dtype)

                if is_matching_split:
                    sample = {
                        'ctrl_idx': ctrl_idx,
                        'h6_idx': h6_idx,
                        'h24_idx': h24_idx,
                        'drug_id': row[self.drug_col],
                        'dose': float(row[self.dose_col]),
                        'timeseries_idx': idx,
                        'source': 'timeseries_data'
                    }
                    self.timeseries_samples.append(sample)

                    self.processed_indices.add(ctrl_idx)
                    if h6_idx: self.processed_indices.add(h6_idx)
                    if h24_idx: self.processed_indices.add(h24_idx)

        print(f"Extracted {len(self.timeseries_samples)} samples from timeseries_data")

    def _extract_additional_samples(self):
        time_cols = [col for col in self.adata.obs.columns if
                     'time' in col.lower() and ('pert' in col.lower() or 'treatment' in col.lower())]
        pert_time_col = time_cols[0] if time_cols else 'pert_time'
        if pert_time_col not in self.adata.obs.columns:
            print(f"Error: time column {pert_time_col} not found, exiting")
            return

        control_cols = [col for col in self.adata.obs.columns if
                        ('control' in col.lower() or 'ctrl' in col.lower()) and (
                                'pair' in col.lower() or 'index' in col.lower())]
        paired_control_col = control_cols[0] if control_cols else 'paired_control_index'
        if paired_control_col not in self.adata.obs.columns:
            print(f"Error: control column {paired_control_col} not found, exiting")
            return

        print(f"Using columns: time={pert_time_col}, control={paired_control_col}")

        time_numeric = pd.to_numeric(self.adata.obs[pert_time_col], errors='coerce')
        mask_6h = (time_numeric == 6) | (time_numeric == 6.0)
        mask_24h = (time_numeric == 24) | (time_numeric == 24.0)

        all_6h_indices = set(self.adata.obs.index[mask_6h])
        all_24h_indices = set(self.adata.obs.index[mask_24h])

        existing_indices = set()
        for sample in self.timeseries_samples:
            if sample['h6_idx']: existing_indices.add(sample['h6_idx'])
            if sample['h24_idx']: existing_indices.add(sample['h24_idx'])

        new_6h_indices = all_6h_indices - existing_indices
        new_24h_indices = all_24h_indices - existing_indices

        print(f"Found new samples: 6h={len(new_6h_indices)}, 24h={len(new_24h_indices)}")

        ctrl_map = {}
        valid_indices = list(new_6h_indices) + list(new_24h_indices)

        if valid_indices:
            ctrl_series = self.adata.obs.loc[valid_indices, paired_control_col]
            ctrl_map = {idx: str(ctrl) for idx, ctrl in ctrl_series.items() if pd.notna(ctrl)}

        drug_id_map = {}
        dose_map = {}

        for col in ['pert_id', 'drug_id', 'cov_drug_name']:
            if col in self.adata.obs.columns:
                drug_series = self.adata.obs.loc[valid_indices, col]
                drug_id_map.update({idx: drug for idx, drug in drug_series.items() if pd.notna(drug)})
                if drug_id_map:
                    break

        if 'dose' in self.adata.obs.columns:
            dose_series = pd.to_numeric(self.adata.obs.loc[valid_indices, 'dose'], errors='coerce')
            dose_map = {idx: dose for idx, dose in dose_series.items() if pd.notna(dose)}

        is_split_required = self.split_key in self.adata.obs.columns
        split_index_dict = {}

        if is_split_required:
            split_series = self.adata.obs[self.split_key].dropna()
            split_index_dict = split_series.to_dict()

        additional_samples = []

        def create_sample(idx, is_6h):
            if idx in self.processed_indices:
                return None

            ctrl_idx = ctrl_map.get(idx)
            if not ctrl_idx or ctrl_idx == 'nan' or ctrl_idx not in self.adata_idx_dict:
                return None

            if is_split_required:
                sample_split = split_index_dict.get(idx)
                if sample_split != self.dtype:
                    return None

            return {
                'ctrl_idx': ctrl_idx,
                'h6_idx': idx if is_6h else None,
                'h24_idx': None if is_6h else idx,
                'drug_id': drug_id_map.get(idx, 'unknown'),
                'dose': dose_map.get(idx, 0.0),
                'timeseries_idx': -1,
                'source': 'additional_fast'
            }

        for idx in new_6h_indices:
            sample = create_sample(idx, True)
            if sample:
                additional_samples.append(sample)

        for idx in new_24h_indices:
            sample = create_sample(idx, False)
            if sample:
                additional_samples.append(sample)

        for sample in additional_samples:
            self.timeseries_samples.append(sample)

            if sample['h6_idx']: self.processed_indices.add(sample['h6_idx'])
            if sample['h24_idx']: self.processed_indices.add(sample['h24_idx'])
            self.processed_indices.add(sample['ctrl_idx'])

        paired_6h_samples = sum(1 for s in additional_samples if s['h6_idx'])
        paired_24h_samples = sum(1 for s in additional_samples if s['h24_idx'])

        print(f"Extracted {len(additional_samples)} additional samples from adata (6h: {paired_6h_samples}, 24h: {paired_24h_samples})")

    def _filter_samples(self):
        if self.data_filter == 'all':
            return

        filtered_samples = []
        complete_count = 0
        partial_6h_count = 0
        partial_24h_count = 0

        for sample in self.timeseries_samples:
            is_complete = sample['h6_idx'] is not None and sample['h24_idx'] is not None
            is_partial_6h = sample['h6_idx'] is not None and sample['h24_idx'] is None
            is_partial_24h = sample['h6_idx'] is None and sample['h24_idx'] is not None

            if self.data_filter == 'complete' and is_complete:
                filtered_samples.append(sample)
                complete_count += 1
            elif self.data_filter == 'partial' and (is_partial_6h or is_partial_24h):
                filtered_samples.append(sample)
                if is_partial_6h:
                    partial_6h_count += 1
                else:
                    partial_24h_count += 1
            elif self.data_filter == 'partial_6h' and is_partial_6h:
                filtered_samples.append(sample)
                partial_6h_count += 1
            elif self.data_filter == 'partial_24h' and is_partial_24h:
                filtered_samples.append(sample)
                partial_24h_count += 1

        self.timeseries_samples = filtered_samples

        print(f"Data filter stats ({self.data_filter}): complete = {complete_count}, "
              f"partial (0-6h) = {partial_6h_count}, partial (0-24h) = {partial_24h_count}")

    def get_sample_type_counts(self):
        complete_count = 0
        partial_6h_count = 0
        partial_24h_count = 0

        for sample in self.timeseries_samples:
            if sample['h6_idx'] is not None and sample['h24_idx'] is not None:
                complete_count += 1
            elif sample['h6_idx'] is not None:
                partial_6h_count += 1
            elif sample['h24_idx'] is not None:
                partial_24h_count += 1

        return {
            'complete': complete_count,
            'partial_6h': partial_6h_count,
            'partial_24h': partial_24h_count,
            'total': len(self.timeseries_samples)
        }

    def _get_smiles_for_index(self, sample_idx):
        if sample_idx in self.adata.obs.index and 'SMILES' in self.adata.obs.columns:
            return self.adata.obs.loc[sample_idx, 'SMILES']
        return None

    def _get_cell_id_for_index(self, sample_idx):
        if sample_idx in self.adata.obs.index and 'cell_id' in self.adata.obs.columns:
            return self.adata.obs.loc[sample_idx, 'cell_id']
        return 'unknown'

    def __len__(self):
        return len(self.timeseries_samples)

    def __getitem__(self, idx):
        sample = self.timeseries_samples[idx]
        ctrl_idx = sample['ctrl_idx']
        h6_idx = sample['h6_idx']
        h24_idx = sample['h24_idx']
        drug_id = sample['drug_id']
        dose = sample['dose']
        data_idx = idx

        ctrl_data = torch.tensor(self.X[self.adata_idx_dict[ctrl_idx]], dtype=torch.float32)

        h6_data = torch.tensor(self.X[self.adata_idx_dict[h6_idx]],
                               dtype=torch.float32) if h6_idx and h6_idx in self.adata_idx_dict else None
        h24_data = torch.tensor(self.X[self.adata_idx_dict[h24_idx]],
                                dtype=torch.float32) if h24_idx and h24_idx in self.adata_idx_dict else None

        smiles = None

        if h6_idx:
            smiles = self._get_smiles_for_index(h6_idx)

        if not smiles and h24_idx:
            smiles = self._get_smiles_for_index(h24_idx)

        if not smiles:
            smiles = self._get_smiles_for_index(ctrl_idx)

        cell_id = None

        cell_id = self._get_cell_id_for_index(ctrl_idx)

        if cell_id == 'unknown' and h6_idx:
            cell_id = self._get_cell_id_for_index(h6_idx)

        if cell_id == 'unknown' and h24_idx:
            cell_id = self._get_cell_id_for_index(h24_idx)

        if smiles and self.use_smile_embedding and hasattr(self, 'smi2emb') and smiles in self.smi2emb:
            smiles_encoding = torch.tensor(self.smi2emb[smiles], dtype=torch.float32)
        else:
            try:
                if smiles:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=2048)
                        smiles_encoding = torch.tensor([float(b) for b in fingerprint.ToBitString()],
                                                       dtype=torch.float32)
                    else:
                        smiles_encoding = torch.zeros(2048, dtype=torch.float32)
                else:
                    smiles_encoding = torch.zeros(2048, dtype=torch.float32)
            except:
                smiles_encoding = torch.zeros(2048, dtype=torch.float32)

        if h6_data is not None and h24_data is not None:
            time_type = "complete"
            timepoints = [0.0, 6.0, 24.0]
            has_timepoints = [True, True, True]
        elif h6_data is not None:
            time_type = "partial_6h"
            timepoints = [0.0, 6.0]
            has_timepoints = [True, True, False]
        elif h24_data is not None:
            time_type = "partial_24h"
            timepoints = [0.0, 24.0]
            has_timepoints = [True, False, True]
        else:
            time_type = "unknown"
            timepoints = [0.0]
            has_timepoints = [True, False, False]

        trajectory = None
        composite_index = None

        if self.trajectory_builder is not None:
            sample_data = {
                'smiles_str': smiles,
                'time_type': time_type,
                'dose': dose,
                'cell_id': cell_id,
                'data_idx': data_idx
            }

            composite_index = self._generate_composite_index(sample_data)
            trajectory = self.trajectory_builder.get_trajectory_by_index(composite_index)

        result = {
            'x0': ctrl_data,
            'x6': h6_data,
            'x24': h24_data,
            'timepoints': torch.tensor(timepoints, dtype=torch.float32),
            'has_timepoints': has_timepoints,
            'smiles': smiles_encoding,
            'smiles_str': smiles if smiles else 'unknown',
            'cell_id': cell_id if cell_id else 'unknown',
            'dose': dose,
            'drug_id': drug_id,
            'time_type': time_type,
            'source': sample['source'] if 'source' in sample else 'unknown',
            'data_idx': data_idx,
            'trajectory': trajectory,
            'composite_index': composite_index,
            'h6_idx': h6_idx,
            'h24_idx': h24_idx,
            'ctrl_idx': ctrl_idx
        }

        return result

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


def custom_collate(batch):
    elem = batch[0]

    special_keys = ['x6', 'x24', 'trajectory']

    result = {}

    time_types = [d['time_type'] for d in batch]
    result['time_type'] = time_types

    for key in elem.keys():
        if key == 'time_type':
            continue

        if key in special_keys:
            non_none_values = [d[key] for d in batch if d[key] is not None]
            if non_none_values:
                if isinstance(non_none_values[0], torch.Tensor):
                    if non_none_values[0].dim() == 3:
                        min_steps = min(traj.shape[0] for traj in non_none_values)
                        non_none_values = [traj[:min_steps] for traj in non_none_values]
                        if all(traj.shape[1:] == non_none_values[0].shape[1:] for traj in non_none_values):
                            result[key] = torch.cat([traj.unsqueeze(1) for traj in non_none_values], dim=1)
                        else:
                            result[key] = non_none_values
                    else:
                        result[key] = torch.stack(non_none_values)
                else:
                    result[key] = non_none_values
            else:
                result[key] = None
        else:
            try:
                values = [d[key] for d in batch]
                if all(v is not None for v in values):
                    if isinstance(values[0], torch.Tensor):
                        result[key] = torch.stack(values)
                    elif isinstance(values[0], (int, float, str, bool)):
                        result[key] = values
                    else:
                        result[key] = torch.utils.data.dataloader.default_collate(values)
                else:
                    result[key] = values
            except:
                result[key] = [d[key] for d in batch]

    return result


def create_enhanced_dataloader(adata, timeseries_data, trajectory_builder=None,
                               split_key='random_split_0', batch_size=32,
                               extract_additional=True, data_filter='all'):
    train_dataset = L1000EnhancedDataset(
        adata=adata,
        timeseries_data=timeseries_data,
        trajectory_builder=trajectory_builder,
        dtype='train',
        split_key=split_key,
        extract_additional=extract_additional,
        data_filter=data_filter
    )

    valid_dataset = L1000EnhancedDataset(
        adata=adata,
        timeseries_data=timeseries_data,
        trajectory_builder=trajectory_builder,
        dtype='valid',
        split_key=split_key,
        extract_additional=extract_additional,
        data_filter=data_filter
    )

    test_dataset = L1000EnhancedDataset(
        adata=adata,
        timeseries_data=timeseries_data,
        trajectory_builder=trajectory_builder,
        dtype='test',
        split_key=split_key,
        extract_additional=extract_additional,
        data_filter=data_filter
    )

    print("\nDataset statistics:")
    train_counts = train_dataset.get_sample_type_counts()
    print(f"Train: complete={train_counts['complete']}, partial(0-6h)={train_counts['partial_6h']}, "
          f"partial(0-24h)={train_counts['partial_24h']}, total={train_counts['total']}")

    valid_counts = valid_dataset.get_sample_type_counts()
    print(f"Valid: complete={valid_counts['complete']}, partial(0-6h)={valid_counts['partial_6h']}, "
          f"partial(0-24h)={valid_counts['partial_24h']}, total={valid_counts['total']}")

    test_counts = test_dataset.get_sample_type_counts()
    print(f"Test: complete={test_counts['complete']}, partial(0-6h)={test_counts['partial_6h']}, "
          f"partial(0-24h)={test_counts['partial_24h']}, total={test_counts['total']}")
    print()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=custom_collate
    )

    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=custom_collate
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=custom_collate
    )

    return train_loader, valid_loader, test_loader, train_dataset, valid_dataset, test_dataset