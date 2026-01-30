import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import os


class TimeEmbedding(nn.Module):

    def __init__(self, time_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, t):
        t = t.unsqueeze(-1).float()
        return self.time_mlp(t)


class GeneEncoder(nn.Module):

    def __init__(self, n_genes, n_latent, hidden_dims=[1200, 800], dropout=0.1):
        super().__init__()

        layers = []
        layers.append(nn.Linear(n_genes, hidden_dims[0]))
        layers.append(nn.LayerNorm(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.LayerNorm(hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.encoder = nn.Sequential(*layers)
        self.mu = nn.Linear(hidden_dims[-1], n_latent)
        self.logvar = nn.Linear(hidden_dims[-1], n_latent)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, logvar


class FeatureProcessing(nn.Module):

    def __init__(self, feature_dim, seq_len, embed_dim, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, seq_len * embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        nn.init.normal_(self.pos_embedding, std=0.02)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, features):
        batch_size = features.shape[0]
        projected = self.feature_proj(features)
        seq_features = projected.view(batch_size, self.seq_len, self.embed_dim)
        seq_features = seq_features + self.pos_embedding
        seq_features = self.norm(seq_features)
        return seq_features


class ImprovedCrossAttention(nn.Module):

    def __init__(self, query_dim, key_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        assert self.head_dim * num_heads == query_dim, "query_dim must be divisible by num_heads"

        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_dim, query_dim)
        self.v_proj = nn.Linear(key_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(dropout)

    def _shape(self, tensor, seq_len, batch_size):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        batch_size, tgt_len, _ = query.size()
        _, src_len, _ = key.size()

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = self._shape(q, tgt_len, batch_size)
        k = self._shape(k, src_len, batch_size)
        v = self._shape(v, src_len, batch_size)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))

        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, tgt_len, -1
        )
        attn_output = self.out_proj(attn_output)

        return attn_output


class EnhancedConditionFusion(nn.Module):

    def __init__(self, latent_dim, feature_dim, embed_dim=256, num_heads=4, dropout=0.1, num_layers=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.dose_processor = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, feature_dim)
        )

        self.z1_processor = FeatureProcessing(latent_dim, seq_len=8, embed_dim=embed_dim, dropout=dropout)
        self.feat_processor = FeatureProcessing(feature_dim, seq_len=16, embed_dim=embed_dim, dropout=dropout)

        self.z1_to_feat_attn_layers = nn.ModuleList([
            ImprovedCrossAttention(embed_dim, embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.feat_to_z1_attn_layers = nn.ModuleList([
            ImprovedCrossAttention(embed_dim, embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.z1_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.feat_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])

        self.z1_fc = nn.Sequential(
            nn.Linear(8 * embed_dim, latent_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim)
        )

        self.feat_fc = nn.Sequential(
            nn.Linear(16 * embed_dim, feature_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim)
        )

        self.condition_combiner = nn.Sequential(
            nn.Linear(latent_dim + feature_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim * 2)
        )

        self.drug_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.SiLU(),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, z1, features, dose=None):
        batch_size = z1.shape[0]

        if dose is not None:
            try:
                if isinstance(dose, torch.Tensor):
                    if dose.dim() == 0:
                        dose = dose.unsqueeze(0).expand(batch_size)
                    elif dose.dim() == 1 and dose.size(0) != batch_size:
                        if dose.size(0) == 1:
                            dose = dose.expand(batch_size)
                        elif dose.size(0) > batch_size:
                            dose = dose[:batch_size]
                        else:
                            padding = torch.zeros(batch_size - dose.size(0), device=dose.device, dtype=dose.dtype)
                            dose = torch.cat([dose, padding])
                    elif dose.dim() > 1:
                        dose = dose.flatten()[:batch_size]
                        if dose.size(0) < batch_size:
                            padding = torch.zeros(batch_size - dose.size(0), device=dose.device, dtype=dose.dtype)
                            dose = torch.cat([dose, padding])
                else:
                    dose_value = float(dose) if isinstance(dose, (int, float)) else 0.0
                    dose = torch.full((batch_size,), dose_value, device=z1.device, dtype=torch.float32)

                log_dose = torch.log10(dose + 1).unsqueeze(-1)
                dose_features = self.dose_processor(log_dose)
                features_enhanced = features * dose_features
            except:
                features_enhanced = features
        else:
            features_enhanced = features

        attention_weights = self.drug_attention(features_enhanced)
        features_enhanced = features_enhanced * attention_weights

        z1_seq = self.z1_processor(z1)
        feat_seq = self.feat_processor(features_enhanced)

        for i in range(self.num_layers):
            z1_attn = self.z1_to_feat_attn_layers[i](z1_seq, feat_seq, feat_seq)
            z1_seq = self.z1_norms[i](z1_seq + z1_attn)

            feat_attn = self.feat_to_z1_attn_layers[i](feat_seq, z1_seq, z1_seq)
            feat_seq = self.feat_norms[i](feat_seq + feat_attn)

        z1_enhanced = z1_seq.reshape(batch_size, -1)
        feat_enhanced = feat_seq.reshape(batch_size, -1)

        z1_final = self.z1_fc(z1_enhanced)
        feat_final = self.feat_fc(feat_enhanced)

        condition = self.condition_combiner(torch.cat([z1_final, feat_final], dim=1))

        return condition, z1_final, feat_final


class ResBlock(nn.Module):

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(min(32, dim), dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.GroupNorm(min(32, dim), dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.block(x)


class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x, context):
        x_seq = x.unsqueeze(1)
        context_seq = context.unsqueeze(1)
        x_norm = self.norm(x_seq)
        attn_output, _ = self.attention(x_norm, context_seq, context_seq)
        output = x_seq + attn_output
        return output.squeeze(1)


class FiLM(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.scale_shift = nn.Linear(dim, dim * 2)

    def forward(self, x, condition):
        scale_shift = self.scale_shift(condition)
        scale, shift = scale_shift.chunk(2, dim=1)
        return x * (1 + scale) + shift


class TransformNet(nn.Module):

    def __init__(self, n_latent, time_dim, condition_dim, dropout=0.1):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_dim)

        hidden_dim = 512

        self.input_layer = nn.Linear(n_latent, hidden_dim)

        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_dim, dropout=dropout)
            for _ in range(4)
        ])

        self.cross_attention_blocks = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, num_heads=8, dropout=dropout)
            for _ in range(4)
        ])

        self.film_layers = nn.ModuleList([
            FiLM(hidden_dim)
            for _ in range(4)
        ])

        self.final_layer = nn.Sequential(
            nn.GroupNorm(min(32, hidden_dim), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_latent)
        )

    def forward(self, x, t, condition):
        t_emb = self.time_embedding(t)
        time_cond = self.time_proj(t_emb)
        cond_signal = self.condition_proj(condition)
        h = self.input_layer(x)

        for res_block, attn_block, film in zip(self.res_blocks, self.cross_attention_blocks, self.film_layers):
            h = h + time_cond
            h = res_block(h)
            h = attn_block(h, cond_signal)
            h = film(h, cond_signal)

        delta = self.final_layer(h)
        return delta


class TrajectoryGuidedProgressiveModel(nn.Module):

    def __init__(self, n_genes, n_latent, features_dim,
                 timesteps=50, hidden_dim=512, dropout=0.1,
                 device="cuda", time_mapping_config=None, vae_kl_weight=0.05,
                 trajectory_memory_dir="./trajectory_memory"):
        super().__init__()
        self.n_genes = n_genes
        self.n_latent = n_latent
        self.features_dim = features_dim
        self.timesteps = timesteps
        self.device = device
        self.trajectory_memory_dir = trajectory_memory_dir
        os.makedirs(trajectory_memory_dir, exist_ok=True)

        self.time_mapping_config = {
            "6h_step": timesteps // 4,
            "24h_step": timesteps - 1,
            "use_nonlinear_mapping": False
        }

        if time_mapping_config is not None:
            self.time_mapping_config.update(time_mapping_config)

        self.gene_encoder_x0 = GeneEncoder(
            n_genes=n_genes,
            n_latent=n_latent,
            hidden_dims=[1200, 800],
            dropout=dropout
        )

        self.gene_encoder_drug = GeneEncoder(
            n_genes=n_genes,
            n_latent=n_latent,
            hidden_dims=[1200, 800],
            dropout=dropout
        )

        self.gene_decoder = nn.Sequential(
            nn.Linear(n_latent, 800),
            nn.LayerNorm(800),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(800, 1200),
            nn.LayerNorm(1200),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1200, n_genes),
            nn.ReLU()
        )

        self.condition_fusion = EnhancedConditionFusion(
            latent_dim=n_latent,
            feature_dim=features_dim,
            embed_dim=256,
            num_heads=4,
            dropout=dropout,
            num_layers=3
        )

        self.transform_net = TransformNet(
            n_latent=n_latent,
            time_dim=n_latent * 2,
            condition_dim=n_latent * 2,
            dropout=dropout
        )

        self.time_mapping = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )

        self.beta = vae_kl_weight
        self.train_config = {
            "vae_weight": 0.1,
            "trajectory_weight": 1.0,
            "endpoint_weight": 2.0
        }

        self._intermediate_states = {}
        self._verbose = False

    def _safe_tensor_to_scalar(self, tensor, name="tensor"):
        if isinstance(tensor, torch.Tensor):
            if tensor.numel() == 1:
                return tensor.item()
            elif tensor.numel() > 1:
                return tensor.mean().item()
            elif tensor.numel() == 0:
                return 0.0
        elif isinstance(tensor, (int, float)):
            return float(tensor)
        else:
            return 0.0

    def _safe_tensor_check(self, tensor, name="tensor"):
        if isinstance(tensor, torch.Tensor):
            if tensor.numel() == 0:
                return False
            if torch.isnan(tensor).any():
                return False
            if torch.isinf(tensor).any():
                return False
        return True

    def _ensure_scalar_loss(self, loss, name="loss"):
        if isinstance(loss, torch.Tensor):
            if loss.numel() == 0:
                return torch.tensor(0.0, device=loss.device, requires_grad=True)
            elif loss.numel() == 1:
                return loss.squeeze()
            else:
                return loss.mean()
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return torch.tensor(float(loss), device=device, requires_grad=True)

    def _safe_dose_processing_fixed(self, dose, batch_size, device):
        if dose is None:
            return torch.zeros(batch_size, dtype=torch.float32, device=device)

        try:
            if isinstance(dose, torch.Tensor):
                dose = dose.to(device)

                if dose.dim() == 0:
                    return dose.expand(batch_size)
                elif dose.dim() == 1:
                    if dose.size(0) == batch_size:
                        return dose
                    elif dose.size(0) == 1:
                        return dose.expand(batch_size)
                    elif dose.size(0) > batch_size:
                        return dose[:batch_size]
                    else:
                        padding = torch.zeros(batch_size - dose.size(0), dtype=dose.dtype, device=device)
                        return torch.cat([dose, padding])
                else:
                    dose_flat = dose.flatten()
                    if dose_flat.size(0) >= batch_size:
                        return dose_flat[:batch_size]
                    else:
                        padding = torch.zeros(batch_size - dose_flat.size(0), dtype=dose.dtype, device=device)
                        return torch.cat([dose_flat, padding])

            elif isinstance(dose, (list, tuple)):
                cleaned_dose = []
                for d in dose:
                    if isinstance(d, (int, float)):
                        cleaned_dose.append(float(d))
                    elif isinstance(d, torch.Tensor):
                        cleaned_dose.append(d.item() if d.numel() == 1 else d.mean().item())
                    else:
                        cleaned_dose.append(0.0)

                if len(cleaned_dose) < batch_size:
                    cleaned_dose.extend([0.0] * (batch_size - len(cleaned_dose)))
                elif len(cleaned_dose) > batch_size:
                    cleaned_dose = cleaned_dose[:batch_size]

                return torch.tensor(cleaned_dose, dtype=torch.float32, device=device)

            else:
                dose_value = float(dose) if isinstance(dose, (int, float)) else 0.0
                return torch.full((batch_size,), dose_value, dtype=torch.float32, device=device)

        except Exception:
            return torch.zeros(batch_size, dtype=torch.float32, device=device)

    def _process_guided_trajectory(self, guided_trajectory, device):
        try:
            if guided_trajectory.device != device:
                guided_trajectory = guided_trajectory.to(device)

            if len(guided_trajectory.shape) != 3:
                return None

            batch_size_traj, traj_length, feature_dim = guided_trajectory.shape

            if feature_dim == self.n_genes:
                guided_trajectory_reshaped = guided_trajectory.view(-1, feature_dim)
                guided_z_reshaped, _, _ = self.encode_gene_expression(
                    guided_trajectory_reshaped, is_drug_treated=True
                )
                guided_z_trajectory = guided_z_reshaped.view(
                    batch_size_traj, traj_length, -1
                ).transpose(0, 1)

            elif feature_dim == self.n_latent:
                guided_z_trajectory = guided_trajectory.transpose(0, 1)

            else:
                return None

            if not self._safe_tensor_check(guided_z_trajectory, "guided_z_trajectory"):
                return None

            return guided_z_trajectory

        except Exception:
            return None

    def _safe_loss_computation_fixed(self, vae_loss, trajectory_loss, endpoint_loss, type_weight):
        try:
            vae_loss = self._ensure_scalar_loss(vae_loss, "vae_loss")
            trajectory_loss = self._ensure_scalar_loss(trajectory_loss, "trajectory_loss")
            endpoint_loss = self._ensure_scalar_loss(endpoint_loss, "endpoint_loss")

            if not self._safe_tensor_check(vae_loss, "vae_loss"):
                vae_loss = torch.tensor(0.0, device=vae_loss.device, requires_grad=True)

            if not self._safe_tensor_check(trajectory_loss, "trajectory_loss"):
                trajectory_loss = torch.tensor(0.0, device=trajectory_loss.device, requires_grad=True)

            if not self._safe_tensor_check(endpoint_loss, "endpoint_loss"):
                endpoint_loss = torch.tensor(0.0, device=endpoint_loss.device, requires_grad=True)

            vae_weight = self.train_config["vae_weight"]
            trajectory_weight = self.train_config["trajectory_weight"] * type_weight
            endpoint_weight = self.train_config["endpoint_weight"] * type_weight

            total_loss = (
                    vae_weight * vae_loss +
                    trajectory_weight * trajectory_loss +
                    endpoint_weight * endpoint_loss
            )

            total_loss = self._ensure_scalar_loss(total_loss, "total_loss")

            if not self._safe_tensor_check(total_loss, "total_loss"):
                device = vae_loss.device if hasattr(vae_loss, 'device') else 'cpu'
                return torch.tensor(1.0, device=device, requires_grad=True)

            return total_loss

        except Exception:
            device = vae_loss.device if hasattr(vae_loss, 'device') and vae_loss.device else 'cpu'
            return torch.tensor(1.0, device=device, requires_grad=True)

    def map_real_time_to_steps(self, time_hours):
        if isinstance(time_hours, (int, float)):
            time = torch.tensor([time_hours], device=self.device).float()
        else:
            time = time_hours.float()

        if self.time_mapping_config["use_nonlinear_mapping"]:
            time = time.view(-1, 1)
            mapped_time = self.time_mapping(time)
            return torch.clamp(torch.floor(torch.sigmoid(mapped_time) * (self.timesteps - 1)), 0,
                               self.timesteps - 1).long()
        else:
            result = torch.zeros_like(time, dtype=torch.long)

            result[time == 0] = 0
            result[time == 6] = self.time_mapping_config["6h_step"]
            result[time == 24] = self.time_mapping_config["24h_step"]

            mask_other = (time != 0) & (time != 6) & (time != 24)
            if torch.any(mask_other):
                mask_0_6 = mask_other & (time < 6)
                if torch.any(mask_0_6):
                    alpha = time[mask_0_6] / 6.0
                    result[mask_0_6] = torch.floor(alpha * self.time_mapping_config["6h_step"]).long()

                mask_6_24 = mask_other & (time >= 6) & (time <= 24)
                if torch.any(mask_6_24):
                    alpha = (time[mask_6_24] - 6) / 18.0
                    step_6h = self.time_mapping_config["6h_step"]
                    step_24h = self.time_mapping_config["24h_step"]
                    result[mask_6_24] = torch.floor(step_6h + alpha * (step_24h - step_6h)).long()

                mask_over_24 = mask_other & (time > 24)
                if torch.any(mask_over_24):
                    result[mask_over_24] = self.time_mapping_config["24h_step"]

            return result

    def encode_gene_expression(self, x, is_drug_treated=False):
        if is_drug_treated:
            return self.gene_encoder_drug(x)
        else:
            return self.gene_encoder_x0(x)

    def decode_to_gene_expression(self, z):
        return self.gene_decoder(z)

    def encode_to_latent(self, x):
        return self.encode_gene_expression(x, is_drug_treated=False)

    def fuse_condition(self, z, features, dose):
        batch_size = z.shape[0]
        device = z.device

        dose_processed = self._safe_dose_processing_fixed(dose, batch_size, device)

        condition, _, _ = self.condition_fusion(z, features, dose_processed)

        return condition

    def vae_loss(self, x, x_rec, mu, logvar):
        mse_loss = F.mse_loss(x_rec, x, reduction='sum') / x.size(0)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = mse_loss + self.beta * kl_loss
        return total_loss, mse_loss, kl_loss

    def _compute_trajectory_following_loss(self, predicted_trajectory, target_trajectory, num_steps=10):
        try:
            batch_size = predicted_trajectory.shape[1]
            pred_timesteps = predicted_trajectory.shape[0]
            target_timesteps = target_trajectory.shape[0]
            timesteps = min(pred_timesteps, target_timesteps)
            device = predicted_trajectory.device

            if timesteps <= 1:
                return torch.tensor(0.0, device=device, requires_grad=True)

            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            total_weight = 0.0

            for step_idx in range(1, timesteps):
                progress = step_idx / (timesteps - 1)
                base_weight = 1.0

                if abs(step_idx - timesteps // 4) <= 1:
                    base_weight = 2.0
                elif step_idx >= timesteps - 2:
                    base_weight = 2.5
                elif step_idx <= 3:
                    base_weight = 1.5
                elif timesteps // 3 <= step_idx <= 2 * timesteps // 3:
                    base_weight = 1.2

                pred_state = predicted_trajectory[step_idx]
                target_state = target_trajectory[step_idx]

                step_loss = F.mse_loss(pred_state, target_state, reduction='mean')

                if torch.isnan(step_loss) or torch.isinf(step_loss):
                    continue

                weighted_step_loss = step_loss * base_weight
                total_loss = total_loss + weighted_step_loss
                total_weight += base_weight

            if total_weight > 0:
                final_loss = total_loss / total_weight
            else:
                final_loss = torch.tensor(0.01, device=device, requires_grad=True)

            return self._ensure_scalar_loss(final_loss, "trajectory_following_loss")

        except Exception as e:
            device = predicted_trajectory.device if hasattr(predicted_trajectory, 'device') else 'cpu'
            return torch.tensor(0.1, device=device, requires_grad=True)

    def _compute_endpoint_loss(self, predicted_trajectory, target_trajectory, time_type="complete"):
        try:
            device = predicted_trajectory.device
            pred_timesteps = predicted_trajectory.shape[0]
            target_timesteps = target_trajectory.shape[0]

            if time_type == "complete":
                t6_idx = min(self.timesteps // 2, pred_timesteps - 1, target_timesteps - 1)
                t24_idx = min(pred_timesteps - 1, target_timesteps - 1)

                z6_pred = predicted_trajectory[t6_idx]
                z6_target = target_trajectory[t6_idx]
                loss_6h = F.mse_loss(z6_pred, z6_target)

                z24_pred = predicted_trajectory[t24_idx]
                z24_target = target_trajectory[t24_idx]
                loss_24h = F.mse_loss(z24_pred, z24_target)

                endpoint_loss = 0.4 * loss_6h + 0.6 * loss_24h

            elif time_type == "partial_6h":
                t6_idx = min(pred_timesteps - 1, target_timesteps - 1)
                z6_pred = predicted_trajectory[t6_idx]
                z6_target = target_trajectory[t6_idx]
                endpoint_loss = F.mse_loss(z6_pred, z6_target)

            elif time_type == "partial_24h":
                t24_idx = min(pred_timesteps - 1, target_timesteps - 1)
                z24_pred = predicted_trajectory[t24_idx]
                z24_target = target_trajectory[t24_idx]
                endpoint_loss = F.mse_loss(z24_pred, z24_target)

            else:
                endpoint_loss = torch.tensor(0.0, device=device, requires_grad=True)

            return self._ensure_scalar_loss(endpoint_loss, "endpoint_loss")

        except Exception:
            device = predicted_trajectory.device if hasattr(predicted_trajectory, 'device') else 'cpu'
            return torch.tensor(0.0, device=device, requires_grad=True)

    def forward(self, x0, drug_features, x6=None, x24=None, dose=None, time_type="complete",
                train_mode=True, guided_trajectory=None, data_indices=None):
        try:
            if x0 is None or drug_features is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                return torch.tensor(1.0, device=device, requires_grad=True)

            batch_size = x0.shape[0]
            device = x0.device

            if drug_features.shape[0] != batch_size:
                if drug_features.shape[0] > batch_size:
                    drug_features = drug_features[:batch_size]
                else:
                    repeat_count = batch_size - drug_features.shape[0]
                    last_sample = drug_features[-1:].repeat(repeat_count, 1)
                    drug_features = torch.cat([drug_features, last_sample], dim=0)

            dose_processed = self._safe_dose_processing_fixed(dose, batch_size, device)

            z0, mu0, logvar0 = self.encode_gene_expression(x0, is_drug_treated=False)

            z6 = None
            z24 = None
            mu6 = None
            logvar6 = None
            mu24 = None
            logvar24 = None

            if x6 is not None and x6.shape[0] == batch_size:
                z6, mu6, logvar6 = self.encode_gene_expression(x6, is_drug_treated=True)

            if x24 is not None and x24.shape[0] == batch_size:
                z24, mu24, logvar24 = self.encode_gene_expression(x24, is_drug_treated=True)

            condition, _, _ = self.condition_fusion(z0, drug_features, dose_processed)

            if train_mode:
                x0_rec = self.decode_to_gene_expression(z0)
                vae_loss_x0, _, _ = self.vae_loss(x0, x0_rec, mu0, logvar0)

                vae_loss_x6 = torch.tensor(0.0, device=device, requires_grad=True)
                if x6 is not None and z6 is not None:
                    x6_rec = self.decode_to_gene_expression(z6)
                    vae_loss_x6, _, _ = self.vae_loss(x6, x6_rec, mu6, logvar6)

                vae_loss_x24 = torch.tensor(0.0, device=device, requires_grad=True)
                if x24 is not None and z24 is not None:
                    x24_rec = self.decode_to_gene_expression(z24)
                    vae_loss_x24, _, _ = self.vae_loss(x24, x24_rec, mu24, logvar24)

                vae_count = 1
                if x6 is not None and z6 is not None:
                    vae_count += 1
                if x24 is not None and z24 is not None:
                    vae_count += 1
                vae_loss = (vae_loss_x0 + vae_loss_x6 + vae_loss_x24) / vae_count

                vae_loss = self._ensure_scalar_loss(vae_loss, "vae_loss")

                if guided_trajectory is None:
                    if not self._safe_tensor_check(vae_loss, "vae_loss_only"):
                        return torch.tensor(1.0, device=device, requires_grad=True)
                    return vae_loss

                guided_z_trajectory = self._process_guided_trajectory(guided_trajectory, device)
                if guided_z_trajectory is None:
                    if not self._safe_tensor_check(vae_loss, "vae_loss_fallback"):
                        return torch.tensor(1.0, device=device, requires_grad=True)
                    return vae_loss

                traj_length = guided_z_trajectory.shape[0]
                predicted_trajectory = self.generate_trajectory_steps(z0, drug_features, traj_length, dose_processed)

                if not self._safe_tensor_check(predicted_trajectory, "predicted_trajectory"):
                    return vae_loss

                if predicted_trajectory.shape[1] != batch_size:
                    return vae_loss

                trajectory_loss = self._compute_trajectory_following_loss(
                    predicted_trajectory, guided_z_trajectory
                )

                endpoint_loss = self._compute_endpoint_loss(
                    predicted_trajectory, guided_z_trajectory, time_type
                )

                if time_type == "complete":
                    type_weight = 2.0
                elif time_type == "partial_6h":
                    type_weight = 1.2
                elif time_type == "partial_24h":
                    type_weight = 1.5
                else:
                    type_weight = 1.0

                total_loss = self._safe_loss_computation_fixed(vae_loss, trajectory_loss, endpoint_loss, type_weight)
                total_loss = self._ensure_scalar_loss(total_loss, "final_total_loss")

                return total_loss
            else:
                if time_type == "partial_6h":
                    traj_length = self.timesteps // 2 + 1
                else:
                    traj_length = self.timesteps

                z_trajectory = self.generate_trajectory_steps(z0, drug_features, traj_length, dose_processed)

                gene_trajectory = []
                for z_t in z_trajectory:
                    x_t = self.decode_to_gene_expression(z_t)
                    gene_trajectory.append(x_t)

                gene_trajectory_tensor = torch.stack(gene_trajectory)

                gene_trajectory_tensor = gene_trajectory_tensor.transpose(0, 1)

                return gene_trajectory_tensor

        except Exception:
            device = x0.device if hasattr(x0, 'device') else 'cpu'
            return torch.tensor(1.0, device=device, requires_grad=True)

    def generate_trajectory_steps(self, z0, drug_features, traj_length=None, dose=None):
        try:
            if traj_length is None:
                traj_length = self.timesteps

            batch_size = z0.shape[0]
            device = z0.device

            condition, _, _ = self.condition_fusion(z0, drug_features, dose)
            current_z = z0.clone()
            all_states = [current_z.clone()]

            for t in range(traj_length - 1):
                t_tensor = torch.full((batch_size,), t, device=device).long()
                delta = self.transform_net(current_z, t_tensor, condition)
                next_z = current_z + delta
                all_states.append(next_z.clone())
                current_z = next_z.clone()

            return torch.stack(all_states)

        except Exception:
            device = z0.device
            traj_length = traj_length or self.timesteps
            return torch.zeros(traj_length, z0.shape[0], self.n_latent, device=device)

    def generate_trajectory(self, x0, drug_features, dose=None, time_type="complete"):
        self.eval()

        with torch.no_grad():
            batch_size = x0.shape[0]
            device = x0.device

            z0, _, _ = self.encode_gene_expression(x0, is_drug_treated=False)

            dose_processed = self._safe_dose_processing_fixed(dose, batch_size, device)

            if time_type == "partial_6h":
                traj_length = self.time_mapping_config["6h_step"] + 1
            elif time_type == "partial_24h":
                traj_length = self.time_mapping_config["24h_step"] + 1
            else:
                traj_length = self.timesteps

            z_trajectory = self.generate_trajectory_steps(z0, drug_features, traj_length, dose_processed)

            gene_trajectory = []
            for z_t in z_trajectory:
                x_t = self.decode_to_gene_expression(z_t)
                gene_trajectory.append(x_t)

            return torch.stack(gene_trajectory)

    def generate_partial_trajectory(self, x0, drug_features, target_time=6):
        self.eval()

        with torch.no_grad():
            z0, _, _ = self.encode_gene_expression(x0, is_drug_treated=False)
            time_step = self.map_real_time_to_steps(torch.tensor([target_time], device=self.device)).item()

            if target_time <= 6:
                time_type = "partial_6h"
                traj_length = self.timesteps // 2 + 1
            else:
                time_type = "partial_24h"
                traj_length = self.timesteps

            z_trajectory = self.generate_trajectory_steps(z0, drug_features, traj_length)
            z_target = z_trajectory[min(time_step, traj_length - 1)]
            x_pred = self.decode_to_gene_expression(z_target)

        return x_pred

    def infer_6h_24h(self, x0, drug_features):
        self.eval()

        with torch.no_grad():
            z0, _, _ = self.encode_gene_expression(x0, is_drug_treated=False)
            z_trajectory = self.generate_trajectory_steps(z0, drug_features)

            t6_idx = self.time_mapping_config["6h_step"]
            t24_idx = self.time_mapping_config["24h_step"]

            z6 = z_trajectory[t6_idx]
            z24 = z_trajectory[t24_idx]

            x6_pred = self.decode_to_gene_expression(z6)
            x24_pred = self.decode_to_gene_expression(z24)

        return x6_pred, x24_pred

    def visualize_trajectory(self, x0, drug_features, target_trajectory=None, x6=None, x24=None,
                             time_type="complete", num_genes=5, save_path=None):
        self.eval()

        with torch.no_grad():
            z0, _, _ = self.encode_gene_expression(x0, is_drug_treated=False)

            z6 = None
            z24 = None

            if x6 is not None:
                z6, _, _ = self.encode_gene_expression(x6, is_drug_treated=True)

            if x24 is not None:
                z24, _, _ = self.encode_gene_expression(x24, is_drug_treated=True)

            if time_type == "partial_6h":
                traj_length = self.timesteps // 2 + 1
            else:
                traj_length = self.timesteps

            predicted_trajectory = self.generate_trajectory_steps(z0, drug_features, traj_length)

            trajectory = []
            for z_t in predicted_trajectory:
                x_t = self.decode_to_gene_expression(z_t)
                trajectory.append(x_t)

            trajectory = torch.stack(trajectory)

            target_trajectory_gene = None
            if target_trajectory is not None:
                target_traj_gene = []
                for z_t in target_trajectory:
                    x_t = self.decode_to_gene_expression(z_t)
                    target_traj_gene.append(x_t)
                target_trajectory_gene = torch.stack(target_traj_gene)

            trajectory_np = trajectory.cpu().numpy()
            x0_np = x0.cpu().numpy()

            if x6 is not None:
                x6_np = x6.cpu().numpy()

            if x24 is not None:
                x24_np = x24.cpu().numpy()

            if target_trajectory_gene is not None:
                target_traj_np = target_trajectory_gene.cpu().numpy()

            if time_type == "complete" and x6 is not None and x24 is not None:
                gene_changes = np.abs(x24_np[0] - x0_np[0])
                top_genes = np.argsort(gene_changes)[-num_genes:]
            elif time_type == "partial_6h" and x6 is not None:
                gene_changes = np.abs(x6_np[0] - x0_np[0])
                top_genes = np.argsort(gene_changes)[-num_genes:]
            elif time_type == "partial_24h" and x24 is not None:
                gene_changes = np.abs(x24_np[0] - x0_np[0])
                top_genes = np.argsort(gene_changes)[-num_genes:]
            else:
                top_genes = np.random.choice(self.n_genes, num_genes, replace=False)

            plt.figure(figsize=(15, 10))

            for i, gene_idx in enumerate(top_genes):
                gene_trajectory = trajectory_np[:, 0, gene_idx]
                plt.plot(gene_trajectory, label=f'Predicted Gene {gene_idx}', linewidth=2.5)

                if target_trajectory_gene is not None:
                    target_gene_traj = target_traj_np[:min(len(target_traj_np), len(trajectory_np)), 0, gene_idx]
                    plt.plot(target_gene_traj, label=f'Target Gene {gene_idx}',
                             linestyle='--', linewidth=2.0, alpha=0.7)

                plt.scatter(0, x0_np[0, gene_idx], color='blue', s=100, zorder=5)

                if time_type == "complete" or time_type == "partial_6h":
                    if x6 is not None:
                        t6_idx = self.time_mapping_config["6h_step"]
                        plt.scatter(t6_idx, x6_np[0, gene_idx],
                                    color='green', s=100, zorder=5)

                if time_type == "complete" or time_type == "partial_24h":
                    if x24 is not None:
                        t24_idx = self.time_mapping_config["24h_step"]
                        plt.scatter(t24_idx, x24_np[0, gene_idx],
                                    color='red', s=100, zorder=5)

            plt.xlabel('Time steps', fontsize=14)
            plt.ylabel('Gene expression', fontsize=14)
            plt.title(f'Gene Expression Trajectory', fontsize=16)

            time_labels = []
            if time_type == "complete":
                time_labels = [(0, "0h"), (self.time_mapping_config["6h_step"], "6h"),
                               (self.time_mapping_config["24h_step"], "24h")]
            elif time_type == "partial_6h":
                time_labels = [(0, "0h"), (self.time_mapping_config["6h_step"], "6h")]
            elif time_type == "partial_24h":
                time_labels = [(0, "0h"), (self.time_mapping_config["24h_step"], "24h")]

            plt.xticks([label[0] for label in time_labels],
                       [label[1] for label in time_labels], fontsize=12)

            plt.grid(True, alpha=0.3)
            plt.legend()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')

            plt.close()

    def eval_x_reconstruction(self, x1, x2, x2_pred, metrics_func=['pearson']):
        x1_np = x1.data.cpu().numpy().astype(float)
        x2_np = x2.data.cpu().numpy().astype(float)
        x2_pred_np = x2_pred.data.cpu().numpy().astype(float)

        DEG_np = x2_np - x1_np
        DEG_pert_np = x2_pred_np - x1_np

        metrics_dict = defaultdict(float)
        metrics_dict_ls = defaultdict(list)

        batch_size = x1_np.shape[0]

        def get_metric_func(metric_name):
            if metric_name == 'pearson':
                def pearson_corr(y_true, y_pred):
                    return np.corrcoef(y_true, y_pred)[0, 1]

                return pearson_corr
            elif metric_name == 'rmse':
                def rmse(y_true, y_pred):
                    return np.sqrt(np.mean((y_true - y_pred) ** 2))

                return rmse
            elif 'precision' in metric_name:
                k = int(metric_name.replace('precision', ''))

                def precision_at_k(y_true, y_pred):
                    true_indices_pos = np.argsort(y_true)[-k:]
                    true_indices_neg = np.argsort(y_true)[:k]

                    pred_indices_pos = np.argsort(y_pred)[-k:]
                    pred_indices_neg = np.argsort(y_pred)[:k]

                    precision_pos = len(np.intersect1d(true_indices_pos, pred_indices_pos)) / k
                    precision_neg = len(np.intersect1d(true_indices_neg, pred_indices_neg)) / k

                    return precision_neg, precision_pos

                return precision_at_k
            else:
                def identity(y_true, y_pred):
                    return 0.0

                return identity

        for m in metrics_func:
            if m in ['precision10', 'precision20', 'precision50', 'precision100', 'precision200']:
                for i in range(batch_size):
                    precision_neg, precision_pos = get_metric_func(m)(x2_np[i, :], x2_pred_np[i, :])
                    metrics_dict_ls['x2_pred_neg_' + m].append(precision_neg)
                    metrics_dict_ls['x2_pred_pos_' + m].append(precision_pos)
            else:
                for i in range(batch_size):
                    metric_value = get_metric_func(m)(x2_np[i, :], x2_pred_np[i, :])
                    metrics_dict_ls['x2_pred_' + m].append(metric_value)

        for m in metrics_func:
            if m in ['precision10', 'precision20', 'precision50', 'precision100', 'precision200']:
                for i in range(batch_size):
                    precision_neg, precision_pos = get_metric_func(m)(DEG_np[i, :], DEG_pert_np[i, :])
                    metrics_dict_ls['DEG_pred_neg_' + m].append(precision_neg)
                    metrics_dict_ls['DEG_pred_pos_' + m].append(precision_pos)
            else:
                for i in range(batch_size):
                    metric_value = get_metric_func(m)(DEG_np[i, :], DEG_pert_np[i, :])
                    metrics_dict_ls['DEG_pred_' + m].append(metric_value)

        for k, v in metrics_dict_ls.items():
            metrics_dict[k] = np.nanmean(v)

        return metrics_dict, metrics_dict_ls

    def save_model(self, path, optimizer=None, scheduler=None, epoch=None, valid_loss=None, args=None):
        model_state = {
            'model_state_dict': self.state_dict(),
            'n_genes': self.n_genes,
            'n_latent': self.n_latent,
            'features_dim': self.features_dim,
            'timesteps': self.timesteps,
            'time_mapping_config': self.time_mapping_config,
            'train_config': self.train_config,
            'beta': self.beta
        }

        if optimizer is not None:
            model_state['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            model_state['scheduler_state_dict'] = scheduler.state_dict()
        if epoch is not None:
            model_state['epoch'] = epoch
        if valid_loss is not None:
            model_state['valid_loss'] = valid_loss
        if args is not None:
            model_state['args'] = args

        torch.save(model_state, path)

    def load_model(self, path, optimizer=None, scheduler=None):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])

        if 'time_mapping_config' in checkpoint:
            self.time_mapping_config = checkpoint['time_mapping_config']
        if 'train_config' in checkpoint:
            self.train_config = checkpoint['train_config']
        if 'beta' in checkpoint:
            self.beta = checkpoint['beta']

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        loaded_info = {}
        for key in ['epoch', 'valid_loss', 'args']:
            if key in checkpoint:
                loaded_info[key] = checkpoint[key]

        return loaded_info