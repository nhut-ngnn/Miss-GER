import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dual_stream_prompt import (
    MissingModalityPromptBank,
    TextGuidedCrossAttentionPromptStream,
    _get_valid_num_heads,
    apply_missing_modality_dropout,
    missing_mod_to_availability_mask,
)


class RelationalGraphLayer(nn.Module):
    """
    Relational message passing over the 3 modality nodes [text, audio, visual].

    Every ordered pair (src -> dst) owns its own transform and attention bias, so
    the layer models 6 typed relations. Messages are only sent from nodes flagged
    as valid sources in `source_mask` [B, 3].
    """

    def __init__(self, d_model: int, dropout: float = 0.0, num_nodes: int = 3):
        super().__init__()
        self.d_model = int(d_model)
        self.num_nodes = int(num_nodes)

        self.rel_weight = nn.Parameter(
            torch.empty(self.num_nodes, self.num_nodes, self.d_model, self.d_model)
        )
        self.rel_bias = nn.Parameter(torch.zeros(self.num_nodes, self.num_nodes))
        self.query = nn.Linear(self.d_model, self.d_model, bias=False)
        self.key = nn.Linear(self.d_model, self.d_model, bias=False)

        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(self.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, 2 * self.d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(2 * self.d_model, self.d_model),
            nn.Dropout(float(dropout)),
        )
        self.ffn_norm = nn.LayerNorm(self.d_model)

        eye = torch.eye(self.num_nodes, dtype=torch.bool)
        self.register_buffer("self_edge", eye, persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        # Start close to identity-scaled transforms so early messages stay stable.
        nn.init.normal_(self.rel_weight, mean=0.0, std=1.0 / math.sqrt(self.d_model))

    def forward(
        self,
        nodes: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # nodes: [B, N, D], source_mask: [B, N] (1 = node may send messages)
        q = self.query(nodes)  # [B, N(dst), D]
        k = self.key(nodes)  # [B, N(src), D]

        # e[b, s, t] = <k_s, q_t> / sqrt(D) + bias[s, t]
        scores = torch.einsum("bsd,btd->bst", k, q) / math.sqrt(self.d_model)
        scores = scores + self.rel_bias.unsqueeze(0)

        valid = (source_mask > 0.5).unsqueeze(2) & ~self.self_edge.unsqueeze(0)  # [B, s, t]
        scores = scores.masked_fill(~valid, -1e4)
        alpha = torch.softmax(scores, dim=1) * valid.to(scores.dtype)  # [B, s, t]

        # msg[b, s, t] = W_{s->t} h_s
        msg = torch.einsum("bsd,stde->bste", nodes, self.rel_weight)
        agg = torch.einsum("bst,bste->bte", alpha, msg)

        out = self.norm(nodes + self.dropout(agg))
        out = self.ffn_norm(out + self.ffn(out))
        return out, alpha


class RelationalGraphPromptNetwork(nn.Module):
    """
    Two-stream relational-graph model for missing-modality emotion recognition.

    Stream A (full modality): modality nodes -> relational graph -> text-guided
        fusion. Trained with cross-modal contrastive learning on samples whose
        three modalities are all present.
    Stream B (missing modality): reconstruct-then-fuse. Missing nodes start from
        learnable missing prompts, are reconstructed from the available nodes by
        relational message passing, then refined and fused.

    Each sample is routed to the stream matching its availability mask. The
    classifier is shared so stage-1 (full-modality) weights transfer to stage 2.

    Forward remains compatible with the repository pipeline:
        forward(x_l, x_a, x_v, missing_mod=None, missing_mask=None, return_aux=False)
    """

    is_relational_graph = True

    def __init__(self, hyp_params):
        super().__init__()
        self.orig_d_l = int(hyp_params.orig_d_l)
        self.orig_d_a = int(hyp_params.orig_d_a)
        self.orig_d_v = int(hyp_params.orig_d_v)
        self.d_model = int(hyp_params.proj_dim)
        self.output_dim = int(hyp_params.output_dim)
        self.embed_dropout = float(getattr(hyp_params, "embed_dropout", 0.0))
        self.dropout = float(getattr(hyp_params, "out_dropout", 0.0))
        self.prompt_dropout = float(getattr(hyp_params, "prompt_dropout", 0.0))
        self.missing_modality_dropout = float(
            getattr(hyp_params, "missing_modality_dropout", 0.0)
        )

        requested_heads = int(getattr(hyp_params, "cross_attn_heads", 0))
        if requested_heads <= 0:
            requested_heads = int(getattr(hyp_params, "num_heads", 1))
        self.cross_attn_heads = _get_valid_num_heads(self.d_model, requested_heads)

        self.proj_l = nn.Linear(self.orig_d_l, self.d_model, bias=False)
        self.proj_a = nn.Linear(self.orig_d_a, self.d_model, bias=False)
        self.proj_v = nn.Linear(self.orig_d_v, self.d_model, bias=False)
        self.proj_norm_l = nn.LayerNorm(self.d_model)
        self.proj_norm_a = nn.LayerNorm(self.d_model)
        self.proj_norm_v = nn.LayerNorm(self.d_model)
        self.input_dropout = nn.Dropout(self.embed_dropout)

        self.prompt_bank = MissingModalityPromptBank(
            d_model=self.d_model,
            dropout=self.prompt_dropout,
        )

        # Stream A: full-modality graph + fusion + contrastive projection.
        self.full_graph = RelationalGraphLayer(self.d_model, self.dropout)
        self.full_fusion = TextGuidedCrossAttentionPromptStream(
            self.d_model, self.cross_attn_heads, self.dropout
        )
        self.contrast_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )

        # Stream B: reconstruct (graph -> head), then refine graph + fusion.
        self.recon_graph = RelationalGraphLayer(self.d_model, self.dropout)
        self.recon_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.fuse_graph = RelationalGraphLayer(self.d_model, self.dropout)
        self.miss_fusion = TextGuidedCrossAttentionPromptStream(
            self.d_model, self.cross_attn_heads, self.dropout
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.output_dim),
        )

    def _project_and_pool(
        self,
        x: torch.Tensor,
        projector: nn.Linear,
        norm: nn.LayerNorm,
        name: str,
    ) -> torch.Tensor:
        if x.dim() == 2:
            return norm(projector(self.input_dropout(x)))
        if x.dim() != 3:
            raise ValueError(f"{name} must have shape [B, D] or [B, T, D].")

        non_padding = (x.abs().sum(dim=-1) > 0.0).to(dtype=x.dtype)
        h = norm(projector(self.input_dropout(x)))
        denom = non_padding.sum(dim=1, keepdim=True).clamp_min(1.0)
        return torch.sum(h * non_padding.unsqueeze(-1), dim=1) / denom

    def _resolve_missing_mask(
        self,
        missing_mod: Optional[torch.Tensor],
        missing_mask: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        source = missing_mask if missing_mask is not None else missing_mod
        mask = missing_mod_to_availability_mask(source, batch_size, device)
        if (mask.sum(dim=1) < 1.0).any():
            raise ValueError("Every sample must have at least one available modality.")
        return apply_missing_modality_dropout(
            mask, self.missing_modality_dropout, self.training
        )

    def _full_stream(self, nodes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ones = torch.ones(nodes.size(0), nodes.size(1), device=nodes.device)
        graph_nodes, _ = self.full_graph(nodes, ones)
        z, _ = self.full_fusion(graph_nodes)
        return z, graph_nodes

    def _missing_stream(
        self,
        h: torch.Tensor,
        missing_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        avail = missing_mask.to(h.dtype).unsqueeze(-1)
        modality_prompt = self.prompt_bank.modality_prompts.unsqueeze(0)
        missing_prompt = self.prompt_bank.missing_prompts.unsqueeze(0)

        # Replace (not scale) missing nodes so held-out features never leak in.
        init_nodes = avail * (h + modality_prompt) + (1.0 - avail) * (
            missing_prompt + modality_prompt
        )

        # Reconstruct: available nodes -> missing nodes through typed relations.
        recon_nodes, _ = self.recon_graph(init_nodes, missing_mask)
        recon = self.recon_head(recon_nodes)

        # Fuse: keep observed nodes, fill missing ones with reconstructions.
        completed = avail * (h + modality_prompt) + (1.0 - avail) * (
            recon + modality_prompt
        )
        ones = torch.ones_like(missing_mask)
        refined, _ = self.fuse_graph(completed, ones)
        z, _ = self.miss_fusion(refined)
        return z, recon, completed

    def forward(
        self,
        x_l: torch.Tensor,
        x_a: torch.Tensor,
        x_v: torch.Tensor,
        missing_mod: Optional[torch.Tensor] = None,
        missing_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        batch_size = x_l.size(0)
        device = x_l.device
        if x_a.size(0) != batch_size or x_v.size(0) != batch_size:
            raise ValueError("All modalities must share the same batch size.")

        text = self._project_and_pool(x_l, self.proj_l, self.proj_norm_l, "text")
        audio = self._project_and_pool(x_a, self.proj_a, self.proj_norm_a, "audio")
        visual = self._project_and_pool(x_v, self.proj_v, self.proj_norm_v, "visual")
        h = torch.stack([text, audio, visual], dim=1)  # [B, 3, D]

        missing_mask = self._resolve_missing_mask(
            missing_mod, missing_mask, batch_size, device
        )
        is_full = missing_mask.min(dim=1).values > 0.5  # [B]

        modality_prompt = self.prompt_bank.modality_prompts.unsqueeze(0)
        z_full, _ = self._full_stream(h + modality_prompt)
        z_miss, recon, _ = self._missing_stream(h, missing_mask)

        z_cross = torch.where(is_full.unsqueeze(-1), z_full, z_miss)
        logits = self.classifier(z_cross)

        if not return_aux:
            return logits

        return {
            "logits": logits,
            "z_cross": z_cross,
            "z_full": z_full,
            "z_miss": z_miss,
            "is_full": is_full,
            "missing_mask": missing_mask,
            "real_nodes": h,
            "recon_nodes": recon,
            "contrast_emb": F.normalize(self.contrast_proj(h), dim=-1),
        }


def cross_modal_contrastive_loss(
    embeddings: torch.Tensor,
    is_full: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Symmetric InfoNCE between every modality pair (t-a, t-v, a-v) on samples
    whose modalities are all present. Positives are the same sample across
    modalities; other samples in the batch are negatives.
    """
    emb = embeddings[is_full]
    if emb.size(0) < 2:
        return embeddings.new_tensor(0.0)

    targets = torch.arange(emb.size(0), device=emb.device)
    loss = emb.new_tensor(0.0)
    pairs = ((0, 1), (0, 2), (1, 2))
    for i, j in pairs:
        logits = emb[:, i] @ emb[:, j].t() / temperature
        loss = loss + 0.5 * (
            F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)
        )
    return loss / len(pairs)


def relational_graph_auxiliary_loss(
    outputs: Dict[str, torch.Tensor],
    lambda_con: float = 0.1,
    lambda_rec: float = 0.5,
    lambda_kd: float = 0.1,
    temperature: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """
    Auxiliary objectives for the two streams.

        con: contrastive alignment of full-modality samples (stream A).
        rec: reconstruction of missing nodes from available ones (stream B).
        kd:  pull the missing-stream representation to the full-stream one.
    """
    is_full = outputs["is_full"]
    mask = outputs["missing_mask"]
    logits = outputs["logits"]
    zero = logits.new_tensor(0.0)

    con = cross_modal_contrastive_loss(outputs["contrast_emb"], is_full, temperature)

    miss_nodes = mask < 0.5  # [B, 3]
    if miss_nodes.any():
        recon = outputs["recon_nodes"][miss_nodes]
        real = outputs["real_nodes"].detach()[miss_nodes]
        rec = F.smooth_l1_loss(recon, real) + (
            1.0 - F.cosine_similarity(recon, real, dim=-1).mean()
        )
    else:
        rec = zero

    miss_samples = ~is_full
    if miss_samples.any():
        kd = 1.0 - F.cosine_similarity(
            outputs["z_miss"][miss_samples],
            outputs["z_full"].detach()[miss_samples],
            dim=-1,
        ).mean()
    else:
        kd = zero

    total = lambda_con * con + lambda_rec * rec + lambda_kd * kd
    return {
        "loss": total,
        "loss_con": con.detach(),
        "loss_rec": rec.detach(),
        "loss_kd": kd.detach(),
    }
