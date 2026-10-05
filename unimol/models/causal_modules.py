# Copyright (c) 2026 The CausalBind Authors.
# Core modules for the CausalBind variants.
# Adapted from CausalDrugCLIP for hyperbolic space.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ConceptExtractor(nn.Module):
    """
    Extracts concept representations from molecular/pocket embeddings.
    Uses a simplified Perceiver-style architecture.

    Input: (batch_size, input_dim) - CLS token representation
    Output: (batch_size, num_concepts, concept_dim) - concept tokens
    """

    def __init__(
        self,
        input_dim: int = 512,
        num_concepts: int = 64,
        concept_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_concepts = num_concepts
        self.concept_dim = concept_dim

        # Learnable concept queries
        self.concept_queries = nn.Parameter(
            torch.randn(num_concepts, concept_dim) * 0.02
        )

        # Project input to concept dimension
        self.input_proj = nn.Linear(input_dim, concept_dim)

        # Cross-attention layers (queries attend to input)
        self.layers = nn.ModuleList([
            PerceiverBlock(
                dim=concept_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(concept_dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        """
        Args:
            x: (batch_size, input_dim) or (batch_size, seq_len, input_dim)
            key_padding_mask: optional (batch_size, seq_len) bool tensor, True at
                padded positions to exclude from cross-attention. Only meaningful
                when x is 3D (per-token input); ignored for 2D (pooled) input.
        Returns:
            concepts: (batch_size, num_concepts, concept_dim)
        """
        batch_size = x.shape[0]

        # Handle both 2D and 3D inputs
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, D)
            key_padding_mask = None

        # Project input
        x = self.input_proj(x)  # (B, seq_len, concept_dim)

        # Expand concept queries for batch
        queries = self.concept_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Apply perceiver layers. Attention is exposed only on request so the
        # training/inference API and checkpoint state remain unchanged.
        attention_layers = []
        for layer in self.layers:
            if return_attention:
                queries, attention = layer(
                    queries,
                    x,
                    key_padding_mask=key_padding_mask,
                    return_attention=True,
                )
                attention_layers.append(attention)
            else:
                queries = layer(queries, x, key_padding_mask=key_padding_mask)

        concepts = self.norm(queries)
        if return_attention:
            # (num_layers, B, num_heads, num_concepts, seq_len)
            return concepts, torch.stack(attention_layers, dim=0)
        return concepts


class PerceiverBlock(nn.Module):
    """Single Perceiver block with cross-attention and FFN."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        kv: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        """
        Args:
            queries: (B, num_queries, dim) - concept queries
            kv: (B, seq_len, dim) - input features
            key_padding_mask: optional (B, seq_len) bool tensor, True at padded
                positions to exclude from attention.
        Returns:
            updated queries: (B, num_queries, dim)
        """
        # Cross-attention
        q = self.norm1(queries)
        kv_normed = self.norm_kv(kv)
        attn_out, attention = self.cross_attn(
            q,
            kv_normed,
            kv_normed,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        queries = queries + attn_out

        # FFN
        queries = queries + self.ffn(self.norm2(queries))

        if return_attention:
            return queries, attention
        return queries


class TokenPoolConceptExtractor(nn.Module):
    """Attention-free token-level control for concept extraction.

    Each valid encoder token is transformed independently by one shared MLP,
    then parameter-free adaptive average pooling resamples the variable-length
    token sequence to ``num_concepts`` slots. Unlike ``ConceptExtractor``, this
    module has no learned concept queries, cross-attention, or Perceiver FFNs.
    It is intended as a matched "simple token features + same KxK mask"
    ablation, not as a new proposed architecture.
    """

    def __init__(
        self,
        input_dim: int = 512,
        num_concepts: int = 64,
        concept_dim: int = 256,
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.token_mlp = nn.Sequential(
            nn.Linear(input_dim, concept_dim),
            nn.GELU(),
            nn.LayerNorm(concept_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.dim() == 2:
            # The protein branch is CLS-pooled in the matched architecture.
            # Repeat its independently projected feature to K slots.
            projected = self.token_mlp(x)
            return projected.unsqueeze(1).expand(
                -1, self.num_concepts, -1
            )

        projected = self.token_mlp(x)
        batch_concepts = []
        for batch_index in range(projected.shape[0]):
            if key_padding_mask is None:
                valid_indices = torch.arange(
                    projected.shape[1], device=projected.device
                )
            else:
                valid_indices = torch.nonzero(
                    ~key_padding_mask[batch_index], as_tuple=False
                ).flatten()

            # Uni-Mol sequences are [CLS], atoms..., [SEP]. Pool only atoms.
            if valid_indices.numel() > 2:
                valid_indices = valid_indices[1:-1]
            elif valid_indices.numel() == 0:
                valid_indices = torch.zeros(
                    1, dtype=torch.long, device=projected.device
                )

            token_features = projected[batch_index, valid_indices]
            pooled = F.adaptive_avg_pool1d(
                token_features.transpose(0, 1).unsqueeze(0),
                self.num_concepts,
            )
            batch_concepts.append(
                pooled.squeeze(0).transpose(0, 1)
            )
        return torch.stack(batch_concepts, dim=0)


class SparseMask(nn.Module):
    """
    Learnable sparse mask with multiple variants selected via ``mask_type``.

    Sparsity variants (tested 2026-04-17):
      - "tanh_plus_1" : original design (tanh(M)+1, hard threshold). Baseline.
      - "relu"        : Option A. ReLU(tanh(M)) in [0, 1]; true zeros.
      - "ste"         : Option B. Straight-through estimator on tanh+1 + threshold.
      - "shrink"      : Option C. ReLU(tanh(M)+1 - tau) with learnable tau.
      - "neg_init"    : Option D. Same as tanh_plus_1 but mask_logits initialized negative.
      - "hard_concrete": Hard Concrete gate (Louizos et al. 2018) with L0 proxy loss.

    Diagnostic variants to isolate why the mask helps even without sparsity:
      - "fixed"       : randomly-initialized mask, FROZEN during training (random projection test).
      - "diagonal"    : learnable 1-D per-dimension weights (only 256 params; tests if per-dim
                        re-weighting alone is sufficient; no cross-dim mixing).
      - "orthogonal"  : orthogonal-initialized M with soft orthogonality penalty (tests whether
                        non-orthogonal rotation matters).
      - "linear"      : plain unconstrained Linear(n, n); no tanh, no +1 (cleanest ablation of
                        "V1 is just an extra linear layer").

    Usage:
        m = SparseMask(num_concepts=256, mask_type="relu")
        out, gate = m(x)
        sparse_loss = m.get_sparsity_loss()
    """

    def __init__(
        self,
        num_concepts: int = 256,
        threshold: float = 0.05,
        init_scale: float = 0.1,
        mask_type: str = "tanh_plus_1",
        mask_rank: int = 8,
        # Hard Concrete params
        hc_beta: float = 2.0 / 3.0,
        hc_gamma: float = -0.1,
        hc_zeta: float = 1.1,
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.threshold = threshold
        self.mask_type = mask_type
        self.mask_rank = mask_rank
        self.hc_beta = hc_beta
        self.hc_gamma = hc_gamma
        self.hc_zeta = hc_zeta

        # Initial logits distribution depends on the variant.
        if mask_type == "relu":
            # Start positive so all entries active; L1 will later drive some negative.
            init_mean = 0.5
        elif mask_type == "neg_init":
            # Start very negative so mask ~= 0 initially; model must learn what to wake up.
            init_mean = -3.0
        elif mask_type == "hard_concrete":
            # log_alpha starts at 0 => E[gate>0] ~= 0.83 with default gamma/zeta.
            init_mean = 0.0
        else:  # tanh_plus_1, ste, shrink
            init_mean = 0.0

        # Parameter shape depends on variant.
        if mask_type == "diagonal":
            # Only a per-dimension gain vector (1D).
            self.mask_logits = nn.Parameter(
                torch.randn(num_concepts) * init_scale + 1.0
            )
        elif mask_type == "orthogonal":
            # Start from a random orthogonal matrix.
            w = torch.empty(num_concepts, num_concepts)
            nn.init.orthogonal_(w)
            self.mask_logits = nn.Parameter(w)
        elif mask_type == "linear":
            # Plain unconstrained Linear(n, n) weights initialized near identity.
            self.mask_logits = nn.Parameter(
                torch.eye(num_concepts) + torch.randn(num_concepts, num_concepts) * init_scale
            )
        elif mask_type == "low_rank":
            # CausalBind-LR: explicit rank-r factorization M_raw = U V^T, with U, V in R^{K x r}.
            # Yields tanh(U V^T) + 1, hard-rank-r structure (no L1 needed for rank
            # control; L1 may still be used as scale shrinkage).
            r = max(1, int(mask_rank))
            # Init small so U V^T ~= 0 ⇒ σ(M) ~= 1 everywhere (matches tanh_plus_1 init).
            scale = init_scale / max(1, r) ** 0.5
            self.mask_U = nn.Parameter(torch.randn(num_concepts, r) * scale)
            self.mask_V = nn.Parameter(torch.randn(num_concepts, r) * scale)
            # Keep a reference attribute for fp16 flatten compatibility (not used).
            self.mask_logits = self.mask_U  # alias; never mutated independently
        else:
            self.mask_logits = nn.Parameter(
                torch.randn(num_concepts, num_concepts) * init_scale + init_mean
            )

        # "fixed" variant: freeze the mask (random projection baseline).
        if mask_type == "fixed":
            self.mask_logits.requires_grad = False

        # Shrinkage threshold (Option C): learnable scalar (as 1-D tensor for
        # compatibility with unicore's FP16 optimizer `flatten_parameters`).
        if mask_type == "shrink":
            self.tau = nn.Parameter(torch.tensor([1.0]))

    def _compute_mask(self) -> torch.Tensor:
        M = self.mask_logits
        if self.mask_type == "tanh_plus_1":
            mask = torch.tanh(M) + 1.0
            mask = mask * (mask > self.threshold).to(mask.dtype)
            return mask

        elif self.mask_type == "relu":
            return F.relu(torch.tanh(M))  # in [0, 1], true zeros when M<0

        elif self.mask_type == "ste":
            mask_soft = torch.tanh(M) + 1.0
            mask_hard = (mask_soft > self.threshold).to(mask_soft.dtype)
            # Straight-through: forward uses hard gate, backward uses soft gradient.
            return mask_soft + (mask_hard - mask_soft).detach()

        elif self.mask_type == "shrink":
            mask_raw = torch.tanh(M) + 1.0  # in [0, 2]
            return F.relu(mask_raw - self.tau)  # shrinkage with learnable tau

        elif self.mask_type == "neg_init":
            mask = torch.tanh(M) + 1.0
            mask = mask * (mask > self.threshold).to(mask.dtype)
            return mask

        elif self.mask_type == "hard_concrete":
            if self.training:
                eps = 1e-6
                u = torch.rand_like(M).clamp_(eps, 1.0 - eps)
                s = torch.sigmoid((torch.log(u) - torch.log(1.0 - u) + M) / self.hc_beta)
            else:
                # Deterministic: take u=0.5
                s = torch.sigmoid(M / self.hc_beta)
            s_bar = s * (self.hc_zeta - self.hc_gamma) + self.hc_gamma
            return s_bar.clamp(0.0, 1.0)

        elif self.mask_type == "fixed":
            # Random frozen mask; apply tanh+1 shape for comparable scale to V1.
            return torch.tanh(M) + 1.0

        elif self.mask_type == "diagonal":
            # Per-dimension weights — return as 1-D; forward uses element-wise mul.
            return M  # shape (n,)

        elif self.mask_type == "orthogonal":
            # Plain matrix; orthogonality is enforced via loss, not reparameterization.
            return M

        elif self.mask_type == "linear":
            # Plain unconstrained linear.
            return M

        elif self.mask_type == "low_rank":
            # CausalBind-LR: M_raw = U V^T, then σ(·) = tanh + 1, then 0.5-gate (same as tanh_plus_1
            # post-processing). The rank-r structural constraint is enforced by
            # construction; the gate is retained so that masks can still produce hard
            # zeros if the bilinear form goes sufficiently negative.
            M_raw = self.mask_U @ self.mask_V.t()           # (K, K), rank ≤ r
            mask = torch.tanh(M_raw) + 1.0
            mask = mask * (mask > self.threshold).to(mask.dtype)
            return mask

        else:
            raise ValueError(f"Unknown mask_type: {self.mask_type}")

    def forward(self, mol_concepts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the mask along the LAST dimension for 2D input (B, D) (V1 style),
        or along the second-to-last dimension (K, concept axis) for 3D input
        (B, K, D) (V2 style, concept-level masking)."""
        mask = self._compute_mask()
        mask = mask.to(mol_concepts.dtype)

        if self.mask_type == "diagonal":
            if mol_concepts.dim() == 2:          # (B, n) * (n,)
                causal_effect = mol_concepts * mask
            else:                                 # (B, K, D) — broadcast along D
                causal_effect = mol_concepts * mask.view(1, -1, 1)
        else:
            if mol_concepts.dim() == 2:          # V1: (B, D) @ (D, D)
                causal_effect = torch.matmul(mol_concepts, mask)
            else:                                 # V2: (B, K, D), mask acts on K axis
                # Apply mask as WEIGHTED MEAN (not sum) over the K axis so that
                # the output scale does NOT blow up by a factor of K. Otherwise
                # with mask ~= 1 everywhere at init, each row produces ~K * mean,
                # inflating concept magnitudes K-fold and crashing the downstream
                # exp_map0 with NaN.
                K = mask.shape[0]
                x_T = mol_concepts.transpose(-1, -2)              # (B, D, K)
                out_T = torch.matmul(x_T, mask) / float(K)        # (B, D, K), /K for mean
                causal_effect = out_T.transpose(-1, -2)           # (B, K, D)
        return causal_effect, mask

    def get_sparsity_loss(self) -> torch.Tensor:
        """
        For tanh-family variants: L1 norm of realized mask (encourages small values).
        For Hard Concrete: expected L0 = E[P(gate>0)].
        For "orthogonal": soft orthogonality penalty ||M^T M - I||_F^2 (reuses the same
            lambda; despite the name, this isn't a sparsity loss but a structural constraint).
        For "fixed": 0 (frozen, no regularization signal needed).

        NOTE (2026-04-18): reduction changed from `.mean()` to `.sum()`. The earlier
        `.mean()` reduction implicitly divided per-entry gradients by K^2 = 65536,
        causing the sparsity signal to be overwhelmed by the contrastive loss even
        at large lambda values. With `.sum()`, each mask entry receives a gradient
        proportional to the stated lambda, so lambda in the range [1e-5, 1e-3]
        gives comparable effective pressure to the mean-reduced lambda in [0.6, 60].
        """
        # IMPORTANT: cast to fp32 before summation — sum over K^2=65536 entries
        # of a tanh+1 mask (each ~1.0) yields ~65536 which exceeds fp16 max 65504.
        if self.mask_type == "hard_concrete":
            t = -self.hc_gamma / (self.hc_zeta - self.hc_gamma)
            log_ratio = math.log(t / (1.0 - t))
            prob_active = torch.sigmoid(self.mask_logits.float() - self.hc_beta * log_ratio)
            return prob_active.sum()
        elif self.mask_type == "orthogonal":
            M = self.mask_logits.float()
            n = M.shape[0]
            eye = torch.eye(n, device=M.device, dtype=torch.float32)
            return ((M.t() @ M - eye) ** 2).sum()
        elif self.mask_type == "fixed":
            return torch.tensor(0.0, device=self.mask_logits.device)
        else:
            mask = self._compute_mask().float()
            return mask.abs().sum()


class ConceptPooler(nn.Module):
    """
    Pools concept tokens to a single vector.
    Supports mean pooling and attention-weighted pooling.
    """

    def __init__(
        self,
        concept_dim: int = 256,
        pooling_type: str = "mean",  # "mean" or "attention"
    ):
        super().__init__()
        self.pooling_type = pooling_type

        if pooling_type == "attention":
            self.attn_weights = nn.Linear(concept_dim, 1)

    def forward(self, concepts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            concepts: (B, num_concepts, concept_dim)
        Returns:
            pooled: (B, concept_dim)
        """
        if self.pooling_type == "mean":
            return concepts.mean(dim=1)
        elif self.pooling_type == "attention":
            weights = F.softmax(self.attn_weights(concepts), dim=1)  # (B, num_concepts, 1)
            return (concepts * weights).sum(dim=1)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")


class ProjectionHead(nn.Module):
    """
    Projects concept embeddings to tangent space for hyperbolic mapping.
    Output is NOT normalized (unlike DrugCLIP) since we use hyperbolic similarity.
    """

    def __init__(
        self,
        input_dim: int = 256,
        output_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim)
        Returns:
            projected: (B, output_dim) - in tangent space (NOT normalized)
        """
        return self.proj(x)
