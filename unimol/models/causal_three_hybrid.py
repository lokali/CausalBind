# Copyright (c) 2026 The CausalBind Authors.
# CausalBind V0 and V1 (CausalBind-EMB), built on the HypSeek three-branch model.
# Operates in tangent space before hyperbolic mapping.

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.data import Dictionary
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from transformers import AutoTokenizer, AutoModelForMaskedLM
from .lorentz import exp_map0, safe_exp_map0
from . import distributed as dist_utils
from .unimol import NonLinearHead, UniMolModel, base_architecture
from .causal_modules import ConceptExtractor, ConceptPooler, SparseMask, ProjectionHead
import numpy as np
import math


@register_model("causal_three_hybrid_v0")
class CausalThreeHybridV0(BaseUnicoreModel):
    """
    CausalBind V0: concept extraction only (control).

    Architecture:
    1. UniMol/ESM2 encoders -> CLS tokens (512-dim)
    2. ConceptExtractor -> (num_concepts, concept_dim)
    3. ConceptPooler -> concept_dim
    4. ProjectionHead -> 128-dim (tangent space)
    5. exp_map0 -> hyperbolic space

    This version adds concept extraction without sparse causal masking,
    serving as a control to measure the benefit of concept extraction alone.
    """

    @staticmethod
    def add_args(parser):
        parser.add_argument("--mol-pooler-dropout", type=float, metavar="D")
        parser.add_argument("--pocket-pooler-dropout", type=float, metavar="D")
        parser.add_argument("--pocket-encoder-layers", type=int)
        parser.add_argument("--recycling", type=int, default=1)
        parser.add_argument("--aperture-eta", type=float, default=1.2)
        parser.add_argument("--curv-init", type=float, default=1.0)
        parser.add_argument("--learn-curv", action="store_true")
        # Causal parameters
        parser.add_argument("--num-concepts", type=int, default=64)
        parser.add_argument("--concept-dim", type=int, default=256)
        parser.add_argument("--concept-layers", type=int, default=4)

    def __init__(self, args, mol_dictionary: Dictionary, pocket_dictionary: Dictionary):
        super().__init__()
        causal_three_hybrid_v0_architecture(args)
        self.args = args

        # Base encoders (same as baseline)
        self.mol_model = UniMolModel(args.mol, mol_dictionary)
        self.pocket_model = UniMolModel(args.pocket, pocket_dictionary)

        # ESM2 model - prefer local path for offline use
        esm2_local_path = os.environ.get("ESM2_PATH", "./pretrain/esm2_t12_35M_UR50D")
        esm2_model_name = esm2_local_path if os.path.exists(esm2_local_path) else "facebook/esm2_t12_35M_UR50D"
        self.tokenizer = AutoTokenizer.from_pretrained(esm2_model_name, use_fast=False, local_files_only=os.path.exists(esm2_local_path))
        self.protein_model = AutoModelForMaskedLM.from_pretrained(esm2_model_name, local_files_only=os.path.exists(esm2_local_path))
        # Freeze ESM2 to reduce NCCL communication overhead
        for param in self.protein_model.parameters():
            param.requires_grad = False

        # Causal parameters
        num_concepts = getattr(args, "num_concepts", 64)
        concept_dim = getattr(args, "concept_dim", 256)
        concept_layers = getattr(args, "concept_layers", 4)

        # Concept extractors for each modality
        self.mol_concept_extractor = ConceptExtractor(
            input_dim=args.mol.encoder_embed_dim,
            num_concepts=num_concepts,
            concept_dim=concept_dim,
            num_layers=concept_layers,
        )
        self.pocket_concept_extractor = ConceptExtractor(
            input_dim=args.pocket.encoder_embed_dim,
            num_concepts=num_concepts,
            concept_dim=concept_dim,
            num_layers=concept_layers,
        )
        self.protein_concept_extractor = ConceptExtractor(
            input_dim=self.protein_model.config.hidden_size,
            num_concepts=num_concepts,
            concept_dim=concept_dim,
            num_layers=concept_layers,
        )
        # Adapter layer for frozen ESM2 output to ensure consistent gradient flow
        self.protein_adapter = nn.Linear(self.protein_model.config.hidden_size, self.protein_model.config.hidden_size)
        nn.init.eye_(self.protein_adapter.weight)
        nn.init.zeros_(self.protein_adapter.bias)

        # Concept poolers
        self.mol_pooler = ConceptPooler(concept_dim=concept_dim, pooling_type="mean")
        self.pocket_pooler = ConceptPooler(concept_dim=concept_dim, pooling_type="mean")
        self.protein_pooler = ConceptPooler(concept_dim=concept_dim, pooling_type="mean")

        # Projection heads (concept_dim -> 128 tangent space)
        self.mol_project = ProjectionHead(input_dim=concept_dim, output_dim=128, hidden_dim=concept_dim)
        self.pocket_project = ProjectionHead(input_dim=concept_dim, output_dim=128, hidden_dim=concept_dim)
        self.protein_project = ProjectionHead(input_dim=concept_dim, output_dim=128, hidden_dim=concept_dim)

        # Hyperbolic parameters
        self.logit_scale = nn.Parameter(torch.ones([1]) * np.log(13))
        self.curv = nn.Parameter(torch.tensor([args.curv_init]).log(), requires_grad=args.learn_curv)
        self._curv_minmax = {
            "max": math.log(args.curv_init * 10),
            "min": math.log(args.curv_init / 10),
        }

        # Learnable scaling factors
        self.mol_alpha = nn.Parameter(torch.tensor([128**-0.5]).log(), requires_grad=True)
        self.pocket_alpha = nn.Parameter(torch.tensor([128**-0.5]).log(), requires_grad=True)
        self.protein_alpha = nn.Parameter(torch.tensor([128**-0.5]).log(), requires_grad=True)

    @classmethod
    def build_model(cls, args, task):
        return cls(args, task.dictionary, task.pocket_dictionary)

    def get_dist_features(self, dist, et, flag):
        if flag == "mol":
            n_node = dist.size(-1)
            gbf_feature = self.mol_model.gbf(dist, et)
            gbf_result = self.mol_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias
        else:
            n_node = dist.size(-1)
            gbf_feature = self.pocket_model.gbf(dist, et)
            gbf_result = self.pocket_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

    def forward(
        self,
        mol_src_tokens,
        mol_src_distance,
        mol_src_edge_type,
        pocket_src_tokens,
        pocket_src_distance,
        pocket_src_edge_type,
        protein_sequences,
        encode=False,
        masked_tokens=None,
        features_only=True,
        is_train=True,
        **kwargs
    ):
        # Clamp parameters
        self.mol_alpha.data = torch.clamp(self.mol_alpha.data, max=0.0)
        self.pocket_alpha.data = torch.clamp(self.pocket_alpha.data, max=0.0)
        self.protein_alpha.data = torch.clamp(self.protein_alpha.data, max=0.0)
        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        κ = self.curv.exp()

        # ——— Mol encoder ———
        mol_padding_mask = mol_src_tokens.eq(self.mol_model.padding_idx)
        mol_x = self.mol_model.embed_tokens(mol_src_tokens)
        mol_graph_attn_bias = self.get_dist_features(mol_src_distance, mol_src_edge_type, "mol")
        mol_outputs = self.mol_model.encoder(mol_x, padding_mask=mol_padding_mask, attn_mask=mol_graph_attn_bias)
        mol_rep_eu = mol_outputs[0][:, 0, :]  # CLS token

        # Causal concept extraction for mol
        mol_concepts = self.mol_concept_extractor(mol_rep_eu)
        mol_pooled = self.mol_pooler(mol_concepts)
        u_mol = self.mol_project(mol_pooled)
        u_mol = u_mol * self.mol_alpha.exp()

        # ——— Pocket encoder ———
        poc_padding_mask = pocket_src_tokens.eq(self.pocket_model.padding_idx)
        poc_x = self.pocket_model.embed_tokens(pocket_src_tokens)
        poc_graph_attn_bias = self.get_dist_features(pocket_src_distance, pocket_src_edge_type, "pocket")
        poc_outputs = self.pocket_model.encoder(poc_x, padding_mask=poc_padding_mask, attn_mask=poc_graph_attn_bias)
        poc_rep_eu = poc_outputs[0][:, 0, :]

        # Causal concept extraction for pocket
        pocket_concepts = self.pocket_concept_extractor(poc_rep_eu)
        pocket_pooled = self.pocket_pooler(pocket_concepts)
        u_poc = self.pocket_project(pocket_pooled)
        u_poc = u_poc * self.pocket_alpha.exp()

        # ——— Protein encoder (ESM2) ———
        # Use padding="max_length" to ensure consistent tensor shapes across all ranks
        inputs = self.tokenizer(
            protein_sequences, return_tensors="pt", padding="max_length", truncation=True, max_length=512
        )
        for k, v in inputs.items():
            inputs[k] = v.cuda()
        # Frozen ESM2 forward (always in fp32 for stability)
        with torch.no_grad():
            with torch.autocast(device_type='cuda', enabled=False):
                prot_outputs = self.protein_model(**inputs, output_hidden_states=True)
                prot_hidden_states = prot_outputs.hidden_states[-1]
                prot_rep_eu = prot_hidden_states[:, 0, :].float()
        # Pass through adapter layer - convert to match adapter dtype for fp16 training
        prot_rep_eu = prot_rep_eu.to(self.protein_adapter.weight.dtype)
        prot_rep_eu = self.protein_adapter(prot_rep_eu)

        # Causal concept extraction for protein
        protein_concepts = self.protein_concept_extractor(prot_rep_eu)
        protein_pooled = self.protein_pooler(protein_concepts)
        u_prot = self.protein_project(protein_pooled)
        u_prot = u_prot * self.protein_alpha.exp()

        # Map to hyperbolic space (use safe version with norm clamping for stability)
        with torch.autocast(u_mol.device.type, dtype=torch.float32):
            h_mol = safe_exp_map0(u_mol, κ, max_norm=10.0)
            h_poc = safe_exp_map0(u_poc, κ, max_norm=10.0)
            h_prot = safe_exp_map0(u_prot, κ, max_norm=10.0)

        return h_prot, h_poc, h_mol

    def set_num_updates(self, num_updates):
        self._num_updates = num_updates

    def get_num_updates(self):
        return getattr(self, "_num_updates", 0)

    def mol_forward(
        self,
        mol_src_tokens,
        mol_src_distance,
        mol_src_edge_type,
        **kwargs
    ):
        """Forward pass for molecule only (for testing)."""
        mol_padding_mask = mol_src_tokens.eq(self.mol_model.padding_idx)
        mol_x = self.mol_model.embed_tokens(mol_src_tokens)
        mol_graph_attn_bias = self.get_dist_features(mol_src_distance, mol_src_edge_type, "mol")
        mol_outputs = self.mol_model.encoder(mol_x, padding_mask=mol_padding_mask, attn_mask=mol_graph_attn_bias)
        mol_rep_eu = mol_outputs[0][:, 0, :]

        mol_concepts = self.mol_concept_extractor(mol_rep_eu)
        mol_pooled = self.mol_pooler(mol_concepts)
        u_mol = self.mol_project(mol_pooled)
        u_mol = u_mol * self.mol_alpha.exp()
        with torch.autocast(u_mol.device.type, dtype=torch.float32):
            h_mol = safe_exp_map0(u_mol, self.curv.exp(), max_norm=10.0)
        return h_mol

    def pocket_forward(
        self,
        pocket_src_tokens,
        pocket_src_distance,
        pocket_src_edge_type,
        **kwargs
    ):
        """Forward pass for pocket only (for testing)."""
        poc_padding_mask = pocket_src_tokens.eq(self.pocket_model.padding_idx)
        poc_x = self.pocket_model.embed_tokens(pocket_src_tokens)
        poc_graph_attn_bias = self.get_dist_features(pocket_src_distance, pocket_src_edge_type, "pocket")
        poc_outputs = self.pocket_model.encoder(poc_x, padding_mask=poc_padding_mask, attn_mask=poc_graph_attn_bias)
        poc_rep_eu = poc_outputs[0][:, 0, :]

        pocket_concepts = self.pocket_concept_extractor(poc_rep_eu)
        pocket_pooled = self.pocket_pooler(pocket_concepts)
        u_poc = self.pocket_project(pocket_pooled)
        u_poc = u_poc * self.pocket_alpha.exp()
        with torch.autocast(u_poc.device.type, dtype=torch.float32):
            h_poc = safe_exp_map0(u_poc, self.curv.exp(), max_norm=10.0)
        return h_poc

    def protein_forward(
        self,
        protein_sequences,
        **kwargs
    ):
        """Forward pass for protein only (for testing)."""
        inputs = self.tokenizer(
            protein_sequences, return_tensors="pt", padding="max_length", truncation=True, max_length=512
        )
        device = self.curv.device
        self.protein_model.to(device)
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        with torch.no_grad():
            with torch.autocast(device_type='cuda', enabled=False):
                prot_outputs = self.protein_model(**inputs, output_hidden_states=True)
                prot_hidden_states = prot_outputs.hidden_states[-1]
                prot_rep_eu = prot_hidden_states[:, 0, :].float()
        prot_rep_eu = prot_rep_eu.to(self.protein_adapter.weight.dtype)
        prot_rep_eu = self.protein_adapter(prot_rep_eu)

        protein_concepts = self.protein_concept_extractor(prot_rep_eu)
        protein_pooled = self.protein_pooler(protein_concepts)
        u_prot = self.protein_project(protein_pooled)
        u_prot = u_prot * self.protein_alpha.exp()
        with torch.autocast(u_prot.device.type, dtype=torch.float32):
            h_prot = safe_exp_map0(u_prot, self.curv.exp(), max_norm=10.0)
        return h_prot


@register_model("causal_three_hybrid_v1")
class CausalThreeHybridV1(BaseUnicoreModel):
    """
    CausalBind-EMB (V1): constrained mask in the pooled embedding space.

    Architecture:
    1. UniMol/ESM2 encoders -> CLS tokens (512-dim)
    2. ConceptExtractor -> (num_concepts, concept_dim)
    3. ConceptPooler -> concept_dim
    4. SparseMask -> concept_dim (with sparsity regularization)
    5. ProjectionHead -> 128-dim (tangent space)
    6. exp_map0 -> hyperbolic space

    This version adds learnable sparse masks to identify important concept dimensions.
    Each modality has its own sparse mask.
    """

    @staticmethod
    def add_args(parser):
        parser.add_argument("--mol-pooler-dropout", type=float, metavar="D")
        parser.add_argument("--pocket-pooler-dropout", type=float, metavar="D")
        parser.add_argument("--pocket-encoder-layers", type=int)
        parser.add_argument("--recycling", type=int, default=1)
        parser.add_argument("--aperture-eta", type=float, default=1.2)
        parser.add_argument("--curv-init", type=float, default=1.0)
        parser.add_argument("--learn-curv", action="store_true")
        # Causal parameters
        parser.add_argument("--num-concepts", type=int, default=64)
        parser.add_argument("--concept-dim", type=int, default=256)
        parser.add_argument("--concept-layers", type=int, default=4)
        parser.add_argument("--sparsity-weight", type=float, default=0.001)
        parser.add_argument(
            "--mask-type",
            type=str,
            default="tanh_plus_1",
            choices=["tanh_plus_1", "relu", "ste", "shrink", "neg_init", "hard_concrete",
                     "fixed", "diagonal", "orthogonal", "linear", "low_rank"],
            help="SparseMask variant (see causal_modules.SparseMask). 'low_rank' is "
                 "the rank-r factorization (CausalBind-LR) M = tanh(UV^T)+1 with rank set by --mask-rank.",
        )
        parser.add_argument(
            "--sparsity-threshold",
            type=float,
            default=0.05,
            help="Threshold for hard-gating in SparseMask (used in forward). Default 0.05; "
                 "larger values induce sparsity more aggressively but with dead-gradient risk.",
        )
        parser.add_argument(
            "--mask-rank",
            type=int,
            default=8,
            help="Rank r for the 'low_rank' mask variant (CausalBind-LR). Mask is constructed as "
                 "tanh(U V^T) + 1 with U, V in R^{K x r}. Ignored for other mask types.",
        )

    def __init__(self, args, mol_dictionary: Dictionary, pocket_dictionary: Dictionary):
        super().__init__()
        causal_three_hybrid_v1_architecture(args)
        self.args = args

        # Store sparsity weight
        self.sparsity_weight = getattr(args, "sparsity_weight", 0.001)

        # Base encoders (same as baseline)
        self.mol_model = UniMolModel(args.mol, mol_dictionary)
        self.pocket_model = UniMolModel(args.pocket, pocket_dictionary)

        # ESM2 model - prefer local path for offline use
        esm2_local_path = os.environ.get("ESM2_PATH", "./pretrain/esm2_t12_35M_UR50D")
        esm2_model_name = esm2_local_path if os.path.exists(esm2_local_path) else "facebook/esm2_t12_35M_UR50D"
        self.tokenizer = AutoTokenizer.from_pretrained(esm2_model_name, use_fast=False, local_files_only=os.path.exists(esm2_local_path))
        self.protein_model = AutoModelForMaskedLM.from_pretrained(esm2_model_name, local_files_only=os.path.exists(esm2_local_path))
        # Freeze ESM2 to reduce NCCL communication overhead
        for param in self.protein_model.parameters():
            param.requires_grad = False

        # Causal parameters
        num_concepts = getattr(args, "num_concepts", 64)
        concept_dim = getattr(args, "concept_dim", 256)
        concept_layers = getattr(args, "concept_layers", 4)

        # Concept extractors for each modality
        self.mol_concept_extractor = ConceptExtractor(
            input_dim=args.mol.encoder_embed_dim,
            num_concepts=num_concepts,
            concept_dim=concept_dim,
            num_layers=concept_layers,
        )
        self.pocket_concept_extractor = ConceptExtractor(
            input_dim=args.pocket.encoder_embed_dim,
            num_concepts=num_concepts,
            concept_dim=concept_dim,
            num_layers=concept_layers,
        )
        self.protein_concept_extractor = ConceptExtractor(
            input_dim=self.protein_model.config.hidden_size,
            num_concepts=num_concepts,
            concept_dim=concept_dim,
            num_layers=concept_layers,
        )
        # Adapter layer for frozen ESM2 output to ensure consistent gradient flow
        self.protein_adapter = nn.Linear(self.protein_model.config.hidden_size, self.protein_model.config.hidden_size)
        nn.init.eye_(self.protein_adapter.weight)
        nn.init.zeros_(self.protein_adapter.bias)

        # Concept poolers
        self.mol_pooler = ConceptPooler(concept_dim=concept_dim, pooling_type="mean")
        self.pocket_pooler = ConceptPooler(concept_dim=concept_dim, pooling_type="mean")
        self.protein_pooler = ConceptPooler(concept_dim=concept_dim, pooling_type="mean")

        # Sparse masks for each modality (independent sparse feature selection)
        mask_type = getattr(args, "mask_type", "tanh_plus_1")
        threshold = getattr(args, "sparsity_threshold", 0.05)
        mask_rank = getattr(args, "mask_rank", 8)
        self.mol_sparse_mask = SparseMask(num_concepts=concept_dim, threshold=threshold, mask_type=mask_type, mask_rank=mask_rank)
        self.pocket_sparse_mask = SparseMask(num_concepts=concept_dim, threshold=threshold, mask_type=mask_type, mask_rank=mask_rank)
        self.protein_sparse_mask = SparseMask(num_concepts=concept_dim, threshold=threshold, mask_type=mask_type, mask_rank=mask_rank)

        # Projection heads (concept_dim -> 128 tangent space)
        self.mol_project = ProjectionHead(input_dim=concept_dim, output_dim=128, hidden_dim=concept_dim)
        self.pocket_project = ProjectionHead(input_dim=concept_dim, output_dim=128, hidden_dim=concept_dim)
        self.protein_project = ProjectionHead(input_dim=concept_dim, output_dim=128, hidden_dim=concept_dim)

        # Hyperbolic parameters
        self.logit_scale = nn.Parameter(torch.ones([1]) * np.log(13))
        self.curv = nn.Parameter(torch.tensor([args.curv_init]).log(), requires_grad=args.learn_curv)
        self._curv_minmax = {
            "max": math.log(args.curv_init * 10),
            "min": math.log(args.curv_init / 10),
        }

        # Learnable scaling factors
        self.mol_alpha = nn.Parameter(torch.tensor([128**-0.5]).log(), requires_grad=True)
        self.pocket_alpha = nn.Parameter(torch.tensor([128**-0.5]).log(), requires_grad=True)
        self.protein_alpha = nn.Parameter(torch.tensor([128**-0.5]).log(), requires_grad=True)

    @classmethod
    def build_model(cls, args, task):
        return cls(args, task.dictionary, task.pocket_dictionary)

    def get_dist_features(self, dist, et, flag):
        if flag == "mol":
            n_node = dist.size(-1)
            gbf_feature = self.mol_model.gbf(dist, et)
            gbf_result = self.mol_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias
        else:
            n_node = dist.size(-1)
            gbf_feature = self.pocket_model.gbf(dist, et)
            gbf_result = self.pocket_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

    def forward(
        self,
        mol_src_tokens,
        mol_src_distance,
        mol_src_edge_type,
        pocket_src_tokens,
        pocket_src_distance,
        pocket_src_edge_type,
        protein_sequences,
        encode=False,
        masked_tokens=None,
        features_only=True,
        is_train=True,
        **kwargs
    ):
        # Clamp parameters
        self.mol_alpha.data = torch.clamp(self.mol_alpha.data, max=0.0)
        self.pocket_alpha.data = torch.clamp(self.pocket_alpha.data, max=0.0)
        self.protein_alpha.data = torch.clamp(self.protein_alpha.data, max=0.0)
        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        κ = self.curv.exp()

        # ——— Mol encoder ———
        mol_padding_mask = mol_src_tokens.eq(self.mol_model.padding_idx)
        mol_x = self.mol_model.embed_tokens(mol_src_tokens)
        mol_graph_attn_bias = self.get_dist_features(mol_src_distance, mol_src_edge_type, "mol")
        mol_outputs = self.mol_model.encoder(mol_x, padding_mask=mol_padding_mask, attn_mask=mol_graph_attn_bias)
        mol_rep_eu = mol_outputs[0][:, 0, :]

        # Causal concept extraction + sparse mask for mol
        mol_concepts = self.mol_concept_extractor(mol_rep_eu)
        mol_pooled = self.mol_pooler(mol_concepts)
        mol_masked, _ = self.mol_sparse_mask(mol_pooled)
        u_mol = self.mol_project(mol_masked)
        u_mol = u_mol * self.mol_alpha.exp()

        # ——— Pocket encoder ———
        poc_padding_mask = pocket_src_tokens.eq(self.pocket_model.padding_idx)
        poc_x = self.pocket_model.embed_tokens(pocket_src_tokens)
        poc_graph_attn_bias = self.get_dist_features(pocket_src_distance, pocket_src_edge_type, "pocket")
        poc_outputs = self.pocket_model.encoder(poc_x, padding_mask=poc_padding_mask, attn_mask=poc_graph_attn_bias)
        poc_rep_eu = poc_outputs[0][:, 0, :]

        # Causal concept extraction + sparse mask for pocket
        pocket_concepts = self.pocket_concept_extractor(poc_rep_eu)
        pocket_pooled = self.pocket_pooler(pocket_concepts)
        pocket_masked, _ = self.pocket_sparse_mask(pocket_pooled)
        u_poc = self.pocket_project(pocket_masked)
        u_poc = u_poc * self.pocket_alpha.exp()

        # ——— Protein encoder (ESM2) ———
        # Use padding="max_length" to ensure consistent tensor shapes across all ranks
        inputs = self.tokenizer(
            protein_sequences, return_tensors="pt", padding="max_length", truncation=True, max_length=512
        )
        for k, v in inputs.items():
            inputs[k] = v.cuda()
        # Frozen ESM2 forward (always in fp32 for stability)
        with torch.no_grad():
            with torch.autocast(device_type='cuda', enabled=False):
                prot_outputs = self.protein_model(**inputs, output_hidden_states=True)
                prot_hidden_states = prot_outputs.hidden_states[-1]
                prot_rep_eu = prot_hidden_states[:, 0, :].float()
        # Pass through adapter layer - convert to match adapter dtype for fp16 training
        prot_rep_eu = prot_rep_eu.to(self.protein_adapter.weight.dtype)
        prot_rep_eu = self.protein_adapter(prot_rep_eu)

        # Causal concept extraction + sparse mask for protein
        protein_concepts = self.protein_concept_extractor(prot_rep_eu)
        protein_pooled = self.protein_pooler(protein_concepts)
        protein_masked, _ = self.protein_sparse_mask(protein_pooled)
        u_prot = self.protein_project(protein_masked)
        u_prot = u_prot * self.protein_alpha.exp()

        # Map to hyperbolic space (use safe version with norm clamping for stability)
        with torch.autocast(u_mol.device.type, dtype=torch.float32):
            h_mol = safe_exp_map0(u_mol, κ, max_norm=10.0)
            h_poc = safe_exp_map0(u_poc, κ, max_norm=10.0)
            h_prot = safe_exp_map0(u_prot, κ, max_norm=10.0)

        return h_prot, h_poc, h_mol

    def get_sparsity_loss(self):
        """Return total sparsity loss from all sparse masks."""
        mol_sparsity = self.mol_sparse_mask.get_sparsity_loss()
        pocket_sparsity = self.pocket_sparse_mask.get_sparsity_loss()
        protein_sparsity = self.protein_sparse_mask.get_sparsity_loss()
        return (mol_sparsity + pocket_sparsity + protein_sparsity) * self.sparsity_weight

    def set_num_updates(self, num_updates):
        self._num_updates = num_updates

    def get_num_updates(self):
        return getattr(self, "_num_updates", 0)

    def mol_forward(
        self,
        mol_src_tokens,
        mol_src_distance,
        mol_src_edge_type,
        **kwargs
    ):
        """Forward pass for molecule only (for testing)."""
        mol_padding_mask = mol_src_tokens.eq(self.mol_model.padding_idx)
        mol_x = self.mol_model.embed_tokens(mol_src_tokens)
        mol_graph_attn_bias = self.get_dist_features(mol_src_distance, mol_src_edge_type, "mol")
        mol_outputs = self.mol_model.encoder(mol_x, padding_mask=mol_padding_mask, attn_mask=mol_graph_attn_bias)
        mol_rep_eu = mol_outputs[0][:, 0, :]

        mol_concepts = self.mol_concept_extractor(mol_rep_eu)
        mol_pooled = self.mol_pooler(mol_concepts)
        mol_masked, _ = self.mol_sparse_mask(mol_pooled)
        u_mol = self.mol_project(mol_masked)
        u_mol = u_mol * self.mol_alpha.exp()
        with torch.autocast(u_mol.device.type, dtype=torch.float32):
            h_mol = safe_exp_map0(u_mol, self.curv.exp(), max_norm=10.0)
        return h_mol

    def pocket_forward(
        self,
        pocket_src_tokens,
        pocket_src_distance,
        pocket_src_edge_type,
        **kwargs
    ):
        """Forward pass for pocket only (for testing)."""
        poc_padding_mask = pocket_src_tokens.eq(self.pocket_model.padding_idx)
        poc_x = self.pocket_model.embed_tokens(pocket_src_tokens)
        poc_graph_attn_bias = self.get_dist_features(pocket_src_distance, pocket_src_edge_type, "pocket")
        poc_outputs = self.pocket_model.encoder(poc_x, padding_mask=poc_padding_mask, attn_mask=poc_graph_attn_bias)
        poc_rep_eu = poc_outputs[0][:, 0, :]

        pocket_concepts = self.pocket_concept_extractor(poc_rep_eu)
        pocket_pooled = self.pocket_pooler(pocket_concepts)
        pocket_masked, _ = self.pocket_sparse_mask(pocket_pooled)
        u_poc = self.pocket_project(pocket_masked)
        u_poc = u_poc * self.pocket_alpha.exp()
        with torch.autocast(u_poc.device.type, dtype=torch.float32):
            h_poc = safe_exp_map0(u_poc, self.curv.exp(), max_norm=10.0)
        return h_poc

    def protein_forward(
        self,
        protein_sequences,
        **kwargs
    ):
        """Forward pass for protein only (for testing)."""
        inputs = self.tokenizer(
            protein_sequences, return_tensors="pt", padding="max_length", truncation=True, max_length=512
        )
        device = self.curv.device
        self.protein_model.to(device)
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        with torch.no_grad():
            with torch.autocast(device_type='cuda', enabled=False):
                prot_outputs = self.protein_model(**inputs, output_hidden_states=True)
                prot_hidden_states = prot_outputs.hidden_states[-1]
                prot_rep_eu = prot_hidden_states[:, 0, :].float()
        prot_rep_eu = prot_rep_eu.to(self.protein_adapter.weight.dtype)
        prot_rep_eu = self.protein_adapter(prot_rep_eu)

        protein_concepts = self.protein_concept_extractor(prot_rep_eu)
        protein_pooled = self.protein_pooler(protein_concepts)
        protein_masked, _ = self.protein_sparse_mask(protein_pooled)
        u_prot = self.protein_project(protein_masked)
        u_prot = u_prot * self.protein_alpha.exp()
        with torch.autocast(u_prot.device.type, dtype=torch.float32):
            h_prot = safe_exp_map0(u_prot, self.curv.exp(), max_norm=10.0)
        return h_prot


@register_model_architecture("causal_three_hybrid_v0", "causal_three_hybrid_v0")
def causal_three_hybrid_v0_architecture(args):
    parser = argparse.ArgumentParser()
    args.mol = parser.parse_args([])
    args.pocket = parser.parse_args([])

    args.mol.encoder_layers = getattr(args, "mol_encoder_layers", 15)
    args.mol.encoder_embed_dim = getattr(args, "mol_encoder_embed_dim", 512)
    args.mol.encoder_ffn_embed_dim = getattr(args, "mol_encoder_ffn_embed_dim", 2048)
    args.mol.encoder_attention_heads = getattr(args, "mol_encoder_attention_heads", 64)
    args.mol.dropout = getattr(args, "mol_dropout", 0.1)
    args.mol.emb_dropout = getattr(args, "mol_emb_dropout", 0.1)
    args.mol.attention_dropout = getattr(args, "mol_attention_dropout", 0.1)
    args.mol.activation_dropout = getattr(args, "mol_activation_dropout", 0.0)
    args.mol.pooler_dropout = getattr(args, "mol_pooler_dropout", 0.0)
    args.mol.max_seq_len = getattr(args, "mol_max_seq_len", 512)
    args.mol.activation_fn = getattr(args, "mol_activation_fn", "gelu")
    args.mol.pooler_activation_fn = getattr(args, "mol_pooler_activation_fn", "tanh")
    args.mol.post_ln = getattr(args, "mol_post_ln", False)
    args.mol.masked_token_loss = -1.0
    args.mol.masked_coord_loss = -1.0
    args.mol.masked_dist_loss = -1.0
    args.mol.x_norm_loss = -1.0
    args.mol.delta_pair_repr_norm_loss = -1.0

    args.pocket.encoder_layers = getattr(args, "pocket_encoder_layers", 15)
    args.pocket.encoder_embed_dim = getattr(args, "pocket_encoder_embed_dim", 512)
    args.pocket.encoder_ffn_embed_dim = getattr(args, "pocket_encoder_ffn_embed_dim", 2048)
    args.pocket.encoder_attention_heads = getattr(args, "pocket_encoder_attention_heads", 64)
    args.pocket.dropout = getattr(args, "pocket_dropout", 0.1)
    args.pocket.emb_dropout = getattr(args, "pocket_emb_dropout", 0.1)
    args.pocket.attention_dropout = getattr(args, "pocket_attention_dropout", 0.1)
    args.pocket.activation_dropout = getattr(args, "pocket_activation_dropout", 0.0)
    args.pocket.pooler_dropout = getattr(args, "pocket_pooler_dropout", 0.0)
    args.pocket.max_seq_len = getattr(args, "pocket_max_seq_len", 512)
    args.pocket.activation_fn = getattr(args, "pocket_activation_fn", "gelu")
    args.pocket.pooler_activation_fn = getattr(args, "pocket_pooler_activation_fn", "tanh")
    args.pocket.post_ln = getattr(args, "pocket_post_ln", False)
    args.pocket.masked_token_loss = -1.0
    args.pocket.masked_coord_loss = -1.0
    args.pocket.masked_dist_loss = -1.0
    args.pocket.x_norm_loss = -1.0
    args.pocket.delta_pair_repr_norm_loss = -1.0

    args.curv_init = getattr(args, "curv_init", 1.0)
    args.learn_curv = getattr(args, "learn_curv", False)
    args.hbce_bounds = getattr(args, "hbce_bounds", [5.0, 7.0, 9.0])
    args.chl_r0 = getattr(args, "chl_r0", 0.5)
    args.chl_dr = getattr(args, "chl_dr", 0.5)
    args.chl_eta0 = getattr(args, "chl_eta0", 0.7)
    args.chl_deta = getattr(args, "chl_deta", 0.2)
    args.lambda_rad = getattr(args, "lambda_rad", 0.5)
    args.lambda_ang = getattr(args, "lambda_ang", 0.5)
    args.gamma_chl = getattr(args, "gamma_chl", 0.1)

    base_architecture(args)


@register_model_architecture("causal_three_hybrid_v1", "causal_three_hybrid_v1")
def causal_three_hybrid_v1_architecture(args):
    causal_three_hybrid_v0_architecture(args)
