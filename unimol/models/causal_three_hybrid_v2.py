"""
CausalBind-SP and CausalBind-LR (architecture name: causal_three_hybrid_v2).

Concept-level K×K cross-modal mask, applied BEFORE the ConceptPooler.
  --mask-type tanh_plus_1 (default): learnable sparse mask   -> CausalBind-SP
  --mask-type low_rank --mask-rank r: rank-r factorization   -> CausalBind-LR

Key architectural difference from V1:
  V1: concepts (B, K, D) → pool → (B, D) → mask(D×D) → (B, D) → project
  V2: concepts (B, K, D) → mask(K×K on concept axis) → (B, K, D) → pool → (B, D) → project

This aligns with the paper's identifiability theory, where the sparse mask
represents interactions between CONCEPTS (indices i, j ∈ {1, ..., K}) rather
than between feature dimensions.

Parameter count per modality: K² = 64² = 4,096  (vs V1's D² = 256² = 65,536, 16× fewer)
"""

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
from .causal_three_hybrid import (
    CausalThreeHybridV1,
    causal_three_hybrid_v1_architecture,
)
import numpy as np
import math


@register_model("causal_three_hybrid_v2")
class CausalThreeHybridV2(CausalThreeHybridV1):
    """
    Concept-level K×K SparseMask (applied BEFORE mean pooling).
    Inherits everything from V1; only the mask placement changes.
    """

    @staticmethod
    def add_args(parser):
        CausalThreeHybridV1.add_args(parser)
        # (Re-registering is fine because argparse allows duplicate adds if guarded; but Unicore
        # calls add_args only once per class, so this just re-exposes V1's args for V2.)

    def __init__(self, args, mol_dictionary: Dictionary, pocket_dictionary: Dictionary):
        # Skip V1's own __init__ parts that create D×D masks; we'll rebuild them K×K.
        nn.Module.__init__(self)
        causal_three_hybrid_v2_architecture(args)
        self.args = args
        self.sparsity_weight = getattr(args, "sparsity_weight", 0.001)

        # Base encoders (same as V1)
        self.mol_model = UniMolModel(args.mol, mol_dictionary)
        self.pocket_model = UniMolModel(args.pocket, pocket_dictionary)

        # ESM2
        esm2_local_path = os.environ.get("ESM2_PATH", "./pretrain/esm2_t12_35M_UR50D")
        esm2_model_name = esm2_local_path if os.path.exists(esm2_local_path) else "facebook/esm2_t12_35M_UR50D"
        self.tokenizer = AutoTokenizer.from_pretrained(
            esm2_model_name, use_fast=False, local_files_only=os.path.exists(esm2_local_path)
        )
        self.protein_model = AutoModelForMaskedLM.from_pretrained(
            esm2_model_name, local_files_only=os.path.exists(esm2_local_path)
        )
        for param in self.protein_model.parameters():
            param.requires_grad = False

        num_concepts = getattr(args, "num_concepts", 64)
        concept_dim = getattr(args, "concept_dim", 256)
        concept_layers = getattr(args, "concept_layers", 4)
        mask_type = getattr(args, "mask_type", "tanh_plus_1")
        threshold = getattr(args, "sparsity_threshold", 0.05)
        mask_rank = getattr(args, "mask_rank", 8)

        # Concept extractors
        self.mol_concept_extractor = ConceptExtractor(
            input_dim=args.mol.encoder_embed_dim,
            num_concepts=num_concepts, concept_dim=concept_dim, num_layers=concept_layers,
        )
        self.pocket_concept_extractor = ConceptExtractor(
            input_dim=args.pocket.encoder_embed_dim,
            num_concepts=num_concepts, concept_dim=concept_dim, num_layers=concept_layers,
        )
        self.protein_concept_extractor = ConceptExtractor(
            input_dim=self.protein_model.config.hidden_size,
            num_concepts=num_concepts, concept_dim=concept_dim, num_layers=concept_layers,
        )
        self.protein_adapter = nn.Linear(
            self.protein_model.config.hidden_size, self.protein_model.config.hidden_size
        )
        nn.init.eye_(self.protein_adapter.weight)
        nn.init.zeros_(self.protein_adapter.bias)

        # Poolers
        self.mol_pooler = ConceptPooler(concept_dim=concept_dim, pooling_type="mean")
        self.pocket_pooler = ConceptPooler(concept_dim=concept_dim, pooling_type="mean")
        self.protein_pooler = ConceptPooler(concept_dim=concept_dim, pooling_type="mean")

        # *** KEY CHANGE: masks are K×K instead of D×D ***
        self.mol_sparse_mask = SparseMask(
            num_concepts=num_concepts, threshold=threshold,
            mask_type=mask_type, mask_rank=mask_rank,
        )
        self.pocket_sparse_mask = SparseMask(
            num_concepts=num_concepts, threshold=threshold,
            mask_type=mask_type, mask_rank=mask_rank,
        )
        self.protein_sparse_mask = SparseMask(
            num_concepts=num_concepts, threshold=threshold,
            mask_type=mask_type, mask_rank=mask_rank,
        )

        # Projection heads (same as V1)
        self.mol_project = ProjectionHead(input_dim=concept_dim, output_dim=128, hidden_dim=concept_dim)
        self.pocket_project = ProjectionHead(input_dim=concept_dim, output_dim=128, hidden_dim=concept_dim)
        self.protein_project = ProjectionHead(input_dim=concept_dim, output_dim=128, hidden_dim=concept_dim)

        # Hyperbolic params (same as V1)
        self.logit_scale = nn.Parameter(torch.ones([1]) * np.log(13))
        self.curv = nn.Parameter(torch.tensor([args.curv_init]).log(), requires_grad=args.learn_curv)
        self._curv_minmax = {
            "max": math.log(args.curv_init * 10),
            "min": math.log(args.curv_init / 10),
        }
        self.mol_alpha = nn.Parameter(torch.tensor([128 ** -0.5]).log(), requires_grad=True)
        self.pocket_alpha = nn.Parameter(torch.tensor([128 ** -0.5]).log(), requires_grad=True)
        self.protein_alpha = nn.Parameter(torch.tensor([128 ** -0.5]).log(), requires_grad=True)

    def forward(
        self, mol_src_tokens, mol_src_distance, mol_src_edge_type,
        pocket_src_tokens, pocket_src_distance, pocket_src_edge_type,
        protein_sequences, encode=False, masked_tokens=None, features_only=True,
        is_train=True, **kwargs,
    ):
        self.mol_alpha.data = torch.clamp(self.mol_alpha.data, max=0.0)
        self.pocket_alpha.data = torch.clamp(self.pocket_alpha.data, max=0.0)
        self.protein_alpha.data = torch.clamp(self.protein_alpha.data, max=0.0)
        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        κ = self.curv.exp()

        # --- Mol: concept → MASK (K×K) → pool → project ---
        mol_padding_mask = mol_src_tokens.eq(self.mol_model.padding_idx)
        mol_x = self.mol_model.embed_tokens(mol_src_tokens)
        mol_graph_attn_bias = self.get_dist_features(mol_src_distance, mol_src_edge_type, "mol")
        mol_outputs = self.mol_model.encoder(mol_x, padding_mask=mol_padding_mask, attn_mask=mol_graph_attn_bias)
        mol_rep_eu = mol_outputs[0][:, 0, :]  # (B, 512)
        mol_concepts = self.mol_concept_extractor(mol_rep_eu)        # (B, K, D)
        mol_concepts_masked, _ = self.mol_sparse_mask(mol_concepts)  # (B, K, D)  ← V2 key step
        mol_pooled = self.mol_pooler(mol_concepts_masked)            # (B, D)
        u_mol = self.mol_project(mol_pooled) * self.mol_alpha.exp()

        # --- Pocket ---
        poc_padding_mask = pocket_src_tokens.eq(self.pocket_model.padding_idx)
        poc_x = self.pocket_model.embed_tokens(pocket_src_tokens)
        poc_graph_attn_bias = self.get_dist_features(pocket_src_distance, pocket_src_edge_type, "pocket")
        poc_outputs = self.pocket_model.encoder(poc_x, padding_mask=poc_padding_mask, attn_mask=poc_graph_attn_bias)
        poc_rep_eu = poc_outputs[0][:, 0, :]
        pocket_concepts = self.pocket_concept_extractor(poc_rep_eu)
        pocket_concepts_masked, _ = self.pocket_sparse_mask(pocket_concepts)
        pocket_pooled = self.pocket_pooler(pocket_concepts_masked)
        u_poc = self.pocket_project(pocket_pooled) * self.pocket_alpha.exp()

        # --- Protein (ESM2, frozen) ---
        inputs = self.tokenizer(
            protein_sequences, return_tensors="pt", padding="max_length", truncation=True, max_length=512
        )
        for k, v in inputs.items():
            inputs[k] = v.cuda()
        with torch.no_grad():
            with torch.autocast(device_type='cuda', enabled=False):
                prot_outputs = self.protein_model(**inputs, output_hidden_states=True)
                prot_rep_eu = prot_outputs.hidden_states[-1][:, 0, :].float()
        prot_rep_eu = prot_rep_eu.to(self.protein_adapter.weight.dtype)
        prot_rep_eu = self.protein_adapter(prot_rep_eu)
        protein_concepts = self.protein_concept_extractor(prot_rep_eu)
        protein_concepts_masked, _ = self.protein_sparse_mask(protein_concepts)
        protein_pooled = self.protein_pooler(protein_concepts_masked)
        u_prot = self.protein_project(protein_pooled) * self.protein_alpha.exp()

        # Hyperbolic map
        with torch.autocast(u_mol.device.type, dtype=torch.float32):
            h_mol = safe_exp_map0(u_mol, κ, max_norm=10.0)
            h_poc = safe_exp_map0(u_poc, κ, max_norm=10.0)
            h_prot = safe_exp_map0(u_prot, κ, max_norm=10.0)

        return h_prot, h_poc, h_mol

    # --- Inference-only forwards (one modality) — also apply the mask at concept level ---
    def mol_forward(self, mol_src_tokens, mol_src_distance, mol_src_edge_type, **kwargs):
        mol_padding_mask = mol_src_tokens.eq(self.mol_model.padding_idx)
        mol_x = self.mol_model.embed_tokens(mol_src_tokens)
        mol_graph_attn_bias = self.get_dist_features(mol_src_distance, mol_src_edge_type, "mol")
        mol_outputs = self.mol_model.encoder(mol_x, padding_mask=mol_padding_mask, attn_mask=mol_graph_attn_bias)
        mol_rep_eu = mol_outputs[0][:, 0, :]
        mol_concepts = self.mol_concept_extractor(mol_rep_eu)
        mol_concepts_masked, _ = self.mol_sparse_mask(mol_concepts)
        mol_pooled = self.mol_pooler(mol_concepts_masked)
        u_mol = self.mol_project(mol_pooled) * self.mol_alpha.exp()
        with torch.autocast(u_mol.device.type, dtype=torch.float32):
            h_mol = safe_exp_map0(u_mol, self.curv.exp(), max_norm=10.0)
        return h_mol

    def pocket_forward(self, pocket_src_tokens, pocket_src_distance, pocket_src_edge_type, **kwargs):
        poc_padding_mask = pocket_src_tokens.eq(self.pocket_model.padding_idx)
        poc_x = self.pocket_model.embed_tokens(pocket_src_tokens)
        poc_graph_attn_bias = self.get_dist_features(pocket_src_distance, pocket_src_edge_type, "pocket")
        poc_outputs = self.pocket_model.encoder(poc_x, padding_mask=poc_padding_mask, attn_mask=poc_graph_attn_bias)
        poc_rep_eu = poc_outputs[0][:, 0, :]
        pocket_concepts = self.pocket_concept_extractor(poc_rep_eu)
        pocket_concepts_masked, _ = self.pocket_sparse_mask(pocket_concepts)
        pocket_pooled = self.pocket_pooler(pocket_concepts_masked)
        u_poc = self.pocket_project(pocket_pooled) * self.pocket_alpha.exp()
        with torch.autocast(u_poc.device.type, dtype=torch.float32):
            h_poc = safe_exp_map0(u_poc, self.curv.exp(), max_norm=10.0)
        return h_poc

    def protein_forward(self, protein_sequences, **kwargs):
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
                prot_rep_eu = prot_outputs.hidden_states[-1][:, 0, :].float()
        prot_rep_eu = prot_rep_eu.to(self.protein_adapter.weight.dtype)
        prot_rep_eu = self.protein_adapter(prot_rep_eu)
        protein_concepts = self.protein_concept_extractor(prot_rep_eu)
        protein_concepts_masked, _ = self.protein_sparse_mask(protein_concepts)
        protein_pooled = self.protein_pooler(protein_concepts_masked)
        u_prot = self.protein_project(protein_pooled) * self.protein_alpha.exp()
        with torch.autocast(u_prot.device.type, dtype=torch.float32):
            h_prot = safe_exp_map0(u_prot, self.curv.exp(), max_norm=10.0)
        return h_prot


@register_model_architecture("causal_three_hybrid_v2", "causal_three_hybrid_v2")
def causal_three_hybrid_v2_architecture(args):
    causal_three_hybrid_v1_architecture(args)
