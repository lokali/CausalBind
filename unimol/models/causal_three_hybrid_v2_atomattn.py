"""
CausalThreeHybridV2-AtomAttn: fixes the concept-extractor input bottleneck.

Diagnosis (2026-07): in causal_three_hybrid_v2, the
ConceptExtractor receives only the pooled CLS/BOS token
(`mol_outputs[0][:, 0, :]`, a single vector), inherited unchanged from the
non-causal HypSeek baseline. Since the Perceiver cross-attention then operates
over a key/value sequence of length 1, softmax attention is trivially uniform
and every one of the K concept queries receives the identical attended value;
we verified empirically (across all K in {32,128,256} and all lambda in
[0, 1]) that the resulting K concept vectors are exactly collinear
(cosine similarity = 1.0000) for every real DUD-E/LIT-PCBA molecule and
pocket tested.

Fix: feed the ConceptExtractor the FULL per-atom / per-residue encoder output
(`mol_outputs[0]`, shape (B, N_tokens, D)) instead of the CLS/BOS slice, with
the padding mask threaded through so cross-attention only attends over real
atoms/residues. This gives the K concept queries an actual set of
differentiated tokens to attend over, so they can (in principle) specialize
to different substructures/pharmacophoric roles instead of collapsing to one
direction. Only the mol and pocket branches are changed here; the protein
(ESM2) branch is left as in V2.

Everything else (mask, pooling, projection, loss) is unchanged, so this is a
minimal, targeted fix rather than a new architecture.
"""

import torch
from unicore.models import register_model, register_model_architecture
from .lorentz import safe_exp_map0
from .causal_three_hybrid_v2 import CausalThreeHybridV2, causal_three_hybrid_v2_architecture


@register_model("causal_three_hybrid_v2_atomattn")
class CausalThreeHybridV2AtomAttn(CausalThreeHybridV2):
    """Same as CausalThreeHybridV2, except the mol/pocket ConceptExtractor
    attends over the full per-token encoder output instead of the pooled
    CLS/BOS token."""

    def _interpret_branch(
        self,
        src_tokens,
        src_distance,
        src_edge_type,
        branch,
    ):
        """Return the normal embedding plus concept-to-token attention.

        This is an analysis-only path: it shares every trained module with the
        regular forward methods and adds no parameters or checkpoint keys.
        """
        if branch == "mol":
            encoder = self.mol_model
            extractor = self.mol_concept_extractor
            sparse_mask = self.mol_sparse_mask
            pooler = self.mol_pooler
            projector = self.mol_project
            alpha = self.mol_alpha
        elif branch == "pocket":
            encoder = self.pocket_model
            extractor = self.pocket_concept_extractor
            sparse_mask = self.pocket_sparse_mask
            pooler = self.pocket_pooler
            projector = self.pocket_project
            alpha = self.pocket_alpha
        else:
            raise ValueError(f"Unsupported interpretation branch: {branch}")

        padding_mask = src_tokens.eq(encoder.padding_idx)
        token_features = encoder.embed_tokens(src_tokens)
        graph_attn_bias = self.get_dist_features(
            src_distance, src_edge_type, branch
        )
        encoder_output = encoder.encoder(
            token_features,
            padding_mask=padding_mask,
            attn_mask=graph_attn_bias,
        )[0]
        concepts, attention_layers = extractor(
            encoder_output,
            key_padding_mask=padding_mask,
            return_attention=True,
        )
        masked_concepts, realized_mask = sparse_mask(concepts)
        tangent = projector(pooler(masked_concepts)) * alpha.exp()
        with torch.autocast(tangent.device.type, dtype=torch.float32):
            embedding = safe_exp_map0(
                tangent, self.curv.exp(), max_norm=10.0
            )
        return {
            "embedding": embedding,
            "concepts": concepts,
            "masked_concepts": masked_concepts,
            "realized_mask": realized_mask,
            "attention_layers": attention_layers,
            "padding_mask": padding_mask,
            "encoder_output": encoder_output,
        }

    def mol_forward_with_attention(
        self, mol_src_tokens, mol_src_distance, mol_src_edge_type, **kwargs
    ):
        return self._interpret_branch(
            mol_src_tokens,
            mol_src_distance,
            mol_src_edge_type,
            "mol",
        )

    def pocket_forward_with_attention(
        self,
        pocket_src_tokens,
        pocket_src_distance,
        pocket_src_edge_type,
        **kwargs,
    ):
        return self._interpret_branch(
            pocket_src_tokens,
            pocket_src_distance,
            pocket_src_edge_type,
            "pocket",
        )

    def embedding_from_concepts(self, concepts, branch):
        """Recompute a branch embedding after an analysis-time concept ablation."""
        if branch == "mol":
            sparse_mask, pooler = self.mol_sparse_mask, self.mol_pooler
            projector, alpha = self.mol_project, self.mol_alpha
        elif branch == "pocket":
            sparse_mask, pooler = self.pocket_sparse_mask, self.pocket_pooler
            projector, alpha = self.pocket_project, self.pocket_alpha
        else:
            raise ValueError(f"Unsupported concept branch: {branch}")
        masked_concepts, _ = sparse_mask(concepts)
        tangent = projector(pooler(masked_concepts)) * alpha.exp()
        with torch.autocast(tangent.device.type, dtype=torch.float32):
            return safe_exp_map0(tangent, self.curv.exp(), max_norm=10.0)

    def embedding_from_encoder_tokens(
        self, encoder_output, key_padding_mask, branch
    ):
        """Recompute an embedding while excluding selected encoder tokens.

        The Uni-Mol encoder output is held fixed. This isolates whether the
        concept bottleneck uses the selected local atom tokens, without
        conflating the result with a second pass through the molecular encoder.
        """
        if branch == "mol":
            extractor = self.mol_concept_extractor
        elif branch == "pocket":
            extractor = self.pocket_concept_extractor
        else:
            raise ValueError(f"Unsupported token branch: {branch}")
        concepts = extractor(
            encoder_output, key_padding_mask=key_padding_mask
        )
        return self.embedding_from_concepts(concepts, branch)

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

        # --- Mol: FULL per-atom sequence → concept extractor (atom-level attention) ---
        mol_padding_mask = mol_src_tokens.eq(self.mol_model.padding_idx)
        mol_x = self.mol_model.embed_tokens(mol_src_tokens)
        mol_graph_attn_bias = self.get_dist_features(mol_src_distance, mol_src_edge_type, "mol")
        mol_outputs = self.mol_model.encoder(mol_x, padding_mask=mol_padding_mask, attn_mask=mol_graph_attn_bias)
        mol_rep_full = mol_outputs[0]  # (B, N_tokens, 512) -- full sequence, NOT [:, 0, :]
        mol_kpm = mol_padding_mask if mol_padding_mask is not None else torch.zeros(
            mol_rep_full.shape[:2], dtype=torch.bool, device=mol_rep_full.device
        )
        mol_concepts = self.mol_concept_extractor(mol_rep_full, key_padding_mask=mol_kpm)
        mol_concepts_masked, _ = self.mol_sparse_mask(mol_concepts)
        mol_pooled = self.mol_pooler(mol_concepts_masked)
        u_mol = self.mol_project(mol_pooled) * self.mol_alpha.exp()

        # --- Pocket: FULL per-residue-atom sequence → concept extractor ---
        poc_padding_mask = pocket_src_tokens.eq(self.pocket_model.padding_idx)
        poc_x = self.pocket_model.embed_tokens(pocket_src_tokens)
        poc_graph_attn_bias = self.get_dist_features(pocket_src_distance, pocket_src_edge_type, "pocket")
        poc_outputs = self.pocket_model.encoder(poc_x, padding_mask=poc_padding_mask, attn_mask=poc_graph_attn_bias)
        poc_rep_full = poc_outputs[0]  # (B, N_tokens, 512)
        poc_kpm = poc_padding_mask if poc_padding_mask is not None else torch.zeros(
            poc_rep_full.shape[:2], dtype=torch.bool, device=poc_rep_full.device
        )
        pocket_concepts = self.pocket_concept_extractor(poc_rep_full, key_padding_mask=poc_kpm)
        pocket_concepts_masked, _ = self.pocket_sparse_mask(pocket_concepts)
        pocket_pooled = self.pocket_pooler(pocket_concepts_masked)
        u_poc = self.pocket_project(pocket_pooled) * self.pocket_alpha.exp()

        # --- Protein (ESM2, frozen) -- unchanged, still CLS-pooled ---
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

        with torch.autocast(u_mol.device.type, dtype=torch.float32):
            h_mol = safe_exp_map0(u_mol, κ, max_norm=10.0)
            h_poc = safe_exp_map0(u_poc, κ, max_norm=10.0)
            h_prot = safe_exp_map0(u_prot, κ, max_norm=10.0)

        return h_prot, h_poc, h_mol

    def mol_forward(self, mol_src_tokens, mol_src_distance, mol_src_edge_type, **kwargs):
        mol_padding_mask = mol_src_tokens.eq(self.mol_model.padding_idx)
        mol_x = self.mol_model.embed_tokens(mol_src_tokens)
        mol_graph_attn_bias = self.get_dist_features(mol_src_distance, mol_src_edge_type, "mol")
        mol_outputs = self.mol_model.encoder(mol_x, padding_mask=mol_padding_mask, attn_mask=mol_graph_attn_bias)
        mol_rep_full = mol_outputs[0]
        mol_kpm = mol_padding_mask if mol_padding_mask is not None else torch.zeros(
            mol_rep_full.shape[:2], dtype=torch.bool, device=mol_rep_full.device
        )
        mol_concepts = self.mol_concept_extractor(mol_rep_full, key_padding_mask=mol_kpm)
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
        poc_rep_full = poc_outputs[0]
        poc_kpm = poc_padding_mask if poc_padding_mask is not None else torch.zeros(
            poc_rep_full.shape[:2], dtype=torch.bool, device=poc_rep_full.device
        )
        pocket_concepts = self.pocket_concept_extractor(poc_rep_full, key_padding_mask=poc_kpm)
        pocket_concepts_masked, _ = self.pocket_sparse_mask(pocket_concepts)
        pocket_pooled = self.pocket_pooler(pocket_concepts_masked)
        u_poc = self.pocket_project(pocket_pooled) * self.pocket_alpha.exp()
        with torch.autocast(u_poc.device.type, dtype=torch.float32):
            h_poc = safe_exp_map0(u_poc, self.curv.exp(), max_norm=10.0)
        return h_poc


@register_model_architecture("causal_three_hybrid_v2_atomattn", "causal_three_hybrid_v2_atomattn")
def causal_three_hybrid_v2_atomattn_architecture(args):
    causal_three_hybrid_v2_architecture(args)
