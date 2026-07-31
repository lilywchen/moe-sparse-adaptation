"""Oracle disentangling block — the CEILING test for the improvement gap.

The phase-diagram result is: under a confound, MoE encodes the batch in its SHARED
representation just as much as dense (identical leakage lift), and additionally routes
by the batch (mi_site up to 0.49) — an extra failure mode that doesn't help. The open
question is whether MoE's STRUCTURE can be turned from liability to asset: can a clean,
batch-invariant shared pathway transfer to an UNSEEN site better than a dense model?

This block answers it with an oracle (a ceiling, not a deployable method):
    out = shared(x) + routed[true_site](x)      # seen sites (site < n_routed)
    out = shared(x)                              # held-out site, or shared_only eval
  * Routing is by the GROUND-TRUTH site label (oracle), not a learned router — so each
    routed expert perfectly absorbs its own site and the shared path is maximally free
    to carry only the content.
  * Expert-dropout (p_drop) randomly forces a fraction of seen-site samples through the
    shared path alone during training, so `shared(x)` becomes a self-sufficient classifier
    pathway and shared-only inference on the unseen site is meaningful (the head has seen
    shared-only inputs).

Decision rule:  shared-only held-out acc  >  dense held-out acc   ->  the gap is REAL and
achievable, build the learned (adversarial-routing) version.  <=  dense  ->  batch & content
aren't separable here even with ground truth -> the remedy is illusory, kill it cheaply.

The routed experts for SEEN sites are NOT available at the unseen site (there is no expert
for site K-1), which is exactly why the unseen-site evaluation is forced onto the shared path.
"""
import copy

import torch
import torch.nn as nn


class SharedRoutedBlock(nn.Module):
    def __init__(self, base_block: nn.Module, n_routed: int, p_drop: float = 0.5):
        super().__init__()
        self.n_routed = n_routed          # one routed expert per SEEN site (= K-1)
        self.p_drop = p_drop              # expert-dropout prob (forces self-sufficient shared path)
        self.shared = copy.deepcopy(base_block)
        self.routed = nn.ModuleList([copy.deepcopy(base_block) for _ in range(n_routed)])
        self._sites = None                # per-batch true site labels, set by the runner
        self.shared_only = False          # True -> ignore routed experts (shared-path inference)

    def set_sites(self, sites: torch.Tensor):
        """Stash this batch's GROUND-TRUTH site labels (LongTensor [B]) before forward.
        Row order is preserved through the network, so site[i] matches input row i here."""
        self._sites = sites

    def forward(self, x):
        shared_out = self.shared(x)
        if self.shared_only:
            return shared_out                       # shared-path-only (sites irrelevant)

        assert self._sites is not None, "oracle routing needs sites; call model.set_sites(site) first"
        s = self._sites.to(x.device)
        B = x.shape[0]
        assert s.shape[0] == B, f"sites ({s.shape[0]}) must match batch ({B})"

        # expert-dropout: in training, a fraction of samples skip their routed expert,
        # so the shared path is trained to classify on its own (needed for shared-only eval).
        if self.training and self.p_drop > 0:
            use_routed = torch.rand(B, device=x.device) > self.p_drop
        else:
            use_routed = torch.ones(B, dtype=torch.bool, device=x.device)

        routed_out = torch.zeros_like(shared_out)
        for r in range(self.n_routed):
            mask = (s == r) & use_routed             # samples of seen-site r that keep their expert
            if mask.any():
                routed_out[mask] = self.routed[r](x[mask]).to(routed_out.dtype)
        # held-out site (s >= n_routed) matches no r -> routed_out stays 0 -> shared-only, automatically.
        return shared_out + routed_out
