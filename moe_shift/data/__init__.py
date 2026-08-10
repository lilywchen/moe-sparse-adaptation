"""Data package. `make_loaders` dispatches on cfg['dataset']:
  rxrx1 -> real WILDS batch effects (rxrx1.make_rxrx1_loaders)
  else  -> injected-nuisance apparatus (sites.make_site_loaders)
Both return (train, test_within, test_heldout, audit) yielding (x, y, site)."""


def make_loaders(cfg):
    d = cfg.get("dataset")
    if d == "rxrx3_core":
        from .rxrx3_core import make_rxrx3_core_loaders
        return make_rxrx3_core_loaders(cfg)
    if d == "rxrx1":
        if cfg.get("rho_engineer", {}).get("enabled"):     # engineered experiment↔siRNA correlation
            from .rxrx1_rho import make_rxrx1_rho_loaders
            return make_rxrx1_rho_loaders(cfg)
        from .rxrx1 import make_rxrx1_loaders
        return make_rxrx1_loaders(cfg)
    if d == "camelyon17":                                  # cross-modality: histopathology hospitals
        from .camelyon17 import make_camelyon17_loaders
        return make_camelyon17_loaders(cfg)
    from .sites import make_site_loaders
    return make_site_loaders(cfg)


def make_val_loader(cfg):
    """OOD validation loader for model/hparam selection (None for the injected-nuisance datasets)."""
    d = cfg.get("dataset")
    if d == "rxrx3_core":
        from .rxrx3_core import make_rxrx3_core_val_loader
        return make_rxrx3_core_val_loader(cfg)
    if d == "rxrx1":
        from .rxrx1 import make_rxrx1_val_loader
        return make_rxrx1_val_loader(cfg)
    if d == "camelyon17":
        from .camelyon17 import make_camelyon17_val_loader
        return make_camelyon17_val_loader(cfg)
    return None
