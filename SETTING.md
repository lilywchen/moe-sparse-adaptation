# Setting and claim boundary

This is supervised full fine-tuning of a pretrained visual foundation model into two scientific
image domains. DINOv2 has not previously seen the scientific task labels and should not be assumed
to contain a batch-invariant biological representation. Pretraining supplies a common visual
starting point; adaptation is where the model first learns the task and where dense versus sparse
capacity can be isolated.

The broad relevance is not “MoE solves microscopy batch effects.” Acquisition shift is a concrete,
high-stakes instance of heterogeneous adaptation: the same scientific concept appears under
different experimental or clinical environments. Conditional capacity might separate incompatible
adaptation demands, or it might waste experts on nuisance. The study is useful if it establishes
which occurs, under what design choices, and how to diagnose it.

RxRx1 is the primary characterization because it has many experiments and a multiclass biological
task. Camelyon17 is a deliberately different replication regime with few hospitals, binary labels,
and histopathology. Two datasets are enough for a focused paired study, but not for universal
claims; dataset-specific conclusions stay dataset-specific.
