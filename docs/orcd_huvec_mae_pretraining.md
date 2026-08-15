# ORCD RxRx1 HUVEC MAE pretraining

This is a standalone self-supervised pretraining arm. It does not use perturbation labels in the
loss and does not run supervised fine-tuning.

The input is the frozen `primary_fold0` registry from the systematic HUVEC study. Only its
`train` role is admitted. The existing source-IID role and all eight target experiments are
excluded before a PyTorch Dataset is created. Ten percent of the admitted wells in each source
experiment are then deterministically reserved for reconstruction validation. Both microscope
sites from a well remain in the same MAE role.

The ORCD raw-image root contains only the 16 source experiment folders. The runner audits all six
channel files used by the MAE partition and refuses to start if another HUVEC experiment folder is
present. This physically seals target experiments while the manifest seal excludes source-IID
sites from loading.

Both dense models use 16x16 patches, 75% masking, a two-block 128-dimensional decoder, AdamW,
and bfloat16 on H100/H200. ViT-Micro has width 128 and six blocks. ViT-Tiny has width 192 and 12
blocks. A fixed validation masking seed makes reconstruction curves comparable across epochs.
The best encoder is selected only by held-out reconstruction loss. Training stops after at least
30 epochs when validation has not improved by 0.001 for 15 epochs, or at the 200-epoch safety cap.

`last.pt` contains model, optimizer, RNG, loader-generator, history, and plateau state. It resumes
automatically. `best_encoder.pt`, periodic encoder snapshots, `curves.jsonl`, `STATUS.json`, and
`RESULT.json` are written under the persistent result root.
