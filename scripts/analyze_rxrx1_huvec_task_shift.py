#!/usr/bin/env python
"""Task-linked, class-conditioned batch analysis from supervised HUVEC folds."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def atomic(path, text):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(path.name+f".{os.getpid()}.tmp"); tmp.write_text(text); os.replace(tmp,path)


def analyze(root):
    root=Path(root).resolve(); registry=json.loads((root/"study_registry.json").read_text())
    rows=[]; fold_rows=[]
    for target in registry["diagnostic_source_pool"]:
        run=root/"runs"/f"huvec_batch12_dense_loo_t{target}"
        if not (run/"RESULT.json").is_file(): continue
        pred=pd.read_parquet(run/"site_predictions.parquet")
        emb=pd.read_parquet(run/"well_embeddings.parquet")
        emb["vector"]=emb.embedding.map(lambda x: np.asarray(x,dtype=np.float32))
        iid=emb[emb.role=="iid_validation"]
        centroids={int(k):np.stack(v.vector).mean(0) for k,v in iid.groupby("label")}
        target_emb=emb[emb.role=="target"]
        distances={}
        for label,g in target_emb.groupby("label"):
            if int(label) not in centroids: continue
            a=np.stack(g.vector).mean(0); b=centroids[int(label)]
            distances[int(label)]=float(1-(a@b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8))
        ip=pred[pred.role=="iid_validation"].groupby("label").agg(iid_acc=("correct_top1","mean"),iid_logp=("true_log_probability","mean"))
        tp=pred[pred.role=="target"].groupby("label").agg(target_acc=("correct_top1","mean"),target_logp=("true_log_probability","mean"))
        joined=ip.join(tp,how="inner"); joined["target"]=int(target); joined["label"]=joined.index.astype(int)
        joined["accuracy_degradation"]=joined.iid_acc-joined.target_acc
        joined["logp_degradation"]=joined.iid_logp-joined.target_logp
        joined["conditioned_embedding_distance"]=joined.label.map(distances)
        rows.append(joined.reset_index(drop=True))
        tg=pred[pred.role=="target"]
        paired=tg.groupby("well_id").filter(lambda x:len(x)==2).groupby("well_id")
        same=float(paired.prediction.nunique().eq(1).mean()) if paired.ngroups else np.nan
        both=float(paired.correct_top1.all().mean()) if paired.ngroups else np.nan
        fold_rows.append({"target":int(target),"site_accuracy":float(tg.correct_top1.mean()),
                          "mean_rank":float(tg.true_class_rank.mean()),
                          "paired_site_same_prediction":same,"paired_site_both_correct":both,
                          "mean_conditioned_embedding_distance":float(np.mean(list(distances.values())))})
    classes=pd.concat(rows,ignore_index=True); folds=pd.DataFrame(fold_rows)
    out=root/"analysis"/"task_linked"; out.mkdir(parents=True,exist_ok=True)
    classes.to_csv(out/"class_conditioned_degradation.csv",index=False); folds.to_csv(out/"batch_difficulty.csv",index=False)
    valid=classes.dropna(subset=["conditioned_embedding_distance","logp_degradation"])
    r=float(valid.conditioned_embedding_distance.corr(valid.logp_degradation))
    profiles=classes.pivot(index="label",columns="target",values="logp_degradation")
    profile_corr=profiles.corr(min_periods=100)
    profile_corr.to_csv(out/"batch_degradation_profile_correlation.csv")
    off=profile_corr.to_numpy()[np.triu_indices(len(profile_corr),1)]
    summary={"folds":len(folds),"class_batch_pairs":len(classes),
             "embedding_distance_vs_logp_degradation_r":r,
             "mean_cross_batch_degradation_profile_correlation":float(np.nanmean(off)),
             "mean_paired_site_prediction_agreement":float(folds.paired_site_same_prediction.mean()),
             "mean_paired_site_both_correct":float(folds.paired_site_both_correct.mean())}
    atomic(out/"SUMMARY.json",json.dumps(summary,indent=2,sort_keys=True))
    plt.figure(figsize=(7,5)); plt.scatter(valid.conditioned_embedding_distance,valid.logp_degradation,s=5,alpha=.18)
    plt.xlabel("class-conditioned supervised embedding distance"); plt.ylabel("true-class log-probability degradation")
    plt.title(f"Task-linked batch displacement (r={r:.2f})"); plt.tight_layout(); plt.savefig(out/"difficulty_vs_degradation.png",dpi=180); plt.close()
    report=f"""<!doctype html><meta charset='utf-8'><title>Task-linked HUVEC batch analysis</title><style>body{{font:16px system-ui;max-width:900px;margin:40px auto;line-height:1.55}}.card{{background:#eef6ff;padding:16px;border-radius:10px}}</style><h1>What makes a HUVEC batch difficult?</h1><p class='card'>This analysis uses the trained supervised ViT itself—not Cell-DINO—and always compares samples within the same perturbation class. Difficulty is therefore tied to loss of perturbation evidence, not generic batch decodability.</p><h2>Measurements</h2><ul><li>Class-conditioned embedding displacement from source-IID support.</li><li>Loss of true-class log probability and accuracy for the same class.</li><li>Whether the two sites of each held-out well yield the same prediction.</li><li>Whether the same perturbations degrade across different batches.</li></ul><h2>Current summary</h2><pre>{json.dumps(summary,indent=2)}</pre><img src='difficulty_vs_degradation.png' style='max-width:100%'><p>Interpretation: a positive distance–degradation correlation supports task-relevant conditional shift. Positive cross-batch profile correlation means some perturbations are systematically fragile; near-zero correlation means difficulty is batch-specific.</p>"""
    atomic(out/"REPORT.html",report); return summary


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--result-root",required=True); a=p.parse_args(); print(json.dumps(analyze(a.result_root),indent=2))
