"""
Phase 3 — Kaplan-Meier Survival Analysis
- High vs low risk KM curves (log-rank test) in GSE49710
- PPEF1 high/low expression KM
- Outputs: GSE49710_survival_df.csv, KM figures
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from config import DATA, OUT, FIG, DPI


def km_curve(durations_hi, events_hi, durations_lo, events_lo,
             title="", save_path=None, labels=("High-risk","Low-risk"),
             colors=("#E53935","#1E88E5")):
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(7, 5))
    for dur, ev, lbl, col in zip(
            [durations_hi, durations_lo], [events_hi, events_lo],
            labels, colors):
        kmf.fit(dur, ev, label=lbl)
        kmf.plot_survival_function(ax=ax, ci_show=True, color=col)

    res = logrank_test(durations_hi, durations_lo,
                       event_observed_A=events_hi,
                       event_observed_B=events_lo)
    ax.set_title(f"{title}\nLog-rank p = {res.p_value:.2e}")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Overall Survival Probability")
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=DPI); print(f"Saved {save_path}")
    plt.close()
    return res.p_value


def stratify_by_gene(expr, meta, gene, cohort="GSE49710"):
    idx   = meta[(meta["cohort"]==cohort)&meta["OS_time"].notna()].index
    e     = expr.loc[gene, idx]
    med   = e.median()
    hi    = e[e >= med].index
    lo    = e[e <  med].index
    return (meta.loc[hi,"OS_time"].values, meta.loc[hi,"OS_event"].values,
            meta.loc[lo,"OS_time"].values, meta.loc[lo,"OS_event"].values)


def run(expr=None, meta=None):
    print("="*60, "\nPHASE 3 — Survival Analysis\n", "="*60)
    if expr is None:
        expr = pd.read_csv(DATA("integrated_expression_811x19860.csv"), index_col=0)
        meta = pd.read_csv(DATA("meta_clinical_unified.csv"), index_col=0)

    # High vs low risk KM
    surv = meta[(meta["cohort"]=="GSE49710") & meta["OS_time"].notna()].copy()
    surv["high_risk_merged"] = surv["high_risk_merged"].astype(float)
    hi = surv[surv["high_risk_merged"]==1]
    lo = surv[surv["high_risk_merged"]==0]

    p = km_curve(hi["OS_time"].values, hi["OS_event"].values,
                 lo["OS_time"].values, lo["OS_event"].values,
                 title="GSE49710 — High vs Low Risk",
                 save_path=FIG("KM_GSE49710_high_risk.png"))
    print(f"High vs Low risk log-rank p = {p:.2e}")

    # PPEF1 KM
    dur_hi, ev_hi, dur_lo, ev_lo = stratify_by_gene(expr, meta, "PPEF1")
    p2 = km_curve(dur_hi, ev_hi, dur_lo, ev_lo,
                  title="GSE49710 — PPEF1 High vs Low",
                  labels=("PPEF1-high","PPEF1-low"),
                  save_path=FIG("KM_PPEF1.png"))
    print(f"PPEF1 KM log-rank p = {p2:.2e}")

    surv.to_csv(OUT("GSE49710_survival_df.csv"))
    print("Phase 3 complete.\n")
    return surv


if __name__ == "__main__": run()
