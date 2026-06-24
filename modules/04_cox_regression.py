"""
Phase 4 — Cox Proportional Hazards Regression
- Univariate Cox for all consensus DEGs
- Multivariate Cox: MYCN + INSS stage + Cox risk score
- Forest plot of top independent predictors
- Outputs: Cox_univariate_results.csv, Cox_risk_scores.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from config import DATA, OUT, FIG, DPI


def univariate_cox(expr, surv_df, genes):
    results = []
    for gene in genes:
        try:
            df = surv_df[["OS_time","OS_event"]].copy()
            df[gene] = expr.loc[gene, df.index]
            df = df.dropna()
            cph = CoxPHFitter()
            cph.fit(df, duration_col="OS_time", event_col="OS_event")
            s = cph.summary.loc[gene]
            results.append({"gene":gene, "HR":np.exp(s["coef"]),
                             "CI_lo":np.exp(s["coef lower 95%"]),
                             "CI_hi":np.exp(s["coef upper 95%"]),
                             "p": s["p"]})
        except Exception:
            pass
    return pd.DataFrame(results).set_index("gene").sort_values("p")


def multivariate_cox(surv_df, covariates):
    df = surv_df[["OS_time","OS_event"] + covariates].dropna()
    cph = CoxPHFitter()
    cph.fit(df, duration_col="OS_time", event_col="OS_event")
    return cph


def forest_plot(cox_res, top_n=20, save_path=None):
    top = cox_res.head(top_n).copy()
    top = top.sort_values("HR")
    fig, ax = plt.subplots(figsize=(7, top_n * 0.35 + 1))
    y = range(len(top))
    ax.scatter(top["HR"], y, color="#E53935", zorder=3, s=30)
    for i, (_, row) in enumerate(top.iterrows()):
        ax.plot([row["CI_lo"], row["CI_hi"]], [i, i],
                color="#E53935", lw=1.5, zorder=2)
    ax.axvline(1, ls="--", color="grey", lw=0.8)
    ax.set_yticks(list(y)); ax.set_yticklabels(top.index, fontsize=8)
    ax.set_xlabel("Hazard Ratio (95% CI)")
    ax.set_title(f"Top {top_n} Univariate Cox — DEG Hazard Ratios")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=DPI); print(f"Saved {save_path}")
    plt.close()


def run(expr=None, meta=None, consensus=None):
    print("="*60, "\nPHASE 4 — Cox Regression\n", "="*60)
    if expr is None:
        expr      = pd.read_csv(DATA("integrated_expression_811x19860.csv"), index_col=0)
        meta      = pd.read_csv(DATA("meta_clinical_unified.csv"), index_col=0)
        consensus = pd.read_csv(OUT("consensus_DEG_filtered.csv"), index_col=0)

    surv = meta[(meta["cohort"]=="GSE49710") & meta["OS_time"].notna()].copy()
    cox_uni = univariate_cox(expr, surv, consensus.index.tolist())
    cox_uni.to_csv(OUT("Cox_univariate_results.csv"))
    print(f"Univariate Cox: {len(cox_uni):,} genes tested")

    # Risk score from top 500 Cox genes
    top500 = cox_uni[cox_uni["p"]<0.05].head(500).index.tolist()
    risk_scores = expr.loc[top500].T.dot(
        np.log(cox_uni.loc[top500,"HR"])
    )
    risk_scores.name = "cox_risk_score"
    risk_scores.to_csv(OUT("Cox_risk_scores.csv"))

    forest_plot(cox_uni, top_n=20, save_path=FIG("Cox_forest_top20.png"))

    # Multivariate Cox
    surv["cox_risk_score"] = risk_scores.reindex(surv.index)
    try:
        mv_cph = multivariate_cox(surv, ["MYCN_amp","INSS_stage","cox_risk_score"])
        mv_cph.print_summary()
    except Exception as e:
        print(f"Multivariate Cox skipped: {e}")

    print("Phase 4 complete.\n")
    return cox_uni, risk_scores


if __name__ == "__main__": run()
