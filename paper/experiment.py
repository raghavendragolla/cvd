"""
Leave-one-site-out external validation of explainable heart-disease models
across the four UCI Heart Disease cohorts.

Measures: distribution shift, transfer degradation, calibration decay,
SHAP explanation stability, and recovery by recalibration / retraining.
"""
import json, warnings, numpy as np, pandas as pd
from scipy import stats, optimize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             roc_curve, confusion_matrix)
from xgboost import XGBClassifier
import shap

warnings.filterwarnings("ignore")
RNG = np.random.RandomState(42)
SEED = 42
RAW = "/mnt/user-data/uploads/Research_Project/data/raw"
OUT = "/tmp/claude-0/-home-claude/18c53616-59bf-5c3b-bdbc-96c0462aa827/scratchpad/results.json"

FEATURES = ["age","sex","cp","trestbps","chol","fbs","restecg",
            "thalach","exang","oldpeak","slope"]
CONT = ["age","trestbps","chol","thalach","oldpeak"]
CAT  = ["sex","cp","fbs","restecg","exang","slope"]
SITES = {"Cleveland":"cleveland_raw.csv","Hungarian":"hungarian_raw.csv",
         "Switzerland":"zurich_raw.csv","VA":"va_raw.csv"}

# ---------------------------------------------------------------- data
def load():
    out = {}
    for name, f in SITES.items():
        d = pd.read_csv(f"{RAW}/{f}").apply(pd.to_numeric, errors="coerce")
        d["target"] = (d["target"] > 0).astype(int)
        d.loc[d["chol"] == 0, "chol"] = np.nan          # UCI missing code
        out[name] = d[FEATURES + ["target"]].reset_index(drop=True)
    return out

# ---------------------------------------------------------------- metrics
def ece(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p); e = 0.0; n = len(y)
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i+1]) if i < bins-1 else (p >= edges[i]) & (p <= edges[i+1])
        if m.sum():
            e += m.sum()/n * abs(y[m].mean() - p[m].mean())
    return float(e)

def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1-eps)
    return np.log(p/(1-p))

def calib_slope_intercept(y, p):
    """Slope: coef of logit(p) in y ~ a + b*logit(p).
       Intercept (calibration-in-the-large): a in y ~ a + offset(logit(p))."""
    lp = logit(p).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    slope = float(LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
                  .fit(lp, y).coef_[0][0])
    off = lp.ravel()
    f = lambda a: -np.sum(y*(a+off) - np.log1p(np.exp(a+off)))
    inter = float(optimize.minimize_scalar(f, bounds=(-10, 10), method="bounded").x)
    return slope, inter

def net_benefit(y, p, pt):
    y = np.asarray(y); n = len(y)
    pred = (p >= pt).astype(int)
    tp = np.sum((pred == 1) & (y == 1)); fp = np.sum((pred == 1) & (y == 0))
    return tp/n - (fp/n)*(pt/(1-pt))

def evaluate(y, p, thr):
    y = np.asarray(y); pred = (p >= thr).astype(int)
    out = {}
    if len(np.unique(y)) > 1:
        out["auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    else:
        out["auc"] = np.nan; out["pr_auc"] = np.nan
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out["sens"] = float(tp/(tp+fn)) if (tp+fn) else np.nan
    out["spec"] = float(tn/(tn+fp)) if (tn+fp) else np.nan
    out["acc"]  = float((tp+tn)/len(y))
    out["f1"]   = float(2*tp/(2*tp+fp+fn)) if (2*tp+fp+fn) else np.nan
    out["brier"] = float(brier_score_loss(y, p))
    out["ece"] = ece(y, p)
    s, i = calib_slope_intercept(y, p)
    out["calib_slope"] = s; out["calib_intercept"] = i
    out["prevalence"] = float(y.mean())
    out["n"] = int(len(y))
    out["nb_20"] = float(net_benefit(y, p, 0.20))
    out["nb_50"] = float(net_benefit(y, p, 0.50))
    return out

def boot_auc_ci(y, p, B=1000):
    y, p = np.asarray(y), np.asarray(p)
    if len(np.unique(y)) < 2: return (np.nan, np.nan)
    rs = np.random.RandomState(SEED); vals = []
    for _ in range(B):
        idx = rs.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2: continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)

# ---------------------------------------------------------------- shift
def psi(a, b, bins=10):
    a = np.asarray(a.dropna()); b = np.asarray(b.dropna())
    if len(a) < 10 or len(b) < 10: return np.nan
    q = np.unique(np.quantile(a, np.linspace(0, 1, bins+1)))
    if len(q) < 3: return np.nan
    ea = np.clip(np.histogram(a, bins=q)[0]/len(a), 1e-4, None)
    eb = np.clip(np.histogram(b, bins=q)[0]/len(b), 1e-4, None)
    return float(np.sum((eb-ea)*np.log(eb/ea)))

def domain_classifier_auc(Xd, Xt, n_perm=200):
    X = np.vstack([Xd, Xt]); y = np.r_[np.zeros(len(Xd)), np.ones(len(Xt))]
    pipe = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=2000, C=1.0))])
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    p = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    obs = roc_auc_score(y, p)
    rs = np.random.RandomState(SEED); null = []
    for _ in range(n_perm):
        yp = rs.permutation(y)
        pp = cross_val_predict(pipe, X, yp, cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
                               method="predict_proba")[:, 1]
        null.append(roc_auc_score(yp, pp))
    pval = (np.sum(np.array(null) >= obs) + 1) / (n_perm + 1)
    return float(obs), float(pval)

# ---------------------------------------------------------------- models
def make_models():
    return {
        "LR":  Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=SEED))]),
        "RF":  RandomForestClassifier(n_estimators=400, max_depth=6, min_samples_leaf=3,
                                      random_state=SEED, n_jobs=-1),
        "XGB": XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                             random_state=SEED, n_jobs=-1, verbosity=0),
    }

def fit_imputer(Xdev):
    """Median for continuous, mode for categorical — fitted on development data only."""
    im_c = SimpleImputer(strategy="median").fit(Xdev[CONT])
    im_k = SimpleImputer(strategy="most_frequent").fit(Xdev[CAT])
    def apply(X):
        Z = X.copy()
        Z[CONT] = im_c.transform(X[CONT]); Z[CAT] = im_k.transform(X[CAT])
        return Z
    return apply

def recalibrate_intercept(y, p):
    off = logit(p)
    f = lambda a: -np.sum(y*(a+off) - np.log1p(np.exp(a+off)))
    a = float(optimize.minimize_scalar(f, bounds=(-10, 10), method="bounded").x)
    return lambda q: 1/(1+np.exp(-(a + logit(q))))

def recalibrate_platt(y, p):
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(logit(p).reshape(-1,1), y)
    return lambda q: lr.predict_proba(logit(q).reshape(-1,1))[:, 1]

def rank_agreement(r1, r2, k=5):
    """r1, r2: dicts feature -> mean|SHAP|."""
    f = list(r1.keys())
    v1 = np.array([r1[x] for x in f]); v2 = np.array([r2[x] for x in f])
    rho, _ = stats.spearmanr(v1, v2); tau, _ = stats.kendalltau(v1, v2)
    t1 = set(np.array(f)[np.argsort(-v1)][:k]); t2 = set(np.array(f)[np.argsort(-v2)][:k])
    jac = len(t1 & t2)/len(t1 | t2)
    return float(rho), float(tau), float(jac)

def mean_abs_shap(model, X):
    ex = shap.TreeExplainer(model)
    sv = ex.shap_values(X)
    sv = np.asarray(sv)
    if sv.ndim == 3: sv = sv[:, :, 1]
    return {f: float(v) for f, v in zip(X.columns, np.abs(sv).mean(axis=0))}

# ================================================================ main
def main():
    data = load()
    R = {"cohorts": {}, "shift": {}, "folds": {}, "explanation": {}}

    for s, d in data.items():
        R["cohorts"][s] = {"n": int(len(d)), "prevalence": float(d.target.mean()),
                           "age_mean": float(d.age.mean()), "male_pct": float(100*d.sex.mean()),
                           "missing_pct": {f: float(100*d[f].isna().mean()) for f in FEATURES}}

    for target in SITES:
        print(f"\n=== FOLD: held-out {target} ===", flush=True)
        dev = pd.concat([data[s] for s in SITES if s != target], ignore_index=True)
        tgt = data[target].copy()
        Xd_raw, yd = dev[FEATURES], dev.target.values
        Xt_raw, yt = tgt[FEATURES], tgt.target.values

        imp = fit_imputer(Xd_raw)
        Xd, Xt = imp(Xd_raw), imp(Xt_raw)

        # ---- shift
        sh = {"features": {}}
        for f in FEATURES:
            a, b = Xd_raw[f].dropna(), Xt_raw[f].dropna()
            if len(b) < 10:
                sh["features"][f] = {"psi": None, "ks_d": None, "ks_p": None,
                                     "wass": None, "note": "insufficient non-missing"}
                continue
            D, pv = stats.ks_2samp(a, b)
            sh["features"][f] = {"psi": psi(a, b), "ks_d": float(D), "ks_p": float(pv),
                                 "wass": float(stats.wasserstein_distance(a, b)/a.std())}
        ks_p = [v["ks_p"] for v in sh["features"].values() if v["ks_p"] is not None]
        if ks_p:
            order = np.argsort(ks_p); m = len(ks_p)
            adj = np.minimum.accumulate((np.array(ks_p)[order]*m/(np.arange(m)+1))[::-1])[::-1]
            names = [k for k, v in sh["features"].items() if v["ks_p"] is not None]
            for nm, a_ in zip(np.array(names)[order], adj):
                sh["features"][nm]["ks_p_bh"] = float(min(a_, 1.0))
            sh["n_sig_bh"] = int(np.sum(adj < 0.05))
        dc_auc, dc_p = domain_classifier_auc(Xd.values, Xt.values)
        sh["domain_auc"] = dc_auc; sh["domain_p"] = dc_p
        tab = np.array([[yd.sum(), len(yd)-yd.sum()], [yt.sum(), len(yt)-yt.sum()]])
        chi2, pv, _, _ = stats.chi2_contingency(tab)
        sh["prev_dev"] = float(yd.mean()); sh["prev_target"] = float(yt.mean())
        sh["prev_chi2"] = float(chi2); sh["prev_p"] = float(pv)
        R["shift"][target] = sh

        # ---- adaptation split of the target
        sss = StratifiedKFold(2, shuffle=True, random_state=SEED)
        ad_idx, ev_idx = next(sss.split(Xt, yt))
        Xa, ya = Xt.iloc[ad_idx], yt[ad_idx]
        Xe, ye = Xt.iloc[ev_idx], yt[ev_idx]

        fold = {}
        for mname, model in make_models().items():
            cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
            p_dev = cross_val_predict(model, Xd, yd, cv=cv, method="predict_proba")[:, 1]
            fpr, tpr, thr = roc_curve(yd, p_dev)
            thr_fixed = float(thr[np.argmax(tpr - fpr)])          # dev Youden threshold
            internal = evaluate(yd, p_dev, thr_fixed)
            internal["auc_ci"] = boot_auc_ci(yd, p_dev)

            model.fit(Xd, yd)
            p_t_all = model.predict_proba(Xt)[:, 1]
            transfer = evaluate(yt, p_t_all, thr_fixed)
            transfer["auc_ci"] = boot_auc_ci(yt, p_t_all)
            # re-optimised threshold on target (upper bound, reported for contrast)
            if len(np.unique(yt)) > 1:
                f2, t2, th2 = roc_curve(yt, p_t_all)
                transfer["sens_reopt"], transfer["spec_reopt"] = float(t2[np.argmax(t2-f2)]), float(1-f2[np.argmax(t2-f2)])

            # adaptation, all judged on the untouched eval half
            p_e = model.predict_proba(Xe)[:, 1]
            frozen = evaluate(ye, p_e, thr_fixed); frozen["auc_ci"] = boot_auc_ci(ye, p_e)
            p_a = model.predict_proba(Xa)[:, 1]

            rc_i = recalibrate_intercept(ya, p_a); rc_p = recalibrate_platt(ya, p_a)
            reint = evaluate(ye, np.clip(rc_i(p_e), 1e-6, 1-1e-6), 0.5)
            replat = evaluate(ye, np.clip(rc_p(p_e), 1e-6, 1-1e-6), 0.5)

            m2 = make_models()[mname]
            retrained = None
            if len(np.unique(ya)) > 1 and ya.sum() >= 5 and (len(ya)-ya.sum()) >= 5:
                m2.fit(Xa, ya)
                p_r = m2.predict_proba(Xe)[:, 1]
                retrained = evaluate(ye, p_r, 0.5); retrained["auc_ci"] = boot_auc_ci(ye, p_r)

            fold[mname] = {"internal": internal, "transfer": transfer, "frozen_eval": frozen,
                           "recal_intercept": reint, "recal_platt": replat, "retrained": retrained,
                           "thr_fixed": thr_fixed}
            print(f"  {mname}: internal AUC {internal['auc']:.3f} -> transfer {transfer['auc']:.3f} "
                  f"| ECE {internal['ece']:.3f} -> {transfer['ece']:.3f} | slope {transfer['calib_slope']:.2f}", flush=True)

        R["folds"][target] = fold

        # ---- explanation stability (XGB; common evaluation set)
        try:
            dev_model = make_models()["XGB"]; dev_model.fit(Xd, yd)
            rank_dev_on_dev = mean_abs_shap(dev_model, Xd)
            rank_dev_on_tgt = mean_abs_shap(dev_model, Xe)     # mix-driven change
            tgt_model = make_models()["XGB"]
            expl = {"mix_driven": None, "model_driven": None}
            rho, tau, jac = rank_agreement(rank_dev_on_dev, rank_dev_on_tgt)
            expl["mix_driven"] = {"spearman": rho, "kendall": tau, "jaccard_top5": jac}
            if len(np.unique(ya)) > 1 and ya.sum() >= 5 and (len(ya)-ya.sum()) >= 5:
                tgt_model.fit(Xa, ya)
                rank_tgt_on_tgt = mean_abs_shap(tgt_model, Xe)  # same data, different model
                rho2, tau2, jac2 = rank_agreement(rank_dev_on_tgt, rank_tgt_on_tgt)
                expl["model_driven"] = {"spearman": rho2, "kendall": tau2, "jaccard_top5": jac2}
            expl["top5_dev"] = sorted(rank_dev_on_dev, key=rank_dev_on_dev.get, reverse=True)[:5]
            expl["top5_tgt"] = sorted(rank_dev_on_tgt, key=rank_dev_on_tgt.get, reverse=True)[:5]
            expl["ranks_dev"] = rank_dev_on_dev
            expl["ranks_tgt"] = rank_dev_on_tgt
            R["explanation"][target] = expl
            print(f"  SHAP: mix-driven tau {tau:.2f}"
                  + (f", model-driven tau {tau2:.2f}" if expl["model_driven"] else ""), flush=True)
        except Exception as e:
            print("  SHAP failed:", e, flush=True)

    # explainer noise floor: repeated SHAP on identical data with different model seeds
    dev = pd.concat([data[s] for s in SITES if s != "Cleveland"], ignore_index=True)
    imp = fit_imputer(dev[FEATURES]); Xd = imp(dev[FEATURES]); yd = dev.target.values
    taus = []
    base = None
    for sd in [1, 2, 3, 4, 5]:
        m = XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.9,
                          colsample_bytree=0.9, eval_metric="logloss", random_state=sd,
                          n_jobs=-1, verbosity=0).fit(Xd, yd)
        r = mean_abs_shap(m, Xd)
        if base is None: base = r
        else: taus.append(rank_agreement(base, r)[1])
    R["noise_floor_kendall"] = {"mean": float(np.mean(taus)), "min": float(np.min(taus))}
    print(f"\nExplainer noise floor: mean Kendall tau {np.mean(taus):.3f}", flush=True)

    json.dump(R, open(OUT, "w"), indent=2, default=lambda o: None if isinstance(o, float) and np.isnan(o) else o)
    print("\nwrote", OUT)

if __name__ == "__main__":
    main()
