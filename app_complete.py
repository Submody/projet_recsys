"""
FitMatch - Programmes sportifs personnalisés (Mini-projet, sujet 11)
====================================================================
Lancer : streamlit run app.py
Design : Style iOS Dark (Glassmorphism & SF Pro)
"""
from __future__ import annotations

import html
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="FitMatch - Programmes sportifs", layout="wide")

# =============================================================================
# 0. CONSTANTES & PALETTE iOS
# =============================================================================
LEVELS = ["Débutant", "Intermédiaire", "Avancé"]
LEVEL_NUM = {n: i + 1 for i, n in enumerate(LEVELS)}
GOALS = ["Perte de poids", "Prise de muscle", "Endurance", "Bien-être & souplesse"]
EQUIPMENTS = ["Haltères", "Élastique", "Tapis", "Barre de traction"]

# Affinité objectif -> catégorie d'exercice
GOAL_WEIGHTS = {
    "Perte de poids": {"HIIT": 1.0, "Cardio": 0.9, "Force": 0.6, "Mobilité": 0.2, "Flexibilité": 0.1},
    "Prise de muscle": {"Force": 1.0, "HIIT": 0.35, "Cardio": 0.15, "Mobilité": 0.3, "Flexibilité": 0.2},
    "Endurance": {"Cardio": 1.0, "HIIT": 0.8, "Force": 0.5, "Mobilité": 0.3, "Flexibilité": 0.2},
    "Bien-être & souplesse": {"Flexibilité": 1.0, "Mobilité": 1.0, "Force": 0.5, "Cardio": 0.4, "HIIT": 0.1},
}

# Couleurs iOS d'accentuation
CAT_STYLE = {
    "Force": ("#5E5CE6", ""),      # Indigo iOS
    "Cardio": ("#64D2FF", ""),     # Teal iOS
    "HIIT": ("#FF453A", ""),       # Red iOS
    "Flexibilité": ("#30D158", ""),# Green iOS
    "Mobilité": ("#FF9F0A", ""),   # Orange iOS
}
IMPACT_COLOR = {"Faible": "#30D158", "Modéré": "#FF9F0A", "Élevé": "#FF453A"}

MODELS = ["Popularité", "Règles & profil", "Contenu (TF-IDF)", "Item-Item kNN", "Factorisation (ALS)", "Hybride"]
MODEL_COLORS = {
    "Popularité": "#8E8E93", "Règles & profil": "#FF9F0A", "Contenu (TF-IDF)": "#30D158",
    "Item-Item kNN": "#64D2FF", "Factorisation (ALS)": "#5E5CE6", "Hybride": "#BF5AF2",
}
DEFAULT_W = (0.40, 0.15, 0.15, 0.30)  # ALS, kNN, contenu, profil

DAY_NAMES = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
SPREAD = {2: [1, 4], 3: [0, 2, 4], 4: [0, 1, 3, 4], 5: [0, 1, 2, 4, 5], 6: [0, 1, 2, 3, 4, 5]}

EXERCISES = [
    ("Pompes sur les genoux", "Force", "Pecs/Bras", 1, "Aucun", "Faible", 4, "reps", "pompes allégées pour pectoraux et triceps, idéal pour débuter"),
    ("Pompes classiques", "Force", "Pecs/Bras", 2, "Aucun", "Faible", 5, "reps", "pompes au sol pour pectoraux, épaules et triceps"),
    ("Pompes diamant", "Force", "Pecs/Bras", 3, "Aucun", "Modéré", 5, "reps", "pompes mains rapprochées ciblant fortement les triceps"),
    ("Dips sur chaise", "Force", "Pecs/Bras", 2, "Aucun", "Modéré", 4, "reps", "dips au poids du corps pour triceps et épaules"),
    ("Développé couché haltères", "Force", "Pecs/Bras", 2, "Haltères", "Faible", 6, "reps", "développé allongé avec haltères pour pectoraux"),
    ("Curl biceps haltères", "Force", "Pecs/Bras", 1, "Haltères", "Faible", 4, "reps", "flexion des bras avec haltères pour biceps"),
    ("Extension triceps élastique", "Force", "Pecs/Bras", 1, "Élastique", "Faible", 4, "reps", "extension des bras avec élastique pour triceps"),
    ("Rowing élastique", "Force", "Dos/Épaules", 1, "Élastique", "Faible", 5, "reps", "tirage horizontal avec élastique pour dos et posture"),
    ("Rowing haltères penché", "Force", "Dos/Épaules", 2, "Haltères", "Faible", 6, "reps", "tirage buste penché avec haltères pour grand dorsal"),
    ("Tractions pronation", "Force", "Dos/Épaules", 3, "Barre de traction", "Modéré", 6, "reps", "tractions à la barre pour dos et biceps"),
    ("Élévations latérales haltères", "Force", "Dos/Épaules", 1, "Haltères", "Faible", 4, "reps", "élévations latérales pour épaules et deltoïdes"),
    ("Superman au sol", "Force", "Dos/Épaules", 1, "Tapis", "Faible", 4, "reps", "extension dorsale allongé pour lombaires et dos"),
    ("Pompes en pike", "Force", "Dos/Épaules", 2, "Aucun", "Modéré", 5, "reps", "pompes corps en V pour épaules et deltoïdes"),
    ("Squats poids du corps", "Force", "Jambes", 1, "Aucun", "Faible", 5, "reps", "squats simples pour cuisses et fessiers"),
    ("Fentes alternées", "Force", "Jambes", 1, "Aucun", "Modéré", 5, "reps", "fentes avant alternées pour quadriceps et fessiers"),
    ("Pont fessier", "Force", "Jambes", 1, "Tapis", "Faible", 4, "reps", "élévation du bassin allongé pour fessiers et ischios"),
    ("Soulevé de terre haltères", "Force", "Jambes", 2, "Haltères", "Faible", 6, "reps", "soulevé de terre pour ischios, fessiers et dos"),
    ("Squat bulgare", "Force", "Jambes", 3, "Aucun", "Modéré", 6, "reps", "squat unilatéral pied arrière surélevé pour jambes"),
    ("Montées sur chaise", "Force", "Jambes", 2, "Aucun", "Modéré", 5, "reps", "montées de marche pour cuisses et fessiers"),
    ("Squat goblet", "Force", "Jambes", 2, "Haltères", "Faible", 5, "reps", "squat avec haltère devant la poitrine pour jambes"),
    ("Chaise murale", "Force", "Jambes", 1, "Aucun", "Faible", 3, "durée", "position assise contre un mur pour cuisses, endurance musculaire"),
    ("Tirage élastique jambes", "Force", "Jambes", 1, "Élastique", "Faible", 4, "reps", "abduction avec élastique pour fessiers et hanches"),
    ("Gainage planche", "Force", "Core", 1, "Aucun", "Faible", 3, "durée", "planche statique pour sangle abdominale"),
    ("Gainage latéral", "Force", "Core", 2, "Tapis", "Faible", 4, "durée", "planche latérale pour obliques et stabilité"),
    ("Crunchs", "Force", "Core", 1, "Tapis", "Faible", 4, "reps", "flexion du buste allongé pour abdominaux"),
    ("Relevés de jambes", "Force", "Core", 2, "Tapis", "Faible", 4, "reps", "relevés de jambes allongé pour bas du ventre"),
    ("Russian twist", "Force", "Core", 2, "Aucun", "Faible", 4, "reps", "rotations du buste assis pour obliques"),
    ("Bird-dog", "Force", "Core", 1, "Tapis", "Faible", 4, "reps", "extension bras-jambe opposés pour gainage et équilibre"),
    ("Hollow body hold", "Force", "Core", 3, "Tapis", "Faible", 3, "durée", "position en banane statique, gainage avancé abdominaux"),
    ("Marche rapide sur place", "Cardio", "Full body", 1, "Aucun", "Faible", 5, "durée", "marche dynamique sur place pour activer le cardio en douceur"),
    ("Jumping Jacks", "Cardio", "Full body", 1, "Aucun", "Modéré", 4, "durée", "sauts écartés cardio pour remise en forme"),
    ("Corde à sauter imaginaire", "Cardio", "Full body", 2, "Aucun", "Élevé", 4, "durée", "sauts rythmés cardio, coordination et mollets"),
    ("Montées de genoux", "Cardio", "Full body", 1, "Aucun", "Modéré", 4, "durée", "course sur place genoux hauts, cardio et cuisses"),
    ("Mountain climbers", "Cardio", "Core", 2, "Aucun", "Modéré", 4, "durée", "gainage dynamique cardio pour abdos et épaules"),
    ("Shadow boxing", "Cardio", "Full body", 1, "Aucun", "Faible", 5, "durée", "enchaînement de coups de boxe dans le vide, cardio et bras"),
    ("Pas chassés latéraux", "Cardio", "Jambes", 1, "Aucun", "Faible", 4, "durée", "déplacements latéraux cardio doux pour jambes"),
    ("Skater jumps", "Cardio", "Jambes", 2, "Aucun", "Élevé", 4, "durée", "sauts latéraux de patineur, cardio et jambes"),
    ("Burpees", "HIIT", "Full body", 3, "Aucun", "Élevé", 5, "durée", "exercice cardio intense sollicitant tout le corps"),
    ("Burpees sans saut", "HIIT", "Full body", 2, "Aucun", "Modéré", 5, "durée", "burpees allégés sans impact pour cardio full body"),
    ("Squats sautés", "HIIT", "Jambes", 2, "Aucun", "Élevé", 4, "durée", "squats explosifs pliométriques pour jambes et cardio"),
    ("Fentes sautées", "HIIT", "Jambes", 3, "Aucun", "Élevé", 4, "durée", "fentes dynamiques pour jambes et cardio intense"),
    ("Sprint sur place", "HIIT", "Full body", 2, "Aucun", "Élevé", 3, "durée", "sprint à fond sur place, intervalles cardio"),
    ("Thrusters haltères", "HIIT", "Full body", 2, "Haltères", "Modéré", 5, "durée", "squat suivi d'un développé avec haltères, full body intense"),
    ("Swing haltère", "HIIT", "Full body", 2, "Haltères", "Modéré", 4, "durée", "balancé d'haltère pour chaîne postérieure et cardio"),
    ("Étirement ischio-jambiers", "Flexibilité", "Jambes", 1, "Tapis", "Faible", 3, "durée", "étirements doux pour l'arrière des jambes"),
    ("Étirement quadriceps", "Flexibilité", "Jambes", 1, "Aucun", "Faible", 2, "durée", "étirement debout de l'avant des cuisses"),
    ("Chien tête en bas", "Flexibilité", "Full body", 1, "Tapis", "Faible", 3, "durée", "posture de yoga et souplesse générale"),
    ("Posture de l'enfant", "Flexibilité", "Dos/Épaules", 1, "Tapis", "Faible", 3, "durée", "posture de yoga relaxante pour étirer le dos"),
    ("Étirement pectoraux au mur", "Flexibilité", "Pecs/Bras", 1, "Aucun", "Faible", 2, "durée", "ouverture de la poitrine contre un mur"),
    ("Torsion allongée", "Flexibilité", "Core", 1, "Tapis", "Faible", 3, "durée", "rotation douce du dos allongé pour la colonne"),
    ("Pigeon", "Flexibilité", "Jambes", 2, "Tapis", "Faible", 3, "durée", "étirement profond des fessiers et des hanches"),
    ("Rotations d'épaules", "Mobilité", "Dos/Épaules", 1, "Aucun", "Faible", 2, "reps", "cercles des bras pour mobiliser les épaules"),
    ("Chat-vache", "Mobilité", "Core", 1, "Tapis", "Faible", 3, "reps", "mobilité de la colonne à quatre pattes"),
    ("Cercles de hanches", "Mobilité", "Jambes", 1, "Aucun", "Faible", 2, "reps", "rotations des hanches pour articulations souples"),
    ("World's greatest stretch", "Mobilité", "Full body", 2, "Aucun", "Faible", 3, "reps", "fente avec rotation pour hanches, dos et épaules"),
    ("Inchworm", "Mobilité", "Full body", 2, "Aucun", "Faible", 3, "reps", "marche des mains jusqu'à la planche, échauffement complet"),
    ("Farmer walk haltères", "Force", "Full body", 1, "Haltères", "Faible", 4, "durée", "marche en portant des haltères pour grip, dos et gainage"),
    ("Squat et développé haltères", "Force", "Full body", 2, "Haltères", "Modéré", 6, "reps", "squat puis développé épaules avec haltères, full body"),
    ("Bear crawl", "Force", "Full body", 2, "Aucun", "Modéré", 4, "durée", "marche à quatre pattes genoux décollés, épaules et gainage"),
    ("Pompes avec rotation", "Force", "Full body", 3, "Aucun", "Modéré", 5, "reps", "pompe suivie d'une rotation latérale, pecs et obliques"),
    ("Planche toucher d'épaule", "Force", "Full body", 2, "Aucun", "Faible", 4, "durée", "planche avec toucher alterné des épaules, stabilité du tronc"),
    ("Good morning élastique", "Force", "Dos/Épaules", 1, "Élastique", "Faible", 4, "reps", "flexion de hanches avec élastique pour lombaires et ischios"),
    ("Marche en fente haltères", "Force", "Jambes", 3, "Haltères", "Modéré", 6, "reps", "fentes marchées avec haltères pour jambes et fessiers"),
]

DAY_FILTERS = {
    "Cardio & HIIT": ("", lambda d: d["category"].isin(["Cardio", "HIIT"])),
    "Haut du corps": ("", lambda d: d["target_muscle"].isin(["Pecs/Bras", "Dos/Épaules"]) & (d["category"] == "Force")),
    "Bas du corps": ("", lambda d: (d["target_muscle"] == "Jambes") & (d["category"] == "Force")),
    "Core & gainage": ("", lambda d: (d["target_muscle"] == "Core") & d["category"].isin(["Force", "Cardio"])),
    "Full body": ("", lambda d: (d["target_muscle"] == "Full body") & d["category"].isin(["Force", "HIIT", "Cardio"])),
    "Mobilité & souplesse": ("", lambda d: d["category"].isin(["Flexibilité", "Mobilité"])),
}
GOAL_PLANS = {
    "Perte de poids": ["Cardio & HIIT", "Full body", "Core & gainage", "Cardio & HIIT", "Bas du corps", "Mobilité & souplesse"],
    "Prise de muscle": ["Haut du corps", "Bas du corps", "Core & gainage", "Haut du corps", "Bas du corps", "Mobilité & souplesse"],
    "Endurance": ["Cardio & HIIT", "Bas du corps", "Cardio & HIIT", "Haut du corps", "Full body", "Mobilité & souplesse"],
    "Bien-être & souplesse": ["Mobilité & souplesse", "Core & gainage", "Full body", "Mobilité & souplesse", "Cardio & HIIT", "Mobilité & souplesse"],
}

# =============================================================================
# 1. DONNÉES SIMULÉES
# =============================================================================
def build_catalog() -> pd.DataFrame:
    df = pd.DataFrame(EXERCISES, columns=[
        "exercise", "category", "target_muscle", "difficulty", "equipment",
        "impact", "minutes", "kind", "description"])
    df.insert(0, "exercise_id", range(1, len(df) + 1))
    df["difficulty_level"] = df["difficulty"].map(dict(enumerate(LEVELS, start=1)))
    df["text"] = (df["category"] + " " + df["target_muscle"] + " " + df["equipment"] + " "
                  + df["difficulty_level"] + " impact " + df["impact"] + " " + df["description"])
    return df

def build_users(rng: np.random.Generator, n: int = 300) -> pd.DataFrame:
    rows = []
    for uid in range(1, n + 1):
        equip = [e for e, p in zip(EQUIPMENTS, [0.45, 0.35, 0.75, 0.15]) if rng.random() < p]
        rows.append({
            "user_id": uid,
            "age": int(rng.integers(18, 66)),
            "fitness_level": str(rng.choice(LEVELS, p=[0.4, 0.4, 0.2])),
            "goal": str(rng.choice(GOALS, p=[0.35, 0.30, 0.20, 0.15])),
            "equipment": ",".join(equip),
            "joint_pain": int(rng.random() < 0.2),
            "time_available": int(rng.choice([20, 30, 45, 60])),
        })
    return pd.DataFrame(rows)

def make_ctx(level_name, goal, equipment, joint_pain, time, user_idx=None, age=None) -> dict:
    return {"level": LEVEL_NUM[level_name], "level_name": level_name, "goal": goal,
            "equipment": set(equipment), "joint_pain": bool(joint_pain), "time": int(time),
            "user_idx": user_idx, "age": age}

def profile_vector(ctx: dict, df: pd.DataFrame):
    diff = df["difficulty"].values - ctx["level"]
    level_fit = np.select([diff < 0, diff == 0, diff == 1], [0.8, 1.0, 0.4], default=0.0)
    goal_fit = df["category"].map(GOAL_WEIGHTS[ctx["goal"]]).values.astype(float)
    owned = set(ctx["equipment"]) | {"Aucun"}
    feasible = df["equipment"].isin(owned).values.copy()
    if ctx["joint_pain"]:
        feasible &= (df["impact"] == "Faible").values
    return 0.55 * goal_fit + 0.45 * level_fit, feasible

def simulate_ratings(rng, df_ex, df_u, PROF, FEAS):
    n_u, n_i = len(df_u), len(df_ex)
    quality = rng.normal(0, 0.35, n_i)
    popw = rng.gamma(2.0, 1.0, n_i)
    muscles = sorted(df_ex["target_muscle"].unique())
    cats = sorted(df_ex["category"].unique())
    mus_idx = df_ex["target_muscle"].map({m: i for i, m in enumerate(muscles)}).values
    cat_idx = df_ex["category"].map({c: i for i, c in enumerate(cats)}).values
    R = np.zeros((n_u, n_i))
    for u in range(n_u):
        taste_m = rng.normal(0, 0.5, len(muscles))
        taste_c = rng.normal(0, 0.4, len(cats))
        latent = (2.3 + 1.9 * PROF[u] + quality + taste_m[mus_idx] + taste_c[cat_idx]
- 1.3 * (~FEAS[u]) + rng.normal(0, 0.55, n_i))
        p = popw * np.where(FEAS[u], 1.0, 0.08)
        p /= p.sum()
        items = rng.choice(n_i, size=int(rng.integers(10, 27)), replace=False, p=p)
        R[u, items] = np.clip(np.rint(latent[items]), 1, 5)
    return R, (R > 0).astype(float)

# =============================================================================
# 2. MODÈLES DE RECOMMANDATION
# =============================================================================
def mm(A):
    A = np.asarray(A, dtype=float)
    lo, hi = A.min(axis=-1, keepdims=True), A.max(axis=-1, keepdims=True)
    return (A - lo) / (hi - lo + 1e-9)

def als(resid, M, k, lam, iters=12, seed=0):
    n_u, n_i = resid.shape
    rng = np.random.default_rng(seed)
    U, V = rng.normal(0, 0.1, (n_u, k)), rng.normal(0, 0.1, (n_i, k))
    reg = lam * np.eye(k)
    for _ in range(iters):
        for u in range(n_u):
            idx = M[u] > 0
            if idx.any():
                Vi = V[idx]
                U[u] = np.linalg.solve(Vi.T @ Vi + reg, Vi.T @ resid[u, idx])
        for i in range(n_i):
            idx = M[:, i] > 0
            if idx.any():
                Ui = U[idx]
                V[i] = np.linalg.solve(Ui.T @ Ui + reg, Ui.T @ resid[idx, i])
    return U, V

def fit_models(S, M, als_k=8, als_lam=8.0, knn_k=15):
    R = S.R * M
    n_u, n_i = R.shape
    cnt_u, cnt_i = M.sum(1), M.sum(0)
    
    total_m = M.sum()
    mu = R.sum() / total_m if total_m > 0 else 3.0

    pop_vec = (5 * mu + R.sum(0)) / (5 + cnt_i)

    bi = (R - mu * M).sum(0) / (cnt_i + 10)
    bu = (R - mu * M - M * bi).sum(1) / (cnt_u + 10)
    base = mu + bu[:, None] + bi[None, :]
    U, V = als((R - base) * M, M, als_k, als_lam)
    pred_als = np.clip(base + U @ V.T, 1, 5)

    user_mean = np.where(cnt_u > 0, R.sum(1) / np.maximum(cnt_u, 1), mu)
    Rc = (R - user_mean[:, None]) * M
    Rc = np.nan_to_num(Rc, nan=0.0)
    
    Sim = cosine_similarity(Rc.T)
    Sim = np.nan_to_num(Sim, nan=0.0)
    np.fill_diagonal(Sim, 0)
    
    abs_sim = np.abs(Sim)
    k_actual = min(knn_k, abs_sim.shape[1])
    if k_actual > 0:
        thr = -np.sort(-abs_sim, axis=1)[:, k_actual - 1][:, None]
        Sim = np.where(abs_sim >= thr, Sim, 0)
    else:
        Sim = np.zeros_like(Sim)
        
    denom = M @ np.abs(Sim).T + 1.0
    pred_knn = np.clip(user_mean[:, None] + (Rc @ Sim.T) / denom, 1, 5)
    pred_knn = np.nan_to_num(pred_knn, nan=mu)

    W = M * (R >= 4) * (R - 3)
    content = cosine_similarity(W @ S.Xd, S.Xd)
    content = np.nan_to_num(content, nan=0.0)

    return SimpleNamespace(pop=np.tile(pop_vec, (n_u, 1)), als=pred_als, knn=pred_knn, content=content,
                           prof=S.PROF * S.FEAS, feas=S.FEAS, pop_vec=pop_vec)

def hybrid_raw(m, w):
    w = np.array(w, dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.array(DEFAULT_W)
    return w[0] * mm(m.als) + w[1] * mm(m.knn) + w[2] * mm(m.content) + w[3] * mm(m.prof)

def model_matrices(m, w):
    pop_n = mm(m.pop)
    return {
        "Popularité": m.pop,
        "Règles & profil": np.where(m.feas, m.prof + 1e-3 * pop_n, -1e9),
        "Contenu (TF-IDF)": m.content + 1e-3 * pop_n,
        "Item-Item kNN": m.knn,
        "Factorisation (ALS)": m.als,
        "Hybride": np.where(m.feas, hybrid_raw(m, w), -1e9),
    }

# =============================================================================
# 3. ÉVALUATION
# =============================================================================
def rank_metrics(scores, S, exclude, k):
    rel = (S.M_test > 0) & (S.R >= 4)
    users = np.where(rel.any(1))[0]
    sc = np.where(exclude > 0, -1e12, scores)
    top = np.argsort(-sc, axis=1)[:, :k]
    hits = rel[np.arange(len(sc))[:, None], top][users]
    nrel = rel[users].sum(1)
    disc = 1 / np.log2(np.arange(k) + 2)
    idcg = np.array([disc[: min(n, k)].sum() for n in nrel])
    div = []
    for u in users:
        sub = S.C[np.ix_(top[u], top[u])]
        div.append(1 - (sub.sum() - np.trace(sub)) / (k * (k - 1)))
    return {
        "Precision": (hits.sum(1) / k).mean(),
        "Recall": (hits.sum(1) / nrel).mean(),
        "NDCG": ((hits * disc).sum(1) / idcg).mean(),
        "Couverture": len(np.unique(top[users])) / S.n_i,
        "Diversité": float(np.mean(div)),
    }

@st.cache_data(show_spinner="Évaluation des modèles...")
def run_eval(k: int, w: tuple) -> pd.DataFrame:
    S = get_system()
    m = S.train
    rows = {name: rank_metrics(sc, S, S.M_train, k) for name, sc in model_matrices(m, w).items()}
    for name in rows:
        rows[name].update({"RMSE": np.nan, "MAE": np.nan})
    for name, pred in {"Popularité": m.pop, "Item-Item kNN": m.knn, "Factorisation (ALS)": m.als}.items():
        err = (pred - S.R)[S.M_test > 0]
        rows[name]["RMSE"] = float(np.sqrt((err ** 2).mean()))
        rows[name]["MAE"] = float(np.abs(err).mean())
    df = pd.DataFrame(rows).T.loc[MODELS, ["RMSE", "MAE", "Precision", "Recall", "NDCG", "Couverture", "Diversité"]]
    return df

@st.cache_data(show_spinner="Réglage des hyperparamètres ALS...")
def tune_als() -> pd.DataFrame:
    S = get_system()
    ks, lams = [2, 4, 8, 12, 16], [1, 4, 8, 16]
    out = pd.DataFrame(index=[f"k={k}" for k in ks], columns=[f"λ={l}" for l in lams], dtype=float)
    for k in ks:
        for l in lams:
            m = fit_models(S, S.M_train, als_k=k, als_lam=l)
            err = (m.als - S.R)[S.M_test > 0]
            out.loc[f"k={k}", f"λ={l}"] = float(np.sqrt((err ** 2).mean()))
    return out

@st.cache_data(show_spinner="Simulation du démarrage à froid...")
def cold_curve(k: int, w: tuple) -> pd.DataFrame:
    S = get_system()
    rng = np.random.default_rng(7)
    rows = []
    for n in [0, 1, 2, 4, 8]:
        keys = rng.random(S.M_train.shape)
        keys[S.M_train == 0] = 2
        rank = keys.argsort(1).argsort(1)
        Mn = S.M_train * (rank < n)
        m = fit_models(S, Mn)
        mats = model_matrices(m, w)
        for name in ["Popularité", "Règles & profil", "Factorisation (ALS)", "Hybride"]:
            rows.append({"Notes connues": n, "Modèle": name,
                         "NDCG": rank_metrics(mats[name], S, S.M_train, k)["NDCG"]})
    return pd.DataFrame(rows)

# =============================================================================
# 4. SYSTÈME INITIALISATION
# =============================================================================
@st.cache_resource(show_spinner="Initialisation du moteur de recommandation...")
def get_system():
    rng = np.random.default_rng(42)
    df_ex, df_u = build_catalog(), build_users(rng)
    ctxs = [make_ctx(r.fitness_level, r.goal, [e for e in r.equipment.split(",") if e],
                     r.joint_pain, r.time_available, i, r.age) for i, r in enumerate(df_u.itertuples())]
    pf = [profile_vector(c, df_ex) for c in ctxs]
    PROF, FEAS = np.array([p[0] for p in pf]), np.array([p[1] for p in pf])
    R, M = simulate_ratings(rng, df_ex, df_u, PROF, FEAS)

    M_test = np.zeros_like(M)
    for u in range(len(df_u)):
        idx = np.where(M[u] > 0)[0]
        M_test[u, rng.choice(idx, max(1, int(round(0.2 * len(idx)))), replace=False)] = 1
    tfidf = TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 2), sublinear_tf=True)
    Xd = tfidf.fit_transform(df_ex["text"]).toarray()

    S = SimpleNamespace(df_ex=df_ex, df_u=df_u, ctxs=ctxs, PROF=PROF, FEAS=FEAS, R=R, M=M,
                        M_test=M_test, M_train=M - M_test, tfidf=tfidf, Xd=Xd, C=cosine_similarity(Xd),
                        n_u=len(df_u), n_i=len(df_ex))
    S.full = fit_models(S, M)
    S.train = fit_models(S, S.M_train)
    return S

# =============================================================================
# 5. LOGIQUE DE RECOMMANDATION POUR L'INTERFACE
# =============================================================================
def user_scores(ctx: dict, w: tuple):
    S = get_system()
    m = S.full
    prof, feas = profile_vector(ctx, S.df_ex)
    u = ctx["user_idx"]
    pop = m.pop_vec
    rules = mm(prof * feas + 1e-3 * mm(pop))
    if u is None:
        return {"Hybride (démarrage à froid)": 0.65 * mm(prof * feas) + 0.35 * mm(pop),
                "Règles & profil": rules, "Popularité": mm(pop)}, feas, None
    ww = np.array(w, float) if sum(w) > 0 else np.array(DEFAULT_W)
    ww = ww / ww.sum()
    hyb = ww[0] * mm(m.als[u]) + ww[1] * mm(m.knn[u]) + ww[2] * mm(m.content[u]) + ww[3] * mm(prof * feas)
    return {"Hybride": hyb, "Popularité": mm(pop), "Règles & profil": rules,
            "Contenu (TF-IDF)": mm(m.content[u] + 1e-3 * mm(pop)), "Item-Item kNN": mm(m.knn[u]),
            "Factorisation (ALS)": mm(m.als[u])}, feas, m.als[u]

def explain(ctx: dict, i: int, pred=None) -> list[str]:
    S = get_system()
    row, reasons, u = S.df_ex.iloc[i], [], ctx["user_idx"]
    if u is not None:
        liked = np.where((S.R[u] >= 4) & (S.M[u] > 0))[0]
        if len(liked):
            j = liked[np.argmax(S.C[i, liked])]
            if S.C[i, j] >= 0.2:
                reasons.append(f"Proche de « {S.df_ex.iloc[j]['exercise']} »")
        if pred is not None and pred >= 4.0:
            reasons.append(f"Plébiscité par des profils proches ({pred:.1f}/5)")
    gw = GOAL_WEIGHTS[ctx["goal"]].get(row["category"], 0)
    if gw >= 0.9:
        reasons.append(f"Idéal pour « {ctx['goal']} »")
    elif gw >= 0.6:
        reasons.append(f"Utile pour « {ctx['goal']} »")
    d = row["difficulty"] - ctx["level"]
    reasons.append("Niveau adapté" if d == 0 else "Facile" if d < 0 else "Progression")
    reasons.append("Sans matériel" if row["equipment"] == "Aucun" else f"Matériel: {row['equipment']}")
    if ctx["joint_pain"] and row["impact"] == "Faible":
        reasons.append("Articulations ménagées")
    return reasons[:4]

def prescription(ctx: dict, row) -> str:
    lvl, goal = ctx["level"], ctx["goal"]
    sets = {1: 2, 2: 3, 3: 4}[lvl]
    if row["category"] == "Flexibilité":
        return "2 × 30 s"
    if row["category"] == "Mobilité":
        return "2 × 10 reps" if row["kind"] == "reps" else "2 × 30 s"
    if row["category"] == "HIIT":
        return f"{sets} × 30 s (repos 30 s)"
    if row["kind"] == "durée":
        return f"{sets} × {({1: 20, 2: 35, 3: 50})[lvl]} s"
    reps = {"Perte de poids": 15, "Prise de muscle": 10, "Endurance": 18, "Bien-être & souplesse": 12}[goal]
    return f"{sets} × {reps} reps"

def build_program(ctx: dict, score: np.ndarray, n_days: int, minutes: int) -> list[dict]:
    S = get_system()
    df = S.df_ex
    _, feas = profile_vector(ctx, df)
    used: dict[int, int] = {}
    diffs, mins = df["difficulty"].values, df["minutes"].values

    def cands(mask):
        base = mask & feas
        for extra in (0, 1, 3):
            c = np.where(base & (diffs <= ctx["level"] + extra))[0]
            if len(c) >= 3:
                return c
        return np.where(base)[0]

    def pick(c, budget, max_items, min_items=3):
        chosen, t = [], 0
        for i in sorted(c, key=lambda i: -(score[i] - 0.35 * used.get(i, 0))):
            if t + mins[i] > budget and len(chosen) >= min_items:
                continue
            chosen.append(int(i))
            t += int(mins[i])
            if t >= budget or len(chosen) >= max_items:
                break
        return chosen, t

    plan = []
    for slot, dtype in zip(SPREAD[n_days], GOAL_PLANS[ctx["goal"]][:n_days]):
        emoji, fn = DAY_FILTERS[dtype]
        blocks, total = [], 0
        warm, cool, tw, tc = [], [], 0, 0
        if dtype != "Mobilité & souplesse":
            warm, tw = pick(cands((df["category"] == "Mobilité").values), 4, 2, 1)
            cool, tc = pick(cands((df["category"] == "Flexibilité").values), 5, 2, 1)
        main, tm = pick(cands(fn(df).values), max(8, minutes - tw - tc), 7)
        for i in warm + main + cool:
            used[i] = used.get(i, 0) + 1
        for label, ids in (("Échauffement", warm), ("Séance", main), ("Retour au calme", cool)):
            if ids:
                blocks.append((label, ids))
        plan.append({"day": DAY_NAMES[slot], "type": dtype, "emoji": emoji, "blocks": blocks, "minutes": tw + tm + tc})
    return plan

# =============================================================================
# 6. DESIGN : CSS STYLE iOS (GLASSMORPHISM & SF PRO)
# =============================================================================
CSS = """
@import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@400;500;600;700;800&family=SF+Pro+Text:wght@400;500;600&display=swap');

:root {
    --ios-bg: #000000;
    --ios-card: rgba(28, 28, 30, 0.65);
    --ios-card-solid: #1C1C1E;
    --ios-card-hover: rgba(44, 44, 46, 0.75);
    --ios-line: rgba(255, 255, 255, 0.12);
    --ios-txt: #F2F2F7;
    --ios-mut: #8E8E93;
    --ios-blue: #0A84FF;
    --ios-green: #30D158;
    --ios-orange: #FF9F0A;
    --ios-red: #FF453A;
    --ios-purple: #BF5AF2;
    --ios-teal: #64D2FF;
    --ios-indigo: #5E5CE6;
}

html, body, [class*="css"], .stApp {
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'SF Pro Display', 'Helvetica Neue', sans-serif !important;
    background-color: var(--ios-bg) !important;
    color: var(--ios-txt);
    -webkit-font-smoothing: antialiased;
}

h1, h2, h3, h4, .hero h1 {
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif !important;
    letter-spacing: -0.022em !important;
}

.stApp {
    background: radial-gradient(circle at 50% 0%, rgba(10, 132, 255, 0.12), transparent 70%),
                radial-gradient(circle at 85% 30%, rgba(191, 90, 242, 0.08), transparent 60%),
                #000000;
}

header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; max-width: 1200px; }

section[data-testid="stSidebar"] {
    background-color: rgba(18, 18, 18, 0.75) !important;
    backdrop-filter: blur(25px);
    -webkit-backdrop-filter: blur(25px);
    border-right: 1px solid var(--ios-line);
}

.hero {
    position: relative;
    background: linear-gradient(135deg, rgba(30, 30, 35, 0.8), rgba(20, 20, 25, 0.6));
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--ios-line);
    border-radius: 24px;
    padding: 2.2rem 2.5rem;
    margin-bottom: 1.8rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

.hero h1 {
    font-size: 2.6rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
    color: #FFFFFF;
}

.hero p {
    color: var(--ios-mut);
    font-size: 1.05rem;
    line-height: 1.5;
    margin: 0;
    max-width: 780px;
}

.hero .pill {
    display: inline-flex;
    align-items: center;
    margin: 1rem 0.5rem 0 0;
    padding: 0.35rem 0.9rem;
    border-radius: 99px;
    font-size: 0.8rem;
    font-weight: 500;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #FFFFFF;
}

.kpi {
    background: var(--ios-card);
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
    border: 1px solid var(--ios-line);
    border-radius: 18px;
    padding: 1.1rem 1.2rem;
}
.kpi span { color: var(--ios-mut); font-size: 0.82rem; font-weight: 500; }
.kpi b { display: block; font-size: 1.8rem; font-weight: 700; margin: 0.2rem 0; color: #FFF; }
.kpi em { font-style: normal; color: var(--ios-mut); font-size: 0.78rem; }

.sec {
    font-size: 1.3rem;
    font-weight: 700;
    margin: 1.4rem 0 0.6rem 0;
    color: #FFFFFF;
}

.chip {
    display: inline-block;
    padding: 0.25rem 0.7rem;
    border-radius: 8px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 0.15rem 0.3rem 0.15rem 0;
    background: rgba(255, 255, 255, 0.08);
    color: var(--ios-txt);
}

.ex-card {
    display: flex;
    gap: 1.2rem;
    align-items: center;
    background: var(--ios-card);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--ios-line);
    border-radius: 20px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.9rem;
    transition: all 0.2s ease;
}
.ex-card:hover {
    background: var(--ios-card-hover);
    transform: translateY(-1px);
}
.ex-rank {
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--ios-blue);
    min-width: 2.2rem;
}
.ex-body { flex: 1; }
.ex-top { display: flex; justify-content: space-between; gap: 1rem; align-items: flex-start; }
.ex-name { font-size: 1.15rem; font-weight: 600; color: #FFFFFF; }
.ex-side { text-align: right; min-width: 130px; }
.ex-side b { font-size: 1.3rem; font-weight: 700; color: var(--ios-blue); }
.ex-side span { display: block; color: var(--ios-mut); font-size: 0.75rem; }

.bar {
    height: 5px;
    border-radius: 5px;
    background: rgba(255, 255, 255, 0.1);
    margin-top: 0.4rem;
    overflow: hidden;
}
.bar i {
    display: block;
    height: 100%;
    border-radius: 5px;
    background: var(--ios-blue);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
    background: rgba(118, 118, 128, 0.24) !important;
    padding: 0.25rem;
    border-radius: 12px;
    border: none;
    backdrop-filter: blur(10px);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 99px;
    padding: 0.45rem 0.9rem;
    color: #8E8E93 !important;
    font-weight: 500;
    font-size: 0.88rem;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #636366 !important;
    color: #FFFFFF !important;
    font-weight: 600;
    box-shadow: 0 3px 8px rgba(0, 0, 0, 0.12);
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

.stButton button, .stDownloadButton button {
    border-radius: 12px !important;
    border: none !important;
    background: var(--ios-blue) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.2rem !important;
    box-shadow: 0 4px 12px rgba(10, 132, 255, 0.3);
    transition: opacity 0.2s ease;
}
.stButton button:hover, .stDownloadButton button:hover { opacity: 0.88; }

.side-card { background: var(--ios-card-solid); border: 1px solid var(--ios-line); border-radius: 18px; padding: 1rem 1.1rem; margin: .4rem 0 1rem 0; }
.side-card h4 { margin: 0 0 .15rem 0; font-size: 1rem; color: #fff; }
.brand { font-size: 1.35rem; font-weight: 800; color: #fff; margin-bottom: .6rem; }
.fav { display: flex; justify-content: space-between; align-items: center; padding: .5rem .7rem; margin-bottom: .4rem; border-radius: 12px; background: var(--ios-card-solid); border: 1px solid var(--ios-line); font-size: .88rem; }
.day { background: var(--ios-card-solid); border: 1px solid var(--ios-line); border-radius: 20px; padding: 1rem 1.1rem; margin-bottom: 1rem; border-top: 3px solid var(--c); }
.day-h { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: .2rem; }
.day-n { font-weight: 700; font-size: 1.1rem; color: #fff; }
.day-m { color: var(--c); font-weight: 600; font-size: .85rem; }
.day-t { color: var(--ios-mut); font-size: .86rem; margin-bottom: .4rem; }
.blk { font-size: .74rem; color: var(--ios-mut); margin: .7rem 0 .25rem 0; font-weight: 600; }
.row { display: flex; justify-content: space-between; gap: .6rem; padding: .32rem 0; border-bottom: 1px dashed rgba(255,255,255,.07); font-size: .86rem; }
.row em { font-style: normal; color: #C9D2FF; white-space: nowrap; }
.tbl-wrap { overflow-x: auto; border: 1px solid var(--ios-line); border-radius: 18px; background: var(--ios-card-solid); }
.tbl { width: 100%; border-collapse: collapse; font-size: .88rem; }
.tbl th { padding: .8rem .9rem; text-align: right; color: var(--ios-mut); font-weight: 600; border-bottom: 1px solid var(--ios-line); }
.tbl th:first-child, .tbl td:first-child { text-align: left; }
.tbl td { padding: .7rem .9rem; text-align: right; border-bottom: 1px solid rgba(255,255,255,.05); color: #DCE3FF; }
.tbl td.best { color: #000; font-weight: 700; background: var(--ios-green); border-radius: 8px; }
.tbl td.na { color: #5C668C; }
.insight { background: rgba(10, 132, 255, 0.1); border: 1px solid var(--ios-line); border-radius: 16px; padding: .9rem 1.1rem; margin: .8rem 0; color: #DCE3FF; font-size: .93rem; line-height: 1.55; }
"""

def esc(x) -> str:
    return html.escape(str(x))

def chip(text, color=None) -> str:
    if color:
        return f'<span class="chip" style="background:{color}20;color:{color}">{esc(text)}</span>'
    return f'<span class="chip">{esc(text)}</span>'

def kpi(label, value, sub="") -> str:
    return f'<div class="kpi"><span>{esc(label)}</span><b>{esc(value)}</b><em>{esc(sub)}</em></div>'

def sec(title, sub=None):
    st.markdown(f'<div class="sec">{title}</div>' + (f'<div class="sub" style="color:var(--ios-mut);font-size:0.88rem;margin-bottom:0.8rem">{sub}</div>' if sub else ""), unsafe_allow_html=True)

def exercise_card(rank, row, match, pred, reasons) -> str:
    col, emo = CAT_STYLE[row["category"]]
    dots = "●" * row["difficulty"] + "○" * (3 - row["difficulty"])
    tags = (chip(f"{row['category']}", col) + chip(row["target_muscle"]) + chip(f"{dots} {row['difficulty_level']}")
            + chip(f"Impact {row['impact'].lower()}", IMPACT_COLOR[row["impact"]]) + chip(f"{row['minutes']} min"))
    pred_html = f"<span>note {pred:.1f}/5</span>" if pred is not None else ""
    why = "".join(chip(r) for r in reasons)
    return (f'<div class="ex-card"><div class="ex-rank">{rank}</div><div class="ex-body">'
            f'<div class="ex-top"><div><div class="ex-name">{esc(row["exercise"])}</div><div style="margin-top:0.3rem">{tags}</div></div>'
            f'<div class="ex-side"><b>{match:.0f}%</b><span>compatibilité</span>{pred_html}'
            f'<div class="bar"><i style="width:{match:.0f}%"></i></div></div></div>'
            f'<div class="why" style="margin-top:0.6rem">{why}</div></div></div>')

def day_card(ctx, d) -> str:
    S = get_system()
    col = CAT_STYLE["Force" if d["type"] in ("Haut du corps", "Bas du corps", "Core & gainage", "Full body")
                    else "Cardio" if d["type"] == "Cardio & HIIT" else "Flexibilité"][0]
    out = (f'<div class="day" style="--c:{col}"><div class="day-h"><span class="day-n">{d["day"]}</span>'
           f'<span class="day-m">{d["minutes"]} min</span></div><div class="day-t">{d["type"]}</div>')
    for label, ids in d["blocks"]:
        out += f'<div class="blk">{label}</div>'
        for i in ids:
            r = S.df_ex.iloc[i]
            out += f'<div class="row"><span>{esc(r["exercise"])}</span><em>{prescription(ctx, r)}</em></div>'
    return out + "</div>"

def html_table(df: pd.DataFrame) -> str:
    lower = {"RMSE", "MAE"}
    best = {c: (df[c].min() if c in lower else df[c].max()) for c in df.columns if df[c].notna().any()}
    head = "".join(f"<th>{c}</th>" for c in ["Modèle"] + list(df.columns))
    rows = ""
    for name, r in df.iterrows():
        cells = ""
        for c in df.columns:
            v = r[c]
            cells += '<td class="na">—</td>' if pd.isna(v) else f'<td{" class=best" if np.isclose(v, best[c]) else ""}>{v:.3f}</td>'
        rows += f'<tr><td class="mname"><i style="background:{MODEL_COLORS[name]}"></i>{name}</td>{cells}</tr>'
    return f'<div class="tbl-wrap"><table class="tbl"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'

def style_fig(fig, h=340):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=h,
                      margin=dict(l=10, r=10, t=46, b=10), font=dict(family="SF Pro Text", color="#C9D2F0"),
                      title_font=dict(family="SF Pro Display", size=15), legend=dict(orientation="h", y=-0.18, title=None))
    fig.update_xaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False)
    return fig

# =============================================================================
# 7. APPLICATION MAIN
# =============================================================================
S = get_system()
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>FitMatch</h1>'
            '<p>Recommandation intelligente de programmes sportifs sur-mesure selon ton profil, tes objectifs et tes contraintes physiques.</p></div>', unsafe_allow_html=True)

# Barre latérale
with st.sidebar:
    st.markdown('<div class="brand">FitMatch</div>', unsafe_allow_html=True)
    mode = st.radio("Profil", ["Utilisateur existant", "Nouveau profil (démarrage à froid)"], label_visibility="collapsed")
    if mode.startswith("Utilisateur"):
        uid = st.selectbox("Utilisateur", S.df_u["user_id"].tolist(), format_func=lambda x: f"Utilisateur #{x:03d}")
        ctx = S.ctxs[uid - 1]
    else:
        lv = st.selectbox("Niveau", LEVELS)
        gl = st.selectbox("Objectif", GOALS)
        eq = st.multiselect("Matériel disponible", EQUIPMENTS, default=["Tapis"])
        jp = st.toggle("Douleurs articulaires", False)
        tm = st.select_slider("Durée d'une séance (min)", [20, 30, 45, 60], value=30)
        ctx = make_ctx(lv, gl, eq, jp, tm)

    n_rated = int((S.M[ctx["user_idx"]] > 0).sum()) if ctx["user_idx"] is not None else 0
    st.markdown(
        f'<div class="side-card"><h4>{esc(ctx["level_name"])} · {esc(ctx["goal"])}</h4>'
        f'<small>{"Sans matériel" if not ctx["equipment"] else esc(", ".join(sorted(ctx["equipment"])))}</small><br>'
        + chip("Douleurs articulaires" if ctx["joint_pain"] else "Aucune douleur")
        + chip(f"{ctx['time']} min")
        + (chip(f"{n_rated} évaluations") if ctx["user_idx"] is not None else chip("Aucun historique")) + "</div>",
        unsafe_allow_html=True)
    top_n = st.slider("Nombre de recommandations", 3, 10, 5)
    use_filter = st.toggle("Appliquer matériel et articulations", True,
                           help="Écarte les exercices impossibles ou risqués pour ce profil.")
    with st.expander("Réglage du modèle hybride"):
        w_als = st.slider("Factorisation (ALS)", 0.0, 1.0, DEFAULT_W[0], 0.05)
        w_knn = st.slider("Item-Item kNN", 0.0, 1.0, DEFAULT_W[1], 0.05)
        w_con = st.slider("Contenu (TF-IDF)", 0.0, 1.0, DEFAULT_W[2], 0.05)
        w_pro = st.slider("Règles & profil", 0.0, 1.0, DEFAULT_W[3], 0.05)
    W = (w_als, w_knn, w_con, w_pro) if (w_als + w_knn + w_con + w_pro) > 0 else DEFAULT_W

scores, feas, als_row = user_scores(ctx, W)
tabs = st.tabs(["Recommandations", "Programme hebdo", "Données", "Évaluation", "Démarrage à froid", "Méthodologie"])

# Onglet 1 : Recommandations
with tabs[0]:
    c1, c2 = st.columns([2.1, 1], gap="large")
    with c1:
        algo = st.selectbox("Algorithme", list(scores.keys()))
        sc = scores[algo].copy()
        if use_filter:
            sc[~feas] = -1
        if ctx["user_idx"] is not None:
            sc[S.M[ctx["user_idx"]] > 0] = -1
        top = np.argsort(-sc)[:top_n]
        sec(f"Pour toi : {top_n} exercices à essayer",
            "Les exercices déjà notés sont masqués. Chaque carte indique pourquoi l'exercice t'est proposé.")
        for r, i in enumerate(top, 1):
            pred = float(als_row[i]) if als_row is not None else None
            st.markdown(exercise_card(r, S.df_ex.iloc[i], max(0.0, scores[algo][i]) * 100, pred, explain(ctx, i, pred)),
                        unsafe_allow_html=True)
        if ctx["user_idx"] is None:
            st.markdown('<div class="insight"><b>Démarrage à froid :</b> sans historique, le système combine ton profil déclaré '
                        '(65 %) et la popularité (35 %). Dès que tu notes des exercices, la factorisation et le kNN prennent le relais.</div>',
                        unsafe_allow_html=True)
    with c2:
        if ctx["user_idx"] is not None:
            u = ctx["user_idx"]
            sec("Tes exercices préférés")
            fav = np.where((S.R[u] >= 4) & (S.M[u] > 0))[0]
            fav = fav[np.argsort(-S.R[u, fav])][:6]
            for i in fav:
                col = CAT_STYLE[S.df_ex.iloc[i]["category"]][0]
                st.markdown(f'<div class="fav" style="border-left:3px solid {col}"><span>{esc(S.df_ex.iloc[i]["exercise"])}</span>'
                            f'<span style="color:{col}">{"Note: " + str(int(S.R[u, i])) + "/5"}</span></div>', unsafe_allow_html=True)
        sec("Légende")
        st.markdown("".join(chip(f"{c}", col) for c, (col, e) in CAT_STYLE.items()), unsafe_allow_html=True)
        st.markdown('<div class="sub" style="margin-top:.8rem;color:var(--ios-mut);font-size:0.85rem">La barre indique la compatibilité (0-100 %) calculée par l\'algorithme choisi.</div>', unsafe_allow_html=True)

# Onglet 2 : Programme hebdomadaire
with tabs[1]:
    sec("Ton programme de la semaine", "Construit à partir du modèle hybride : un thème par séance, échauffement et retour au calme inclus.")
    p1, p2 = st.columns(2)
    n_days = p1.slider("Séances par semaine", 2, 6, 4)
    minutes = p2.select_slider("Durée d'une séance (min)", [20, 30, 45, 60], value=ctx["time"])
    plan = build_program(ctx, list(scores.values())[0], n_days, minutes)

    tot_min = sum(d["minutes"] for d in plan)
    uniq = len({i for d in plan for _, ids in d["blocks"] for i in ids})
    k1, k2, k3 = st.columns(3)
    k1.markdown(kpi("Séances", n_days, f"{7 - n_days} jours de repos"), unsafe_allow_html=True)
    k2.markdown(kpi("Temps d'entraînement", f"{tot_min} min", "par semaine"), unsafe_allow_html=True)
    k3.markdown(kpi("Exercices différents", uniq, "variété sur la semaine"), unsafe_allow_html=True)
    st.write("")
    cols = st.columns(3)
    for n, d in enumerate(plan):
        cols[n % 3].markdown(day_card(ctx, d), unsafe_allow_html=True)

    cat_min = {}
    for d in plan:
        for _, ids in d["blocks"]:
            for i in ids:
                r = S.df_ex.iloc[i]
                cat_min[r["category"]] = cat_min.get(r["category"], 0) + int(r["minutes"])
    g1, g2 = st.columns([1, 1.3])
    fig = go.Figure(go.Pie(labels=list(cat_min), values=list(cat_min.values()), hole=0.62,
                           marker=dict(colors=[CAT_STYLE[c][0] for c in cat_min]), textinfo="percent"))
    g1.plotly_chart(style_fig(fig.update_layout(title="Répartition du temps par catégorie"), 300), use_container_width=True)
    with g2:
        txt = f"PROGRAMME FITMATCH - {ctx['goal']} ({ctx['level_name']})\n\n"
        for d in plan:
            txt += f"{d['day']} - {d['type']} ({d['minutes']} min)\n"
            for label, ids in d["blocks"]:
                txt += f"  {label} :\n" + "".join(f"    - {S.df_ex.iloc[i]['exercise']} : {prescription(ctx, S.df_ex.iloc[i])}\n" for i in ids)
            txt += "\n"
        st.markdown('<div class="insight">Les prescriptions s\'adaptent au niveau (séries, durée) et à l\'objectif.</div>', unsafe_allow_html=True)
        st.download_button("Télécharger le programme (.txt)", txt, file_name="programme_fitmatch.txt")

# Onglet 3 : Données (EDA)
with tabs[2]:
    sec("Analyse exploratoire", "Données simulées via un modèle latent.")
    n_r = int(S.M.sum())
    sparsity = 100 * (1 - n_r / (S.n_u * S.n_i))
    cols = st.columns(5)
    for c, (a, b, s_) in zip(cols, [("Exercices", S.n_i, "5 catégories"), ("Utilisateurs", S.n_u, "profils variés"),
                                     ("Évaluations", f"{n_r:,}".replace(",", " "), f"{n_r / S.n_u:.1f} par personne"),
                                     ("Sparsité", f"{sparsity:.1f} %", "cases vides"),
                                     ("Note moyenne", f"{S.R[S.M > 0].mean():.2f}", "sur 5")]):
        c.markdown(kpi(a, b, s_), unsafe_allow_html=True)
    st.write("")
    e1, e2 = st.columns(2)
    dist = pd.Series(S.R[S.M > 0]).value_counts().sort_index()
    e1.plotly_chart(style_fig(px.bar(x=dist.index, y=dist.values, labels={"x": "Note", "y": "Nombre"},
                                     title="Distribution des notes", color_discrete_sequence=["#0A84FF"])), use_container_width=True)
    pop = pd.DataFrame({"Exercice": S.df_ex["exercise"], "Catégorie": S.df_ex["category"], "Notes": S.M.sum(0)}).sort_values("Notes", ascending=False)
    e2.plotly_chart(style_fig(px.bar(pop, x="Exercice", y="Notes", color="Catégorie", title="Longue traîne de popularité",
                                     color_discrete_map={c: v[0] for c, v in CAT_STYLE.items()}).update_xaxes(showticklabels=False)),
                    use_container_width=True)
    e3, e4 = st.columns(2)
    order_u, order_i = np.argsort(-S.M.sum(1)), np.argsort(-S.M.sum(0))
    heat = go.Figure(go.Heatmap(z=S.M[np.ix_(order_u, order_i)], colorscale=[[0, "#1C1C1E"], [1, "#0A84FF"]], showscale=False))
    heat.update_layout(title="Matrice utilisateurs × exercices", xaxis_title="Exercices (popularité)", yaxis_title="Utilisateurs (activité)")
    e3.plotly_chart(style_fig(heat), use_container_width=True)
    rc = pd.DataFrame({"Catégorie": np.tile(S.df_ex["category"].values, (S.n_u, 1))[S.M > 0], "Note": S.R[S.M > 0]})
    e4.plotly_chart(style_fig(px.box(rc, x="Catégorie", y="Note", color="Catégorie", title="Notes par catégorie",
                                     color_discrete_map={c: v[0] for c, v in CAT_STYLE.items()}).update_layout(showlegend=False)),
                    use_container_width=True)
    e5, e6 = st.columns(2)
    e5.plotly_chart(style_fig(px.sunburst(S.df_u, path=["goal", "fitness_level"], title="Profils : objectifs et niveaux",
                                          color_discrete_sequence=px.colors.qualitative.Pastel)), use_container_width=True)
    ratings_u = S.M.sum(1)
    e6.plotly_chart(style_fig(px.histogram(x=ratings_u, nbins=17, title="Nombre de notes par utilisateur",
                                           labels={"x": "Notes données"}, color_discrete_sequence=["#FF453A"])), use_container_width=True)
    with st.expander("Catalogue d'exercices"):
        st.dataframe(S.df_ex.drop(columns=["text", "difficulty"]), use_container_width=True, hide_index=True)
    with st.expander("Profils utilisateurs"):
        st.dataframe(S.df_u, use_container_width=True, hide_index=True)

# Onglet 4 : Évaluation
with tabs[3]:
    sec("Comparaison des algorithmes", "Découpage 80 / 20 par utilisateur. Un exercice est « pertinent » s'il a reçu une note ≥ 4 dans le jeu de test.")
    K = st.slider("K (taille de la liste recommandée)", 3, 10, 5, key="K")
    res = run_eval(K, W)
    show = res.rename(columns={"Precision": f"Precision@{K}", "Recall": f"Recall@{K}", "NDCG": f"NDCG@{K}"})
    st.markdown(html_table(show), unsafe_allow_html=True)
    st.caption("En vert : meilleure valeur de la colonne.")
    best_ndcg, best_div = res["NDCG"].idxmax(), res["Diversité"].idxmax()
    gain = res.loc[best_ndcg, "NDCG"] / res.loc["Popularité", "NDCG"] - 1
    st.markdown(f'<div class="insight"><b>{best_ndcg}</b> obtient le meilleur NDCG@{K} ({res.loc[best_ndcg, "NDCG"]:.3f}), '
                f'soit {gain:+.0%} par rapport à la popularité.</div>',
                unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    long = res.reset_index(names="Modèle").melt(id_vars="Modèle", value_vars=["Precision", "Recall", "NDCG"], var_name="Mesure")
    m1.plotly_chart(style_fig(px.bar(long, x="Mesure", y="value", color="Modèle", barmode="group", title=f"Qualité du classement (K = {K})",
                                     color_discrete_map=MODEL_COLORS, labels={"value": "Score"})), use_container_width=True)
    sc_df = res.reset_index(names="Modèle")
    fg = px.scatter(sc_df, x="Diversité", y="NDCG", color="Modèle", size="Couverture", text="Modèle",
                    color_discrete_map=MODEL_COLORS, title="Compromis précision / diversité")
    fg.update_traces(textposition="top center", marker=dict(sizemode="diameter", sizeref=0.03)).update_layout(showlegend=False)
    m2.plotly_chart(style_fig(fg), use_container_width=True)

    sec("Réglage des hyperparamètres", "RMSE de la factorisation ALS sur le jeu de test selon k et λ.")
    tune = tune_als()
    hm = px.imshow(tune.round(3), text_auto=".3f", aspect="auto", color_continuous_scale=["#30D158", "#5E5CE6", "#FF453A"])
    hm.update_layout(coloraxis_showscale=False)
    t1, t2 = st.columns([1.5, 1])
    t1.plotly_chart(style_fig(hm, 300), use_container_width=True)
    bk, bl = np.unravel_index(np.argmin(tune.values), tune.shape)
    t2.markdown(f'<div class="insight">Meilleure configuration : <b>{tune.index[bk]}</b> et <b>{tune.columns[bl]}</b> (RMSE {tune.values[bk, bl]:.3f}).</div>', unsafe_allow_html=True)

# Onglet 5 : Démarrage à froid
with tabs[4]:
    sec("Nouvel utilisateur : combien de notes faut-il ?", "NDCG@K quand on ne connaît que n évaluations par personne.")
    K2 = st.slider("K", 3, 10, 5, key="K2")
    cc = cold_curve(K2, W)
    fc = px.line(cc, x="Notes connues", y="NDCG", color="Modèle", markers=True, color_discrete_map=MODEL_COLORS,
                 title="Qualité selon la taille de l'historique")
    fc.update_traces(line=dict(width=3), marker=dict(size=9))
    st.plotly_chart(style_fig(fc, 380), use_container_width=True)

    sec("Nouvel exercice : où le placer ?", "Un exercice sans aucune note est positionné grâce à sa description (TF-IDF).")
    f1, f2 = st.columns(2)
    nm = f1.text_input("Nom", "Planche avec toucher d'épaule")
    ct = f1.selectbox("Catégorie", list(CAT_STYLE), index=0)
    mu_ = f2.selectbox("Muscle ciblé", sorted(S.df_ex["target_muscle"].unique()), index=1)
    ds = f2.text_area("Description", "gainage dynamique avec toucher alterné des épaules pour stabiliser le tronc")
    v = S.tfidf.transform([f"{ct} {mu_} Aucun {ds}"]).toarray()
    sims = cosine_similarity(v, S.Xd)[0]
    nn = np.argsort(-sims)[:5]
    prior = float((sims[nn] * S.full.pop_vec[nn]).sum() / (sims[nn].sum() + 1e-9))
    st.markdown(f'<div class="insight"><b>{esc(nm)}</b> démarre avec une note estimée de <b>{prior:.2f}/5</b>.</div>', unsafe_allow_html=True)
    for r, i in enumerate(nn, 1):
        st.markdown(exercise_card(r, S.df_ex.iloc[i], float(sims[i]) * 100 / max(sims[nn[0]], 1e-9), None,
                                  [f"Similarité {sims[i]:.2f}", f"Note moyenne {S.full.pop_vec[i]:.2f}/5"]), unsafe_allow_html=True)

# Onglet 6 : Méthodologie
with tabs[5]:
    sec("Méthodologie et limites")
    st.markdown("""
**Données.** 60 exercices, 300 profils et ~5 500 notes générées par un modèle latent.

**Algorithmes.**
- *Popularité* : moyenne bayésienne des notes.
- *Règles & profil* : filtres durs (matériel, douleurs) + score objectif/niveau.
- *Contenu* : TF-IDF sur la description des exercices.
- *Item-Item kNN* : cosinus sur notes centrées (15 voisins).
- *Factorisation (ALS)* : biais utilisateur/exercice + facteurs latents.
- *Hybride* : combinaison pondérée des scores.

**Mesures.** RMSE, MAE, Precision@K, Recall@K, NDCG@K, couverture et diversité.
""")