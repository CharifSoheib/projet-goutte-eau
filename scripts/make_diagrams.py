"""Genere les diagrammes du rapport (architecture, composants, base de donnees, Gantt) avec matplotlib."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, TEAL, SAND, GREY, RED = "#D6E6F5", "#D4EFE7", "#FBEBD0", "#ECECEC", "#F8D9D6"


def box(ax, x, y, w, h, text, color, fs=8.5, bold=False, ls="-", ec="#33475B"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec=ec, lw=1.2, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, p1, p2, label="", rad=0.0, fs=7, off=(0, 0.12)):
    ax.annotate("", xy=p2, xytext=p1, arrowprops=dict(arrowstyle="-|>", color="#33475B", lw=1.2, connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((p1[0] + p2[0]) / 2 + off[0], (p1[1] + p2[1]) / 2 + off[1], label, ha="center", fontsize=fs, style="italic",
                bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.85))


def poly(ax, pts, label="", lpos=None, fs=7):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color="#33475B", lw=1.2)
    ax.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle="-|>", color="#33475B", lw=1.2))
    if label:
        ax.text(*lpos, label, ha="center", va="center", fontsize=fs, style="italic", bbox=dict(fc="white", ec="none", pad=0.5))


def canvas(w, h, title):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    return fig, ax


def architecture():
    fig, ax = canvas(12, 6.6, "Architecture générale du MVP Goutte d'Eau")
    for x, lab in [(0.2, "Sources"), (2.6, "Collecte"), (5.0, "Stockage"), (7.2, "Modèle ML"), (9.5, "Service & clients")]:
        ax.text(x + 1.1, 6.0, lab, ha="center", fontsize=10, fontweight="bold", color="#33475B")
    box(ax, 0.2, 4.2, 2.1, 1.2, "API Infoclimat\n(StatIC + SYNOP)\ntoken personnel", SAND)
    box(ax, 0.2, 2.3, 2.1, 1.2, "Capteurs IoT\n(phase ultérieure)", GREY, ls="--")
    box(ax, 0.2, 0.4, 2.1, 1.2, "Jeu de démo\nSYNTHÉTIQUE\n(hors ligne)", RED)
    box(ax, 2.6, 3.0, 2.1, 2.0, "Collecteur\n(retry, timeout,\ndécoupage 30 j)\n+ Validation\n(plages, dates, doublons)", BLUE)
    box(ax, 5.0, 3.0, 2.0, 2.0, "Base SQLite\nstations\nobservations\ncollection_runs\nmodel_runs\nprediction_log", TEAL)
    box(ax, 7.2, 3.6, 2.0, 1.4, "Préparation\n(agrégation jour,\nvariables, cible J+1)", BLUE)
    box(ax, 7.2, 1.6, 2.0, 1.5, "Entraînement\n+ évaluation\n(split temporel)", BLUE)
    box(ax, 7.2, 0.2, 2.0, 1.0, "Modèle .joblib\n+ metrics.json", TEAL)
    box(ax, 9.7, 3.0, 2.1, 2.0, "API REST FastAPI\n/predict /history\n/stations /model\n/health\n(clé API optionnelle)", BLUE, bold=True)
    box(ax, 9.7, 0.9, 2.1, 1.3, "Interface Streamlit\n(MVP de test)", SAND)
    box(ax, 9.7, 5.2, 2.1, 0.6, "Application cible (phase 2)", GREY, ls="--", fs=8)
    arrow(ax, (2.3, 4.8), (2.6, 4.4), "HTTPS/JSON", off=(0, 0.2))
    arrow(ax, (2.3, 1.0), (2.6, 3.3), "", rad=-0.2)
    arrow(ax, (2.3, 2.9), (2.6, 3.5), "", rad=0.1)
    arrow(ax, (4.7, 4.0), (5.0, 4.0))
    arrow(ax, (7.0, 4.3), (7.2, 4.3))
    arrow(ax, (8.2, 3.6), (8.2, 3.1))
    arrow(ax, (8.2, 1.6), (8.2, 1.2))
    poly(ax, [(9.2, 0.7), (9.58, 0.7), (9.58, 3.6), (9.7, 3.6)])
    ax.text(9.58, 0.25, "charge\nle modèle", ha="center", fontsize=7, style="italic")
    poly(ax, [(6.0, 3.0), (6.0, 1.4), (9.42, 1.4), (9.42, 3.3), (9.7, 3.3)], "lit les observations", (7.7, 1.4))
    arrow(ax, (10.75, 2.2), (10.75, 3.0), "HTTP/JSON")
    arrow(ax, (10.75, 5.2), (10.75, 5.0), "")
    fig.tight_layout(); fig.savefig(OUT / "architecture_generale.png", dpi=160); plt.close(fig)


def components():
    fig, ax = canvas(12, 6.8, "Diagramme de composants (modules Python et interfaces)")
    box(ax, 0.2, 5.3, 2.6, 1.0, "data_collection.py\nInfoclimatClient\ncollect_real / collect_demo", BLUE)
    box(ax, 0.2, 3.7, 2.6, 1.0, "synthetic_data.py\n(mode démo)", RED)
    box(ax, 3.6, 5.3, 2.6, 1.0, "data_validation.py\nvalidate_observations()", BLUE)
    box(ax, 3.6, 3.5, 2.6, 1.3, "database.py\nDépôt SQLite\ninsert_observations()\nload_observations()", TEAL)
    box(ax, 7.0, 5.3, 2.6, 1.0, "preprocessing.py\nhourly_to_daily()\nbuild_features()", BLUE)
    box(ax, 7.0, 3.7, 2.6, 1.0, "train_model.py\ntrain() / evaluate()", BLUE)
    box(ax, 10.0, 1.9, 1.9, 1.1, "models/\nrain_model.joblib\nmetrics.json", TEAL)
    box(ax, 0.2, 1.6, 2.6, 1.3, "streamlit_app.py\nInterface de test\n(client HTTP)", SAND)
    box(ax, 3.6, 1.6, 2.6, 1.3, "api.py (FastAPI)\nauth_dep, conn_dep\n/predict /history ...", BLUE, bold=True)
    box(ax, 7.0, 1.6, 2.6, 1.3, "predict.py\nload_bundle()\npredict_risk()\nreplay_history()", BLUE, bold=True)
    box(ax, 3.6, 0.1, 2.6, 1.0, "schemas.py\n(Pydantic)", GREY)
    box(ax, 7.0, 0.1, 2.6, 1.0, "config.py + region.json\n.env (variables)", GREY)
    arrow(ax, (1.5, 4.7), (1.5, 5.3), "DataFrame", off=(0.6, 0))
    arrow(ax, (2.8, 5.8), (3.6, 5.8), "DataFrame")
    arrow(ax, (4.9, 5.3), (4.9, 4.8), "lignes valides", off=(0.8, 0))
    arrow(ax, (6.2, 4.5), (7.0, 5.4), "SQL", off=(-0.25, 0.05))
    arrow(ax, (8.3, 5.3), (8.3, 4.7), "features", off=(0.5, 0))
    arrow(ax, (9.6, 4.0), (10.9, 3.0), "joblib.dump", rad=-0.2, off=(0.55, 0.15))
    arrow(ax, (6.2, 3.7), (7.6, 2.9), "SQL (lecture)", off=(-0.35, 0.25))
    arrow(ax, (2.8, 2.25), (3.6, 2.25), "HTTP", off=(0, 0.17))
    arrow(ax, (6.2, 2.25), (7.0, 2.25), "appelle", off=(0, 0.17))
    arrow(ax, (10.0, 2.45), (9.6, 2.45), "joblib.load", off=(0, -0.3))
    arrow(ax, (4.9, 1.6), (4.9, 1.1))
    arrow(ax, (8.3, 1.6), (8.3, 1.1))
    fig.tight_layout(); fig.savefig(OUT / "diagramme_composants.png", dpi=160); plt.close(fig)


def erd():
    fig, ax = canvas(12, 6.6, "Modèle de données (SQLite)")
    tables = {
        "stations": (0.3, 3.6, ["PK station_id TEXT", "name TEXT NOT NULL", "region TEXT NOT NULL", "latitude REAL", "longitude REAL", "elevation_m REAL"]),
        "observations": (4.1, 2.9, ["PK station_id TEXT  (FK)", "PK obs_time_utc TEXT", "temperature_c REAL", "pressure_hpa REAL", "humidity_pct REAL", "wind_mean_kmh REAL", "wind_gust_kmh REAL", "rain_1h_mm REAL (>=0)", "data_origin TEXT (CHECK)", "ingested_at TEXT"]),
        "collection_runs": (8.4, 3.8, ["PK run_id INTEGER", "started_at / finished_at", "data_origin TEXT", "period_start / period_end", "rows_received / rows_valid", "rows_inserted INTEGER", "status / message"]),
        "model_runs": (0.3, 0.3, ["PK model_id TEXT", "trained_at TEXT", "data_origin TEXT", "algorithm TEXT", "metrics_json TEXT", "model_path TEXT"]),
        "prediction_log": (4.4, 0.2, ["PK id INTEGER", "requested_at TEXT", "station_id TEXT", "target_date TEXT", "probability REAL", "mode TEXT", "model_id TEXT"]),
    }
    for name, (x, y, cols) in tables.items():
        h = 0.34 * (len(cols) + 1)
        ax.add_patch(FancyBboxPatch((x, y), 3.4, h, boxstyle="round,pad=0.01,rounding_size=0.05", fc="white", ec="#33475B", lw=1.2))
        ax.add_patch(FancyBboxPatch((x, y + h - 0.34), 3.4, 0.34, boxstyle="round,pad=0.01,rounding_size=0.05", fc=TEAL, ec="#33475B", lw=1.2))
        ax.text(x + 1.7, y + h - 0.17, name, ha="center", va="center", fontweight="bold", fontsize=9)
        for i, c in enumerate(cols):
            ax.text(x + 0.12, y + h - 0.34 - 0.34 * (i + 0.55), c, fontsize=7.5, va="center")
    arrow(ax, (3.7, 5.0), (4.1, 5.0), "1 — N", off=(0, 0.15))
    ax.text(8.4, 2.9, "Index : idx_obs_time(obs_time_utc)\n           idx_obs_origin(data_origin)\nClé primaire (station, heure) = anti-doublon", fontsize=8, va="top",
            bbox=dict(fc=SAND, ec="#33475B"))
    fig.tight_layout(); fig.savefig(OUT / "modele_donnees.png", dpi=160); plt.close(fig)


GANTT = [
    ("Phase 0 – Cadrage", None, None, True),
    ("Cahier des charges complémentaire, périmètre", "2026-11-02", "2026-11-13", False),
    ("Méthode, outils collaboratifs, budget", "2026-11-02", "2026-11-13", False),
    ("Phase 1 – Conception", None, None, True),
    ("Architecture, composants, sécurité", "2026-11-09", "2026-11-27", False),
    ("Conception base de données", "2026-11-16", "2026-11-27", False),
    ("Phase 2 – Données", None, None, True),
    ("Accès API (token) et collecte historique", "2026-11-23", "2026-12-04", False),
    ("Validation, nettoyage, préparation", "2026-11-30", "2026-12-11", False),
    ("Phase 3 – IA et services", None, None, True),
    ("Modélisation et évaluation", "2026-12-07", "2027-01-08", False),
    ("API REST", "2026-12-14", "2027-01-15", False),
    ("Interface graphique de test", "2027-01-04", "2027-01-22", False),
    ("Phase 4 – Qualité et mise en service", None, None, True),
    ("Tests, qualité, sécurité", "2027-01-11", "2027-01-29", False),
    ("Documentation et rapport", "2027-01-04", "2027-02-05", False),
    ("Recette, pilote agriculteurs, mise en service", "2027-02-01", "2027-02-12", False),
]
MILESTONES = [("J1 Cadrage validé", "2026-11-13"), ("J2 Architecture validée", "2026-11-27"), ("J3 Données en base", "2026-12-04"),
              ("J4 Modèle validé", "2027-01-08"), ("J5 MVP complet", "2027-01-29"), ("J6 Mise en service pilote", "2027-02-12")]


def gantt():
    fig, ax = plt.subplots(figsize=(12, 6.4))
    n = len(GANTT)
    for i, (name, s, e, ph) in enumerate(GANTT):
        y = n - i
        if ph:
            ax.text(pd.Timestamp("2026-10-31"), y, name, fontweight="bold", fontsize=8.5, ha="right", va="center")
            ax.axhspan(y - 0.5, y + 0.5, color="#EEF3F8", zorder=0)
        else:
            s_, e_ = pd.Timestamp(s), pd.Timestamp(e) + pd.Timedelta(days=1)
            ax.barh(y, (e_ - s_).days, left=mdates.date2num(s_), height=0.55, color="#4A90C2", ec="#2B5F86")
            ax.text(pd.Timestamp("2026-10-31"), y, name, fontsize=8, ha="right", va="center")
    for lab, d in MILESTONES:
        x = mdates.date2num(pd.Timestamp(d) + pd.Timedelta(days=1))
        ax.axvline(x, color="#C0392B", ls=":", lw=1)
        ax.plot(x, n + 1.1, marker="D", color="#C0392B")
        ax.text(x, n + 1.6, lab.split(" ", 1)[0], ha="center", fontsize=8, color="#C0392B", fontweight="bold")
    ax.set_ylim(0.3, n + 2); ax.set_yticks([])
    ax.set_xlim(mdates.date2num(pd.Timestamp("2026-10-31")), mdates.date2num(pd.Timestamp("2027-02-16")))
    ax.xaxis_date(); ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m")); ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", alpha=0.3); ax.set_title("Diagramme de Gantt – 15 semaines (2 nov. 2026 – 12 fév. 2027)", fontweight="bold")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.subplots_adjust(left=0.30, right=0.98, top=0.9, bottom=0.08)
    fig.savefig(OUT / "gantt.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    architecture(); components(); erd(); gantt()
    print("diagrammes generes dans", OUT)
