"""Interface de demonstration (Streamlit) : consomme l'API REST, ne lit jamais la base directement.

Lancement : streamlit run app/streamlit_app.py   (l'API doit tourner sur GOUTTE_API_URL)
"""
import os
from datetime import date, timedelta

import httpx
import pandas as pd
import streamlit as st

API_URL = os.getenv("GOUTTE_API_URL", "http://127.0.0.1:8001")
API_KEY = os.getenv("GOUTTE_API_KEY", "")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}
LEVEL_TEXT = {"faible": "Risque FAIBLE", "modere": "Risque MODÉRÉ", "eleve": "Risque ÉLEVÉ"}
MODE_TEXT = {
    "next_day_forecast": "Prévision à 1 jour (observations les plus récentes)",
    "historical_replay": "Rejeu historique (veille seulement) – pas une prévision",
    "climatology": "Climatologie mensuelle – pas une prévision",
}

st.set_page_config(page_title="Goutte d'Eau – risque de pluie", page_icon="💧", layout="centered")
st.title("Goutte d'Eau – risque de pluie")
st.caption("Démonstrateur MVP – région Bretagne. Estimation de la probabilité d'au moins 1 mm de pluie sur la journée (UTC).")


def api_get(path: str, **params):
    try:
        r = httpx.get(f"{API_URL}{path}", params=params, headers=HEADERS, timeout=10)
        r.raise_for_status()
        return r.json(), None
    except httpx.HTTPStatusError as exc:
        detail = exc.response.json().get("detail", exc.response.text) if exc.response.content else str(exc)
        return None, f"Erreur {exc.response.status_code} : {detail}"
    except httpx.HTTPError:
        return None, f"API injoignable ({API_URL}). Démarrer l'API puis recharger la page."


health, err = api_get("/health")
if err:
    st.error(err)
    st.stop()
if health["data_origin"] == "synthetic_demo":
    st.warning("Les données et le modèle de cette démonstration sont SYNTHÉTIQUES : "
               "les résultats n'ont aucune valeur météorologique réelle.")

stations, err = api_get("/stations")
if err:
    st.error(err)
    st.stop()
names = {f"{s['name']} ({s['station_id']})": s["station_id"] for s in stations}

_first = api_get("/history", station_id=stations[0]["station_id"], days=7)[0]
default_date = (pd.to_datetime(_first["points"][-1]["date"]).date() + timedelta(days=1)) if _first and _first["points"] else date.today()
st.caption(f"Pour tester la prévision à 1 jour, choisir le lendemain de la dernière observation disponible "
           f"(proposé par défaut : {default_date.strftime('%d/%m/%Y')}). Au-delà, l'API renvoie la climatologie.")

with st.form("form_prediction"):
    choice = st.selectbox("Station météorologique", list(names), help="Stations de la région retenue.")
    target = st.date_input("Date à estimer", value=default_date, min_value=date(2000, 1, 1),
                           max_value=date.today() + timedelta(days=366), format="DD/MM/YYYY")
    submitted = st.form_submit_button("Estimer le risque de pluie")

if submitted:
    res, err = api_get("/predict", date=target.isoformat(), station_id=names[choice])
    if err:
        st.error(err)
    else:
        pct = round(res["probability"] * 100)
        st.subheader(f"{LEVEL_TEXT[res['risk_level']]} – {pct} % de probabilité")
        st.progress(res["probability"], text=f"{pct} %")
        st.write(f"**Méthode :** {MODE_TEXT[res['mode']]}.")
        st.write(f"**Seuil de pluie :** {res['rain_threshold_mm']} mm sur la journée.")
        if res.get("rain_observed_mm") is not None:
            st.write(f"**Pluie observée ce jour-là :** {res['rain_observed_mm']:.1f} mm.")
        for w in res["warnings"]:
            st.info(w)
        if res.get("features_used"):
            with st.expander("Variables utilisées (observations de la veille)"):
                st.table(pd.DataFrame(res["features_used"].items(), columns=["Variable", "Valeur"]))

st.divider()
st.subheader("Fiabilité récente du modèle")
hist, err = api_get("/history", station_id=names[choice], days=30)
if err:
    st.error(err)
elif hist["points"]:
    df = pd.DataFrame(hist["points"])
    df["date"] = pd.to_datetime(df["date"])
    st.line_chart(df.set_index("date")[["probability"]].rename(columns={"probability": "Probabilité estimée"}))
    st.caption("Courbe : probabilité estimée la veille pour chaque jour. Le tableau ci-dessous donne la pluie réellement observée.")
    show = df.assign(
        **{"Probabilité (%)": (df["probability"] * 100).round(0).astype(int),
           "Pluie observée (mm)": df["rain_observed_mm"].round(1),
           "Jour pluvieux": df["rain_observed"].map({True: "Oui", False: "Non"})}
    )[["date", "Probabilité (%)", "Pluie observée (mm)", "Jour pluvieux"]]
    show["date"] = show["date"].dt.strftime("%d/%m/%Y")
    st.dataframe(show, hide_index=True, width="stretch")
else:
    st.write("Pas assez d'observations pour afficher l'historique.")
