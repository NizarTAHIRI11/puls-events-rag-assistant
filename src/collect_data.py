import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPEN_AGENDA_API")
AGENDAS_URL = "https://api.openagenda.com/v2/agendas"

def get_agendas(search: str) -> pd.DataFrame:
    """
    Récupère tous les agendas correspondant à une recherche.
    """
    all_agendas = []
    after = None
    while True:
        params = {
            "search": search,
            "size": 100,
            "key": API_KEY
        }
        if after is not None:
            params["after"] = after
        response = requests.get(
            AGENDAS_URL,
            params=params,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        all_agendas.extend(data.get("agendas", []))
        after = data.get("after")
        if not after:
            break
    return pd.json_normalize(all_agendas)

def get_events(agenda_uid: int) -> pd.DataFrame:
    """
    Récupère tous les événements d'un agenda OpenAgenda.
    Traitements appliqués :
    - pagination avec `after` ;
    - suppression des doublons par `uid` ;
    - conversion des dates ;
    - conservation des événements terminés depuis moins d'un an
      ainsi que des événements futurs.
    """
    url = f"https://api.openagenda.com/v2/agendas/{agenda_uid}/events"
    all_events = []
    after = None
    while True:
        params = {
            "key": API_KEY,
            "size": 100,
        }
        if after:
            params["after"] = after
        response = requests.get(
            url,
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        all_events.extend(data.get("events", []))
        after = data.get("after")
        if not after:
            break
    df_events = pd.json_normalize(all_events)
    if df_events.empty:
        return df_events
    # Suppression des doublons
    if "uid" in df_events.columns:
        df_events = df_events.drop_duplicates(
            subset="uid",
            keep="first",
        )
    # Conversion des dates
    date_columns = [
        "firstTiming.begin",
        "firstTiming.end",
        "lastTiming.begin",
        "lastTiming.end",
        "nextTiming.begin",
        "nextTiming.end",
    ]
    for column in date_columns:
        if column in df_events.columns:
            df_events[column] = pd.to_datetime(
                df_events[column],
                errors="coerce",
                utc=True,
            )
    # Conservation des événements récents et futurs
    if "lastTiming.end" in df_events.columns:
        limite = (
            pd.Timestamp.now(tz="Europe/Paris")
            - pd.DateOffset(years=1)
        )
        df_events = df_events[
            df_events["lastTiming.end"].notna()
            & (df_events["lastTiming.end"] >= limite)
        ]
    return df_events.reset_index(drop=True)
if __name__ == "__main__":
    if not API_KEY:
        raise ValueError("La clé OPEN_AGENDA_API est absente du fichier .env")
    print("Tu cherches les agendas d'une ville.")
    ville = input("Ville : ").strip()
    df_agendas = get_agendas(ville)
    print("Nombre d'agendas :", len(df_agendas))
    if df_agendas.empty:
        print("Aucun agenda trouvé.")
    else:
        all_events = []
        for uid in df_agendas["uid"]:
            print(f"Récupération des événements de l'agenda {uid}...")
            try:
                df = get_events(uid)
                if not df.empty:
                    all_events.append(df)
            except Exception as e:
                print(f"Erreur pour l'agenda {uid} : {e}")
        if all_events:
            df_events = pd.concat(all_events, ignore_index=True)
            # Suppression des doublons entre agendas
            df_events = df_events.drop_duplicates(subset="uid")
            print("\nNombre total d'événements :", len(df_events))
            date_columns = [
                col
                for col in df_events.columns
                if "Timing." in col
            ]
            for col in date_columns:
                print(f"\n===== {col} =====")
                print("Date minimale :", df_events[col].min())
                print("Date maximale :", df_events[col].max())
        else:
            print("Aucun événement trouvé.")
    print("les evenement sont :",df_events.head())
    print("les evenement sont :",df_events.columns)