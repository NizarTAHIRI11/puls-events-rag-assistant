import pandas as pd
import json
from src.collect_data import get_agendas, get_events



def collect_all_events(search: str) -> pd.DataFrame:
    """
    Récupère les événements de tous les agendas
    correspondant à la recherche.
    """

    df_agendas = get_agendas(search)
    all_events = []

    print(f"Nombre d'agendas trouvés : {len(df_agendas)}")

    for position, agenda_uid in enumerate(df_agendas["uid"], start=1):
        print(
            f"Récupération de l'agenda "
            f"{position}/{len(df_agendas)} : {agenda_uid}"
        )

        try:
            df_agenda_events = get_events(agenda_uid)

            if not df_agenda_events.empty:
                df_agenda_events["agenda_uid"] = agenda_uid
                all_events.append(df_agenda_events)

        except Exception as error:
            print(f"Erreur pour l'agenda {agenda_uid} : {error}")

    if not all_events:
        return pd.DataFrame()

    return pd.concat(all_events, ignore_index=True)


def clean_events(df: pd.DataFrame, ville: str) -> pd.DataFrame:
    """
    Nettoie les événements :
    - suppression des doublons ;
    - filtre géographique.
    """
    if df.empty:
        return df
    df = df.copy()
    # Suppression des doublons selon l'identifiant de l'événement
    if "uid" in df.columns:
        df = df.drop_duplicates(subset="uid")
    # Conserver uniquement les événements dont la ville correspond
    # à la ville recherchée.
    if "location.city" in df.columns:
        df = df[
            df["location.city"]
            .fillna("")
            .str.strip()
            .str.casefold()
            == ville.strip().casefold()
        ]
    return df.reset_index(drop=True)

def prepare_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    def normalize_value(value):
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False)
        if pd.isna(value):
            return None
        return str(value)
    for column in df.select_dtypes(include=["object", "string"]).columns:
        df[column] = df[column].apply(normalize_value)
        df[column] = df[column].astype("string")
    return df
if __name__ == "__main__":
    ville = "paris"
    # 1. Collecte
    df_events_raw = collect_all_events(ville)
    print("Événements bruts :", len(df_events_raw))
    # 2. Nettoyage
    df_events_clean = clean_events(
        df=df_events_raw,
        ville=ville,
    )
    print("Événements après nettoyage :", len(df_events_clean))
    ville_fichier = ville.lower().replace(" ", "_")
    # Préparation pour Parquet
    df_events_raw_parquet = prepare_for_parquet(df_events_raw)
    df_events_clean_parquet = prepare_for_parquet(df_events_clean)
    # 3. Sauvegarde
    df_events_raw.to_csv(
        f"data/raw/events_{ville_fichier}_raw.csv",
        index=False,
    )
    df_events_raw_parquet.to_parquet(
        f"data/raw/events_{ville_fichier}_raw.parquet",
        index=False,
    )
    df_events_clean.to_csv(
        f"data/processed/events_{ville_fichier}_clean.csv",
        index=False,
    )
    df_events_clean_parquet.to_parquet(
        f"data/processed/events_{ville_fichier}_clean.parquet",
        index=False,
    )