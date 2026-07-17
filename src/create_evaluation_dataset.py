from pathlib import Path

import pandas as pd


# Chemins du projet
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "events_paris_clean.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation"
OUTPUT_FILE = OUTPUT_DIR / "test_questions_answers.csv"

# Nombre maximal d'événements utilisés.
# Trois questions seront créées pour chaque événement.
NUMBER_OF_EVENTS = 7


def find_column(dataframe: pd.DataFrame, possible_names: list[str]) -> str | None:
    """
    Retourne la première colonne existante parmi plusieurs noms possibles.

    Args:
        dataframe: DataFrame contenant les événements.
        possible_names: Liste des noms de colonnes recherchés.

    Returns:
        Le nom de la première colonne trouvée, sinon None.
    """
    for column_name in possible_names:
        if column_name in dataframe.columns:
            return column_name

    return None


def clean_value(value: object, default: str = "Non renseigné") -> str:
    """
    Nettoie une valeur provenant du DataFrame.

    Args:
        value: Valeur à nettoyer.
        default: Valeur utilisée lorsque l'information est absente.

    Returns:
        Une chaîne de caractères nettoyée.
    """
    if pd.isna(value):
        return default

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "null"}:
        return default

    return text


def format_date(value: object) -> str:
    """
    Convertit une date en format lisible français.

    Args:
        value: Date provenant du fichier CSV.

    Returns:
        Date au format JJ/MM/AAAA ou « Non renseignée ».
    """
    if pd.isna(value):
        return "Non renseignée"

    parsed_date = pd.to_datetime(value, errors="coerce", utc=True)

    if pd.isna(parsed_date):
        return clean_value(value, "Non renseignée")

    return parsed_date.strftime("%d/%m/%Y")


def create_expected_answer(
    title: str,
    location: str,
    address: str,
    city: str,
    start_date: str,
    end_date: str,
) -> str:
    """
    Construit la réponse humaine de référence pour un événement.
    """
    return (
        f"L'événement « {title} » a lieu à {location}, "
        f"à l'adresse {address}, {city}. "
        f"Il est prévu du {start_date} au {end_date}."
    )


def main() -> None:
    """
    Crée un jeu de données annoté de questions/réponses à partir
    des événements nettoyés de Paris.
    """
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Le fichier source est introuvable : {INPUT_FILE}"
        )

    dataframe = pd.read_csv(
    INPUT_FILE,
    low_memory=False,
)

    if dataframe.empty:
        raise ValueError("Le fichier d'événements nettoyés est vide.")

    # Recherche flexible des colonnes utilisées dans le projet
    uid_column = find_column(dataframe, ["uid", "event_uid"])
    title_column = find_column(
        dataframe,
        ["title.fr", "title", "event_title"],
    )
    location_column = find_column(
        dataframe,
        ["location.name", "location_name", "venue"],
    )
    address_column = find_column(
        dataframe,
        ["location.address", "address"],
    )
    city_column = find_column(
        dataframe,
        ["location.city", "city"],
    )
    start_column = find_column(
        dataframe,
        ["firstTiming.begin", "first_begin", "start_date"],
    )
    end_column = find_column(
        dataframe,
        ["lastTiming.end", "last_end", "end_date"],
    )

    required_columns = {
        "titre": title_column,
        "lieu": location_column,
        "ville": city_column,
        "date de début": start_column,
        "date de fin": end_column,
    }

    missing_columns = [
        label
        for label, column_name in required_columns.items()
        if column_name is None
    ]

    if missing_columns:
        raise ValueError(
            "Colonnes nécessaires introuvables : "
            + ", ".join(missing_columns)
            + f"\nColonnes disponibles : {list(dataframe.columns)}"
        )

    # On conserve uniquement les lignes suffisamment complètes
    selected_dataframe = dataframe.dropna(
        subset=[
            title_column,
            location_column,
            city_column,
            start_column,
            end_column,
        ]
    ).copy()

    selected_dataframe = selected_dataframe.drop_duplicates(
        subset=[title_column]
    )

    if selected_dataframe.empty:
        raise ValueError(
            "Aucun événement suffisamment complet pour créer le jeu de test."
        )

    number_to_select = min(NUMBER_OF_EVENTS, len(selected_dataframe))

    # random_state garantit le même échantillon à chaque exécution
    selected_dataframe = selected_dataframe.sample(
        n=number_to_select,
        random_state=42,
    )

    evaluation_rows: list[dict[str, str]] = []

    for _, event in selected_dataframe.iterrows():
        event_uid = (
            clean_value(event[uid_column])
            if uid_column
            else "Non renseigné"
        )

        title = clean_value(event[title_column])
        location = clean_value(event[location_column])
        address = (
            clean_value(event[address_column])
            if address_column
            else "Non renseignée"
        )
        city = clean_value(event[city_column])
        start_date = format_date(event[start_column])
        end_date = format_date(event[end_column])

        complete_answer = create_expected_answer(
            title=title,
            location=location,
            address=address,
            city=city,
            start_date=start_date,
            end_date=end_date,
        )

        # Question 1 : recherche à partir du titre
        evaluation_rows.append(
            {
                "question": f"Où et quand a lieu l'événement « {title} » ?",
                "expected_answer": complete_answer,
                "expected_event_uid": event_uid,
                "expected_title": title,
                "expected_city": city,
                "category": "titre",
            }
        )

        # Question 2 : recherche à partir du lieu
        evaluation_rows.append(
            {
                "question": (
                    f"Quel événement culturel est organisé à {location} ?"
                ),
                "expected_answer": complete_answer,
                "expected_event_uid": event_uid,
                "expected_title": title,
                "expected_city": city,
                "category": "lieu",
            }
        )

        # Question 3 : recherche à partir de la période
        evaluation_rows.append(
            {
                "question": (
                    f"Quel événement est prévu à {city} "
                    f"entre le {start_date} et le {end_date} ?"
                ),
                "expected_answer": complete_answer,
                "expected_event_uid": event_uid,
                "expected_title": title,
                "expected_city": city,
                "category": "date",
            }
        )

    evaluation_dataframe = pd.DataFrame(evaluation_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    evaluation_dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 65)
    print("JEU DE DONNÉES D'ÉVALUATION CRÉÉ")
    print("=" * 65)
    print(f"Fichier source : {INPUT_FILE}")
    print(f"Fichier généré : {OUTPUT_FILE}")
    print(f"Événements sélectionnés : {number_to_select}")
    print(f"Questions créées : {len(evaluation_dataframe)}")
    print("\nRépartition par catégorie :")
    print(evaluation_dataframe["category"].value_counts().to_string())


if __name__ == "__main__":
    main()