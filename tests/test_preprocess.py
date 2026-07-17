import json
import sys
from pathlib import Path

import pandas as pd
import pytest


# Ajout de la racine du projet au chemin Python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import clean_events, prepare_for_parquet


@pytest.fixture
def sample_events() -> pd.DataFrame:
    """
    Crée un jeu de données fictif pour tester le preprocessing.
    """
    return pd.DataFrame(
        {
            "uid": [101, 101, 102, 103, 104],
            "title.fr": [
                "Concert à Paris",
                "Concert à Paris",
                "Exposition à Paris",
                "Festival à Amiens",
                "Événement sans ville",
            ],
            "location.city": [
                "Paris",
                "Paris",
                " paris ",
                "Amiens",
                None,
            ],
            "keywords.fr": [
                ["musique", "concert"],
                ["musique", "concert"],
                ["art", "exposition"],
                ["festival"],
                None,
            ],
            "image.variants": [
                {"small": "image1.jpg"},
                {"small": "image1.jpg"},
                {"small": "image2.jpg"},
                None,
                None,
            ],
        }
    )


def test_clean_events_returns_dataframe(
    sample_events: pd.DataFrame,
) -> None:
    """
    Vérifie que clean_events retourne un DataFrame.
    """
    result = clean_events(sample_events, ville="Paris")

    assert isinstance(result, pd.DataFrame)


def test_clean_events_removes_duplicates(
    sample_events: pd.DataFrame,
) -> None:
    """
    Vérifie que les doublons sont supprimés selon uid.
    """
    result = clean_events(sample_events, ville="Paris")

    assert result["uid"].duplicated().sum() == 0


def test_clean_events_filters_selected_city(
    sample_events: pd.DataFrame,
) -> None:
    """
    Vérifie que seuls les événements de Paris sont conservés.
    """
    result = clean_events(sample_events, ville="Paris")

    cities = (
        result["location.city"]
        .fillna("")
        .str.strip()
        .str.casefold()
    )

    assert (cities == "paris").all()


def test_clean_events_ignores_case_and_spaces(
    sample_events: pd.DataFrame,
) -> None:
    """
    Vérifie que le filtre accepte les différences de casse
    et les espaces inutiles.
    """
    result = clean_events(sample_events, ville="PARIS")

    assert 102 in result["uid"].values


def test_clean_events_expected_count(
    sample_events: pd.DataFrame,
) -> None:
    """
    Vérifie le nombre d'événements après nettoyage.
    """
    result = clean_events(sample_events, ville="Paris")

    assert len(result) == 2


def test_clean_events_empty_dataframe() -> None:
    """
    Vérifie que la fonction gère un DataFrame vide.
    """
    empty_df = pd.DataFrame()

    result = clean_events(empty_df, ville="Paris")

    assert result.empty


def test_prepare_for_parquet_converts_lists(
    sample_events: pd.DataFrame,
) -> None:
    """
    Vérifie que les listes sont converties en chaînes JSON.
    """
    result = prepare_for_parquet(sample_events)

    value = result.loc[0, "keywords.fr"]

    assert isinstance(value, str)
    assert json.loads(value) == ["musique", "concert"]


def test_prepare_for_parquet_converts_dictionaries(
    sample_events: pd.DataFrame,
) -> None:
    """
    Vérifie que les dictionnaires sont convertis en chaînes JSON.
    """
    result = prepare_for_parquet(sample_events)

    value = result.loc[0, "image.variants"]

    assert isinstance(value, str)
    assert json.loads(value) == {"small": "image1.jpg"}


def test_prepare_for_parquet_keeps_simple_values(
    sample_events: pd.DataFrame,
) -> None:
    """
    Vérifie que les valeurs simples ne sont pas modifiées.
    """
    result = prepare_for_parquet(sample_events)

    assert result.loc[0, "uid"] == 101
    assert result.loc[0, "title.fr"] == "Concert à Paris"


def test_original_dataframe_is_not_modified(
    sample_events: pd.DataFrame,
) -> None:
    """
    Vérifie que le DataFrame original reste inchangé.
    """
    original_df = sample_events.copy(deep=True)

    clean_events(sample_events, ville="Paris")
    prepare_for_parquet(sample_events)

    pd.testing.assert_frame_equal(sample_events, original_df)