import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.collect_data import get_agendas, get_events


load_dotenv()

API_KEY = os.getenv("OPEN_AGENDA_API")
SEARCH_CITY = "Amiens"


@pytest.fixture(scope="module")
def agendas_df() -> pd.DataFrame:
    """
    Récupère les agendas correspondant à la recherche Amiens.
    Le résultat est réutilisé par les différents tests.
    """
    if not API_KEY:
        pytest.skip("La clé OPEN_AGENDA_API est absente du fichier .env")

    return get_agendas(SEARCH_CITY)


@pytest.fixture(scope="module")
def events_df(agendas_df: pd.DataFrame) -> pd.DataFrame:
    """
    Récupère les événements du premier agenda contenant des événements.
    """
    if agendas_df.empty:
        pytest.fail("Aucun agenda trouvé pour la recherche Amiens.")

    for agenda_uid in agendas_df["uid"]:
        df = get_events(int(agenda_uid))

        if not df.empty:
            return df

    pytest.fail("Aucun événement trouvé dans les agendas récupérés.")


def test_agendas_not_empty(agendas_df: pd.DataFrame) -> None:
    """
    Vérifie que l'API retourne au moins un agenda.
    """
    assert not agendas_df.empty, (
        "L'API OpenAgenda n'a retourné aucun agenda."
    )


def test_agendas_have_uid_column(agendas_df: pd.DataFrame) -> None:
    """
    Vérifie que les agendas possèdent une colonne uid.
    """
    assert "uid" in agendas_df.columns, (
        "La colonne uid est absente des agendas."
    )


def test_events_not_empty(events_df: pd.DataFrame) -> None:
    """
    Vérifie que l'API retourne au moins un événement.
    """
    assert not events_df.empty, (
        "L'API OpenAgenda n'a retourné aucun événement."
    )


def test_events_have_uid_column(events_df: pd.DataFrame) -> None:
    """
    Vérifie que les événements possèdent une colonne uid.
    """
    assert "uid" in events_df.columns, (
        "La colonne uid est absente des événements."
    )


def test_events_have_no_duplicate_uid(events_df: pd.DataFrame) -> None:
    """
    Vérifie que get_events supprime les doublons selon uid.
    """
    duplicate_count = events_df["uid"].duplicated().sum()
    assert duplicate_count == 0, (
        f"{duplicate_count} événement(s) dupliqué(s) détecté(s)."
    )