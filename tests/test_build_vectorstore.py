import sys
from pathlib import Path

import faiss
import pandas as pd
import pytest
from langchain_core.documents import Document


# Ajout de la racine du projet au chemin Python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.build_vectorstore import (
    DATA_PATH,
    VECTORSTORE_PATH,
    clean_value,
    create_documents,
    load_events,
    split_documents,
)


@pytest.fixture
def sample_events() -> pd.DataFrame:
    """
    Crée des événements fictifs pour tester la création
    des Documents LangChain.
    """
    return pd.DataFrame(
        {
            "uid": [101, 102],
            "title.fr": [
                "Concert du 20 août",
                "Exposition culturelle",
            ],
            "description.fr": [
                "Concert de musique organisé à Paris.",
                "Exposition consacrée à l'art contemporain.",
            ],
            "keywords.fr": [
                '["concert", "musique"]',
                '["exposition", "art"]',
            ],
            "location.city": [
                "Paris",
                "Paris",
            ],
            "location.name": [
                "Accor Arena",
                "Musée de Paris",
            ],
            "location.address": [
                "8 boulevard de Bercy, 75012 Paris",
                "10 rue de Rivoli, 75001 Paris",
            ],
            "location.latitude": [
                48.8386,
                48.8606,
            ],
            "location.longitude": [
                2.3786,
                2.3376,
            ],
            "firstTiming.begin": [
                "2026-08-20T18:00:00+02:00",
                "2026-09-10T10:00:00+02:00",
            ],
            "firstTiming.end": [
                "2026-08-20T21:00:00+02:00",
                "2026-09-10T18:00:00+02:00",
            ],
            "lastTiming.begin": [
                "2026-08-20T18:00:00+02:00",
                "2026-09-30T10:00:00+02:00",
            ],
            "lastTiming.end": [
                "2026-08-20T21:00:00+02:00",
                "2026-09-30T18:00:00+02:00",
            ],
            "status": [
                1,
                1,
            ],
            "originAgenda.uid": [
                5001,
                5002,
            ],
            "originAgenda.title": [
                "Agenda culturel Paris",
                "Musées de Paris",
            ],
        }
    )


def test_load_events_returns_dataframe() -> None:
    """
    Vérifie que le fichier Parquet est chargé sous forme de DataFrame.
    """
    result = load_events(DATA_PATH)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_loaded_events_are_in_paris() -> None:
    """
    Vérifie que les événements nettoyés concernent uniquement Paris.
    """
    df_events = load_events(DATA_PATH)

    cities = (
        df_events["location.city"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    assert (cities == "paris").all()


def test_loaded_events_are_recent_or_future() -> None:
    """
    Vérifie que les événements se sont terminés depuis moins d'un an
    ou qu'ils sont prévus dans le futur.
    """
    df_events = load_events(DATA_PATH)

    last_end = pd.to_datetime(
        df_events["lastTiming.end"],
        errors="coerce",
        utc=True,
    )

    limit_date = (
        pd.Timestamp.now(tz="UTC")
        - pd.DateOffset(years=1)
    )

    assert last_end.notna().all()
    assert (last_end >= limit_date).all()


def test_clean_value_handles_missing_value() -> None:
    """
    Vérifie que les valeurs manquantes deviennent des chaînes vides.
    """
    assert clean_value(None) == ""
    assert clean_value(pd.NA) == ""
    assert clean_value(float("nan")) == ""


def test_create_documents_returns_documents(
    sample_events: pd.DataFrame,
) -> None:
    """
    Vérifie que chaque événement devient un Document LangChain.
    """
    documents = create_documents(sample_events)

    assert len(documents) == len(sample_events)
    assert all(
        isinstance(document, Document)
        for document in documents
    )


def test_document_contains_event_information(
    sample_events: pd.DataFrame,
) -> None:
    """
    Vérifie que le texte contient les informations nécessaires
    pour répondre à une question par date et par lieu.
    """
    documents = create_documents(sample_events)
    content = documents[0].page_content

    assert "Concert du 20 août" in content
    assert "Paris" in content
    assert "Accor Arena" in content
    assert "8 boulevard de Bercy" in content
    assert "2026-08-20" in content


def test_document_metadata_contains_required_fields(
    sample_events: pd.DataFrame,
) -> None:
    """
    Vérifie que les métadonnées contiennent les champs nécessaires
    aux filtres par date et localisation.
    """
    documents = create_documents(sample_events)
    metadata = documents[0].metadata

    required_metadata = {
        "uid",
        "title",
        "description",
        "keywords",
        "city",
        "location_name",
        "address",
        "latitude",
        "longitude",
        "first_begin",
        "first_end",
        "last_begin",
        "last_end",
        "status",
        "agenda_uid",
        "agenda_title",
    }

    assert required_metadata.issubset(metadata.keys())


def test_document_metadata_has_correct_city_and_date(
    sample_events: pd.DataFrame,
) -> None:
    """
    Vérifie que la ville et les dates sont correctement conservées.
    """
    documents = create_documents(sample_events)
    metadata = documents[0].metadata

    assert metadata["city"] == "Paris"
    assert "2026-08-20" in metadata["first_begin"]
    assert "2026-08-20" in metadata["last_end"]


def test_split_documents_returns_chunks(
    sample_events: pd.DataFrame,
) -> None:
    """
    Vérifie que le découpage retourne des Documents LangChain.
    """
    documents = create_documents(sample_events)
    chunks = split_documents(documents)

    assert chunks
    assert all(
        isinstance(chunk, Document)
        for chunk in chunks
    )


def test_chunks_preserve_metadata(
    sample_events: pd.DataFrame,
) -> None:
    """
    Vérifie que les chunks conservent les métadonnées
    de l'événement original.
    """
    documents = create_documents(sample_events)
    chunks = split_documents(documents)

    first_chunk_metadata = chunks[0].metadata

    assert first_chunk_metadata["uid"] == "101"
    assert first_chunk_metadata["city"] == "Paris"
    assert "2026-08-20" in first_chunk_metadata["first_begin"]


def test_saved_faiss_index_exists() -> None:
    """
    Vérifie que les fichiers de la base FAISS ont été sauvegardés.
    """
    vectorstore_path = Path(VECTORSTORE_PATH)

    assert vectorstore_path.exists()
    assert (vectorstore_path / "index.faiss").exists()
    assert (vectorstore_path / "index.pkl").exists()


def test_saved_faiss_index_contains_vectors() -> None:
    """
    Vérifie que l'index FAISS sauvegardé contient des vecteurs.
    """
    index_path = Path(VECTORSTORE_PATH) / "index.faiss"

    if not index_path.exists():
        pytest.fail(
            "Le fichier index.faiss est introuvable. "
            "Exécutez d'abord : py -m src.build_vectorstore"
        )

    index = faiss.read_index(str(index_path))

    assert index.ntotal > 0
    assert index.ntotal == 4402