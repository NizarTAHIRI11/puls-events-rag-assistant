import sys
from datetime import date
from pathlib import Path
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document


# Ajout de la racine du projet au chemin Python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_chain import (
    answer_question,
    convert_metadata_date,
    document_matches_date,
    extract_explicit_date,
    format_documents,
    retrieve_documents,
)


@pytest.fixture
def sample_documents() -> list[Document]:
    """
    Crée des événements fictifs pour tester le système RAG.
    """
    return [
        Document(
            page_content=(
                "Titre : Concert du 20 août\n"
                "Description : Concert de musique à Paris.\n"
                "Ville : Paris\n"
                "Lieu : Accor Arena\n"
                "Adresse : 8 boulevard de Bercy\n"
                "Première date de début : "
                "2026-08-20T18:00:00+02:00\n"
                "Dernière date de fin : "
                "2026-08-20T21:00:00+02:00"
            ),
            metadata={
                "uid": "101",
                "title": "Concert du 20 août",
                "city": "Paris",
                "location_name": "Accor Arena",
                "address": "8 boulevard de Bercy",
                "first_begin": "2026-08-20T18:00:00+02:00",
                "last_end": "2026-08-20T21:00:00+02:00",
            },
        ),
        Document(
            page_content=(
                "Titre : Exposition culturelle\n"
                "Description : Exposition d'art contemporain.\n"
                "Ville : Paris\n"
                "Lieu : Musée de Paris\n"
                "Adresse : 10 rue de Rivoli\n"
                "Première date de début : "
                "2026-09-10T10:00:00+02:00\n"
                "Dernière date de fin : "
                "2026-09-30T18:00:00+02:00"
            ),
            metadata={
                "uid": "102",
                "title": "Exposition culturelle",
                "city": "Paris",
                "location_name": "Musée de Paris",
                "address": "10 rue de Rivoli",
                "first_begin": "2026-09-10T10:00:00+02:00",
                "last_end": "2026-09-30T18:00:00+02:00",
            },
        ),
    ]


def test_format_documents_contains_event_information(
    sample_documents: list[Document],
) -> None:
    """
    Vérifie que le contexte contient les informations des événements.
    """
    context = format_documents(sample_documents)

    assert "Concert du 20 août" in context
    assert "Accor Arena" in context
    assert "Exposition culturelle" in context
    assert "Musée de Paris" in context


def test_format_documents_separates_events(
    sample_documents: list[Document],
) -> None:
    """
    Vérifie que les événements sont séparés dans le contexte.
    """
    context = format_documents(sample_documents)

    assert "ÉVÉNEMENT 1" in context
    assert "ÉVÉNEMENT 2" in context
    assert "---" in context


def test_format_documents_handles_empty_list() -> None:
    """
    Vérifie que l'absence de documents est correctement gérée.
    """
    context = format_documents([])

    assert "Aucun événement" in context


@pytest.mark.parametrize(
    ("question", "expected_date"),
    [
        (
            "Quels événements sont prévus le 20/08/2026 ?",
            date(2026, 8, 20),
        ),
        (
            "Quels événements sont prévus le 20-08-2026 ?",
            date(2026, 8, 20),
        ),
        (
            "Donne-moi les événements du 20 août 2026.",
            date(2026, 8, 20),
        ),
        (
            "Que faire le 5 février 2027 ?",
            date(2027, 2, 5),
        ),
    ],
)
def test_extract_explicit_date(
    question: str,
    expected_date: date,
) -> None:
    """
    Vérifie l'extraction des différents formats de date.
    """
    assert extract_explicit_date(question) == expected_date


def test_extract_explicit_date_returns_none_without_date() -> None:
    """
    Vérifie qu'une question sans date explicite retourne None.
    """
    result = extract_explicit_date(
        "Quels sont les événements culturels à Paris ?"
    )

    assert result is None


def test_extract_explicit_date_rejects_invalid_date() -> None:
    """
    Vérifie qu'une date impossible n'est pas acceptée.
    """
    result = extract_explicit_date(
        "Quels événements sont prévus le 32 août 2026 ?"
    )

    assert result is None


def test_convert_metadata_date() -> None:
    """
    Vérifie la conversion d'une date de métadonnée.
    """
    result = convert_metadata_date(
        "2026-08-20T18:00:00+02:00"
    )

    assert result == date(2026, 8, 20)


def test_convert_metadata_date_handles_missing_value() -> None:
    """
    Vérifie la gestion des dates absentes ou incorrectes.
    """
    assert convert_metadata_date(None) is None
    assert convert_metadata_date("") is None
    assert convert_metadata_date("date incorrecte") is None


def test_document_matches_requested_date(
    sample_documents: list[Document],
) -> None:
    """
    Vérifie qu'un événement est retenu pour sa date exacte.
    """
    result = document_matches_date(
        sample_documents[0],
        date(2026, 8, 20),
    )

    assert result is True


def test_document_does_not_match_another_date(
    sample_documents: list[Document],
) -> None:
    """
    Vérifie qu'un événement est rejeté pour une autre date.
    """
    result = document_matches_date(
        sample_documents[0],
        date(2026, 8, 21),
    )

    assert result is False


def test_document_matches_date_inside_period(
    sample_documents: list[Document],
) -> None:
    """
    Vérifie qu'une date comprise dans la période est acceptée.
    """
    result = document_matches_date(
        sample_documents[1],
        date(2026, 9, 20),
    )

    assert result is True


def test_retrieve_documents_without_explicit_date(
    sample_documents: list[Document],
) -> None:
    """
    Vérifie la recherche sémantique sans filtre de date.
    """
    vectorstore = Mock()

    vectorstore.similarity_search.return_value = sample_documents

    result = retrieve_documents(
        vectorstore=vectorstore,
        question="Je cherche un événement culturel à Paris",
    )

    vectorstore.similarity_search.assert_called_once_with(
        "Je cherche un événement culturel à Paris",
        k=20,
    )

    assert result == sample_documents


def test_retrieve_documents_filters_explicit_date(
    sample_documents: list[Document],
) -> None:
    """
    Vérifie le filtre sur une date explicitement demandée.
    """
    vectorstore = Mock()

    vectorstore.similarity_search.return_value = sample_documents

    result = retrieve_documents(
        vectorstore=vectorstore,
        question="Événement du 20 août 2026",
    )

    assert len(result) == 1
    assert result[0].metadata["uid"] == "101"


def test_retrieve_documents_returns_empty_when_date_not_found(
    sample_documents: list[Document],
) -> None:
    """
    Vérifie qu'aucun événement n'est retourné lorsque la date
    demandée ne correspond à aucun document.
    """
    vectorstore = Mock()

    vectorstore.similarity_search.return_value = sample_documents

    result = retrieve_documents(
        vectorstore=vectorstore,
        question="Événement du 1 janvier 2027",
    )

    assert result == []


def test_answer_question_rejects_empty_question() -> None:
    """
    Vérifie qu'une question vide est refusée.
    """
    result = answer_question(
        question="   ",
        vectorstore=Mock(),
        generation_chain=Mock(),
    )

    assert result == "Veuillez saisir une question."


def test_answer_question_calls_generation_chain(
    sample_documents: list[Document],
) -> None:
    """
    Vérifie que la question et le contexte sont transmis
    à la chaîne de génération.
    """
    vectorstore = Mock()
    vectorstore.similarity_search.return_value = sample_documents

    generation_chain = Mock()
    generation_chain.invoke.return_value = (
        "Voici les événements trouvés."
    )

    result = answer_question(
        question="Quels événements sont disponibles à Paris ?",
        vectorstore=vectorstore,
        generation_chain=generation_chain,
    )

    assert result == "Voici les événements trouvés."
    generation_chain.invoke.assert_called_once()

    invocation_data = generation_chain.invoke.call_args.args[0]

    assert "context" in invocation_data
    assert "question" in invocation_data
    assert "current_date" in invocation_data

    assert invocation_data["question"] == (
        "Quels événements sont disponibles à Paris ?"
    )

    assert "Concert du 20 août" in invocation_data["context"]