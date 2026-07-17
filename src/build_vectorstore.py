import os
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

DATA_PATH = "data/processed/events_paris_clean.parquet"
VECTORSTORE_PATH = "vectorstore/paris_faiss"

REQUIRED_COLUMNS = [
    "uid",
    "title.fr",
    "description.fr",
    "location.city",
    "firstTiming.begin",
    "lastTiming.end",
]


def load_events(file_path: str) -> pd.DataFrame:
    """
    Charge les événements nettoyés depuis un fichier Parquet.

    Args:
        file_path: Chemin du fichier Parquet.

    Returns:
        DataFrame contenant les événements.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        ValueError: Si le fichier est vide ou incomplet.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Le fichier de données est introuvable : {file_path}"
        )

    df_events = pd.read_parquet(file_path)

    if df_events.empty:
        raise ValueError(
            "Le fichier ne contient aucun événement."
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df_events.columns
    ]

    if missing_columns:
        raise ValueError(
            "Colonnes obligatoires absentes : "
            + ", ".join(missing_columns)
        )

    return df_events


def clean_value(value: Any) -> str:
    """
    Convertit une valeur en chaîne compatible avec les documents
    et les métadonnées LangChain.

    Args:
        value: Valeur à nettoyer.

    Returns:
        Chaîne vide si la valeur est manquante,
        sinon représentation textuelle de la valeur.
    """
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def create_documents(df_events: pd.DataFrame) -> list[Document]:
    """
    Transforme chaque événement en Document LangChain.

    Le contenu textuel contient les informations utiles pour la
    recherche sémantique. Les métadonnées conservent les informations
    structurées nécessaires aux filtres par date et localisation.

    Args:
        df_events: DataFrame contenant les événements nettoyés.

    Returns:
        Liste de Documents LangChain.
    """
    documents = []

    for _, row in df_events.iterrows():
        uid = clean_value(row.get("uid"))

        title = clean_value(
            row.get("title.fr")
        )

        description = clean_value(
            row.get("description.fr")
        )

        city = clean_value(
            row.get("location.city")
        )

        location_name = clean_value(
            row.get("location.name")
        )

        address = clean_value(
            row.get("location.address")
        )

        latitude = clean_value(
            row.get("location.latitude")
        )

        longitude = clean_value(
            row.get("location.longitude")
        )

        first_begin = clean_value(
            row.get("firstTiming.begin")
        )

        first_end = clean_value(
            row.get("firstTiming.end")
        )

        last_begin = clean_value(
            row.get("lastTiming.begin")
        )

        last_end = clean_value(
            row.get("lastTiming.end")
        )

        keywords = clean_value(
            row.get("keywords.fr")
        )

        event_status = clean_value(
            row.get("status")
        )

        agenda_uid = clean_value(
            row.get("originAgenda.uid")
        )

        agenda_title = clean_value(
            row.get("originAgenda.title")
        )

        content = f"""
Titre : {title}
Description : {description}
Mots-clés : {keywords}

Ville : {city}
Lieu : {location_name}
Adresse : {address}
Latitude : {latitude}
Longitude : {longitude}

Première date de début : {first_begin}
Première date de fin : {first_end}
Dernière date de début : {last_begin}
Dernière date de fin : {last_end}

Statut : {event_status}
Agenda d'origine : {agenda_title}
""".strip()

        metadata = {
            "uid": uid,
            "title": title,
            "description": description,
            "keywords": keywords,
            "city": city,
            "location_name": location_name,
            "address": address,
            "latitude": latitude,
            "longitude": longitude,
            "first_begin": first_begin,
            "first_end": first_end,
            "last_begin": last_begin,
            "last_end": last_end,
            "status": event_status,
            "agenda_uid": agenda_uid,
            "agenda_title": agenda_title,
        }

        documents.append(
            Document(
                page_content=content,
                metadata=metadata,
            )
        )

    return documents


def split_documents(
    documents: list[Document],
) -> list[Document]:
    """
    Découpe les documents en chunks avant vectorisation.

    Les métadonnées du document original sont conservées
    automatiquement dans chaque chunk.

    Args:
        documents: Liste de Documents LangChain.

    Returns:
        Liste de chunks.
    """
    if not documents:
        raise ValueError(
            "Aucun document disponible pour le découpage."
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    return text_splitter.split_documents(documents)


def build_vectorstore(
    chunks: list[Document],
) -> FAISS:
    """
    Génère les embeddings Mistral et construit l'index FAISS.

    Args:
        chunks: Documents découpés à indexer.

    Returns:
        Base vectorielle FAISS.
    """
    if not MISTRAL_API_KEY:
        raise ValueError(
            "La clé MISTRAL_API_KEY est absente du fichier .env"
        )

    if not chunks:
        raise ValueError(
            "Aucun chunk disponible pour la vectorisation."
        )

    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=MISTRAL_API_KEY,
    )

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return vectorstore


def save_vectorstore(
    vectorstore: FAISS,
    output_path: str,
) -> None:
    """
    Sauvegarde la base vectorielle FAISS localement.

    Args:
        vectorstore: Base FAISS à sauvegarder.
        output_path: Dossier de destination.
    """
    vectorstore.save_local(output_path)


def main() -> None:
    """
    Exécute le pipeline complet de création de la base vectorielle.
    """
    # 1. Chargement
    df_events = load_events(DATA_PATH)

    print(
        "Nombre d'événements chargés :",
        len(df_events),
    )

    # 2. Création des documents
    documents = create_documents(df_events)

    print(
        "Nombre de documents créés :",
        len(documents),
    )

    max_document_size = max(
        len(document.page_content)
        for document in documents
    )

    print(
        "Taille maximale d'un document :",
        max_document_size,
    )

    # 3. Découpage
    chunks = split_documents(documents)

    print(
        "Nombre de chunks créés :",
        len(chunks),
    )

    # 4. Construction FAISS
    vectorstore = build_vectorstore(chunks)

    # 5. Vérification
    number_of_vectors = vectorstore.index.ntotal

    if number_of_vectors != len(chunks):
        raise ValueError(
            "Le nombre de vecteurs ne correspond pas "
            "au nombre de chunks."
        )

    # 6. Sauvegarde
    save_vectorstore(
        vectorstore=vectorstore,
        output_path=VECTORSTORE_PATH,
    )

    print(
        "Base vectorielle sauvegardée dans :",
        VECTORSTORE_PATH,
    )

    print(
        "Nombre de vecteurs indexés :",
        number_of_vectors,
    )


if __name__ == "__main__":
    main()