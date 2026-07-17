import os
import re
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings


load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

VECTORSTORE_PATH = "vectorstore/paris_faiss"

FRENCH_MONTHS = {
    "janvier": 1,
    "février": 2,
    "fevrier": 2,
    "mars": 3,
    "avril": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7,
    "août": 8,
    "aout": 8,
    "septembre": 9,
    "octobre": 10,
    "novembre": 11,
    "décembre": 12,
    "decembre": 12,
}


def format_documents(documents) -> str:
    """
    Transforme les documents récupérés en contexte pour le LLM.
    """
    if not documents:
        return "Aucun événement correspondant n'a été trouvé."

    formatted_documents = []

    for position, document in enumerate(documents, start=1):
        formatted_documents.append(
            f"ÉVÉNEMENT {position}\n"
            f"{document.page_content}"
        )

    return "\n\n---\n\n".join(formatted_documents)


def extract_explicit_date(question: str) -> date | None:
    """
    Extrait une date explicite depuis une question.

    Formats pris en charge :
    - 20/08/2026 ;
    - 20-08-2026 ;
    - 20 août 2026.
    """
    numeric_match = re.search(
        r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b",
        question,
    )

    if numeric_match:
        day, month, year = map(
            int,
            numeric_match.groups(),
        )

        try:
            return date(year, month, day)
        except ValueError:
            return None

    text_match = re.search(
        r"\b(\d{1,2})\s+"
        r"(janvier|février|fevrier|mars|avril|mai|juin|"
        r"juillet|août|aout|septembre|octobre|novembre|"
        r"décembre|decembre)"
        r"\s+(\d{4})\b",
        question.casefold(),
    )

    if text_match:
        day = int(text_match.group(1))
        month = FRENCH_MONTHS[text_match.group(2)]
        year = int(text_match.group(3))

        try:
            return date(year, month, day)
        except ValueError:
            return None

    return None


def convert_metadata_date(value) -> date | None:
    """
    Convertit une date stockée dans les métadonnées.
    """
    if value is None or value == "":
        return None

    converted_date = pd.to_datetime(
        value,
        errors="coerce",
        utc=True,
    )

    if pd.isna(converted_date):
        return None

    return converted_date.date()


def document_matches_date(
    document,
    requested_date: date,
) -> bool:
    """
    Vérifie si la date demandée se trouve dans la période
    de l'événement.
    """
    metadata = document.metadata

    first_begin = convert_metadata_date(
        metadata.get("first_begin")
    )

    last_end = convert_metadata_date(
        metadata.get("last_end")
    )

    if first_begin is None or last_end is None:
        return False

    return first_begin <= requested_date <= last_end


def retrieve_documents(
    vectorstore: FAISS,
    question: str,
    number_of_results: int = 20,
):
    """
    Recherche les événements proches sémantiquement.

    Si une date explicite est détectée, les résultats sont ensuite
    filtrés grâce aux métadonnées temporelles.
    """
    documents = vectorstore.similarity_search(
        question,
        k=number_of_results,
    )

    requested_date = extract_explicit_date(question)

    if requested_date is None:
        return documents[:5]

    matching_documents = [
        document
        for document in documents
        if document_matches_date(
            document,
            requested_date,
        )
    ]

    return matching_documents[:10]


def create_rag_system():
    """
    Charge FAISS et construit le système RAG avec Mistral.
    """
    if not MISTRAL_API_KEY:
        raise ValueError(
            "La clé MISTRAL_API_KEY est absente du fichier .env"
        )

    if not os.path.exists(VECTORSTORE_PATH):
        raise FileNotFoundError(
            "La base vectorielle est introuvable : "
            f"{VECTORSTORE_PATH}"
        )

    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=MISTRAL_API_KEY,
    )

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    prompt = ChatPromptTemplate.from_template(
        """
Tu es l'assistant culturel de Puls-Events.

La date actuelle est le {current_date}.

Ta mission est de recommander des événements culturels
uniquement à partir du contexte fourni.

Règles obligatoires :

1. N'invente jamais un événement, une date, un lieu ou une adresse.
2. Respecte précisément la date et le lieu demandés.
3. Ne recommande pas un événement qui ne correspond pas à la demande.
4. Si le contexte ne contient aucun résultat correspondant, réponds
   clairement que tu n'as trouvé aucun événement pertinent.
5. Pour chaque événement recommandé, indique :
   - le titre ;
   - la date ou la période ;
   - le lieu ;
   - l'adresse ;
   - une courte description.
6. Ne mentionne pas les informations absentes du contexte.
7. Évite les doublons.

Contexte des événements :

{context}

Question de l'utilisateur :

{question}

Réponse :
""".strip()
    )

    llm = ChatMistralAI(
        model="mistral-small-latest",
        api_key=MISTRAL_API_KEY,
        temperature=0,
    )

    generation_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    return vectorstore, generation_chain


def answer_question(
    question: str,
    vectorstore: FAISS,
    generation_chain,
) -> str:
    """
    Recherche les événements et génère la réponse finale.
    """
    if not question.strip():
        return "Veuillez saisir une question."

    documents = retrieve_documents(
        vectorstore=vectorstore,
        question=question,
    )

    context = format_documents(documents)

    current_date = datetime.now(
        ZoneInfo("Europe/Paris")
    ).strftime("%d/%m/%Y")

    return generation_chain.invoke(
        {
            "context": context,
            "question": question,
            "current_date": current_date,
        }
    )


def main() -> None:
    """
    Lance un test interactif du système RAG.
    """
    vectorstore, generation_chain = create_rag_system()

    question = input("Question : ").strip()

    response = answer_question(
        question=question,
        vectorstore=vectorstore,
        generation_chain=generation_chain,
    )

    print("\nRéponse :")
    print(response)


if __name__ == "__main__":
    main()