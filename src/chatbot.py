import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.rag_chain import (
    answer_question,
    create_rag_system,
)


st.set_page_config(
    page_title="Assistant culturel Paris",
    page_icon="🎭",
    layout="centered",
)

st.title("🎭 Assistant culturel — Paris")

st.write(
    """
Posez une question pour trouver des événements culturels à Paris.

Vous pouvez effectuer une recherche par type d'événement,
par lieu ou par date.
"""
)


@st.cache_resource
def load_rag_system():
    """
    Charge une seule fois la base FAISS et la chaîne Mistral.
    """
    return create_rag_system()


try:
    vectorstore, generation_chain = load_rag_system()

except Exception as error:
    st.error(
        "Impossible de charger le système RAG : "
        f"{error}"
    )
    st.stop()


question = st.text_input(
    "Votre recherche",
    placeholder=(
        "Exemple : Quels événements sont prévus "
        "à Paris le 20 août 2026 ?"
    ),
)


st.caption(
    "Exemples : concert demain, exposition en septembre, "
    "événements du 20 août 2026."
)


if st.button("Rechercher", type="primary"):
    if not question.strip():
        st.warning(
            "Veuillez saisir une question."
        )

    else:
        try:
            with st.spinner(
                "Recherche des événements en cours..."
            ):
                response = answer_question(
                    question=question,
                    vectorstore=vectorstore,
                    generation_chain=generation_chain,
                )

            st.subheader("Recommandations")

            st.markdown(response)

        except Exception as error:
            st.error(
                "Une erreur est survenue pendant la recherche : "
                f"{error}"
            )