import hashlib
import os
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from pypdf.errors import DependencyError, PdfReadError


load_dotenv()


CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
MAX_CONTEXT_CHUNKS = 5
MIN_USEFUL_TEXT_CHARS = 20


@dataclass
class Chunk:
    text: str
    page: int
    source: str


def get_client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def file_hash(files: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode("utf-8"))
        digest.update(file.getvalue())
    return digest.hexdigest()


def extract_pdf_chunks(uploaded_files: list[Any]) -> list[Chunk]:
    chunks: list[Chunk] = []

    for uploaded_file in uploaded_files:
        try:
            reader = PdfReader(uploaded_file)
            for page_number, page in enumerate(reader.pages, start=1):
                page_text = clean_extracted_text(page.extract_text() or "")
                if not page_text:
                    continue

                chunks.extend(
                    Chunk(text=chunk, page=page_number, source=uploaded_file.name)
                    for chunk in split_text(page_text)
                )
        except DependencyError as exc:
            st.error(
                f"Could not read `{uploaded_file.name}` Install the required dependencies"
                " Run `pip install -r requirements.txt`, then restart the app."
            )
            st.caption(str(exc))
        except PdfReadError as exc:
            st.error(f"Could not read `{uploaded_file.name}` as a valid PDF.")
            st.caption(str(exc))

    return chunks


def clean_extracted_text(text: str) -> str:
    text = "".join(char if char.isprintable() else " " for char in text)
    text = " ".join(text.split())

    useful_chars = sum(char.isalnum() for char in text)
    if useful_chars < MIN_USEFUL_TEXT_CHARS:
        return ""

    return text


def split_text(text: str, chunk_size: int = 1400, overlap: int = 220) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(end - overlap, 0)
    return chunks


def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    vectors = np.array([item.embedding for item in response.data], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def build_index(client: OpenAI, uploaded_files: list[Any]) -> None:
    chunks = extract_pdf_chunks(uploaded_files)
    if not chunks:
        st.session_state.pdf_chunks = []
        st.session_state.pdf_embeddings = None
        return

    with st.spinner("Reading and indexing your PDFs..."):
        embeddings = embed_texts(client, [chunk.text for chunk in chunks])

    st.session_state.pdf_chunks = chunks
    st.session_state.pdf_embeddings = embeddings


def retrieve_context(client: OpenAI, question: str) -> list[tuple[Chunk, float]]:
    chunks: list[Chunk] = st.session_state.get("pdf_chunks", [])
    embeddings = st.session_state.get("pdf_embeddings")
    if not chunks or embeddings is None:
        return []

    query_embedding = embed_texts(client, [question])[0]
    scores = embeddings @ query_embedding
    top_indices = np.argsort(scores)[::-1][:MAX_CONTEXT_CHUNKS]
    return [(chunks[index], float(scores[index])) for index in top_indices]


def answer_question(client: OpenAI, question: str, matches: list[tuple[Chunk, float]]) -> str:
    context = "\n\n".join(
        f"[{index}] Source: {chunk.source}, page {chunk.page}\n{chunk.text}"
        for index, (chunk, _score) in enumerate(matches, start=1)
    )

    prompt = f"""
You are a careful PDF question-answering assistant.
Answer using only the provided PDF context.
Use any explicit facts, dates, labels, headings, fields, table values, and document metadata that appear in the context.
When answering date or time questions, distinguish between different kinds of dates, such as document date,
sent date, effective date, due date, purchase date, event date, delivery date, or signature date.
If the context contains only a related date but not the exact date being asked for, provide the related date
and clearly say that the exact requested date is not shown.
If the requested fact is not present in the context, say exactly what is missing rather than guessing.
Include short citations like [1] or [2] for facts you use.

PDF context:
{context}

Question:
{question}
""".strip()

    response = client.responses.create(
        model=CHAT_MODEL,
        input=prompt,
    )
    return response.output_text


def initialize_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pdf_file_hash", None)
    st.session_state.setdefault("pdf_chunks", [])
    st.session_state.setdefault("pdf_embeddings", None)


def main() -> None:
    st.set_page_config(page_title="AI PDF Chatbot", page_icon=":material/description:", layout="wide")
    initialize_state()

    st.title("AI PDF Chatbot")
    st.caption("Upload PDFs, ask questions, and get answers grounded in the document text.")

    with st.sidebar:
        st.header("Setup")
        st.write(f"Chat model: `{CHAT_MODEL}`")
        st.write(f"Embedding model: `{EMBEDDING_MODEL}`")

        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    client = get_client()
    if not client:
        st.info("Set OPENAI_API_KEY in your `.env` file, then restart the app.")
        return

    if uploaded_files:
        current_hash = file_hash(uploaded_files)
        if current_hash != st.session_state.pdf_file_hash:
            build_index(client, uploaded_files)
            st.session_state.pdf_file_hash = current_hash
            st.session_state.messages = []

        chunk_count = len(st.session_state.pdf_chunks)
        if chunk_count:
            st.success(f"Indexed {chunk_count} text chunks from {len(uploaded_files)} PDF file(s).")
        else:
            st.warning("No extractable text was found. Scanned PDFs may need OCR first.")
    else:
        st.info("Upload at least one PDF to start chatting.")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about your PDFs")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the PDFs and writing an answer..."):
            matches = retrieve_context(client, question)
            answer = answer_question(client, question, matches)
        st.markdown(answer)

        with st.expander("Retrieved context"):
            for chunk, score in matches:
                st.markdown(f"**{chunk.source}, page {chunk.page}** - similarity `{score:.3f}`")
                st.write(chunk.text)

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()