import os
import json
import pickle
import hashlib
import logging
from dataclasses import dataclass, asdict
from typing import List, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from groq import Groq
import streamlit as st
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#updates


LLM_PROVIDER = st.secrets.get("LLM_PROVIDER")
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
OPENAI_MODEL = st.secrets["OPENAI_MODEL"]
EMBEDDING_MODEL = st.secrets["EMBEDDING_MODEL"]
TOP_K = 5
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
HYBRID_ALPHA = 0.5
DOCS_DIR = "RPD-en-US"
INDEX_DIR = "./data/index"


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_name: str
    doc_path: str
    page_number: Optional[int]
    section_title: Optional[str]
    content: str
    char_start: int
    char_end: int
    chunk_index: int
    total_chunks: int

    def to_citation(self) -> str:
        parts = [f"Document: {self.doc_name}"]
        if self.page_number is not None:
            parts.append(f"Page: {self.page_number}")
        parts.append(f"Chunk ID: {self.chunk_id}")
        return " | ".join(parts)

    def to_dict(self):
        return asdict(self)


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_name: str
    doc_path: str
    page_number: Optional[int]
    section_title: Optional[str]
    content: str
    semantic_score: float
    keyword_score: float
    hybrid_score: float
    citation: str

# PDF LOAD
def load_and_chunk_pdfs(folder_path: str,
                        chunk_size: int = 500,
                        chunk_overlap: int = 50) -> List[DocumentChunk]:

    docs = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file_name)
            loader = PDFMinerLoader(pdf_path, concatenate_pages=False)
            documents = loader.load()
            docs.extend(documents)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    split_docs = text_splitter.split_documents(docs)

    chunks = []

    for idx, doc in enumerate(split_docs):
        doc_name = os.path.basename(doc.metadata.get("source", "Unknown"))
        page_number = doc.metadata.get("page", None)

        raw_id = f"{doc_name}_{page_number}_{idx}"
        chunk_id = hashlib.md5(raw_id.encode()).hexdigest()[:12]

        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                doc_name=doc_name,
                doc_path=doc.metadata.get("source", ""),
                page_number=page_number,
                section_title=None,
                content=doc.page_content,
                char_start=0,
                char_end=len(doc.page_content),
                chunk_index=idx,
                total_chunks=len(split_docs)
            )
        )
    return chunks


# HYBRID RETRIEVER

class HybridRetriever:

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", alpha: float = 0.5):
        self.alpha = alpha
        self.embedder = SentenceTransformer(embedding_model)
        self.faiss_index = None
        self.bm25_index = None
        self.chunks: List[DocumentChunk] = []
        self.chunk_texts: List[str] = []
        self._initialized = False

    def index_exists(self, index_dir: str) -> bool:
        required = ["faiss.index", "bm25.pkl", "chunks.json"]
        return all(os.path.exists(os.path.join(index_dir, f)) for f in required)

    def build_index(self, chunks: List[DocumentChunk], index_dir: str):
        os.makedirs(index_dir, exist_ok=True)

        self.chunks = chunks
        self.chunk_texts = [c.content for c in chunks]

        embeddings = self.embedder.encode(
            self.chunk_texts,
            convert_to_numpy=True
        ).astype(np.float32)

        faiss.normalize_L2(embeddings)
        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.faiss_index.add(embeddings)

        tokenized = [text.lower().split() for text in self.chunk_texts]
        self.bm25_index = BM25Okapi(tokenized)

        faiss.write_index(self.faiss_index, os.path.join(index_dir, "faiss.index"))
        with open(os.path.join(index_dir, "bm25.pkl"), "wb") as f:
            pickle.dump(self.bm25_index, f)
        with open(os.path.join(index_dir, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in chunks], f)

        self._initialized = True

    def load_index(self, index_dir: str):
        self.faiss_index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
        with open(os.path.join(index_dir, "bm25.pkl"), "rb") as f:
            self.bm25_index = pickle.load(f)
        with open(os.path.join(index_dir, "chunks.json"), encoding="utf-8") as f:
            data = json.load(f)

        self.chunks = [DocumentChunk(**d) for d in data]
        self.chunk_texts = [c.content for c in self.chunks]
        self._initialized = True

    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:

        query_vec = self.embedder.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(query_vec)

        semantic_scores, semantic_indices = self.faiss_index.search(query_vec, top_k * 2)

        tokens = query.lower().split()
        keyword_scores = self.bm25_index.get_scores(tokens)

        rrf_scores = {}

        # Semantic
        for rank, (idx, score) in enumerate(zip(semantic_indices[0], semantic_scores[0])):
            if idx < 0:
                continue
            rrf_scores.setdefault(idx, {"semantic": 0, "keyword": 0, "hybrid": 0})
            rrf_scores[idx]["semantic"] = float(score)
            rrf_scores[idx]["hybrid"] += self.alpha * (1 / (60 + rank + 1))

        # Keyword
        top_keyword_indices = np.argsort(keyword_scores)[::-1][:top_k * 2]
        for rank, idx in enumerate(top_keyword_indices):
            score = keyword_scores[idx]
            if score <= 0:
                continue
            rrf_scores.setdefault(idx, {"semantic": 0, "keyword": 0, "hybrid": 0})
            rrf_scores[idx]["keyword"] = float(score)
            rrf_scores[idx]["hybrid"] += (1 - self.alpha) * (1 / (60 + rank + 1))

        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1]["hybrid"], reverse=True)[:top_k]

        results = []
        for idx, scores in sorted_items:
            chunk = self.chunks[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_name=chunk.doc_name,
                    doc_path=chunk.doc_path,
                    page_number=chunk.page_number,
                    section_title=None,
                    content=chunk.content,
                    semantic_score=scores["semantic"],
                    keyword_score=scores["keyword"],
                    hybrid_score=scores["hybrid"],
                    citation=chunk.to_citation()
                )
            )

        return results


# LLM CLIENT

class LLMClient:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    def complete(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()


# RAG PIPELINE

class RAGPipeline:

    def __init__(self, retriever: HybridRetriever, llm: LLMClient):
        self.retriever = retriever
        self.llm = llm

    def run(self, question: str, top_k: int = 5):

        retrieved_chunks = self.retriever.search(question, top_k=top_k)

        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_blocks.append(f"[{i}] {chunk.citation}\n{chunk.content}")

        context_text = "\n\n".join(context_blocks)

        system_prompt = """
You are a helpful assistant.
Answer ONLY using the provided context.
If not found, say you don't know.
Always cite sources like [1], [2] if you get the answer then only else do not cite.
"""

        user_prompt = f"""
Question:
{question}

Context:
{context_text}

Provide a clear structured answer.
"""

        answer = self.llm.complete(system_prompt, user_prompt)

        return {
            "question": question,
            "answer": answer,
            "chunks": retrieved_chunks
        }



# INITIALIZER

def initialize_retriever():
    load_dotenv()

    pdf_dir = os.getenv("PDF_DIR", "RPD-en-US")
    index_dir = os.getenv("INDEX_DIR", "./data/index")
    build = os.getenv("BUILD_INDEX", "0") == "1"

    retriever = HybridRetriever()

    if build or not retriever.index_exists(index_dir):
        chunks = load_and_chunk_pdfs(pdf_dir)
        retriever.build_index(chunks, index_dir)
    else:
        retriever.load_index(index_dir)

    return retriever

retriever = initialize_retriever()
llm = LLMClient()
rag = RAGPipeline(retriever, llm)

response = rag.run("How to create workflow")
print(response["answer"])