import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser


class RAGService:

    def __init__(self, groq_api_key: str):

        self.index_dir = Path("indexes")
        self.index_dir.mkdir(exist_ok=True)

        # ====================================================
        # EMBEDDINGS
        # ====================================================

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # ====================================================
        # LLM
        # ====================================================

        self.llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0.2,
            groq_api_key=groq_api_key
        )

        # ====================================================
        # SPLITTER
        # ====================================================

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=500
        )

        # ====================================================
        # PROMPT
        # ====================================================

        self.prompt = PromptTemplate(
            template="""
You are a helpful assistant answering questions about a PDF.

Use ONLY the information provided in the PDF CONTEXT.

IMPORTANT INSTRUCTIONS:

1. Carefully read the entire provided context before answering.
2. The PDF may contain OCR errors, formatting problems, broken words,
   or page numbers mixed into the text.
3. Ignore page numbers when counting items.
4. If the question asks "how many", determine whether the PDF gives:
   - a numerical quantity, OR
   - a list of categories/types/languages.
5. If a list is present, count the actual listed items.
6. Do NOT treat a page number as an item count.
7. Do NOT invent information.
8. If the PDF does not provide enough information, say:
   "I don't know based on the provided PDF."
9. If the PDF gives categories but not the total number of individual
   objects, clearly distinguish between the two.

PDF CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""",
            input_variables=[
                "context",
                "question"
            ]
        )

        self.parser = StrOutputParser()


    # ========================================================
    # INDEX PDF
    # ========================================================

    def index_pdf(
        self,
        document_id: str,
        pdf_path: str
    ):

        # ----------------------------------------------------
        # Load PDF
        # ----------------------------------------------------

        loader = PyPDFLoader(pdf_path)

        documents = loader.load()

        # ----------------------------------------------------
        # Split
        # ----------------------------------------------------

        chunks = self.splitter.split_documents(
            documents
        )

        # ----------------------------------------------------
        # Create FAISS
        # ----------------------------------------------------

        vector_store = FAISS.from_documents(
            chunks,
            self.embeddings
        )

        # ----------------------------------------------------
        # Save index
        # ----------------------------------------------------

        index_path = self.index_dir / document_id

        vector_store.save_local(
            str(index_path)
        )

        return {
            "pages": len(documents),
            "chunks": len(chunks)
        }


    # ========================================================
    # LOAD VECTOR STORE
    # ========================================================

    def get_vector_store(
        self,
        document_id: str
    ):

        index_path = self.index_dir / document_id

        if not index_path.exists():
            raise ValueError(
                "Document not found."
            )

        vector_store = FAISS.load_local(
            str(index_path),
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        return vector_store


    # ========================================================
    # FORMAT DOCUMENTS
    # ========================================================

    def format_docs(
        self,
        retrieved_docs
    ):

        return "\n\n".join(
            doc.page_content
            for doc in retrieved_docs
        )


    # ========================================================
    # ASK
    # ========================================================

    def ask(
        self,
        document_id: str,
        question: str
    ):

        vector_store = self.get_vector_store(
            document_id
        )

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 2
            }
        )

        # ----------------------------------------------------
        # RAG CHAIN
        # ----------------------------------------------------

        parallel_chain = RunnableParallel(
            {
                "context": (
                    retriever
                    | RunnableLambda(
                        self.format_docs
                    )
                ),
                "question": RunnablePassthrough()
            }
        )

        main_chain = (
            parallel_chain
            | self.prompt
            | self.llm
            | self.parser
        )

        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        answer = main_chain.invoke(
            question
        )

        # ----------------------------------------------------
        # Get sources
        # ----------------------------------------------------

        retrieved_docs = retriever.invoke(
            question
        )

        sources = []

        for doc in retrieved_docs:

            page = doc.metadata.get(
                "page"
            )

            sources.append(
                {
                    "page": (
                        page + 1
                        if isinstance(page, int)
                        else page
                    ),
                    "content": doc.page_content
                }
            )

        return {
            "answer": answer,
            "sources": sources
        }
