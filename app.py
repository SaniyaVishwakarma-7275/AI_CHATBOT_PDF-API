import os
import json
import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser


class RAGService:

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self, groq_api_key: str):

        self.groq_api_key = groq_api_key

        self.vectorstore_dir = Path("vectorstore")
        self.processed_file = Path("processed.json")

        self.vectorstore_dir.mkdir(
            exist_ok=True
        )

        # ----------------------------------------------------
        # Embedding model
        # ----------------------------------------------------

        print("🔄 Loading embedding model...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        print("✅ Embedding model loaded")


        # ----------------------------------------------------
        # Groq LLM
        # ----------------------------------------------------

        print("🔄 Loading Groq LLM...")

        self.llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0.2,
            groq_api_key=self.groq_api_key
        )

        print("✅ Groq LLM loaded")


        # ----------------------------------------------------
        # Vectorstore
        # ----------------------------------------------------

        self.vectorstore = self.load_vectorstore()


    # ========================================================
    # FILE HASH
    # ========================================================

    def get_file_hash(self, file_path):

        sha256 = hashlib.sha256()

        with open(file_path, "rb") as file:

            while True:

                chunk = file.read(8192)

                if not chunk:
                    break

                sha256.update(chunk)

        return sha256.hexdigest()


    # ========================================================
    # LOAD PROCESSED FILES
    # ========================================================

    def load_processed_files(self):

        if not self.processed_file.exists():

            return {}

        try:

            with open(
                self.processed_file,
                "r",
                encoding="utf-8"
            ) as file:

                return json.load(file)

        except Exception:

            return {}


    # ========================================================
    # SAVE PROCESSED FILES
    # ========================================================

    def save_processed_files(
        self,
        processed_files
    ):

        with open(
            self.processed_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                processed_files,
                file,
                indent=4
            )


    # ========================================================
    # LOAD EXISTING FAISS
    # ========================================================

    def load_vectorstore(self):

        index_file = (
            self.vectorstore_dir /
            "index.faiss"
        )

        if not index_file.exists():

            print("ℹ️ No existing FAISS index found")

            return None

        try:

            print("🔄 Loading existing FAISS index...")

            vectorstore = FAISS.load_local(
                str(self.vectorstore_dir),
                self.embeddings,
                allow_dangerous_deserialization=True
            )

            print("✅ Existing FAISS loaded")

            return vectorstore

        except Exception as e:

            print(
                f"⚠️ Could not load FAISS: {e}"
            )

            return None


    # ========================================================
    # PROCESS UPLOAD FOLDER
    # ========================================================

    def process_upload_folder(
        self,
        upload_dir: str
    ):

        upload_path = Path(upload_dir)

        upload_path.mkdir(
            exist_ok=True
        )

        processed_files = (
            self.load_processed_files()
        )

        new_documents = []

        pdf_files = list(
            upload_path.glob("*.pdf")
        )

        if not pdf_files:

            print("📂 No PDF files found")

            return self.vectorstore


        print(
            f"📚 Found {len(pdf_files)} PDF file(s)"
        )


        # ====================================================
        # CHECK EACH PDF
        # ====================================================

        for pdf_path in pdf_files:

            filename = pdf_path.name

            current_hash = (
                self.get_file_hash(
                    pdf_path
                )
            )


            # ------------------------------------------------
            # Already indexed
            # ------------------------------------------------

            if (
                filename in processed_files
                and
                processed_files[filename]
                == current_hash
            ):

                print(
                    f"⏭️ SKIP: {filename}"
                )

                continue


            # ------------------------------------------------
            # New or modified PDF
            # ------------------------------------------------

            print(
                f"📄 PROCESSING: {filename}"
            )


            try:

                # --------------------------------------------
                # Extract text
                # --------------------------------------------

                loader = PyPDFLoader(
                    str(pdf_path)
                )

                documents = loader.load()


                # --------------------------------------------
                # Add metadata
                # --------------------------------------------

                for document in documents:

                    document.metadata[
                        "source_file"
                    ] = filename


                new_documents.extend(
                    documents
                )


                # --------------------------------------------
                # Save hash
                # --------------------------------------------

                processed_files[
                    filename
                ] = current_hash


                print(
                    f"   ✅ Pages: {len(documents)}"
                )


            except Exception as e:

                print(
                    f"❌ Error processing "
                    f"{filename}: {e}"
                )


        # ====================================================
        # NO NEW FILES
        # ====================================================

        if not new_documents:

            print(
                "✅ No new PDFs to process"
            )

            return self.vectorstore


        # ====================================================
        # TEXT SPLITTING
        # ====================================================

        print("✂️ Creating chunks...")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=500
        )

        chunks = splitter.split_documents(
            new_documents
        )

        print(
            f"✅ Created {len(chunks)} chunks"
        )


        # ====================================================
        # EMBEDDINGS + FAISS
        # ====================================================

        print(
            "🧠 Creating embeddings..."
        )


        if self.vectorstore is None:

            # ----------------------------------------------
            # First PDF / first index
            # ----------------------------------------------

            self.vectorstore = (
                FAISS.from_documents(
                    chunks,
                    self.embeddings
                )
            )

            print(
                "✅ New FAISS index created"
            )

        else:

            # ----------------------------------------------
            # Add new PDFs to existing FAISS
            # ----------------------------------------------

            self.vectorstore.add_documents(
                chunks
            )

            print(
                "✅ New embeddings added to FAISS"
            )


        # ====================================================
        # SAVE FAISS
        # ====================================================

        self.vectorstore.save_local(
            str(self.vectorstore_dir)
        )

        print(
            "💾 FAISS index saved"
        )


        # ====================================================
        # SAVE PROCESSED FILES
        # ====================================================

        self.save_processed_files(
            processed_files
        )

        print(
            "💾 Processed file information saved"
        )


        return self.vectorstore


    # ========================================================
    # RETRIEVER
    # ========================================================

    def get_retriever(self):

        if self.vectorstore is None:

            raise ValueError(
                "No PDFs have been indexed yet."
            )

        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 2
            }
        )


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
    # PROMPT
    # ========================================================

    def create_prompt(self):

        return PromptTemplate(
            template="""

You are a helpful assistant answering questions
about the uploaded PDF documents.

Use ONLY the information provided in the PDF CONTEXT.

IMPORTANT INSTRUCTIONS:

1. Carefully read the provided context before answering.

2. The PDF may contain OCR errors, formatting problems,
   broken words, or page numbers mixed into the text.

3. Ignore page numbers when counting items.

4. If the question asks "how many", determine whether
   the PDF gives:
   - a numerical quantity, OR
   - a list of categories/types/languages.

5. If a list is present, count the actual listed items.

6. Do NOT treat a page number as an item count.

7. Do NOT invent information.

8. If the PDF does not provide enough information, say:

"I don't know based on the provided PDF."

9. If the PDF gives categories but not the total number
   of individual objects, clearly distinguish between them.

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


    # ========================================================
    # ASK QUESTION
    # ========================================================

    def ask(
        self,
        question: str
    ):

        retriever = self.get_retriever()

        prompt = self.create_prompt()


        # ----------------------------------------------------
        # RAG chain
        # ----------------------------------------------------

        parallel_chain = RunnableParallel(
            {
                "context":
                    retriever
                    | RunnableLambda(
                        self.format_docs
                    ),

                "question":
                    RunnablePassthrough()
            }
        )


        chain = (
            parallel_chain
            | prompt
            | self.llm
            | StrOutputParser()
        )


        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        answer = chain.invoke(
            question
        )


        # ----------------------------------------------------
        # Retrieve sources
        # ----------------------------------------------------

        retrieved_docs = retriever.invoke(
            question
        )


        sources = []


        for doc in retrieved_docs:

            page_number = doc.metadata.get(
                "page",
                None
            )

            if isinstance(
                page_number,
                int
            ):

                page_number += 1


            sources.append(
                {
                    "file":
                        doc.metadata.get(
                            "source_file",
                            "Unknown"
                        ),

                    "page":
                        page_number,

                    "content":
                        doc.page_content
                }
            )


        return {
            "answer": answer,
            "sources": sources
        }





























# import os
# import tempfile

# import streamlit as st
# from dotenv import load_dotenv

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain_community.vectorstores import FAISS

# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_groq import ChatGroq

# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import (
#     RunnableParallel,
#     RunnablePassthrough,
#     RunnableLambda
# )
# from langchain_core.output_parsers import StrOutputParser
# from fastapi import FastAPI


# app = FastAPI()

# # CORS - React frontend ke liye
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://localhost:3000",
#         "http://127.0.0.1:3000",
#         "http://localhost:5173",
#         "http://127.0.0.1:5173",
#     ],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # ============================================================
# # 1. LOAD ENVIRONMENT VARIABLES
# # ============================================================

# load_dotenv()

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# if not GROQ_API_KEY:
#     st.error("GROQ_API_KEY not found. Please add it to your .env file.")
#     st.stop()


# # ============================================================
# # 2. PAGE CONFIG
# # ============================================================

# st.set_page_config(
#     page_title="PDF RAG Chatbot",
#     page_icon="📚",
#     layout="wide"
# )

# st.title("📚 PDF Question Answering with RAG")
# st.write(
#     "Upload a PDF and ask questions about its contents."
# )


# # ============================================================
# # 3. UPLOAD PDF
# # ============================================================

# uploaded_file = st.file_uploader(
#     "Upload your PDF",
#     type=["pdf"]
# )


# # ============================================================
# # 4. CREATE EMBEDDING MODEL
# # ============================================================

# @st.cache_resource
# def load_embedding_model():

#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )

#     return embeddings


# embeddings = load_embedding_model()


# # ============================================================
# # 5. CREATE GROQ LLM
# # ============================================================

# @st.cache_resource
# def load_llm():

#     llm = ChatGroq(
#         model="openai/gpt-oss-20b",
#         temperature=0.2,
#         groq_api_key=GROQ_API_KEY
#     )

#     return llm


# llm = load_llm()


# # ============================================================
# # 6. PROCESS PDF
# # ============================================================

# if uploaded_file is not None:

#     st.success(f"Uploaded: {uploaded_file.name}")

#     # Save uploaded PDF temporarily
#     with tempfile.NamedTemporaryFile(
#         delete=False,
#         suffix=".pdf"
#     ) as temp_file:

#         temp_file.write(uploaded_file.getvalue())

#         pdf_path = temp_file.name


#     # ========================================================
#     # STEP 1 — PDF TEXT EXTRACTION
#     # ========================================================

#     with st.spinner("Reading PDF..."):

#         loader = PyPDFLoader(pdf_path)

#         documents = loader.load()

#     st.write(f"📄 Pages loaded: {len(documents)}")


#     # ========================================================
#     # STEP 2 — TEXT SPLITTING
#     # ========================================================

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1500,
#         chunk_overlap=500
#     )

#     chunks = splitter.split_documents(documents)

#     st.write(f"✂️ Number of chunks: {len(chunks)}")


#     # ========================================================
#     # STEP 3 — EMBEDDINGS + FAISS
#     # ========================================================

#     with st.spinner("Creating embeddings..."):

#         vector_store = FAISS.from_documents(
#             chunks,
#             embeddings
#         )

#     st.success("✅ PDF indexed successfully!")


#     # ========================================================
#     # STEP 4 — RETRIEVER
#     # ========================================================

#     retriever = vector_store.as_retriever(
#         search_type="similarity",
#         search_kwargs={
#             "k": 2
#         }
#     )


#     # ========================================================
#     # STEP 5 — PROMPT
#     # ========================================================

#     prompt = PromptTemplate(
#     template="""
# You are a helpful assistant answering questions about a PDF.

# Use ONLY the information provided in the PDF CONTEXT.

# IMPORTANT INSTRUCTIONS:

# 1. Carefully read the entire provided context before answering.
# 2. The PDF may contain OCR errors, formatting problems, broken words,
#    or page numbers mixed into the text.
# 3. Ignore page numbers when counting items.
# 4. If the question asks "how many", determine whether the PDF gives:
#    - a numerical quantity, OR
#    - a list of categories/types/languages.
# 5. If a list is present, count the actual listed items.
# 6. Do NOT treat a page number as an item count.
# 7. Do NOT invent information.
# 8. If the PDF does not provide enough information, say:
#    "I don't know based on the provided PDF."
# 9. If the PDF gives categories but not the total number of individual
#    objects, clearly distinguish between the two.

# PDF CONTEXT:
# {context}

# QUESTION:
# {question}

# ANSWER:
# """,
#     input_variables=["context", "question"]
# )




#     # ========================================================
#     # STEP 6 — FORMAT RETRIEVED DOCUMENTS
#     # ========================================================

#     def format_docs(retrieved_docs):

#         context_text = "\n\n".join(
#             doc.page_content
#             for doc in retrieved_docs
#         )

#         return context_text


#     # ========================================================
#     # STEP 7 — RAG CHAIN
#     # ========================================================

#     parallel_chain = RunnableParallel(
#         {
#             "context": retriever | RunnableLambda(format_docs),
#             "question": RunnablePassthrough()
#         }
#     )


#     parser = StrOutputParser()


#     main_chain = (
#         parallel_chain
#         | prompt
#         | llm
#         | parser
#     )


#     # ========================================================
#     # STEP 8 — QUESTION INPUT
#     # ========================================================

#     st.divider()

#     question = st.text_input(
#         "Ask a question about your PDF:"
#     )


#     # ========================================================
#     # STEP 9 — GENERATION
#     # ========================================================

#     if question:

#         with st.spinner("Searching the PDF and generating answer..."):

#             answer = main_chain.invoke(question)

#         st.subheader("🤖 Answer")

#         st.write(answer)


#         # ====================================================
#         # SHOW SOURCES
#         # ====================================================

#         st.subheader("📖 Retrieved Sources")

#         retrieved_docs = retriever.invoke(question)

#         for i, doc in enumerate(retrieved_docs):

#             page_number = doc.metadata.get(
#                 "page",
#                 "Unknown"
#             )

#             with st.expander(
#                 f"Source {i + 1} — Page {page_number + 1 if isinstance(page_number, int) else page_number}"
#             ):

#                 st.write(doc.page_content)
