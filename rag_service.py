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
        self.pdf_pages = {}

        # ====================================================
        # PROMPT
        # ====================================================

        self.prompt = PromptTemplate(
    template="""
You are a helpful PDF assistant.

Answer the user's question using ONLY the information in the PDF CONTEXT.

Your response must feel natural, conversational, and concise.

========================
ANSWER STYLE
========================

1. Answer the user's exact question directly.
2. For a simple question, give a short answer (usually 1-3 sentences).
3. Do not give a long explanation unless the user asks for details.
4. Do not repeat information unnecessarily.
5. Do not repeat the user's question.
6. Do not add information that is not in the PDF.
7. Do not make assumptions.
8. If the user asks for "steps", "how to", or "procedure",
provide the steps as a short numbered list.
9. If the user asks for a fee, give the relevant fee directly.
Do not list every fee in the PDF unless the user asks for all fees.
10. If the user asks for a location, give only the relevant location.
11. If the user asks for membership, explain the membership process briefly.
12. If the user asks "how can I become a member?",
    give the essential steps, not every detail from the PDF.
13. Use bullet points or numbered lists only when they improve readability.
14. Do not use Markdown tables.
15. Do not create unnecessary headings.
16. Use simple language.
17. If the user speaks in Hinglish, you may respond in simple Hinglish.
18. Do not say "according to the PDF" unless necessary.
19. Do not offer additional information at the end unless it is genuinely useful.
20. Never invent links, phone numbers, fees, dates, procedures, or other facts.
36. When the user asks "how many" or "total", and the PDF contains a
    list of items for that category, count ALL items in the relevant list.

37. Do not assume that the number of items visible in one retrieved chunk
    is the total number in the PDF.

38. If a list appears to continue across multiple pages or chunks,
    combine the relevant chunks before counting.

39. For counting questions, carefully check all retrieved context for
    every item belonging to the requested category.

40. Never answer a counting question based only on the first few items
    found in the context.

========================
IMPORTANT PDF RULES
========================

- The PDF may contain OCR errors, broken words, or page numbers.
- Ignore page numbers when interpreting content.
- Carefully interpret the retrieved context.
- If the answer is not available in the context, say:
"I couldn't find this information in the provided PDF."
- If the PDF provides multiple categories but not a total number,
clearly distinguish between categories and total quantity.

========================
PDF CONTEXT
========================

{context}

========================
USER QUESTION
========================

{question}

========================
ANSWER
========================

Give only the answer to the user's question.
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
        self.pdf_pages[document_id] = documents

        # ----------------------------------------------------
        # Split
        # ----------------------------------------------------

        chunks = self.splitter.split_documents(documents)

        # Har chunk mein page number preserve karo
        for chunk in chunks:
            page = chunk.metadata.get("page")

            if isinstance(page, int):
                chunk.metadata["page"] = page

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

        formatted_docs = []

        for doc in retrieved_docs:

            page = doc.metadata.get("page")

            if isinstance(page, int):
                page = page + 1

            formatted_docs.append(
                f"--- PDF PAGE {page} ---\n"
                f"{doc.page_content}"
            )
        return "\n\n".join(formatted_docs)
                


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
                "k": 5
            }
        )

        # ----------------------------------------------------
        # Retrieve relevant chunks
        # ----------------------------------------------------

        retrieved_docs = retriever.invoke(question)

        # ----------------------------------------------------
        # Add previous + current + next pages
        # ----------------------------------------------------

        expanded_docs = []

        for doc in retrieved_docs:

            page = doc.metadata.get("page")

            # Agar page number available nahi hai
            if not isinstance(page, int):

                expanded_docs.append(doc)
                continue

            # Current page ke saath previous aur next page
            target_pages = {
                page - 1,
                page,
                page + 1
            }

            # Original PDF pages mein search karo
            for other_doc in self.pdf_pages.get(
                document_id,
                []
            ):

                other_page = other_doc.metadata.get("page")

                if other_page in target_pages:

                    expanded_docs.append(
                        other_doc
                    )

        # ----------------------------------------------------
        # Remove duplicate pages
        # ----------------------------------------------------

        unique_pages = {}

        for doc in expanded_docs:

            page = doc.metadata.get("page")

            if page not in unique_pages:

                unique_pages[page] = doc

        expanded_docs = list(
            unique_pages.values()
        )

        # ----------------------------------------------------
        # Sort pages
        # ----------------------------------------------------

        expanded_docs.sort(
            key=lambda doc: doc.metadata.get(
                "page",
                999999
            )
        )

        # ----------------------------------------------------
        # Create context
        # ----------------------------------------------------

        context = self.format_docs(
            expanded_docs
        )

        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        answer = (
            self.prompt
            | self.llm
            | self.parser
        ).invoke(
            {
                "context": context,
                "question": question
            }
        )

        # ----------------------------------------------------
        # Get sources
        # ----------------------------------------------------

        sources = []

        for doc in expanded_docs:

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

        # ----------------------------------------------------
        # Return response
        # ----------------------------------------------------

        return {
            "answer": answer,
            "sources": sources
        }
