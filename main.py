import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app import RAGService


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in .env")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="PDF RAG API",
    description="Backend API for PDF question answering",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DIRECTORIES
# ============================================================

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ============================================================
# RAG SERVICE
# ============================================================

rag_service = RAGService(
    groq_api_key=GROQ_API_KEY
)


# ============================================================
# REQUEST MODEL
# ============================================================

class AskRequest(BaseModel):
    question: str


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    print("\n" + "=" * 60)
    print("🔍 Checking PDF folder...")
    print("=" * 60)

    rag_service.process_upload_folder(
        upload_dir=str(UPLOAD_DIR)
    )

    print("=" * 60)
    print("✅ PDF processing completed")
    print("🚀 API is ready")
    print("=" * 60 + "\n")


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "message": "PDF RAG API is running"
    }


@app.get("/health")
def health():

    return {
        "status": "ok"
    }


# ============================================================
# ASK QUESTION
# ============================================================

@app.post("/ask")
def ask_question(request: AskRequest):

    try:

        result = rag_service.ask(
            question=request.question
        )

        return {
            "success": True,
            "question": request.question,
            "answer": result["answer"],
            "sources": result["sources"]
        }

    except ValueError as e:

        raise HTTPException(
            status_code=404,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )








# import os
# import shutil
# import uuid
# from pathlib import Path

# from dotenv import load_dotenv
# from fastapi import FastAPI, File, HTTPException, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel

# from rag_service import RAGService


# # ============================================================
# # ENVIRONMENT
# # ============================================================

# load_dotenv()

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# if not GROQ_API_KEY:
#     raise RuntimeError("GROQ_API_KEY not found in .env")


# # ============================================================
# # FASTAPI
# # ============================================================

# app = FastAPI(
#     title="PDF RAG API",
#     description="Backend API for PDF question answering",
#     version="1.0.0"
# )


# # ============================================================
# # CORS
# # ============================================================

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
# # DIRECTORIES
# # ============================================================

# UPLOAD_DIR = Path("uploads")
# UPLOAD_DIR.mkdir(exist_ok=True)


# # ============================================================
# # RAG SERVICE
# # ============================================================

# rag_service = RAGService(
#     groq_api_key=GROQ_API_KEY
# )


# # ============================================================
# # REQUEST MODEL
# # ============================================================

# class AskRequest(BaseModel):
#     document_id: str
#     question: str


# # ============================================================
# # HEALTH CHECK
# # ============================================================

# @app.get("/")
# def root():
#     return {
#         "message": "PDF RAG API is running"
#     }


# @app.get("/health")
# def health():
#     return {
#         "status": "ok"
#     }


# # ============================================================
# # UPLOAD PDF
# # ============================================================

# @app.post("/upload")
# async def upload_pdf(
#     file: UploadFile = File(...)
# ):

#     if not file.filename.lower().endswith(".pdf"):
#         raise HTTPException(
#             status_code=400,
#             detail="Only PDF files are supported."
#         )

#     document_id = str(uuid.uuid4())

#     pdf_path = UPLOAD_DIR / f"{document_id}.pdf"

#     try:

#         with open(pdf_path, "wb") as buffer:
#             shutil.copyfileobj(
#                 file.file,
#                 buffer
#             )

#         result = rag_service.index_pdf(
#             document_id=document_id,
#             pdf_path=str(pdf_path)
#         )

#         return {
#             "success": True,
#             "document_id": document_id,
#             "filename": file.filename,
#             "pages": result["pages"],
#             "chunks": result["chunks"]
#         }

#     except Exception as e:

#         if pdf_path.exists():
#             pdf_path.unlink()

#         raise HTTPException(
#             status_code=500,
#             detail=str(e)
#         )


# # ============================================================
# # ASK QUESTION
# # ============================================================

# @app.post("/ask")
# def ask_question(request: AskRequest):

#     try:

#         result = rag_service.ask(
#             document_id=request.document_id,
#             question=request.question
#         )

#         return {
#             "success": True,
#             "document_id": request.document_id,
#             "question": request.question,
#             "answer": result["answer"],
#             "sources": result["sources"]
#         }

#     except ValueError as e:

#         raise HTTPException(
#             status_code=404,
#             detail=str(e)
#         )

#     except Exception as e:

#         raise HTTPException(
#             status_code=500,
#             detail=str(e)
#         )






# # import os
# # import shutil
# # import uuid
# # from pathlib import Path

# # from dotenv import load_dotenv
# # from fastapi import FastAPI, File, HTTPException, UploadFile
# # from pydantic import BaseModel

# # from rag_service import RAGService
# # from fastapi.middleware.cors import CORSMiddleware

# # app = FastAPI()

# # # CORS - React frontend ke liye
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=[
# #         "http://localhost:3000",
# #         "http://127.0.0.1:3000",
# #         "http://localhost:5173",
# #         "http://127.0.0.1:5173",
# #     ],
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )




# # # ============================================================
# # # ENVIRONMENT
# # # ============================================================

# # load_dotenv()

# # GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# # if not GROQ_API_KEY:
# #     raise RuntimeError("GROQ_API_KEY not found in .env")


# # # ============================================================
# # # DIRECTORIES
# # # ============================================================

# # UPLOAD_DIR = Path("uploads")
# # UPLOAD_DIR.mkdir(exist_ok=True)


# # # ============================================================
# # # FASTAPI
# # # ============================================================

# # app = FastAPI(
# #     title="PDF RAG API",
# #     description="Backend API for PDF question answering",
# #     version="1.0.0"
# # )


# # # ============================================================
# # # RAG SERVICE
# # # ============================================================

# # rag_service = RAGService(
# #     groq_api_key=GROQ_API_KEY
# # )


# # # ============================================================
# # # REQUEST MODEL
# # # ============================================================

# # class AskRequest(BaseModel):
# #     document_id: str
# #     question: str


# # # ============================================================
# # # HEALTH CHECK
# # # ============================================================

# # @app.get("/")
# # def root():
# #     return {
# #         "message": "PDF RAG API is running"
# #     }


# # @app.get("/health")
# # def health():
# #     return {
# #         "status": "ok"
# #     }


# # # ============================================================
# # # UPLOAD PDF
# # # ============================================================

# # @app.post("/upload")
# # async def upload_pdf(
# #     file: UploadFile = File(...)
# # ):

# #     if not file.filename.lower().endswith(".pdf"):
# #         raise HTTPException(
# #             status_code=400,
# #             detail="Only PDF files are supported."
# #         )

# #     document_id = str(uuid.uuid4())

# #     pdf_path = UPLOAD_DIR / f"{document_id}.pdf"

# #     try:

# #         with open(pdf_path, "wb") as buffer:
# #             shutil.copyfileobj(
# #                 file.file,
# #                 buffer
# #             )

# #         result = rag_service.index_pdf(
# #             document_id=document_id,
# #             pdf_path=str(pdf_path)
# #         )

# #         return {
# #             "success": True,
# #             "document_id": document_id,
# #             "filename": file.filename,
# #             "pages": result["pages"],
# #             "chunks": result["chunks"]
# #         }

# #     except Exception as e:

# #         if pdf_path.exists():
# #             pdf_path.unlink()

# #         raise HTTPException(
# #             status_code=500,
# #             detail=str(e)
# #         )


# # # ============================================================
# # # ASK QUESTION
# # # ============================================================

# # @app.post("/ask")
# # def ask_question(request: AskRequest):

# #     try:

# #         result = rag_service.ask(
# #             document_id=request.document_id,
# #             question=request.question
# #         )

# #         return {
# #             "success": True,
# #             "document_id": request.document_id,
# #             "question": request.question,
# #             "answer": result["answer"],
# #             "sources": result["sources"]
# #         }

# #     except ValueError as e:

# #         raise HTTPException(
# #             status_code=404,
# #             detail=str(e)
# #         )

# #     except Exception as e:

# #         raise HTTPException(
# #             status_code=500,
# #             detail=str(e)
# #         )
