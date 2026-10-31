from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.documents import router as documents_router

app = FastAPI(title="DocClear AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.include_router(documents_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "DocClear AI"}
