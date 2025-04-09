from fastapi import FastAPI
from api.routes import router

app = FastAPI(title="Smart Vision API (LLaVA-Pro)")
app.include_router(router)
