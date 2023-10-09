from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import app_router


main_app = FastAPI(title='Сервер модели ML')

main_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

main_app.include_router(app_router)




