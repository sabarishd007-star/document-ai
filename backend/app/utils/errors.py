from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

class DocuMindException(Exception):
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code

def register_exception_handlers(app: FastAPI):
    @app.exception_handler(DocuMindException)
    async def documind_exception_handler(request: Request, exc: DocuMindException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.message}
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"error": f"An unexpected error occurred: {str(exc)}"}
        )

