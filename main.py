from fastapi import FastAPI
from s3_service import S3AccessService 
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
s3_service = S3AccessService()

# TODO: Add WITD frontend domain here 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_to_s3(file_path: str):
    return s3_service.upload_file(file_path)


