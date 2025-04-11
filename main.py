import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from eda_report_generator import EDAReportGenerator
from s3_service import S3AccessService 
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os

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
async def upload_dataset(file: UploadFile = File(...)):
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file.file)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file.file)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a .csv or .xlsx file.")
        
        report_path = f"uploads/{uuid.uuid4()}.pdf"
        EDAReportGenerator.generate(df, report_path)
        s3_path = s3_service.upload_file(report_path)
        os.remove(report_path)

        return {
            "resCode": 200,
            "resMsg": "File uploaded and parsed successfully.",
            "resData": {
                "report_url": s3_path,
                "ttl": "5",
            }
        }

    except Exception as e:
        return {
            "resCode": 500,
            "resMsg": "Error reading the file.",
            "resData": {
                "error": str(e)
            }
        }


 