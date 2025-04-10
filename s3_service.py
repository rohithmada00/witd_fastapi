import boto3
import os
from dotenv import load_dotenv
import uuid
from botocore.exceptions import ClientError


class S3AccessService:
    def __init__(self):
        load_dotenv()
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION")
        )
        self.bucket_name = os.getenv("BUCKET_NAME")

    def upload_file(self, file_path):
        _, file_extension = os.path.splitext(file_path)
        file_key = f"uploads/{uuid.uuid4()}{file_extension}"

        try:
            self.s3.upload_file(file_path, self.bucket_name, file_key)
            presigned_url = self.s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': file_key},
                ExpiresIn=300  # 300 seconds = 5 minutes
            )

            return {
                "resCode": 200,
                "resMsg": "File uploaded successfully.",
                "resData": {
                    "fileUrl": presigned_url
                }
            }

        except ClientError as e:
            return {
                "resCode": 500,
                "resMsg": "File upload failed.",
                "resData": {
                    "error": str(e)
                }
            }

