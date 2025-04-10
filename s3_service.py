import boto3
import os
from dotenv import load_dotenv

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
        self.s3.upload_file(file_path, self.bucket_name, file_path)
        presigned_url = self.s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': self.bucket_name, 'Key': file_path},
            ExpiresIn=300  # 300 seconds = 5 minutes
        )

        return presigned_url



