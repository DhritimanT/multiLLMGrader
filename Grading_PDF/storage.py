
from config import s3_client, AWS_S3_BUCKET

def s3_presign_url(bucket_key: str, expires_in: int = 3600) -> str:
    if not s3_client or not AWS_S3_BUCKET:
        raise Exception("S3 is not configured")
    return s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": AWS_S3_BUCKET, "Key": bucket_key},
        ExpiresIn=expires_in,
    )