import os
import boto3
from botocore.client import Config as BotoConfig

# Basic logging configuration with file output
import sys
import logging
from logging.handlers import RotatingFileHandler

# Ensure stdout/stderr use UTF-8 on platforms where the default is cp1252 (e.g., Windows)
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    # Best-effort; if reconfigure isn't supported, continue with defaults
    pass

# Ensure log directory exists
log_file_path = "/var/log/multiLLMGrader/multiLLMGrader.log"
log_dir = os.path.dirname(log_file_path)
os.makedirs(log_dir, exist_ok=True)

# Configure logging with both file and console handlers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        # File handler with rotation
        RotatingFileHandler(
            log_file_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        ),
        # Console handler for development
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)

AWS_S3_BUCKET = os.environ.get("AWS_S3_BUCKET", "")
AWS_S3_REGION = os.environ.get("AWS_S3_REGION", "us-east-1")
AWS_S3_ENDPOINT = os.environ.get("AWS_S3_ENDPOINT", "")

def create_s3_client():
    session = boto3.session.Session()
    if AWS_S3_ENDPOINT:
        return session.client(
            "s3",
            region_name=AWS_S3_REGION,
            endpoint_url=AWS_S3_ENDPOINT,
            config=BotoConfig(s3={"addressing_style": "virtual"}),
        )
    return session.client("s3", region_name=AWS_S3_REGION)



s3_client = None
try:
    if AWS_S3_BUCKET:
        s3_client = create_s3_client()
        logger.info(f"S3 client initialized for bucket: {AWS_S3_BUCKET}")
    else:
        logger.warning("AWS_S3_BUCKET not set. S3 features disabled.")
except Exception as e:
    logger.error(f"Failed to initialize S3 client: {e}")
    s3_client = None