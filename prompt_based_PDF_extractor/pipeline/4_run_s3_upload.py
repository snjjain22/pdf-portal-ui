#!/usr/bin/env python3
"""
Stage 4: S3 Uploader for PDF Extractor

Uploads extracted product crop images to AWS S3.
Adds 's3_url' field to the product JSON.

Usage:
    python 4_run_s3_upload.py --input outputs/run_X/3_seo_optimized.json --output outputs/run_X/4_s3_uploaded.json
"""

import os
import sys
import json
import logging
import argparse
import boto3
from pathlib import Path
from typing import List, Dict, Any
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class S3Uploader:
    def __init__(self):
        self.access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self.secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.bucket_name = os.getenv("S3_BUCKET_NAME")
        self.region_name = os.getenv("S3_REGION_NAME", "ap-south-1")
        
        if not all([self.access_key, self.secret_key, self.bucket_name]):
            logger.error("Missing AWS credentials in .env file")
            raise ValueError("Missing AWS credentials")
            
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region_name
            )
            logger.info(f"Initialized S3 client for bucket: {self.bucket_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Boto3 client: {e}")
            raise

    def upload_file(self, file_path: str, object_name: str) -> str:
        """Upload a file to an S3 bucket and return the public URL."""
        if not os.path.isfile(file_path):
            logger.warning(f"File not found: {file_path}")
            return None

        try:
            # Determine content type
            ext = Path(file_path).suffix.lower()
            content_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
            
            logger.info(f"Uploading {Path(file_path).name} to {object_name}...")
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=object_name,
                Body=open(file_path, 'rb'),
                ContentType=content_type,
                # Remove ACL='public-read' if bucket policies handle public access
                # For many setups, public-read ACL is required for public access
                # ACL='public-read' 
            )
            
            # Construct public URL
            # Format: https://{bucket}.s3.{region}.amazonaws.com/{key}
            url = f"https://{self.bucket_name}.s3.{self.region_name}.amazonaws.com/{object_name}"
            return url
            
        except ClientError as e:
            logger.error(f"S3 Upload error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error uploading {file_path}: {e}")
            return None

def run_s3_upload(input_file: str, output_file: str) -> Dict[str, Any]:
    """Run the S3 upload stage."""
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    logger.info(f"Loading products from {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Handle both direct list and wrapper dict formats
    if isinstance(data, dict) and "products" in data:
        products = data["products"]
        wrapper = data
    else:
        products = data
        wrapper = {"products": products}
        
    logger.info(f"Processing {len(products)} products for S3 upload...")
    
    try:
        uploader = S3Uploader()
    except Exception as e:
        logger.error(f"Skipping S3 upload due to initialization failure: {e}")
        # Return original products but with status error
        return {
            "status": "error", 
            "error": str(e), 
            "products_processed": 0,
            "uploads_success": 0
        }

    success_count = 0
    fail_count = 0
    skipped_count = 0
    
    # Determine S3 prefix based on PDF name (extracted from output path)
    # Output path structure: outputs/PDF_NAME/4_s3_uploaded.json
    # So parent is the PDF output dir
    pdf_dir_name = output_path.parent.name

    # Check for override env var (useful for testing)
    override_folder = os.getenv("S3_FOLDER_OVERRIDE")
    if override_folder:
        s3_folder = override_folder
        logger.warning(f"Using S3 folder override from env: {s3_folder}")
    else:
        # Use a cleaner version of the PDF name for S3
        s3_folder = pdf_dir_name.replace("outputs_", "").replace("_results", "")
    
    base_prefix = f"pdf_extractor/{s3_folder}"
    
    for product in products:
        crop_path = product.get("crop_path")
        
        # Check if we have a valid local image to upload
        if not crop_path:
            logger.warning(f"Product {product.get('sku', 'unknown')} has no crop_path")
            skipped_count += 1
            continue
            
        if product.get("s3_url"):
            logger.info(f"Product {product.get('sku')} already has s3_url, skipping upload")
            skipped_count += 1
            continue

        # Prepare S3 key (path)
        filename = Path(crop_path).name
        s3_key = f"{base_prefix}/{filename}"
        
        # Upload
        s3_url = uploader.upload_file(crop_path, s3_key)
        
        if s3_url:
            product["s3_url"] = s3_url
            # Also update image_src for compatibility with pipeline logic
            product["image_src"] = s3_url
            success_count += 1
        else:
            fail_count += 1
            
    # Update wrapper stats
    wrapper["s3_upload"] = {
        "success_count": success_count,
        "fail_count": fail_count,
        "skipped_count": skipped_count
    }
            
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(wrapper, f, indent=2)
        
    logger.info(f"S3 Upload Complete. Success: {success_count}, Failed: {fail_count}, Skipped: {skipped_count}")
    logger.info(f"Saved to {output_path}")
    
    return {
        "status": "success",
        "products_processed": len(products),
        "uploads_success": success_count,
        "uploads_failed": fail_count,
        "uploads_skipped": skipped_count,
        "output_file": str(output_path)
    }

def main():
    parser = argparse.ArgumentParser(description="Stage 4: S3 Upload")
    parser.add_argument("--input", required=True, help="Input JSON file (from SEO stage)")
    parser.add_argument("--output", required=True, help="Output JSON file (with s3_urls)")
    
    args = parser.parse_args()
    
    try:
        run_s3_upload(args.input, args.output)
    except Exception as e:
        logger.error(f"S3 Upload Stage Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
