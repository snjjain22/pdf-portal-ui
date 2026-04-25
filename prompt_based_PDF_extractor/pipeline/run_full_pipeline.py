#!/usr/bin/env python3
"""
Full Pipeline Orchestrator for Prompt-Based PDF Extractor

Runs all pipeline stages in sequence:
1. Stage 0: PDF to Images
2. Stage 1: Detection (Grounded SAM + VLM classification + OCR)
3. Stage 2: Prompt-Based Field Mapping (THE KEY INNOVATION)
4. Stage 3: SEO Content Generation
5. Stage 4: S3 Upload (product images)
6. Stage 5: Application Image Generation (room + furniture renders)
7. Stage 6: Matrixify CSV Export
8. Stage 7: Validate & Clean

Usage:
    python run_full_pipeline.py <pdf_path>
    python run_full_pipeline.py PDFs/my_catalog.pdf
    python run_full_pipeline.py --all  # Process all PDFs
"""

import argparse
import sys
import time
import shutil
import logging
import importlib.util
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import pipeline modules by loading them directly (since they have numeric prefixes)
def load_module(name, path):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

pipeline_dir = Path(__file__).parent

# Load pipeline stage modules
stage0 = load_module("stage0", pipeline_dir / "0_run_pdf_extraction.py")
stage1 = load_module("stage1", pipeline_dir / "1_run_detection.py")
stage2 = load_module("stage2", pipeline_dir / "2_run_field_mapping.py")
stage3 = load_module("stage3", pipeline_dir / "3_run_seo.py")
stage4 = load_module("stage4", pipeline_dir / "4_run_s3_upload.py")
stage5 = load_module("stage5", pipeline_dir / "5_run_app_images.py")
stage6 = load_module("stage6", pipeline_dir / "6_run_matrixify.py")
stage7 = load_module("stage7", pipeline_dir / "7_validate_and_clean.py")
stage8 = load_module("stage8", pipeline_dir / "8_generate_qr_codes.py")

# Get the main functions
extract_pdf = stage0.extract_pdf
run_detection = stage1.run_detection
run_field_mapping = stage2.run_field_mapping
run_seo = stage3.run_seo
run_s3_upload = stage4.run_s3_upload
run_app_images = stage5.run_app_images
run_matrixify = stage6.run_matrixify
validate_and_clean = stage7.validate_and_clean
run_qr_generation = stage8.run_qr_generation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_full_pipeline(
    pdf_path: str,
    config_dir: str = None,
    skip_stages: list = None,
    limit: int = None,
    vendor: str = None,
    generate_qr: bool = False,
) -> dict:
    """
    Run the complete PDF extraction pipeline.
    
    Args:
        pdf_path: Path to PDF file
        config_dir: Optional path to config directory
        skip_stages: List of stage numbers to skip (e.g., [0] to skip extraction)
        limit: Optional limit on number of pages to process (for testing)
    
    Returns:
        Dict with pipeline results
    """
    pdf_path = Path(pdf_path)
    skip_stages = skip_stages or []
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    # Clean up old output directory if it exists AND we are not skipping stages
    pdf_name = pdf_path.stem.replace(" ", "_").replace("-", "_").upper() + "_PDF"
    old_output_dir = Path(__file__).parent.parent / "outputs" / pdf_name
    
    if old_output_dir.exists() and not skip_stages:
        logger.info(f"Cleaning up old output: {old_output_dir}")
        shutil.rmtree(old_output_dir)
        logger.info("✓ Old output deleted")
    elif old_output_dir.exists() and skip_stages:
        logger.info(f"Resuming in existing output directory: {old_output_dir}")
    
    start_time = time.time()
    
    logger.info("=" * 70)
    logger.info("PROMPT-BASED PDF EXTRACTOR - FULL PIPELINE")
    logger.info("=" * 70)
    logger.info(f"PDF: {pdf_path.name}")
    if vendor:
        logger.info(f"Vendor: {vendor}")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    
    results = {
        "pdf_path": str(pdf_path),
        "pdf_name": pdf_path.stem,
        "start_time": datetime.now().isoformat(),
        "stages": {}
    }
    
    output_dir = None
    
    # Stage 0: PDF Extraction
    if 0 not in skip_stages:
        logger.info("\n" + "=" * 50)
        logger.info("STAGE 0: PDF EXTRACTION")
        logger.info("=" * 50)
        
        try:
            stage_start = time.time()
            stage_result = extract_pdf(str(pdf_path), config_dir, limit=limit)
            output_dir = stage_result.get("output_dir")
            
            results["stages"]["0_pdf_extraction"] = {
                "status": "success",
                "duration_sec": round(time.time() - stage_start, 2),
                "page_count": stage_result.get("page_count")
            }
            logger.info(f"✓ Stage 0 complete: {stage_result.get('page_count')} pages")
            
        except Exception as e:
            logger.error(f"✗ Stage 0 failed: {e}")
            results["stages"]["0_pdf_extraction"] = {"status": "error", "error": str(e)}
            results["status"] = "failed"
            return results
    else:
        logger.info("\n[SKIPPED] Stage 0: PDF Extraction")
        # Determine output dir from PDF name
        from common.output_manager import OutputManager
        om = OutputManager("outputs")
        output_dir = str(om.init_pdf_output(str(pdf_path)))
    
    # Stage 1: Detection
    if 1 not in skip_stages:
        logger.info("\n" + "=" * 50)
        logger.info("STAGE 1: DETECTION")
        logger.info("=" * 50)
        
        try:
            stage_start = time.time()
            stage_result = run_detection(output_dir, config_dir, limit=limit)
            
            results["stages"]["1_detection"] = {
                "status": "success",
                "duration_sec": round(time.time() - stage_start, 2),
                "product_count": stage_result.get("product_count")
            }
            logger.info(f"✓ Stage 1 complete: {stage_result.get('product_count')} products detected")
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"✗ Stage 1 failed: {type(e).__name__}: {e}")
            logger.error(f"Traceback:\n{tb}")
            results["stages"]["1_detection"] = {"status": "error", "error": str(e), "traceback": tb}
            results["status"] = "failed"
            return results
    else:
        logger.info("\n[SKIPPED] Stage 1: Detection")
    
    # Stage 2: Field Mapping (THE KEY PROMPT-BASED STAGE)
    if 2 not in skip_stages:
        logger.info("\n" + "=" * 50)
        logger.info("STAGE 2: PROMPT-BASED FIELD MAPPING")
        logger.info("(This is where LLM prompts replace hardcoded Python!)")
        logger.info("=" * 50)
        
        try:
            stage_start = time.time()
            stage_result = run_field_mapping(output_dir, config_dir)
            
            results["stages"]["2_field_mapping"] = {
                "status": "success",
                "duration_sec": round(time.time() - stage_start, 2),
                "product_count": stage_result.get("product_count")
            }
            logger.info(f"✓ Stage 2 complete: {stage_result.get('product_count')} products mapped")
            
        except Exception as e:
            logger.error(f"✗ Stage 2 failed: {e}")
            results["stages"]["2_field_mapping"] = {"status": "error", "error": str(e)}
            results["status"] = "failed"
            return results
    else:
        logger.info("\n[SKIPPED] Stage 2: Field Mapping")
    
    # Stage 3: SEO Generation
    if 3 not in skip_stages:
        logger.info("\n" + "=" * 50)
        logger.info("STAGE 3: SEO CONTENT GENERATION")
        logger.info("=" * 50)
        
        try:
            stage_start = time.time()
            stage_result = run_seo(output_dir, config_dir)
            
            results["stages"]["3_seo"] = {
                "status": "success",
                "duration_sec": round(time.time() - stage_start, 2),
                "seo_success": stage_result.get("seo_success")
            }
            logger.info(f"✓ Stage 3 complete: SEO generated for {stage_result.get('seo_success')} products")
            
        except Exception as e:
            logger.error(f"✗ Stage 3 failed: {e}")
            results["stages"]["3_seo"] = {"status": "error", "error": str(e)}
            results["status"] = "failed"
            return results
    else:
        logger.info("\n[SKIPPED] Stage 3: SEO Generation")
    
    # Stage 4: S3 Upload
    if 4 not in skip_stages:
        logger.info("\n" + "=" * 50)
        logger.info("STAGE 4: S3 UPLOAD")
        logger.info("=" * 50)
        
        try:
            stage_start = time.time()
            # Input is from SEO stage (3), Output is for Matrixify stage (5)
            # Both live in the 'data' subdirectory of the output folder
            input_file = Path(output_dir) / "data" / "3_seo_results.json"
            output_file = Path(output_dir) / "data" / "4_s3_uploaded.json"
            
            stage_result = run_s3_upload(str(input_file), str(output_file))
            
            results["stages"]["4_s3_upload"] = {
                "status": "success",
                "duration_sec": round(time.time() - stage_start, 2),
                "uploads_success": stage_result.get("uploads_success"),
                "uploads_skipped": stage_result.get("uploads_skipped")
            }
            logger.info(f"✓ Stage 4 complete: {stage_result.get('uploads_success')} images uploaded, {stage_result.get('uploads_skipped')} skipped")
            
        except Exception as e:
            logger.error(f"✗ Stage 4 failed: {e}")
            results["stages"]["4_s3_upload"] = {"status": "error", "error": str(e)}
            # Consider if failed upload should block pipeline or just log error
            # For now we block, as image URLs are critical
            results["status"] = "failed"
            return results
    else:
        logger.info("\n[SKIPPED] Stage 4: S3 Upload")

    # Stage 5: Application Image Generation
    if 5 not in skip_stages:
        logger.info("\n" + "=" * 50)
        logger.info("STAGE 5: APPLICATION IMAGE GENERATION")
        logger.info("=" * 50)
        
        try:
            stage_start = time.time()
            stage_result = run_app_images(output_dir, config_dir)
            
            results["stages"]["5_app_images"] = {
                "status": "success",
                "duration_sec": round(time.time() - stage_start, 2),
                "images_success": stage_result.get("images_success"),
                "images_failed": stage_result.get("images_failed"),
                "images_skipped": stage_result.get("images_skipped")
            }
            logger.info(f"✓ Stage 5 complete: {stage_result.get('images_success')} application images generated")
            
        except Exception as e:
            logger.error(f"✗ Stage 5 failed: {e}")
            results["stages"]["5_app_images"] = {"status": "error", "error": str(e)}
            # Non-blocking: continue pipeline even if app images fail
            logger.warning("Continuing pipeline without application images...")
    else:
        logger.info("\n[SKIPPED] Stage 5: Application Image Generation")

    # Stage 6: Matrixify Export
    if 6 not in skip_stages:
        logger.info("\n" + "=" * 50)
        logger.info("STAGE 6: MATRIXIFY CSV EXPORT")
        logger.info("=" * 50)
        
        try:
            stage_start = time.time()
            stage_result = run_matrixify(output_dir, config_dir, vendor=vendor)
            
            results["stages"]["6_matrixify"] = {
                "status": "success",
                "duration_sec": round(time.time() - stage_start, 2),
                "csv_path": stage_result.get("csv_path"),
                "product_count": stage_result.get("product_count")
            }
            logger.info(f"✓ Stage 6 complete: {stage_result.get('product_count')} products exported")
            
        except Exception as e:
            logger.error(f"✗ Stage 6 failed: {e}")
            results["stages"]["6_matrixify"] = {"status": "error", "error": str(e)}
            results["status"] = "failed"
            return results
    else:
        logger.info("\n[SKIPPED] Stage 6: Matrixify Export")
    
    # Stage 7: Validate & Clean
    if 7 not in skip_stages:
        logger.info("\n" + "=" * 50)
        logger.info("STAGE 7: VALIDATE & CLEAN MATRIXIFY CSV")
        logger.info("=" * 50)
        
        try:
            stage_start = time.time()
            stage_result = validate_and_clean(output_dir, config_dir)
            
            results["stages"]["7_validation"] = {
                "status": "success",
                "duration_sec": round(time.time() - stage_start, 2),
                "csvs_processed": stage_result.get("csvs_processed")
            }
            logger.info(f"✓ Stage 7 complete: {stage_result.get('csvs_processed')} CSV(s) validated")
            
        except Exception as e:
            logger.error(f"✗ Stage 7 failed: {e}")
            results["stages"]["7_validation"] = {"status": "error", "error": str(e)}
            # Don't return here - validation failure shouldn't block the pipeline
            logger.warning("Continuing despite validation error...")
    else:
        logger.info("\n[SKIPPED] Stage 7: Validate & Clean")

    # Stage 8: QR Code Generation (optional, enabled with --qr flag)
    if generate_qr and 8 not in skip_stages:
        logger.info("\n" + "=" * 50)
        logger.info("STAGE 8: QR CODE GENERATION")
        logger.info("=" * 50)

        try:
            stage_start = time.time()
            stage_result = run_qr_generation(output_dir)

            results["stages"]["8_qr_generation"] = {
                "status": "success",
                "duration_sec": round(time.time() - stage_start, 2),
                "product_count": stage_result.get("product_count"),
                "qr_dir": stage_result.get("qr_dir"),
                "pdf_path": stage_result.get("pdf_path"),
            }
            logger.info(f"✓ Stage 8 complete: {stage_result.get('product_count')} QR codes → {stage_result.get('qr_dir')}")

        except Exception as e:
            logger.error(f"✗ Stage 8 failed: {e}")
            results["stages"]["8_qr_generation"] = {"status": "error", "error": str(e)}
            logger.warning("Continuing despite QR generation error...")
    else:
        logger.info("\n[SKIPPED] Stage 8: QR Code Generation")

    # Calculate total time
    total_time = time.time() - start_time
    results["total_duration_sec"] = round(total_time, 2)
    results["status"] = "success"
    results["output_dir"] = output_dir
    results["end_time"] = datetime.now().isoformat()
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"PDF: {pdf_path.name}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    logger.info("")
    logger.info("Stage Summary:")
    for stage_name, stage_data in results["stages"].items():
        status = "✓" if stage_data.get("status") == "success" else "✗"
        duration = stage_data.get("duration_sec", 0)
        logger.info(f"  {status} {stage_name}: {duration}s")
    logger.info("=" * 70)
    
    return results


def process_all_pdfs(pdfs_dir: str = "PDFs", config_dir: str = None):
    """Process all PDFs in the PDFs directory."""
    pdfs_dir = Path(pdfs_dir)
    
    if not pdfs_dir.exists():
        pdfs_dir.mkdir(parents=True)
        logger.info(f"Created PDFs directory: {pdfs_dir}")
        logger.info("Please add PDF files and run again.")
        return []
    
    pdf_files = list(pdfs_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdfs_dir}")
        return []
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    results = []
    for pdf_path in pdf_files:
        try:
            result = run_full_pipeline(str(pdf_path), config_dir)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")
            results.append({
                "pdf_path": str(pdf_path),
                "status": "error",
                "error": str(e)
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run the full prompt-based PDF extraction pipeline"
    )
    parser.add_argument(
        "pdf_path",
        nargs="?",
        help="Path to PDF file"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all PDFs in PDFs/ directory"
    )
    parser.add_argument(
        "--pdfs-dir",
        default="PDFs",
        help="Directory containing PDF files"
    )
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Path to config directory"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of pages to process (for testing)"
    )
    parser.add_argument(
        "--skip",
        type=int,
        nargs="+",
        default=[],
        help="Stage numbers to skip: 0=PDF extraction, 1=Detection, 2=VLM, 3=SEO, 4=S3 Upload, 5=App Images, 6=Matrixify, 7=Validation"
    )
    parser.add_argument(
        "--vendor",
        type=str,
        default=None,
        help="Vendor / brand name to write into the Matrixify CSV (e.g. 'Xterio')"
    )
    parser.add_argument(
        "--qr",
        action="store_true",
        help="Also generate QR code sheet after pipeline completes (Stage 8)"
    )

    args = parser.parse_args()

    # If vendor not supplied via flag, ask interactively
    vendor = args.vendor
    if not vendor:
        try:
            vendor = input("Enter vendor name (press Enter to use default 'Next Level Decor'): ").strip()
        except EOFError:
            vendor = ""
        if not vendor:
            vendor = None  # Fall back to schema default
    
    if args.all:
        results = process_all_pdfs(args.pdfs_dir, args.config_dir)
        successful = sum(1 for r in results if r.get("status") == "success")
        logger.info(f"\nProcessed {successful}/{len(results)} PDFs successfully")
    elif args.pdf_path:
        run_full_pipeline(args.pdf_path, args.config_dir, args.skip, limit=args.limit, vendor=vendor, generate_qr=args.qr)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
