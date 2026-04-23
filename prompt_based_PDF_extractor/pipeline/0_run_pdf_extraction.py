#!/usr/bin/env python3
"""
Stage 0: PDF to Image Extraction

Converts PDF catalog pages to high-resolution images for processing.

Usage:
    python 0_run_pdf_extraction.py <pdf_path>
    python 0_run_pdf_extraction.py --all  # Process all PDFs in PDFs/ folder
"""

import argparse
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.config_manager import get_config_manager
from common.output_manager import OutputManager
from common.pdf_converter import PDFConverter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_pdf(pdf_path: str, config_dir: str = None, limit: int = None) -> dict:
    """
    Extract pages from a PDF as images.
    
    Args:
        pdf_path: Path to PDF file
        config_dir: Optional path to config directory
        limit: Optional limit on number of pages to process
    
    Returns:
        Dict with extraction results
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    logger.info(f"=" * 60)
    logger.info(f"Stage 0: PDF Extraction")
    logger.info(f"PDF: {pdf_path.name}")
    logger.info(f"=" * 60)
    
    # Load configuration
    config_manager = get_config_manager(config_dir)
    settings = config_manager.get_output_settings()
    website_config = config_manager.load_website_config()
    
    # Get PDF extraction settings
    pdf_config = website_config.get("pdf_extraction", {})
    dpi = pdf_config.get("dpi", 600)  # 600 DPI for high quality
    image_format = pdf_config.get("image_format", "jpeg").lower()
    poppler_path = pdf_config.get("poppler_path")
    
    # Initialize output manager
    output_manager = OutputManager(settings.get("output_dir", "outputs"))
    output_dir = output_manager.init_pdf_output(str(pdf_path))
    
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"DPI: {dpi}")
    
    # Convert PDF to images
    converter = PDFConverter(dpi=dpi, poppler_path=poppler_path)
    
    page_count = converter.get_page_count(str(pdf_path))
    logger.info(f"PDF has {page_count} pages")
    
    if limit:
        logger.info(f"Limiting to first {limit} pages")
    
    image_paths = converter.convert_pdf_to_images(
        str(pdf_path),
        str(output_manager.pages_dir),
        fmt=image_format,
        last_page=limit
    )
    
    # Save extraction metadata
    result = {
        "pdf_path": str(pdf_path),
        "pdf_name": pdf_path.stem,
        "page_count": page_count,
        "dpi": dpi,
        "output_dir": str(output_dir),
        "page_images": [str(p) for p in image_paths],
        "status": "success"
    }
    
    output_manager.save_json(result, "0_extraction_results.json")
    output_manager.log("pdf_extraction", f"Extracted {page_count} pages", "INFO")
    
    logger.info(f"✓ Extraction complete: {page_count} pages")
    logger.info(f"Images saved to: {output_manager.pages_dir}")
    
    return result


def process_all_pdfs(pdfs_dir: str = "PDFs", config_dir: str = None):
    """Process all PDFs in the PDFs directory."""
    pdfs_dir = Path(pdfs_dir)
    
    if not pdfs_dir.exists():
        pdfs_dir.mkdir(parents=True)
        logger.info(f"Created PDFs directory: {pdfs_dir}")
        logger.info("Please add PDF files to this directory and run again.")
        return []
    
    pdf_files = list(pdfs_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdfs_dir}")
        return []
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    results = []
    for pdf_path in pdf_files:
        try:
            result = extract_pdf(str(pdf_path), config_dir)
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
        description="Stage 0: Extract PDF pages to images"
    )
    parser.add_argument(
        "pdf_path",
        nargs="?",
        help="Path to PDF file (or --all to process all)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all PDFs in PDFs/ directory"
    )
    parser.add_argument(
        "--pdfs-dir",
        default="PDFs",
        help="Directory containing PDF files (default: PDFs)"
    )
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Path to config directory"
    )
    
    args = parser.parse_args()
    
    if args.all:
        results = process_all_pdfs(args.pdfs_dir, args.config_dir)
        successful = sum(1 for r in results if r.get("status") == "success")
        logger.info(f"\nProcessed {successful}/{len(results)} PDFs successfully")
    elif args.pdf_path:
        extract_pdf(args.pdf_path, args.config_dir)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
