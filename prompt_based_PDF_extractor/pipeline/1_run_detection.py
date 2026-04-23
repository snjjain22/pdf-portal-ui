#!/usr/bin/env python3
"""
Stage 1: Detection Pipeline

Runs Grounded SAM detection on page images to find product swatches.
Also classifies pages using VLM and extracts SKUs using OCR.

Usage:
    python 1_run_detection.py <pdf_output_dir>
    python 1_run_detection.py outputs/MY_CATALOG_PDF
"""

import argparse
import json
import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.config_manager import get_config_manager
from common.output_manager import OutputManager
from common.grounded_sam_detector import GroundedSAMDetector, VLMPageClassifier
from common.sku_extractor import SKUExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_detection(output_dir: str, config_dir: str = None, limit: int = None) -> Dict[str, Any]:
    """
    Run detection on all page images in a PDF output directory.
    
    Args:
        output_dir: Path to PDF output directory (e.g., outputs/MY_CATALOG_PDF)
        config_dir: Optional path to config directory
        limit: Optional limit on number of pages to process
    
    Returns:
        Dict with detection results
    """
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
    logger.info(f"=" * 60)
    logger.info(f"Stage 1: Detection")
    logger.info(f"Directory: {output_dir.name}")
    logger.info(f"=" * 60)
    
    # Determine PDF name for config overrides
    pdf_name = output_dir.name.replace("_PDF", "")
    
    # Load configuration with PDF-specific overrides
    config_manager = get_config_manager(config_dir)
    website_config = config_manager.load_website_config(pdf_name=pdf_name)
    detection_config = website_config.get("grounded_sam_detection", {})
    
    # Initialize output manager
    output_manager = OutputManager(str(output_dir.parent))
    output_manager._current_pdf_dir = output_dir
    output_manager._current_pdf_name = pdf_name
    
    # Get page images (support both PNG and JPEG formats)
    pages_dir = output_dir / "pages"
    page_images = sorted(list(pages_dir.glob("page_*.png")) + list(pages_dir.glob("page_*.jpeg")) + list(pages_dir.glob("page_*.jpg")))
    
    if not page_images:
        logger.warning(f"No page images found in {pages_dir}")
        return {"status": "error", "error": "No page images found"}
    
    logger.info(f"Found {len(page_images)} page images")
    
    if limit:
        page_images = page_images[:limit]
        logger.info(f"Limiting to first {limit} pages for testing")
    
    # =========================================================================
    # PHASE 1: Dynamic Prompt Generation (NEW!)
    # =========================================================================
    logger.info(f"\n{'=' * 60}")
    logger.info(f"PHASE 1: Analyzing catalog to generate optimal detection prompt")
    logger.info(f"{'=' * 60}")
    
    # Initialize page classifier for prompt generation
    page_classifier = VLMPageClassifier()
    
    # Generate dynamic prompt by analyzing sample pages
    # This iterates through pages until it finds 5 ACTUAL PRODUCT PAGES
    # (skips covers, application photos, text pages automatically)
    page_paths_str = [str(p) for p in page_images]
    dynamic_prompt = page_classifier.analyze_sample_pages_for_prompt(
        page_paths_str, 
        num_product_pages=5
    )
    
    # Determine which prompt to use
    config_prompt = detection_config.get("prompt", "product swatch . material sample")
    if dynamic_prompt:
        detection_prompt = dynamic_prompt
        logger.info(f"\n✓ Using VLM-generated prompt: \"{detection_prompt}\"")
    else:
        detection_prompt = config_prompt
        logger.info(f"\n→ Using config default prompt: \"{detection_prompt}\"")
    
    # =========================================================================
    # PHASE 2: Initialize Detector with Optimized Prompt
    # =========================================================================
    logger.info(f"\n{'=' * 60}")
    logger.info(f"PHASE 2: Running detection on all pages")
    logger.info(f"{'=' * 60}")
    
    # Initialize detector with the determined prompt
    models_dir = Path(__file__).parent.parent / "models"
    detector = GroundedSAMDetector(
        dino_weights=str(models_dir / detection_config.get("dino_weights", "groundingdino_swint_ogc.pth")),
        sam_checkpoint=str(models_dir / detection_config.get("sam_checkpoint", "sam_vit_b_01ec64.pth")),
        prompt=detection_prompt,  # Use the dynamic or config prompt
        box_threshold=detection_config.get("box_threshold", 0.15),
        text_threshold=detection_config.get("text_threshold", 0.20),
        nms_threshold=detection_config.get("nms_threshold", 0.3),
        min_area_pct=detection_config.get("min_area_pct", 2.0),
        max_area_pct=detection_config.get("max_area_pct", 70.0),
        min_aspect=detection_config.get("min_aspect", 0.4), # Default 0.4
        use_sam=detection_config.get("use_sam_refinement", False)
    )
    
    # Set up page classifier with classification prompt (different from prompt generation)
    classification_prompt = config_manager.get_prompt_template("page_classification")
    page_classifier.set_prompt(classification_prompt)
    
    # Initialize SKU extractor with config
    # Re-use already loaded config or re-load with pdf_name if needed
    # We already have website_config from earlier, let's just use it?
    # Actually, let's re-load consistently to be safe if earlier code mutated it? 
    # But usually config should be immutable. Let's use pdf_name.
    website_config_ocr = config_manager.load_website_config(pdf_name=pdf_name)
    ocr_config = website_config_ocr.get("ocr_extraction", {})
    sku_extractor = SKUExtractor(
        engine=ocr_config.get("engine", "easyocr"),
        languages=ocr_config.get("languages", ["en"]),
        sku_patterns=ocr_config.get("sku_patterns"),
        search_radius_ratio=ocr_config.get("search_radius_ratio", 0.3),
        prefer_position=ocr_config.get("prefer_position", "below"),
        auto_sku_template=ocr_config.get("auto_sku_template", "{pdf_name}-P{page:03d}-S{swatch:02d}")
    )
    
    # Process each page
    all_results = []
    products = []
    product_index = 0
    pdf_name = output_manager._current_pdf_name.replace("_", "-")
    
    for page_idx, page_path in enumerate(page_images):
        page_num = page_idx + 1
        logger.info(f"\nProcessing page {page_num}/{len(page_images)}: {page_path.name}")
        
        # Classify page with VLM
        page_type = page_classifier.classify(str(page_path))
        logger.info(f"  Page type: {page_type}")
        
        if page_type in ["application", "skip"]:
            logger.info(f"  Skipping {page_type} page")
            all_results.append({
                "page_num": page_num,
                "page_path": str(page_path),
                "page_type": page_type,
                "detections": [],
                "skipped": True
            })
            continue
        
        # Run detection
        result = detector.detect(str(page_path))
        logger.info(f"  Found {result['detection_count']} detections")
        
        # RETRY FALLBACK: If product page has 0 detections, generate page-specific prompt
        if result["detection_count"] == 0 and page_type == "product":
            logger.info(f"  ⚡ Retrying with page-specific VLM prompt + lower threshold...")
            
            page_specific_prompt = page_classifier.generate_page_specific_prompt(str(page_path))
            
            if page_specific_prompt:
                # Create a temporary detector with the page-specific prompt and lower threshold
                retry_detector = GroundedSAMDetector(
                    dino_weights=str(models_dir / detection_config.get("dino_weights", "groundingdino_swint_ogc.pth")),
                    sam_checkpoint=str(models_dir / detection_config.get("sam_checkpoint", "sam_vit_b_01ec64.pth")),
                    prompt=page_specific_prompt,
                    box_threshold=0.10,  # Even lower for retry
                    text_threshold=0.15,
                    nms_threshold=detection_config.get("nms_threshold", 0.3),
                    min_area_pct=detection_config.get("min_area_pct", 2.0),
                    max_area_pct=detection_config.get("max_area_pct", 70.0),
                    min_aspect=detection_config.get("min_aspect", 0.4),
                    use_sam=False
                )
                # Share the already-loaded DINO model to avoid reloading
                retry_detector._dino_model = detector._dino_model
                retry_detector._device = detector._device
                
                result = retry_detector.detect(str(page_path))
                logger.info(f"  ⚡ Retry found {result['detection_count']} detections")
        
        if result["detection_count"] == 0:
            all_results.append({
                "page_num": page_num,
                "page_path": str(page_path),
                "page_type": page_type,
                "detections": [],
                "skipped": False
            })
            continue
        
        # Crop detections
        detections = detector.crop_detections(
            str(page_path),
            result["detections"],
            str(output_manager.products_dir),
            padding=5
        )
        
        # Extract SKUs using OCR on FULL PAGE and spatial matching
        # This properly associates SKU labels (below/beside swatches) with detections
        detections = sku_extractor.extract_skus(
            image_path=str(page_path),
            detections=detections,
            pdf_name=pdf_name,
            page_number=page_num
        )
        
        # Rename crop files to use SKU as filename
        import shutil
        filtered_detections = []
        
        for det in detections:
            crop_path = det.get("crop_path")
            
            # 1. VERIFY CROP IS A PRODUCT SWATCH (Filtering Step)
            if crop_path and Path(crop_path).exists():
                is_valid = page_classifier.verify_crop(crop_path)
                if not is_valid:
                    logger.info(f"  Dropped non-swatch crop (application/noise): {Path(crop_path).name}")
                    # Delete the rejected file
                    try:
                        os.remove(crop_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete rejected crop: {e}")
                    continue
            
            # 2. Rename file with SKU
            sku = det.get("sku")
            if crop_path and sku and Path(crop_path).exists():
                # Create new filename using SKU
                old_path = Path(crop_path)
                # Sanitize SKU for filename (remove invalid chars)
                safe_sku = "".join(c if c.isalnum() or c in "-_" else "_" for c in sku)
                new_crop_path = old_path.parent / f"{safe_sku}.png"
                
                # Handle duplicates by adding suffix
                if new_crop_path.exists() and str(new_crop_path) != str(old_path):
                    counter = 1
                    while new_crop_path.exists():
                        new_crop_path = old_path.parent / f"{safe_sku}_{counter}.png"
                        counter += 1
                    # Update SKU to reflect the unique suffix (e.g., 442246 -> 442246-1)
                    det["sku"] = f"{sku}-{counter - 1}"
                
                # Rename the file
                shutil.move(str(old_path), str(new_crop_path))
                det["crop_path"] = str(new_crop_path)
                logger.debug(f"Renamed {old_path.name} -> {new_crop_path.name}")
            else:
                logger.warning(f"Could not rename {crop_path}: SKU='{sku}', Exists={Path(crop_path).exists() if crop_path else 'N/A'}")

            filtered_detections.append(det)
        
        detections = filtered_detections
        
        # Create product entries
        for det in detections:
            crop_path = det.get("crop_path")
            if crop_path:
                # Create product entry
                product = {
                    "product_index": product_index,
                    "sku": det.get("sku", f"{pdf_name}-P{page_num:03d}-S{det.get('index', 0):02d}"),
                    "page_num": page_num,
                    "page_path": str(page_path),
                    "crop_path": crop_path,
                    "bbox": det.get("bbox_pixels"),
                    "confidence": det.get("confidence"),
                    "page_type": page_type,
                    "sku_source": det.get("sku_source", "auto")
                }
                products.append(product)
                product_index += 1
        
        # Save debug visualization
        debug_path = output_dir / "data" / f"page_{page_num:03d}_detections.png"
        detector.visualize_detections(str(page_path), detections, str(debug_path))
        
        all_results.append({
            "page_num": page_num,
            "page_path": str(page_path),
            "page_type": page_type,
            "detections": detections,
            "detection_count": len(detections),
            "skipped": False
        })
    
    # Save results
    result = {
        "output_dir": str(output_dir),
        "page_count": len(page_images),
        "product_count": len(products),
        "detection_prompt": detection_prompt,  # Save the prompt used (dynamic or config)
        "prompt_source": "vlm_generated" if dynamic_prompt else "config_default",
        "pages": all_results,
        "products": products,
        "status": "success"
    }
    
    output_manager.save_json(result, "1_detection_results.json")
    output_manager.log("detection", f"Detected {len(products)} products from {len(page_images)} pages", "INFO")
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Detection complete!")
    logger.info(f"  Pages processed: {len(page_images)}")
    logger.info(f"  Products detected: {len(products)}")
    logger.info(f"  Results saved to: {output_manager.data_dir / '1_detection_results.json'}")
    logger.info(f"{'=' * 60}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Run detection on PDF page images"
    )
    parser.add_argument(
        "output_dir",
        help="Path to PDF output directory (e.g., outputs/MY_CATALOG_PDF)"
    )
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Path to config directory"
    )
    
    args = parser.parse_args()
    
    try:
        run_detection(args.output_dir, args.config_dir)
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise


if __name__ == "__main__":
    main()
