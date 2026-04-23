#!/usr/bin/env python3
"""
Stage 2: Prompt-Based Field Mapping

The KEY stage that uses LLM prompts instead of hardcoded Python logic.
Takes detection results and derives all product fields using natural language prompts.

This replaces ~250 lines of hardcoded dictionaries (FINISH_MAP, APPEARANCE_KEYWORDS,
COLOR_MAP, etc.) with a single comprehensive Jinja2 prompt template.

Usage:
    python 2_run_field_mapping.py <pdf_output_dir>
    python 2_run_field_mapping.py outputs/MY_CATALOG_PDF
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.config_manager import get_config_manager
from common.output_manager import OutputManager
from common.llm_client import LLMClient, get_text_llm_client, get_vision_llm_client

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PromptBasedFieldMapper:
    """
    Uses LLM prompts to derive product fields from raw data.
    
    Replaces hardcoded Python dictionaries with natural language rules
    in Jinja2 prompt templates.
    """
    
    def __init__(self, config_manager):
        """
        Initialize field mapper.
        
        Args:
            config_manager: ConfigManager instance
        """
        self.config_manager = config_manager
        
        # Load configuration
        website_config = config_manager.load_website_config()
        llm_config = website_config.get("llm", {})
        
        # Initialize LLM clients
        # Text LLM for field mapping (DeepSeek - cheap and fast)
        self.text_llm = get_text_llm_client(
            model=llm_config.get("text_model", "deepseek-chat"),
            max_retries=llm_config.get("max_retries", 3),
            timeout=llm_config.get("timeout", 120)
        )
        
        # Vision LLM for color extraction from images (OpenRouter/Llama-4)
        self.vision_llm = get_vision_llm_client(
            model=llm_config.get("vision_model", "meta-llama/llama-4-maverick"),
            max_retries=llm_config.get("max_retries", 3),
            timeout=llm_config.get("timeout", 120)
        )
        
        # Load prompt templates
        self.field_mapping_prompt = config_manager.get_prompt_template("field_mapping_derivation")
        self.vlm_classification_prompt = config_manager.get_prompt_template("vlm_classification")
        
        # Load product defaults
        self.product_defaults = config_manager.get_product_defaults()
        
        # Load taxonomies for context
        self.taxonomies = config_manager.load_all_taxonomies()
        
        logger.info("PromptBasedFieldMapper initialized")
        logger.info(f"  Text LLM: {self.text_llm.model}")
        logger.info(f"  Vision LLM: {self.vision_llm.model}")
    
    def extract_visual_attributes(self, crop_path: str) -> Dict[str, Any]:
        """
        Use VLM to extract visual attributes from product image.
        
        Args:
            crop_path: Path to cropped product image
        
        Returns:
            Dict with colors, appearance, finish from visual analysis
        """
        try:
            response = self.vision_llm.chat_with_image(
                prompt=self.vlm_classification_prompt,
                image_path=crop_path,
                temperature=0.2
            )
            
            return self.vision_llm.parse_json_response(response)
        
        except Exception as e:
            logger.warning(f"VLM extraction failed for {crop_path}: {e}")
            return {
                "color": "Multi Colour",
                "appearance": "Solid Colour",
                "finish": "unknown",
                "confidence": "low"
            }
    
    def derive_fields(self, product: Dict[str, Any], visual_attrs: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Use LLM to derive all product fields using prompt-based rules.
        
        Args:
            product: Raw product data (sku, page_num, etc.)
            visual_attrs: Visual attributes from VLM (colors, appearance, etc.)
        
        Returns:
            Product dict with all derived fields
        """
        # Merge visual attributes with product data
        product_data = {**product}
        if visual_attrs:
            product_data["visual_color"] = visual_attrs.get("color", "Multi Colour")
            product_data["visual_appearance"] = visual_attrs.get("appearance")
            product_data["visual_finish"] = visual_attrs.get("finish")
        
        # Add defaults
        product_data["defaults"] = self.product_defaults
        
        # Render the prompt with product data
        prompt = self.config_manager.render_prompt(
            "field_mapping_derivation",
            product=product_data,
            taxonomies=self.taxonomies
        )
        
        try:
            # Call LLM
            response = self.text_llm.chat(
                prompt=prompt,
                temperature=0.1,  # Low temperature for consistency
                max_tokens=2048
            )
            
            # Parse JSON response
            derived = self.text_llm.parse_json_response(response)
            
            # Merge derived fields with original
            return {**product, **derived}
        
        except Exception as e:
            logger.error(f"Field derivation failed for {product.get('sku')}: {e}")
            
            # Return with defaults on failure
            return {
                **product,
                "title": product.get("sku", "Unknown"),
                "handle": product.get("sku", "unknown").lower().replace(" ", "-"),
                "category": self.product_defaults.get("category", "laminate"),
                "sub_category": self.product_defaults.get("sub_category", "Decorative"),
                "appearance": visual_attrs.get("appearance", "Solid Colour") if visual_attrs else "Solid Colour",
                "finish": self.product_defaults.get("finish", "Texture"),
                "color": visual_attrs.get("color", "Multi Colour") if visual_attrs else "Multi Colour",
                "size": self.product_defaults.get("size", "8*4 Feet"),
                "thickness": self.product_defaults.get("thickness", "1.0 mm"),
                "derivation_error": str(e)
            }
    
    def process_products(self, products: List[Dict]) -> List[Dict]:
        """
        Process all products through the field mapping pipeline.
        
        Args:
            products: List of raw product dicts from detection
        
        Returns:
            List of products with all fields derived
        """
        processed = []
        
        for i, product in enumerate(products):
            logger.info(f"Processing product {i+1}/{len(products)}: {product.get('sku')}")
            
            # Extract visual attributes from crop
            visual_attrs = None
            crop_path = product.get("crop_path")
            if crop_path and Path(crop_path).exists():
                visual_attrs = self.extract_visual_attributes(crop_path)
                logger.info(f"  Visual: {visual_attrs.get('appearance')} / {visual_attrs.get('colors')}")
            
            # Derive all fields using LLM
            derived = self.derive_fields(product, visual_attrs)
            logger.info(f"  Derived: {derived.get('category')} / {derived.get('appearance')} / {derived.get('finish')}")
            
            processed.append(derived)
        
        return processed


def run_field_mapping(output_dir: str, config_dir: str = None) -> Dict[str, Any]:
    """
    Run prompt-based field mapping on detected products.
    
    Args:
        output_dir: Path to PDF output directory
        config_dir: Optional path to config directory
    
    Returns:
        Dict with field mapping results
    """
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
    logger.info(f"=" * 60)
    logger.info(f"Stage 2: Prompt-Based Field Mapping")
    logger.info(f"Directory: {output_dir.name}")
    logger.info(f"=" * 60)
    
    # Load configuration
    config_manager = get_config_manager(config_dir)
    
    # Initialize output manager
    output_manager = OutputManager(str(output_dir.parent))
    output_manager._current_pdf_dir = output_dir
    output_manager._current_pdf_name = output_dir.name.replace("_PDF", "")
    
    # Load detection results
    detection_path = output_dir / "data" / "1_detection_results.json"
    if not detection_path.exists():
        raise FileNotFoundError(f"Detection results not found: {detection_path}")
    
    with open(detection_path) as f:
        detection_results = json.load(f)
    
    products = detection_results.get("products", [])
    logger.info(f"Found {len(products)} products to process")
    
    if not products:
        logger.warning("No products to process")
        return {"status": "error", "error": "No products found"}
    
    # Initialize field mapper
    mapper = PromptBasedFieldMapper(config_manager)
    
    # Process products
    processed_products = mapper.process_products(products)
    
    # Save results
    result = {
        "output_dir": str(output_dir),
        "product_count": len(processed_products),
        "products": processed_products,
        "status": "success"
    }
    
    output_manager.save_json(result, "2_field_mapping_results.json")
    output_manager.log("field_mapping", f"Processed {len(processed_products)} products", "INFO")
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Field mapping complete!")
    logger.info(f"  Products processed: {len(processed_products)}")
    logger.info(f"  Results saved to: {output_manager.data_dir / '2_field_mapping_results.json'}")
    logger.info(f"{'=' * 60}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Prompt-based field mapping for detected products"
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
        run_field_mapping(args.output_dir, args.config_dir)
    except Exception as e:
        logger.error(f"Field mapping failed: {e}")
        raise


if __name__ == "__main__":
    main()
