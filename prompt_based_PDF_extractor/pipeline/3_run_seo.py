#!/usr/bin/env python3
"""
Stage 3: SEO Content Generation

Uses LLM prompts to generate SEO-optimized content:
- Product titles
- URL handles
- Body HTML (descriptions)
- Meta descriptions
- Image alt text
- Tags

Usage:
    python 3_run_seo.py <pdf_output_dir>
    python 3_run_seo.py outputs/MY_CATALOG_PDF
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
from common.llm_client import get_text_llm_client

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SEOGenerator:
    """
    Generates SEO-optimized content using LLM prompts.
    """
    
    def __init__(self, config_manager):
        """
        Initialize SEO generator.
        
        Args:
            config_manager: ConfigManager instance
        """
        self.config_manager = config_manager
        
        # Load configuration
        website_config = config_manager.load_website_config()
        llm_config = website_config.get("llm", {})
        seo_config = config_manager.load_seo_config()
        
        # Initialize LLM client
        self.llm = get_text_llm_client(
            model=llm_config.get("text_model", "deepseek-chat"),
            max_retries=llm_config.get("max_retries", 3),
            timeout=llm_config.get("timeout", 120)
        )
        
        # Load prompt template
        self.seo_prompt = config_manager.get_prompt_template("seo_generation")
        
        # SEO settings
        self.seo_settings = seo_config.get("generation", {})
        
        logger.info("SEOGenerator initialized")
        logger.info(f"  LLM: {self.llm.model}")
    
    def generate_seo_content(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate SEO content for a single product.
        
        Args:
            product: Product dict with derived fields
        
        Returns:
            Product dict with added SEO fields
        """
        # Render prompt with product data
        prompt = self.config_manager.render_prompt(
            "seo_generation",
            product=product
        )
        
        try:
            # Call LLM
            response = self.llm.chat(
                prompt=prompt,
                temperature=0.5,  # Slightly higher for creative content
                max_tokens=2048
            )
            
            # Parse JSON response
            seo_content = self.llm.parse_json_response(response)
            
            # Merge with product
            return {
                **product,
                "title": seo_content.get("title", product.get("title")),
                "handle": seo_content.get("handle", self._generate_handle(product)),
                "body_html": seo_content.get("body_html", ""),
                "meta_description": seo_content.get("meta_description", ""),
                "image_alt": seo_content.get("image_alt", ""),
                "tags": seo_content.get("tags", ""),
                "faq_titles": seo_content.get("faq_titles", []),
                "faq_descriptions": seo_content.get("faq_descriptions", []),
                "seo_generated": True
            }
        
        except Exception as e:
            logger.warning(f"SEO generation failed for {product.get('sku')}: {e}")
            
            # Generate fallback SEO content
            return {
                **product,
                "title": self._generate_fallback_title(product),
                "handle": self._generate_handle(product),
                "body_html": self._generate_fallback_description(product),
                "meta_description": self._generate_fallback_meta(product),
                "image_alt": f"{product.get('appearance', '')} {product.get('finish', '')} {product.get('category', 'Laminate')} - {product.get('sku', '')} by Next Level Decor",
                "tags": f"{product.get('category', 'laminate')}, {product.get('appearance', '')}, {product.get('finish', '')}",
                "faq_titles": [],
                "faq_descriptions": [],
                "seo_generated": False,
                "seo_error": str(e)
            }
    
    def _generate_handle(self, product: Dict) -> str:
        """Generate URL handle from product data."""
        parts = [
            product.get("sub_category", ""),
            product.get("appearance", ""),
            product.get("category", "laminate"),
            product.get("sku", "")
        ]
        handle = "-".join(p for p in parts if p)
        handle = handle.lower().replace(" ", "-").replace("&", "and")
        # Remove special characters
        handle = "".join(c if c.isalnum() or c == "-" else "" for c in handle)
        # Remove consecutive dashes
        while "--" in handle:
            handle = handle.replace("--", "-")
        return handle.strip("-")
    
    def _generate_fallback_title(self, product: Dict) -> str:
        """Generate fallback title."""
        parts = [
            product.get("sub_category", ""),
            product.get("appearance", ""),
            product.get("category", "Laminate"),
            product.get("size", "8*4 ft"),
            product.get("thickness", ""),
            f"- {product.get('sku', '')}"
        ]
        return " ".join(p for p in parts if p)
    
    def _generate_fallback_description(self, product: Dict) -> str:
        """Generate fallback body HTML."""
        p1 = f"Premium {product.get('appearance', '')} {product.get('category', 'laminate')} in {product.get('size', '8*4 ft')} size with {product.get('finish', 'textured')} finish."
        p2 = f"Ideal for kitchen cabinets, wardrobes, and wall panels. Easy to maintain and durable."
        return f"<p>{p1}</p><p>{p2}</p>"
    
    def _generate_fallback_meta(self, product: Dict) -> str:
        """Generate fallback meta description."""
        return f"Premium {product.get('appearance', '')} {product.get('category', 'laminate')} {product.get('size', '8*4 ft')} with {product.get('finish', '')} finish. Ideal for interiors by Next Level Decor."[:158]
    
    def process_products(self, products: List[Dict]) -> List[Dict]:
        """
        Generate SEO content for all products.
        
        Args:
            products: List of products with derived fields
        
        Returns:
            List of products with SEO content added
        """
        processed = []
        
        for i, product in enumerate(products):
            logger.info(f"Generating SEO {i+1}/{len(products)}: {product.get('sku')}")
            
            result = self.generate_seo_content(product)
            logger.info(f"  Title: {result.get('title', '')[:50]}...")
            
            processed.append(result)
        
        return processed


def run_seo(output_dir: str, config_dir: str = None) -> Dict[str, Any]:
    """
    Run SEO content generation on mapped products.
    
    Args:
        output_dir: Path to PDF output directory
        config_dir: Optional path to config directory
    
    Returns:
        Dict with SEO generation results
    """
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
    logger.info(f"=" * 60)
    logger.info(f"Stage 3: SEO Content Generation")
    logger.info(f"Directory: {output_dir.name}")
    logger.info(f"=" * 60)
    
    # Load configuration
    config_manager = get_config_manager(config_dir)
    
    # Initialize output manager
    output_manager = OutputManager(str(output_dir.parent))
    output_manager._current_pdf_dir = output_dir
    output_manager._current_pdf_name = output_dir.name.replace("_PDF", "")
    
    # Load field mapping results
    mapping_path = output_dir / "data" / "2_field_mapping_results.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Field mapping results not found: {mapping_path}")
    
    with open(mapping_path) as f:
        mapping_results = json.load(f)
    
    products = mapping_results.get("products", [])
    logger.info(f"Found {len(products)} products to process")
    
    if not products:
        logger.warning("No products to process")
        return {"status": "error", "error": "No products found"}
    
    # Initialize SEO generator
    seo_generator = SEOGenerator(config_manager)
    
    # Process products
    processed_products = seo_generator.process_products(products)
    
    # Count successes
    seo_success = sum(1 for p in processed_products if p.get("seo_generated", False))
    
    # Save results
    result = {
        "output_dir": str(output_dir),
        "product_count": len(processed_products),
        "seo_success": seo_success,
        "products": processed_products,
        "status": "success"
    }
    
    output_manager.save_json(result, "3_seo_results.json")
    output_manager.log("seo", f"Generated SEO for {seo_success}/{len(processed_products)} products", "INFO")
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"SEO generation complete!")
    logger.info(f"  Products processed: {len(processed_products)}")
    logger.info(f"  SEO generated: {seo_success}")
    logger.info(f"  Results saved to: {output_manager.data_dir / '3_seo_results.json'}")
    logger.info(f"{'=' * 60}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: Generate SEO content for products"
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
        run_seo(args.output_dir, args.config_dir)
    except Exception as e:
        logger.error(f"SEO generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
