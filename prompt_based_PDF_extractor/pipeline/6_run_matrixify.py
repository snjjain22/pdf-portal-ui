#!/usr/bin/env python3
"""
Stage 4: Matrixify CSV Export

Transforms processed products into Shopify Matrixify CSV format.
Applies the matrixify schema to generate a ready-to-import file.

Usage:
    python 4_run_matrixify.py <pdf_output_dir>
    python 4_run_matrixify.py outputs/MY_CATALOG_PDF
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.config_manager import get_config_manager
from common.output_manager import OutputManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MatrixifyTransformer:
    """
    Transforms product data into Shopify Matrixify CSV format.
    """
    
    def __init__(self, config_manager):
        """
        Initialize transformer.
        
        Args:
            config_manager: ConfigManager instance
        """
        self.config_manager = config_manager
        
        # Load matrixify schema (nested under 'matrixify' key)
        full_schema = config_manager.load_matrixify_schema()
        self.schema = full_schema.get("matrixify", full_schema)
        
        # Get column names (simple list of strings)
        self.column_names = self.schema.get("columns", [])
        
        # Get field mappings (product_field -> column_name)
        self.field_mappings = self.schema.get("field_mappings", {})
        
        # Reverse mapping for easier lookup (column_name -> product_field)
        self.column_to_field = {v: k for k, v in self.field_mappings.items()}
        
        # Get fixed values (column_name -> value)
        self.fixed_values = self.schema.get("fixed_values", {})
        
        # Get default values
        self.defaults = self.schema.get("defaults", {})
        
        # Product defaults from website config
        website_config = config_manager.load_website_config()
        self.product_defaults = website_config.get("product_defaults", {})
        
        logger.info(f"MatrixifyTransformer initialized with {len(self.column_names)} columns")
    
    def transform_product(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform a single product to Matrixify row format.
        
        Args:
            product: Product dict with all fields
        
        Returns:
            Dict with Matrixify column values
        """
        row = {}
        
        for col_name in self.column_names:
            value = ""
            
            # 1. Check if this column has a fixed value
            if col_name in self.fixed_values:
                value = self.fixed_values[col_name]
                if "scratch_resistance" in col_name:
                    logger.info(f"DEBUG: Fixed value for {col_name} -> '{value}'")
            
            # 2. Check if this column is mapped to a product field
            elif col_name in self.column_to_field:
                product_field = self.column_to_field[col_name]
                value = product.get(product_field, "")
                
                # Handle list values (e.g., colors -> comma-separated)
                if isinstance(value, list):
                    value = ", ".join(str(v) for v in value)
            
            # 3. Special handling for specific columns
            if col_name == "Handle":
                value = product.get("handle", "") or self._generate_handle(product)
            
            elif col_name == "Title":
                value = product.get("title", "") or product.get("product_name", "")
            
            elif col_name == "Body HTML":
                value = product.get("body_html", "") or product.get("seo_description", "")
            
            elif col_name == "Tags":
                value = product.get("tags", "")
            
            elif col_name == "Variant SKU":
                value = product.get("sku", "")
            
            elif col_name == "Image Src":
                # Product image first, then application images (semicolon-separated)
                product_image = product.get("s3_url", "") or product.get("image_src", "") or product.get("crop_path", "")
                app_images = product.get("application_image_urls", [])
                all_images = [product_image] + list(app_images)
                value = ";".join(url for url in all_images if url)
            
            elif col_name == "Variant Image":
                value = ""  # Always empty
            
            elif col_name == "Image Alt Text":
                value = product.get("image_alt", "") or product.get("title", "")
            
            elif col_name == "Variant Price":
                value = str(product.get("price", self.product_defaults.get("price", "0")))
            
            # 4. Handle color metafield (single string, not array)
            elif col_name == "Metafield: custom.colour_family [single_line_text_field]":
                color = product.get("color", "")
                # Ensure it's a single string, not array
                if isinstance(color, list):
                    color = color[0] if color else "Multi Colour"
                value = str(color) if color else "Multi Colour"
            
            # 5. Handle FAQ metafields (list type requires JSON array format)
            elif col_name == "Metafield: custom.faq_title [list.single_line_text_field]":
                faq_titles = product.get("faq_titles", [])
                if faq_titles and isinstance(faq_titles, list):
                    value = json.dumps(faq_titles)
                else:
                    value = ""
            
            elif col_name == "Metafield: custom.faq_description [list.single_line_text_field]":
                faq_descriptions = product.get("faq_descriptions", [])
                if faq_descriptions and isinstance(faq_descriptions, list):
                    value = json.dumps(faq_descriptions)
                else:
                    value = ""
            
            # 6. Fallback for Title Tag (SEO Title)
            elif col_name == "Metafield: title_tag [string]":
                # Use mapped value (seo_title) or fallback to main Title
                if not value:
                    value = product.get("title", "") or product.get("product_name", "")

            # 7. Lowercase Type and Appearance values
            if col_name in ("Type", "Metafield: custom.appearance [single_line_text_field]"):
                value = str(value).lower() if value else value

            row[col_name] = value
        
        return row
    
    def _generate_handle(self, product: Dict) -> str:
        """Generate URL handle from product data."""
        parts = [
            product.get("sub_category", ""),
            product.get("appearance", ""),
            product.get("category", "laminate"),
            product.get("sku", "")
        ]
        handle = "-".join(p.lower() for p in parts if p)
        handle = handle.replace(" ", "-").replace("&", "and")
        # Remove special characters
        handle = "".join(c if c.isalnum() or c == "-" else "" for c in handle)
        # Remove consecutive dashes
        while "--" in handle:
            handle = handle.replace("--", "-")
        return handle.strip("-")
    
    def transform_products(self, products: List[Dict]) -> pd.DataFrame:
        """
        Transform all products to Matrixify DataFrame.
        
        Args:
            products: List of product dicts
        
        Returns:
            DataFrame with Matrixify format
        """
        rows = []
        
        for product in products:
            row = self.transform_product(product)
            rows.append(row)
        
        # Create DataFrame with all columns
        df = pd.DataFrame(rows, columns=self.column_names)
        
        # Fill NaN with empty string
        df = df.fillna("")
        
        return df
    
    def validate_csv(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the generated CSV.
        
        Args:
            df: Matrixify DataFrame
        
        Returns:
            Validation results dict
        """
        issues = []
        
        # Check required columns
        required = ["Handle", "Title", "Vendor"]
        for col in required:
            if col not in df.columns:
                issues.append(f"Missing required column: {col}")
            elif df[col].isna().all() or (df[col] == "").all():
                issues.append(f"Column '{col}' has no values")
        
        # Check for duplicate handles
        if "Handle" in df.columns:
            duplicates = df[df["Handle"].duplicated()]["Handle"].tolist()
            if duplicates:
                issues.append(f"Duplicate handles found: {duplicates[:5]}")
        
        # Check price is numeric
        if "Variant Price" in df.columns:
            non_numeric = []
            for idx, val in df["Variant Price"].items():
                try:
                    if val:
                        float(val)
                except ValueError:
                    non_numeric.append(val)
            if non_numeric:
                issues.append(f"Non-numeric prices found: {non_numeric[:5]}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "row_count": len(df),
            "column_count": len(df.columns)
        }


def run_matrixify(output_dir: str, config_dir: str = None, vendor: str = None) -> Dict[str, Any]:
    """
    Transform products to Matrixify CSV format.
    
    Args:
        output_dir: Path to PDF output directory
        config_dir: Optional path to config directory
        vendor: Optional vendor/brand name to override the schema default
    
    Returns:
        Dict with transformation results
    """
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
    logger.info(f"=" * 60)
    logger.info(f"Stage 4: Matrixify CSV Export")
    logger.info(f"Directory: {output_dir.name}")
    logger.info(f"=" * 60)
    
    # Load configuration
    config_manager = get_config_manager(config_dir)
    
    # Initialize output manager
    output_manager = OutputManager(str(output_dir.parent))
    output_manager._current_pdf_dir = output_dir
    output_manager._current_pdf_name = output_dir.name.replace("_PDF", "")
    
    # Load input data (prefer app images > S3 uploaded > SEO results)
    app_path = output_dir / "data" / "5_app_images.json"
    s3_path = output_dir / "data" / "4_s3_uploaded.json"
    seo_path = output_dir / "data" / "3_seo_results.json"
    
    if app_path.exists():
        input_path = app_path
    elif s3_path.exists():
        input_path = s3_path
    else:
        input_path = seo_path
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input data not found. Checked {app_path}, {s3_path} and {seo_path}")
    
    logger.info(f"Loading data from: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both wrapper dict and list formats
    if isinstance(data, dict):
        products = data.get("products", [])
    else:
        products = data
        
    logger.info(f"Found {len(products)} products to export")
    
    if not products:
        logger.warning("No products to export")
        return {"status": "error", "error": "No products found"}
    
    # Initialize transformer
    transformer = MatrixifyTransformer(config_manager)

    # Override vendor if supplied
    if vendor:
        transformer.fixed_values["Vendor"] = vendor
        logger.info(f"Vendor set to: {vendor}")
    
    # Transform to DataFrame
    df = transformer.transform_products(products)
    logger.info(f"Created DataFrame: {len(df)} rows x {len(df.columns)} columns")
    
    # Validate
    validation = transformer.validate_csv(df)
    if not validation["valid"]:
        logger.warning(f"Validation issues: {validation['issues']}")
    else:
        logger.info("Validation passed!")
    
    # Save CSV
    csv_filename = f"matrixify_{output_manager._current_pdf_name}.csv"
    csv_path = output_manager.data_dir / csv_filename
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"Saved CSV: {csv_path}")
    
    # Also save to root of output dir for easy access
    root_csv_path = output_dir / csv_filename
    df.to_csv(root_csv_path, index=False, encoding='utf-8-sig')
    
    # Save results
    result = {
        "output_dir": str(output_dir),
        "product_count": len(products),
        "csv_path": str(csv_path),
        "validation": validation,
        "status": "success"
    }
    
    output_manager.save_json(result, "4_matrixify_results.json")
    output_manager.log("matrixify", f"Exported {len(df)} products to CSV", "INFO")
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Matrixify export complete!")
    logger.info(f"  Products exported: {len(df)}")
    logger.info(f"  Columns: {len(df.columns)}")
    logger.info(f"  CSV saved to: {csv_path}")
    logger.info(f"  Also at: {root_csv_path}")
    logger.info(f"{'=' * 60}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Stage 4: Export products to Matrixify CSV"
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
        run_matrixify(args.output_dir, args.config_dir)
    except Exception as e:
        logger.error(f"Matrixify export failed: {e}")
        raise


if __name__ == "__main__":
    main()
