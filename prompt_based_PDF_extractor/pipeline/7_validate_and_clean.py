#!/usr/bin/env python3
"""
Stage 5: Validation and Cleanup for Matrixify Export

Validates the Matrixify CSV against the expected Shopify column format:
- Removes any unwanted/extra columns
- Adds any missing columns as empty
- Reorders columns to match Shopify Matrixify import format
- Generates a detailed validation report

Usage:
    python 5_validate_and_clean.py <pdf_output_dir>
    python 5_validate_and_clean.py outputs/MY_CATALOG_PDF
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

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


# Expected Shopify Matrixify column order (exact match required for clean import)
EXPECTED_COLUMNS = [
    'ID',
    'Title',
    'Body HTML',
    'Vendor',
    'Type',
    'Tags',
    'Status',
    'Published',
    'Total Inventory Qty',
    'Category',
    'Option1 Name',
    'Option1 Value',
    'Option2 Name',
    'Option2 Value',
    'Option3 Name',
    'Option3 Value',
    'Variant Position',
    'Variant SKU',
    'Variant Barcode',
    'Variant Image',
    'Variant Weight',
    'Variant Weight Unit',
    'Variant Price',
    'Variant Compare At Price',
    'Variant Taxable',
    'Variant Tax Code',
    'Variant Inventory Tracker',
    'Variant Inventory Policy',
    'Variant Fulfillment Service',
    'Variant Requires Shipping',
    'Inventory Available: New 215 (old 146),opp nehru indoor stadium ,syndemans road, Periyamedu, Periyamet, Choolai, Chennai,',
    'Metafield: title_tag [string]',
    'Metafield: description_tag [string]',
    'Metafield: custom.colour_family [single_line_text_field]',
    'Metafield: custom.applications [list.single_line_text_field]',
    'Metafield: custom.finish [single_line_text_field]',
    'Metafield: custom.collection_name [single_line_text_field]',
    'Metafield: custom.base_material [single_line_text_field]',
    'Metafield: custom.width [single_line_text_field]',
    'Metafield: custom.thickness [single_line_text_field]',
    'Metafield: custom.size [single_line_text_field]',
    'Metafield: custom.length [single_line_text_field]',
    'Metafield: custom.coverage_area [single_line_text_field]',
    'Metafield: custom.indoor_outdoor [single_line_text_field]',
    'Metafield: custom.project_type [list.single_line_text_field]',
    'Metafield: custom.sub_category [single_line_text_field]',
    'Metafield: custom.appearance [single_line_text_field]',
    'Metafield: custom.scratch_resistance [single_line_text_field]',
    'Metafield: custom.water_moisture_resistant [single_line_text_field]',
    'Metafield: custom.fire_resistant [single_line_text_field]',
    'Metafield: custom.uv_resistant [single_line_text_field]',
    'Metafield: custom.anti_fingerprint [single_line_text_field]',
    'Metafield: custom.grade [single_line_text_field]',
    'Image Src',
    'Image Type',
    'Published Scope',
    'Image Alt Text',
    'Image Position',
    'Metafield: custom.faq_title [list.single_line_text_field]',
    'Metafield: custom.faq_description [list.single_line_text_field]'
]


def generate_report(
    input_path: Path,
    output_path: Path,
    original_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    removed_cols: list,
    missing_cols: list,
    report_path: Path
) -> str:
    """Generate a detailed validation report."""
    
    report = []
    report.append("=" * 80)
    report.append("MATRIXIFY EXPORT VALIDATION REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Input file: {input_path}")
    report.append(f"Output file: {output_path}")
    report.append("")
    
    # Summary
    report.append("SUMMARY")
    report.append("-" * 80)
    report.append(f"Original CSV: {len(original_df)} rows, {len(original_df.columns)} columns")
    report.append(f"Cleaned CSV:  {len(cleaned_df)} rows, {len(cleaned_df.columns)} columns")
    removed_rows = len(original_df) - len(cleaned_df)
    if removed_rows > 0:
        report.append(f"Removed Rows: {removed_rows} (Missing application images)")
    report.append(f"Expected:     {len(EXPECTED_COLUMNS)} columns")
    report.append("")
    
    # Removed columns
    if removed_cols:
        report.append(f"REMOVED COLUMNS ({len(removed_cols)})")
        report.append("-" * 80)
        for col in removed_cols:
            report.append(f"  ✗ {col}")
        report.append("")
    else:
        report.append("REMOVED COLUMNS: None")
        report.append("")
    
    # Missing columns (added as empty)
    if missing_cols:
        report.append(f"MISSING COLUMNS - Added as empty ({len(missing_cols)})")
        report.append("-" * 80)
        for col in missing_cols:
            report.append(f"  ⚠ {col}")
        report.append("")
    else:
        report.append("MISSING COLUMNS: None")
        report.append("")
    
    # Column order validation
    report.append("COLUMN ORDER VALIDATION")
    report.append("-" * 80)
    if list(cleaned_df.columns) == EXPECTED_COLUMNS:
        report.append("✔ Column order matches Shopify format")
    else:
        report.append("✗ Column order mismatch detected!")
        report.append("")
        report.append("First 10 mismatches:")
        mismatch_count = 0
        for i, (expected, actual) in enumerate(zip(EXPECTED_COLUMNS, cleaned_df.columns)):
            if expected != actual:
                report.append(f"  Position {i+1}: Expected '{expected}', got '{actual}'")
                mismatch_count += 1
                if mismatch_count >= 10:
                    break
    report.append("")
    
    # Key field validation
    report.append("KEY FIELD VALIDATION")
    report.append("-" * 80)
    
    # Define columns that MUST be filled (based on production requirements)
    CRITICAL_COLUMNS = [
        'Title',
        'Body HTML',
        'Vendor',
        'Type',
        'Tags',
        'Status',
        'Published',
        'Total Inventory Qty',
        'Category',
        'Option1 Name',
        'Option1 Value',
        'Variant Position',
        'Variant SKU',
        'Metafield: title_tag [string]',
        'Metafield: description_tag [string]',
        'Metafield: custom.colour_family [single_line_text_field]',
        'Metafield: custom.finish [single_line_text_field]',
        'Metafield: custom.base_material [single_line_text_field]',
        'Metafield: custom.width [single_line_text_field]',
        'Metafield: custom.thickness [single_line_text_field]',
        'Metafield: custom.size [single_line_text_field]',
        'Metafield: custom.length [single_line_text_field]',
        'Metafield: custom.coverage_area [single_line_text_field]',
        'Metafield: custom.indoor_outdoor [single_line_text_field]',
        'Metafield: custom.appearance [single_line_text_field]',
        'Metafield: custom.scratch_resistance [single_line_text_field]',
        'Metafield: custom.water_moisture_resistant [single_line_text_field]',
        'Metafield: custom.fire_resistant [single_line_text_field]',
        'Metafield: custom.uv_resistant [single_line_text_field]',
        'Metafield: custom.anti_fingerprint [single_line_text_field]',
        'Image Src',
        'Image Type',
        'Published Scope',
        'Image Alt Text',
        'Image Position',
        'Metafield: custom.faq_title [list.single_line_text_field]',
        'Metafield: custom.faq_description [list.single_line_text_field]'
    ]

    all_critical_passed = True
    for col in CRITICAL_COLUMNS:
        if col in cleaned_df.columns:
            non_empty = cleaned_df[col].notna() & (cleaned_df[col].astype(str).str.strip() != '') & (cleaned_df[col].astype(str).str.strip() != 'nan')
            fill_count = non_empty.sum()
            total_rows = len(cleaned_df)
            
            if fill_count == total_rows:
                report.append(f"✔ {col}: {fill_count}/{total_rows} filled (100%)")
            else:
                report.append(f"✗ {col}: {fill_count}/{total_rows} filled - WARNING: MISSING DATA!")
                all_critical_passed = False
        else:
            report.append(f"✗ {col}: NOT FOUND IN CSV")
            all_critical_passed = False
    
    if all_critical_passed:
        report.append("\n✔ ALL CRITICAL FIELDS PASSED VALIDATION")
    else:
        report.append("\n⚠ SOME CRITICAL FIELDS HAVE MISSING DATA")
    
    report.append("")
    
    # Image URL check
    if 'Image Src' in cleaned_df.columns:
        non_empty_img = cleaned_df['Image Src'].notna() & (cleaned_df['Image Src'].astype(str).str.strip() != '')
        if non_empty_img.sum() > 0:
            first_img = str(cleaned_df[non_empty_img]['Image Src'].iloc[0])
            if first_img.startswith('http'):
                report.append("  ✔ Image Src contains URLs (ready for Shopify)")
            else:
                report.append("  ⚠ Image Src contains local paths (upload to S3 before Shopify import)")
            report.append(f"    Sample: {first_img[:80]}...")
            
    report.append("")
    report.append("VALUE FORMAT VALIDATION")
    report.append("-" * 80)
    
    # Check lowercase Type
    if 'Type' in cleaned_df.columns:
        type_vals = cleaned_df['Type'].dropna().astype(str)
        non_lower = type_vals[~type_vals.str[0].str.islower() & (type_vals != '')]
        if non_lower.empty:
            report.append("✔ Type values start with lowercase")
        else:
            report.append(f"✗ Type has {len(non_lower)} values not starting with lowercase (These have been auto-corrected in the output CSV)")
            
    # Check lowercase Appearance
    appearance_col = 'Metafield: custom.appearance [single_line_text_field]'
    if appearance_col in cleaned_df.columns:
        app_vals = cleaned_df[appearance_col].dropna().astype(str)
        non_lower_app = app_vals[~app_vals.str[0].str.islower() & (app_vals != '')]
        if non_lower_app.empty:
            report.append("✔ Appearance values start with lowercase")
        else:
            report.append(f"✗ Appearance has {len(non_lower_app)} values not starting with lowercase (These have been auto-corrected in the output CSV)")

    report.append("")
    report.append("SKU COUNTS BY ATTRIBUTE")
    report.append("-" * 80)
    
    attributes_to_count = {
        'Color': 'Metafield: custom.colour_family [single_line_text_field]',
        'Appearance': 'Metafield: custom.appearance [single_line_text_field]',
        'Finish': 'Metafield: custom.finish [single_line_text_field]',
        'Thickness': 'Metafield: custom.thickness [single_line_text_field]',
        'Category': 'Category',
        'Subcategory': 'Metafield: custom.sub_category [single_line_text_field]'
    }
    
    for attr_name, col_name in attributes_to_count.items():
        report.append(f"By {attr_name}:")
        if col_name in cleaned_df.columns:
            counts = cleaned_df[col_name].fillna('Empty/None').value_counts()
            for val, count in counts.items():
                report.append(f"  - {val}: {count}")
        else:
            report.append(f"  Column '{col_name}' missing")
        report.append("")
    
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    
    # Write report
    report_text = "\n".join(report)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    return report_text


def validate_and_clean(output_dir: str, config_dir: str = None) -> Dict[str, Any]:
    """
    Validate and clean the Matrixify CSV export.
    
    Finds the matrixify CSV in the output directory, validates columns,
    removes extras, adds missing, reorders, and generates a report.
    
    Args:
        output_dir: Path to PDF output directory (e.g., outputs/MY_CATALOG_PDF)
        config_dir: Optional path to config directory
    
    Returns:
        Dict with validation results
    """
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
    logger.info(f"=" * 60)
    logger.info(f"Stage 5: Validate & Clean Matrixify CSV")
    logger.info(f"Directory: {output_dir.name}")
    logger.info(f"=" * 60)
    
    # Find the matrixify CSV file(s)
    data_dir = output_dir / "data"
    root_csvs = list(output_dir.glob("matrixify_*.csv"))
    data_csvs = list(data_dir.glob("matrixify_*.csv")) if data_dir.exists() else []
    
    all_csvs = root_csvs + data_csvs
    
    if not all_csvs:
        raise FileNotFoundError(
            f"No matrixify CSV found in {output_dir} or {data_dir}.\n"
            "Run Stage 4 (Matrixify Export) first."
        )
    
    logger.info(f"Found {len(all_csvs)} CSV file(s) to validate")
    
    # Process each CSV (typically there's one in data/ and a copy in root)
    results_list = []
    
    for csv_path in all_csvs:
        logger.info(f"\nProcessing: {csv_path}")
        
        # Load original
        df_original = pd.read_csv(csv_path)
        df = df_original.copy()
        
        original_cols = len(df.columns)
        original_rows = len(df)
        
        logger.info(f"  Input: {original_rows} rows, {original_cols} columns")
        logger.info(f"  Expected: {len(EXPECTED_COLUMNS)} columns")
        
        # Rename Variant SKU [ID] to Variant SKU if present
        if 'Variant SKU [ID]' in df.columns:
            df.rename(columns={'Variant SKU [ID]': 'Variant SKU'}, inplace=True)
        
        # Find extra columns to remove
        extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]
        if extra_cols:
            logger.info(f"  Removing {len(extra_cols)} extra columns:")
            for col in extra_cols:
                logger.info(f"    ✗ {col}")
            df = df.drop(columns=extra_cols)
        
        # Find missing columns to add
        missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        if missing_cols:
            logger.info(f"  Adding {len(missing_cols)} missing columns (empty):")
            for col in missing_cols:
                logger.info(f"    ⚠ {col}")
                df[col] = ''
        
        # Enforce lowercase for Type and Appearance
        if 'Type' in df.columns:
            df['Type'] = df['Type'].astype(str).apply(lambda x: x[:1].lower() + x[1:] if pd.notna(x) and x != 'nan' and len(x) > 0 else x)
        appearance_col = 'Metafield: custom.appearance [single_line_text_field]'
        if appearance_col in df.columns:
            df[appearance_col] = df[appearance_col].astype(str).apply(lambda x: x[:1].lower() + x[1:] if pd.notna(x) and x != 'nan' and len(x) > 0 else x)

        # Filter out rows that do not have application images (indicated by missing ';' in Image Src)
        if 'Image Src' in df.columns:
            initial_row_count = len(df)
            df = df[df['Image Src'].astype(str).str.contains(';', na=False)]
            removed_count = initial_row_count - len(df)
            if removed_count > 0:
                logger.info(f"  Removing {removed_count} rows that are missing application images")

        # Reorder to match expected column order
        df = df[EXPECTED_COLUMNS]
        
        # Overwrite the CSV with cleaned version
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"  ✓ Cleaned CSV saved: {csv_path}")
        logger.info(f"  Final: {len(df)} rows, {len(df.columns)} columns")
        
        # Validate column order
        if list(df.columns) == EXPECTED_COLUMNS:
            logger.info(f"  ✓ Column order matches Shopify format")
        else:
            logger.warning(f"  ✗ Column order mismatch!")
        
        results_list.append({
            "csv_path": str(csv_path),
            "original_columns": original_cols,
            "final_columns": len(df.columns),
            "rows": len(df),
            "removed_columns": extra_cols,
            "missing_columns": missing_cols,
            "column_order_valid": list(df.columns) == EXPECTED_COLUMNS
        })
        
        # Generate report (only for root CSV to avoid duplication)
        if csv_path.parent == output_dir:
            report_path = output_dir / "data" / "5_validation_report.txt"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"\nGenerating validation report...")
            report_text = generate_report(
                csv_path, csv_path, df_original, df,
                extra_cols, missing_cols, report_path
            )
            
            logger.info(f"  ✓ Report saved: {report_path}")
            
            # Print report summary
            for line in report_text.split('\n'):
                logger.info(f"  {line}")
    
    # Build final result
    result = {
        "output_dir": str(output_dir),
        "csvs_processed": len(all_csvs),
        "validations": results_list,
        "expected_columns": len(EXPECTED_COLUMNS),
        "status": "success"
    }
    
    # Save results JSON
    results_json_path = output_dir / "data" / "5_validation_results.json"
    import json
    with open(results_json_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Validation & cleanup complete!")
    logger.info(f"  CSVs processed: {len(all_csvs)}")
    logger.info(f"  All columns: {len(EXPECTED_COLUMNS)}")
    logger.info(f"{'=' * 60}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Stage 5: Validate and clean Matrixify CSV export"
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
        validate_and_clean(args.output_dir, args.config_dir)
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise


if __name__ == "__main__":
    main()
