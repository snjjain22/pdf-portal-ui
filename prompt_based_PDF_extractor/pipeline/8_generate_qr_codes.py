#!/usr/bin/env python3
"""
Stage 8: QR Code Generation

For each product in the Matrixify CSV, generates:
- A QR code PNG pointing to the Railway redirect URL (go.nextleveldecor.in/{SKU})
- A printable A4 PDF sheet (4x6 grid, 24 QR codes per page) with SKU labels
- A qr_manifest.csv (SKU, Handle, Redirect_URL, Shopify_URL, QR_Image_Path)
- Merges SKU → handle entries into redirect_service/redirects.json at repo root

QR codes point to the Railway redirect service, not directly to Shopify.
This allows QR codes to be printed before products go live on Shopify.

Usage:
    python 8_generate_qr_codes.py --output-dir outputs/CROMA1_MM_PDF
    python 8_generate_qr_codes.py --all
    python 8_generate_qr_codes.py --all --force
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REDIRECT_BASE_URL = "https://go.nextleveldecor.in/"
SHOPIFY_BASE_URL = "https://nextleveldecor.in/products/"

GRID_COLS = 4
GRID_ROWS = 6
CELLS_PER_PAGE = GRID_COLS * GRID_ROWS
QR_SIZE_PX = 300
CELL_W_MM = 48.0
CELL_H_MM = 45.0
LABEL_H_MM = 6.0
PAGE_MARGIN_MM = 8.0
HEADER_H_MM = 10.0


def slugify_title(title: str, sku: str) -> str:
    """Derive Shopify-compatible URL handle from product title."""
    if not title:
        return re.sub(r"[^a-z0-9]+", "-", sku.lower()).strip("-")
    s = title.lower()
    s = s.replace("*", "x")
    s = s.replace(".", "")
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


def load_products_from_csv(csv_path: Path) -> list:
    """Parse matrixify CSV and return list of product dicts."""
    products = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        header_stripped = [h.strip() for h in header]

        try:
            title_idx = header_stripped.index("Title")
        except ValueError:
            title_idx = 1

        try:
            sku_idx = header_stripped.index("Variant SKU [ID]")
        except ValueError:
            try:
                sku_idx = header_stripped.index("Variant SKU")
            except ValueError:
                sku_idx = 17

        for row in reader:
            if len(row) <= sku_idx:
                continue
            sku = row[sku_idx].strip()
            if not sku or len(sku) > 25 or sku.startswith('"'):
                continue
            title = row[title_idx].strip().strip('"') if len(row) > title_idx else ""
            handle = slugify_title(title, sku)
            products.append({
                "sku": sku,
                "title": title,
                "handle": handle,
                "shopify_url": SHOPIFY_BASE_URL + handle,
                "redirect_url": REDIRECT_BASE_URL + sku,
            })

    return products


def generate_qr_image(url: str, output_path: Path, size_px: int = QR_SIZE_PX) -> Path:
    """Generate a single QR code PNG for the given URL."""
    import qrcode

    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=2,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img = img.resize((size_px, size_px))
    img.save(str(output_path))
    return output_path


def generate_qr_images_batch(products: list, qr_dir: Path, force: bool = False) -> list:
    """Generate QR PNG files for all products. Adds qr_image_path to each dict."""
    qr_dir.mkdir(parents=True, exist_ok=True)
    for product in products:
        safe_sku = product["sku"].replace("/", "_").replace("\\", "_")
        output_path = qr_dir / f"{safe_sku}.png"
        if not force and output_path.exists():
            product["qr_image_path"] = output_path
            continue
        generate_qr_image(product["redirect_url"], output_path)
        product["qr_image_path"] = output_path
    return products


def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def generate_pdf_sheet(products: list, pdf_path: Path, catalog_name: str) -> Path:
    """Generate printable A4 PDF with QR codes in a 4x6 grid."""
    from fpdf import FPDF

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=False)

    qr_w = CELL_W_MM - 10
    qr_h = CELL_H_MM - LABEL_H_MM - 4

    for page_num, page_products in enumerate(_chunks(products, CELLS_PER_PAGE)):
        pdf.add_page()

        pdf.set_font("Helvetica", "B", 9)
        pdf.set_xy(PAGE_MARGIN_MM, PAGE_MARGIN_MM)
        pdf.cell(0, HEADER_H_MM - 2, f"{catalog_name}  |  Page {page_num + 1}", align="C")

        for i, product in enumerate(page_products):
            col = i % GRID_COLS
            row = i // GRID_COLS
            x = PAGE_MARGIN_MM + col * CELL_W_MM
            y = PAGE_MARGIN_MM + HEADER_H_MM + row * CELL_H_MM

            qr_path = product.get("qr_image_path")
            if qr_path and Path(qr_path).exists():
                pdf.image(str(qr_path), x=x + 5, y=y + 2, w=qr_w, h=qr_h)

            pdf.set_font("Courier", size=7)
            pdf.set_xy(x, y + qr_h + 3)
            pdf.cell(CELL_W_MM, LABEL_H_MM, product["sku"], align="C")

    pdf.output(str(pdf_path))
    return pdf_path


def write_manifest_csv(products: list, manifest_path: Path) -> Path:
    """Write SKU/Handle/URL/QR_Image_Path manifest CSV."""
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["SKU", "Handle", "Redirect_URL", "Shopify_URL", "QR_Image_Path"],
        )
        writer.writeheader()
        for p in products:
            writer.writerow({
                "SKU": p["sku"],
                "Handle": p["handle"],
                "Redirect_URL": p["redirect_url"],
                "Shopify_URL": p["shopify_url"],
                "QR_Image_Path": str(p.get("qr_image_path", "")),
            })
    return manifest_path


def update_redirects_json(products: list, repo_root: Path) -> Path:
    """Merge new SKU→handle entries into redirect_service/redirects.json."""
    redirects_path = repo_root / "redirect_service" / "redirects.json"
    redirects_path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if redirects_path.exists():
        with open(redirects_path) as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}

    for p in products:
        existing[p["sku"]] = p["handle"]

    with open(redirects_path, "w") as f:
        json.dump(existing, f, indent=2, sort_keys=True)

    return redirects_path


def run_qr_generation(output_dir: str, config_dir: str = None, force: bool = False) -> dict:
    """Main entry point for QR generation stage (mirrors other stage signatures)."""
    output_dir = Path(output_dir)
    catalog_name = output_dir.name.replace("_PDF", "").replace("_", " ").title()

    csv_files = list(output_dir.glob("matrixify_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No matrixify CSV found in {output_dir}")
    csv_path = csv_files[0]

    products = load_products_from_csv(csv_path)
    if not products:
        return {"status": "no_products", "output_dir": str(output_dir), "product_count": 0}

    qr_dir = output_dir / "qr_codes"
    products = generate_qr_images_batch(products, qr_dir, force=force)

    pdf_path = qr_dir / f"qr_sheet_{output_dir.name}.pdf"
    generate_pdf_sheet(products, pdf_path, catalog_name)

    manifest_path = qr_dir / "qr_manifest.csv"
    write_manifest_csv(products, manifest_path)

    # prompt_based_PDF_extractor/pipeline/ → prompt_based_PDF_extractor/ → next_level_website/
    repo_root = Path(__file__).parent.parent.parent
    redirects_path = update_redirects_json(products, repo_root)

    return {
        "status": "success",
        "output_dir": str(output_dir),
        "catalog_name": catalog_name,
        "product_count": len(products),
        "qr_dir": str(qr_dir),
        "pdf_path": str(pdf_path),
        "manifest_path": str(manifest_path),
        "redirects_json": str(redirects_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate QR codes for product samples")
    parser.add_argument(
        "--output-dir",
        help="Catalog output directory (e.g. outputs/CROMA1_MM_PDF)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all catalogs in outputs/",
    )
    parser.add_argument(
        "--outputs-dir",
        default=None,
        help="Base outputs directory (default: outputs/ relative to script)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate existing QR images",
    )
    args = parser.parse_args()

    base = Path(__file__).parent.parent

    if args.all:
        outputs_dir = Path(args.outputs_dir) if args.outputs_dir else base / "outputs"
        catalogs = [
            d for d in sorted(outputs_dir.iterdir())
            if d.is_dir() and list(d.glob("matrixify_*.csv"))
        ]
        if not catalogs:
            print(f"No catalogs found in {outputs_dir}")
            sys.exit(1)
        total = 0
        for catalog in catalogs:
            try:
                result = run_qr_generation(str(catalog), force=args.force)
                print(f"✓ {catalog.name}: {result.get('product_count')} QR codes")
                total += result.get("product_count", 0)
            except Exception as e:
                print(f"✗ {catalog.name}: {e}")
        print(f"\nTotal: {total} QR codes generated")

    elif args.output_dir:
        result = run_qr_generation(args.output_dir, force=args.force)
        print(f"✓ {result.get('product_count')} QR codes → {result.get('qr_dir')}")
        print(f"  PDF:       {result.get('pdf_path')}")
        print(f"  Manifest:  {result.get('manifest_path')}")
        print(f"  Redirects: {result.get('redirects_json')}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
