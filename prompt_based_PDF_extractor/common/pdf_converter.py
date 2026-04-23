"""
PDF to Image Converter for Prompt-Based PDF Extractor
Uses PyMuPDF (fitz) — no poppler or system dependencies required.
"""

import os
from pathlib import Path
from typing import List, Optional
from PIL import Image
import fitz  # PyMuPDF


class PDFConverter:
    """Converts PDF files to images using PyMuPDF."""

    def __init__(self, dpi: int = 600, poppler_path: Optional[str] = None):
        # poppler_path kept for API compatibility — unused with PyMuPDF
        self.dpi = dpi

    def get_page_count(self, pdf_path: str) -> int:
        """Return the number of pages in a PDF."""
        with fitz.open(str(pdf_path)) as doc:
            return len(doc)

    def convert_pdf_to_images(
        self,
        pdf_path: str,
        output_dir: str,
        first_page: Optional[int] = None,
        last_page: Optional[int] = None,
        fmt: str = "jpeg",
    ) -> List[Path]:
        """
        Convert PDF pages to images and save them to output_dir.

        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save images
            first_page: First page (1-indexed, None = 1)
            last_page: Last page (1-indexed, None = last)
            fmt: Output format ('jpeg' or 'png')

        Returns:
            List of saved image paths
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        print(f"Converting PDF: {pdf_path.name}")
        print(f"  DPI: {self.dpi}")
        print(f"  Output: {output_dir}")

        # Scale factor: PDF units are 72 DPI by default
        scale = self.dpi / 72.0
        mat = fitz.Matrix(scale, scale)

        saved_paths = []

        with fitz.open(str(pdf_path)) as doc:
            total_pages = len(doc)
            start = (first_page or 1) - 1       # convert to 0-indexed
            end = min(last_page or total_pages, total_pages)  # inclusive page number

            for page_idx in range(start, end):
                page = doc[page_idx]
                pix = page.get_pixmap(matrix=mat, alpha=False)

                # Convert to PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                page_num = page_idx + 1
                ext = "jpg" if fmt.lower() in ("jpeg", "jpg") else fmt.lower()
                output_path = output_dir / f"page_{page_num:03d}.{ext}"

                pil_fmt = "JPEG" if ext == "jpg" else fmt.upper()
                img.save(str(output_path), pil_fmt)
                saved_paths.append(output_path)
                print(f"  Saved: {output_path.name}")

        print(f"Converted {len(saved_paths)} pages")
        return saved_paths

    def convert_single_page(
        self,
        pdf_path: str,
        page_num: int,
        output_path: Optional[str] = None,
    ) -> Image.Image:
        """
        Convert a single page from a PDF.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (1-indexed)
            output_path: Optional path to save the image

        Returns:
            PIL Image object
        """
        scale = self.dpi / 72.0
        mat = fitz.Matrix(scale, scale)

        with fitz.open(str(pdf_path)) as doc:
            page = doc[page_num - 1]
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        if output_path:
            img.save(str(output_path))

        return img


def convert_pdf(
    pdf_path: str,
    output_dir: str,
    dpi: int = 600,
    poppler_path: Optional[str] = None,
) -> List[Path]:
    """Convenience function to convert a PDF to images."""
    converter = PDFConverter(dpi=dpi)
    return converter.convert_pdf_to_images(pdf_path, output_dir)
