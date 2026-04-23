"""
PDF to Image Converter for Prompt-Based PDF Extractor
Converts PDF pages to high-resolution images using pdf2image.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple
from pdf2image import convert_from_path
from PIL import Image


class PDFConverter:
    """Converts PDF files to images."""
    
    def __init__(self, dpi: int = 600, poppler_path: Optional[str] = None):
        """
        Initialize PDF converter.
        
        Args:
            dpi: Resolution for converted images (higher = better quality but larger files)
            poppler_path: Path to Poppler binaries (required on Windows)
        """
        self.dpi = dpi
        self.poppler_path = poppler_path
        
        # Auto-detect Poppler on Windows if not provided
        if self.poppler_path is None and os.name == 'nt':
            # Common Poppler installation paths on Windows
            common_paths = [
                r"C:\Program Files\poppler\Library\bin",
                r"C:\Program Files\poppler\bin",
                r"C:\poppler\Library\bin",
                r"C:\poppler\bin",
                os.path.expanduser(r"~\poppler\Library\bin"),
            ]
            for path in common_paths:
                if os.path.exists(path):
                    self.poppler_path = path
                    break
    
    def convert_pdf_to_images(
        self, 
        pdf_path: str, 
        output_dir: str,
        first_page: Optional[int] = None,
        last_page: Optional[int] = None,
        fmt: str = "png"
    ) -> List[Path]:
        """
        Convert PDF pages to images.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save images
            first_page: First page to convert (1-indexed, None = first)
            last_page: Last page to convert (1-indexed, None = last)
            fmt: Output image format (png, jpeg, etc.)
        
        Returns:
            List of paths to generated images
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        print(f"Converting PDF: {pdf_path.name}")
        print(f"  DPI: {self.dpi}")
        print(f"  Output: {output_dir}")
        
        # Convert PDF to images
        convert_kwargs = {
            "pdf_path": str(pdf_path),
            "dpi": self.dpi,
            "fmt": fmt,
        }
        
        if self.poppler_path:
            convert_kwargs["poppler_path"] = self.poppler_path
        
        if first_page:
            convert_kwargs["first_page"] = first_page
        if last_page:
            convert_kwargs["last_page"] = last_page
        
        # Try conversion, but handle very large page images that trigger
        # Pillow's DecompressionBombError by retrying at lower DPI values.
        images = None
        dpi_fallbacks = [self.dpi]
        if self.dpi and self.dpi > 300:
            dpi_fallbacks.append(300)
        if 150 not in dpi_fallbacks:
            dpi_fallbacks.append(150)

        last_exc = None
        for dpi_try in dpi_fallbacks:
            convert_kwargs["dpi"] = dpi_try
            try:
                print(f"Attempting conversion at {dpi_try} DPI")
                images = convert_from_path(**convert_kwargs)
                break
            except Image.DecompressionBombError as e:
                print(f"DecompressionBombError at {dpi_try} DPI: {e}. Retrying at lower DPI.")
                last_exc = e

        if images is None:
            # If all retries failed, surface a clear error
            raise RuntimeError("PDF conversion failed due to large image size (DecompressionBomb).") from last_exc
        
        # Save images with consistent naming
        saved_paths = []
        start_page = first_page or 1
        
        for i, image in enumerate(images):
            page_num = start_page + i
            output_path = output_dir / f"page_{page_num:03d}.{fmt}"
            image.save(str(output_path), fmt.upper())
            saved_paths.append(output_path)
            print(f"  Saved: page_{page_num:03d}.{fmt}")
        
        print(f"Converted {len(saved_paths)} pages")
        return saved_paths
    
    def get_page_count(self, pdf_path: str) -> int:
        """
        Get the number of pages in a PDF.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Number of pages
        """
        from pdf2image.pdf2image import pdfinfo_from_path
        
        info_kwargs = {"pdf_path": str(pdf_path)}
        if self.poppler_path:
            info_kwargs["poppler_path"] = self.poppler_path
        
        info = pdfinfo_from_path(**info_kwargs)
        return info.get("Pages", 0)
    
    def convert_single_page(
        self, 
        pdf_path: str, 
        page_num: int,
        output_path: Optional[str] = None
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
        convert_kwargs = {
            "pdf_path": str(pdf_path),
            "dpi": self.dpi,
            "first_page": page_num,
            "last_page": page_num,
        }
        
        if self.poppler_path:
            convert_kwargs["poppler_path"] = self.poppler_path
        
        # Handle potential DecompressionBombError for very large single pages
        images = None
        dpi_fallbacks = [self.dpi]
        if self.dpi and self.dpi > 300:
            dpi_fallbacks.append(300)
        if 150 not in dpi_fallbacks:
            dpi_fallbacks.append(150)

        last_exc = None
        for dpi_try in dpi_fallbacks:
            convert_kwargs["dpi"] = dpi_try
            try:
                images = convert_from_path(**convert_kwargs)
                break
            except Image.DecompressionBombError as e:
                print(f"DecompressionBombError converting single page at {dpi_try} DPI: {e}. Retrying at lower DPI.")
                last_exc = e

        if images is None:
            raise RuntimeError("Single-page PDF conversion failed due to large image size (DecompressionBomb).") from last_exc
        
        if not images:
            raise ValueError(f"Failed to convert page {page_num}")
        
        image = images[0]
        
        if output_path:
            image.save(output_path)
        
        return image


def convert_pdf(
    pdf_path: str, 
    output_dir: str, 
    dpi: int = 600,
    poppler_path: Optional[str] = None
) -> List[Path]:
    """
    Convenience function to convert a PDF to images.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save images
        dpi: Resolution for converted images
        poppler_path: Path to Poppler binaries
    
    Returns:
        List of paths to generated images
    """
    converter = PDFConverter(dpi=dpi, poppler_path=poppler_path)
    return converter.convert_pdf_to_images(pdf_path, output_dir)
