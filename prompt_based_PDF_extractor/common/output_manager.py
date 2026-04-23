"""
Output Manager for Prompt-Based PDF Extractor
Manages per-PDF output directories and file organization.
"""

import os
import shutil
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List


class OutputManager:
    """Manages output directories and files for PDF extraction."""
    
    def __init__(self, base_output_dir: str = "outputs"):
        """
        Initialize OutputManager.
        
        Args:
            base_output_dir: Base directory for all outputs
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        self._current_pdf_dir: Optional[Path] = None
        self._current_pdf_name: Optional[str] = None
    
    def init_pdf_output(self, pdf_path: str, clean: bool = False) -> Path:
        """
        Initialize output directory for a PDF.
        
        Args:
            pdf_path: Path to the PDF file
            clean: If True, remove existing output directory
        
        Returns:
            Path to the PDF's output directory
        """
        pdf_name = Path(pdf_path).stem
        # Sanitize directory name
        safe_name = self._sanitize_name(pdf_name)
        
        pdf_output_dir = self.base_output_dir / f"{safe_name}_PDF"
        
        if clean and pdf_output_dir.exists():
            shutil.rmtree(pdf_output_dir)
        
        # Create subdirectories
        subdirs = [
            "pages",           # Full page images
            "products",        # Cropped product images
            "thumbnails",      # Product thumbnails
            "data",            # JSON/CSV data files
            "logs",            # Processing logs
        ]
        
        for subdir in subdirs:
            (pdf_output_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        self._current_pdf_dir = pdf_output_dir
        self._current_pdf_name = safe_name
        
        return pdf_output_dir
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize a name for use as a directory name."""
        # Replace spaces and special characters
        safe = name.replace(" ", "_").replace("-", "_")
        # Remove any non-alphanumeric characters except underscores
        safe = "".join(c if c.isalnum() or c == "_" else "" for c in safe)
        return safe.upper()
    
    @property
    def current_dir(self) -> Path:
        """Get current PDF output directory."""
        if self._current_pdf_dir is None:
            raise RuntimeError("No PDF output initialized. Call init_pdf_output first.")
        return self._current_pdf_dir
    
    @property
    def pages_dir(self) -> Path:
        """Get pages subdirectory."""
        return self.current_dir / "pages"
    
    @property
    def products_dir(self) -> Path:
        """Get products subdirectory."""
        return self.current_dir / "products"
    
    @property
    def thumbnails_dir(self) -> Path:
        """Get thumbnails subdirectory."""
        return self.current_dir / "thumbnails"
    
    @property
    def data_dir(self) -> Path:
        """Get data subdirectory."""
        return self.current_dir / "data"
    
    @property
    def logs_dir(self) -> Path:
        """Get logs subdirectory."""
        return self.current_dir / "logs"
    
    # File path helpers
    def get_page_image_path(self, page_num: int, extension: str = "png") -> Path:
        """Get path for a page image."""
        return self.pages_dir / f"page_{page_num:03d}.{extension}"
    
    def get_product_image_path(self, sku: str, extension: str = "png") -> Path:
        """Get path for a cropped product image."""
        safe_sku = sku.replace("/", "_").replace("\\", "_")
        return self.products_dir / f"{safe_sku}.{extension}"
    
    def get_thumbnail_path(self, sku: str, extension: str = "png") -> Path:
        """Get path for a product thumbnail."""
        safe_sku = sku.replace("/", "_").replace("\\", "_")
        return self.thumbnails_dir / f"{safe_sku}_thumb.{extension}"
    
    # Data file management
    def save_json(self, data: Any, filename: str):
        """Save data as JSON in the data directory."""
        path = self.data_dir / filename
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path
    
    def load_json(self, filename: str) -> Any:
        """Load JSON data from the data directory."""
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_csv(self, df: pd.DataFrame, filename: str):
        """Save DataFrame as CSV in the data directory."""
        path = self.data_dir / filename
        df.to_csv(path, index=False, encoding='utf-8')
        return path
    
    def load_csv(self, filename: str) -> pd.DataFrame:
        """Load CSV from the data directory."""
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        return pd.read_csv(path, encoding='utf-8')
    
    # Stage output file names
    def get_detection_output_path(self) -> Path:
        """Path for detection stage output."""
        return self.data_dir / "1_detection_results.json"
    
    def get_field_mapping_output_path(self) -> Path:
        """Path for field mapping stage output."""
        return self.data_dir / "2_field_mapping_results.json"
    
    def get_seo_output_path(self) -> Path:
        """Path for SEO stage output."""
        return self.data_dir / "3_seo_results.json"
    
    def get_matrixify_output_path(self) -> Path:
        """Path for Matrixify CSV output."""
        return self.data_dir / f"4_matrixify_{self._current_pdf_name}.csv"
    
    # Logging
    def log(self, stage: str, message: str, level: str = "INFO"):
        """Append a log message."""
        log_file = self.logs_dir / "pipeline.log"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] [{stage}] {message}\n"
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    
    def save_stage_log(self, stage: str, data: Dict[str, Any]):
        """Save detailed stage execution log."""
        log_file = self.logs_dir / f"{stage}_log.json"
        data["timestamp"] = datetime.now().isoformat()
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Utility methods
    def list_page_images(self) -> List[Path]:
        """List all page images in order."""
        images = list(self.pages_dir.glob("page_*.png"))
        return sorted(images)
    
    def list_product_images(self) -> List[Path]:
        """List all cropped product images."""
        return list(self.products_dir.glob("*.png"))
    
    def get_pdf_info(self) -> Dict[str, Any]:
        """Get info about current PDF processing."""
        return {
            "pdf_name": self._current_pdf_name,
            "output_dir": str(self.current_dir),
            "page_count": len(self.list_page_images()),
            "product_count": len(self.list_product_images()),
        }
    
    def cleanup_temp_files(self):
        """Remove any temporary files created during processing."""
        # Remove any .tmp files
        for tmp_file in self.current_dir.rglob("*.tmp"):
            tmp_file.unlink()


# Convenience function
def get_output_manager(base_output_dir: str = "outputs") -> OutputManager:
    """Create and return an OutputManager instance."""
    return OutputManager(base_output_dir)
