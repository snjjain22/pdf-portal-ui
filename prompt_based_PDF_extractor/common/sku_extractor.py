#!/usr/bin/env python3
"""
OCR-based SKU Extractor for PDF Catalog Pages

Extracts SKU/product codes from text near detected product swatches.
Uses spatial matching to associate text labels with the correct swatch.

Based on the proven approach from pdf-catalog-extractor:
1. Run OCR on FULL PAGE to get all text with bounding boxes
2. Use spatial matching to associate SKU text with nearest swatch
3. Prioritize text BELOW the swatch (most common label position)
4. Fall back to auto-generated SKU if no match found
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class SKUExtractor:
    """
    Extracts SKU/product codes from catalog page images using OCR.
    
    Associates detected text with nearby product swatches using spatial matching.
    """
    
    def __init__(
        self,
        engine: str = "easyocr",
        languages: List[str] = None,
        sku_patterns: List[str] = None,
        search_radius_ratio: float = 0.3,
        prefer_position: str = "below",
        auto_sku_template: str = "{pdf_name}-P{page:03d}-S{swatch:02d}"
    ):
        """
        Initialize SKU extractor.
        
        Args:
            engine: OCR engine - "easyocr" or "tesseract"
            languages: List of language codes (e.g., ["en"])
            sku_patterns: List of regex patterns for SKU detection
            search_radius_ratio: Search radius as ratio of swatch height
            prefer_position: Where to look for SKU first - "below", "above", "right", "any"
            auto_sku_template: Template for auto-generated SKUs when OCR fails
        """
        self.engine = engine
        self.languages = languages or ["en"]
        self.search_radius_ratio = search_radius_ratio
        self.prefer_position = prefer_position
        self.auto_sku_template = auto_sku_template
        
        # Compile SKU patterns - common laminate catalog formats
        default_patterns = [
            r"[A-Z]{2,3}[-\s]?\d{3,5}",  # UT-013, SAP-14502, ACR-001
            r"\d{4,6}",  # 4-6 digit codes (3697, 10234)
            r"\d{4,5}\s*[A-Z]{1,3}",  # Digit + suffix (3697 SU)
            r"[A-Z]+\d+",  # Letters followed by numbers (ACR123)
        ]
        self.sku_patterns = [re.compile(p, re.IGNORECASE) for p in (sku_patterns or default_patterns)]
        
        # OCR reader (lazy initialization)
        self._reader = None
    
    def _init_ocr(self):
        """Initialize OCR engine."""
        if self._reader is not None:
            return
        
        if self.engine == "easyocr":
            try:
                import easyocr
                logger.info(f"Initializing EasyOCR with languages: {self.languages}")
                self._reader = easyocr.Reader(self.languages, gpu=True)
                logger.info("✓ EasyOCR initialized (GPU enabled)")
            except Exception as e:
                logger.warning(f"EasyOCR GPU init failed, trying CPU: {e}")
                try:
                    import easyocr
                    self._reader = easyocr.Reader(self.languages, gpu=False)
                    logger.info("✓ EasyOCR initialized (CPU)")
                except ImportError:
                    raise ImportError("easyocr not installed. Install with: pip install easyocr")
        
        elif self.engine == "tesseract":
            try:
                import pytesseract
                pytesseract.get_tesseract_version()
                self._reader = pytesseract
                logger.info("✓ Tesseract OCR initialized")
            except Exception as e:
                raise ImportError(f"Tesseract not available: {e}")
        
        else:
            raise ValueError(f"Unknown OCR engine: {self.engine}")
    
    def extract_all_text(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Extract all text regions from an image.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            List of text regions:
            [
                {
                    "text": str,
                    "bbox": [x_min, y_min, x_max, y_max],
                    "confidence": float
                },
                ...
            ]
        """
        self._init_ocr()
        
        image_path = Path(image_path)
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return []
        
        text_regions = []
        
        if self.engine == "easyocr":
            # Pass file path string directly — EasyOCR loads it itself, avoids cv2 issues
            results = self._reader.readtext(str(image_path))
            
            for bbox, text, confidence in results:
                # EasyOCR bbox is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                text_regions.append({
                    "text": text.strip(),
                    "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
                    "confidence": float(confidence)
                })
        
        elif self.engine == "tesseract":
            from PIL import Image
            import pytesseract
            
            with Image.open(image_path) as img:
                data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])
                
                if text and conf > 0:
                    x = data['left'][i]
                    y = data['top'][i]
                    w = data['width'][i]
                    h = data['height'][i]
                    
                    text_regions.append({
                        "text": text,
                        "bbox": [x, y, x + w, y + h],
                        "confidence": conf / 100.0
                    })
        
        logger.debug(f"Extracted {len(text_regions)} text regions from {image_path.name}")
        return text_regions
    
    def find_sku_in_text(self, text: str) -> Optional[str]:
        """
        Find SKU pattern in text string.
        
        Args:
            text: Text to search
        
        Returns:
            Matched SKU or None
        """
        for pattern in self.sku_patterns:
            match = pattern.search(text)
            if match:
                return match.group(0).strip().upper().replace(" ", "-")
        return None
    
    def _get_center(self, bbox: List[int]) -> Tuple[float, float]:
        """Get center point of a bounding box."""
        x_min, y_min, x_max, y_max = bbox
        return (x_min + x_max) / 2, (y_min + y_max) / 2
    
    def _get_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _is_below(self, text_bbox: List[int], swatch_bbox: List[int]) -> bool:
        """Check if text is below the swatch."""
        return text_bbox[1] > swatch_bbox[3]  # text top > swatch bottom
    
    def _is_above(self, text_bbox: List[int], swatch_bbox: List[int]) -> bool:
        """Check if text is above the swatch."""
        return text_bbox[3] < swatch_bbox[1]  # text bottom < swatch top
    
    def _is_right(self, text_bbox: List[int], swatch_bbox: List[int]) -> bool:
        """Check if text is to the right of the swatch."""
        return text_bbox[0] > swatch_bbox[2]  # text left > swatch right
    
    def _is_left(self, text_bbox: List[int], swatch_bbox: List[int]) -> bool:
        """Check if text is to the left of the swatch."""
        return text_bbox[2] < swatch_bbox[0]  # text right < swatch left
    
    def match_skus_to_detections(
        self,
        detections: List[Dict[str, Any]],
        text_regions: List[Dict[str, Any]],
        image_width: int,
        image_height: int,
        pdf_name: str = "catalog",
        page_number: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Match SKUs from text regions to detected swatches.
        
        Uses spatial proximity and position preference to associate
        the correct SKU with each swatch.
        
        Args:
            detections: List of detections (with bbox_pixels)
            text_regions: List of OCR text regions
            image_width: Image width in pixels
            image_height: Image height in pixels
            pdf_name: PDF name for auto-SKU generation
            page_number: Page number for auto-SKU generation
        
        Returns:
            Updated detections with 'sku' field added
        """
        if not detections:
            return detections
        
        # Filter text regions that contain potential SKUs
        sku_regions = []
        for region in text_regions:
            sku = self.find_sku_in_text(region["text"])
            if sku:
                sku_regions.append({
                    **region,
                    "sku": sku
                })
        
        logger.info(f"Found {len(sku_regions)} potential SKU regions on page")
        
        # Track which SKUs have been assigned
        used_skus = set()
        
        # Process each detection
        for det_idx, detection in enumerate(detections):
            swatch_bbox = detection.get("bbox_pixels") or detection.get("bbox")
            if not swatch_bbox:
                continue
                
            swatch_height = swatch_bbox[3] - swatch_bbox[1]
            swatch_width = swatch_bbox[2] - swatch_bbox[0]
            
            # Find the bottom center of the swatch (most common SKU position)
            swatch_bottom_center = (
                (swatch_bbox[0] + swatch_bbox[2]) / 2,
                swatch_bbox[3]  # bottom edge
            )
            
            # Score each SKU region
            candidates = []
            
            for sku_region in sku_regions:
                if sku_region["sku"] in used_skus:
                    continue
                
                text_bbox = sku_region["bbox"]
                text_center = self._get_center(text_bbox)
                
                # Calculate distance from swatch bottom center to text center
                distance = self._get_distance(swatch_bottom_center, text_center)
                
                # Calculate position score (higher = better)
                position_score = 0
                if self._is_below(text_bbox, swatch_bbox):
                    position_score = 3  # Highest priority - SKU usually below
                elif self._is_right(text_bbox, swatch_bbox):
                    position_score = 2
                elif self._is_left(text_bbox, swatch_bbox):
                    position_score = 1
                elif self._is_above(text_bbox, swatch_bbox):
                    position_score = 1
                
                # Check horizontal alignment (SKU should be roughly centered under swatch)
                swatch_center_x = (swatch_bbox[0] + swatch_bbox[2]) / 2
                text_center_x = (text_bbox[0] + text_bbox[2]) / 2
                horizontal_offset = abs(swatch_center_x - text_center_x)
                
                # Alignment bonus if text is within swatch width
                alignment_score = 2 if horizontal_offset < swatch_width else 0
                
                # Normalize distance by swatch height
                normalized_distance = distance / swatch_height
                
                candidates.append({
                    "sku": sku_region["sku"],
                    "distance": normalized_distance,
                    "position_score": position_score,
                    "alignment_score": alignment_score,
                    "confidence": sku_region["confidence"],
                    "raw_text": sku_region["text"]
                })
            
            # Sort candidates by: position_score (desc), alignment_score (desc), distance (asc)
            candidates.sort(key=lambda c: (-c["position_score"], -c["alignment_score"], c["distance"]))
            
            # Assign the best candidate
            if candidates:
                best = candidates[0]
                # Only assign if within reasonable distance and has good position/alignment
                if best["distance"] < 3.0 and (best["position_score"] > 0 or best["distance"] < 1.0):
                    detection["sku"] = best["sku"]
                    detection["sku_confidence"] = best["confidence"]
                    detection["sku_source"] = "ocr"
                    used_skus.add(best["sku"])
                    logger.debug(f"Detection {det_idx}: Assigned SKU '{best['sku']}' (dist={best['distance']:.2f}, pos={best['position_score']})")
                else:
                    logger.debug(f"Detection {det_idx}: No suitable SKU (best dist={best['distance']:.2f}, pos={best['position_score']})")
            
            # Generate auto-SKU if no match found
            if "sku" not in detection:
                auto_sku = self.auto_sku_template.format(
                    pdf_name=pdf_name,
                    page=page_number,
                    swatch=det_idx + 1
                )
                detection["sku"] = auto_sku
                detection["sku_confidence"] = 0.0
                detection["sku_source"] = "auto"
                logger.debug(f"Detection {det_idx}: Auto-generated SKU '{auto_sku}'")
        
        return detections
    
    def extract_skus(
        self,
        image_path: str,
        detections: List[Dict[str, Any]],
        pdf_name: str = "catalog",
        page_number: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Extract and match SKUs for all detections in an image.
        
        This is the main entry point. It:
        1. Runs OCR on the full page image
        2. Finds all text that matches SKU patterns
        3. Spatially matches each SKU to the nearest swatch
        
        Args:
            image_path: Path to the catalog page image
            detections: List of detections with bbox_pixels
            pdf_name: PDF name for auto-SKU generation
            page_number: Page number for auto-SKU generation
        
        Returns:
            Updated detections with SKU information
        """
        from PIL import Image
        
        if not detections:
            return detections
        
        image_path = Path(image_path)
        
        # Get image dimensions
        with Image.open(image_path) as img:
            img_width, img_height = img.size
        
        # Extract all text from the FULL PAGE
        text_regions = self.extract_all_text(str(image_path))
        
        # Match SKUs to detections using spatial proximity
        updated_detections = self.match_skus_to_detections(
            detections=detections,
            text_regions=text_regions,
            image_width=img_width,
            image_height=img_height,
            pdf_name=pdf_name,
            page_number=page_number
        )
        
        # Log summary
        ocr_count = sum(1 for d in updated_detections if d.get("sku_source") == "ocr")
        auto_count = sum(1 for d in updated_detections if d.get("sku_source") == "auto")
        logger.info(f"SKU extraction: {ocr_count} OCR-matched, {auto_count} auto-generated")
        
        return updated_detections


class SKUExtractorFactory:
    """Factory for creating SKU extractors from config."""
    
    @staticmethod
    def from_config(config: Dict[str, Any]) -> SKUExtractor:
        """
        Create an SKUExtractor from configuration dictionary.
        
        Args:
            config: OCR extraction configuration from website.yaml
        
        Returns:
            Configured SKUExtractor instance
        """
        ocr_config = config.get("ocr_extraction", {})
        
        return SKUExtractor(
            engine=ocr_config.get("engine", "easyocr"),
            languages=ocr_config.get("languages", ["en"]),
            sku_patterns=ocr_config.get("sku_patterns"),
            search_radius_ratio=ocr_config.get("search_radius_ratio", 0.3),
            prefer_position=ocr_config.get("prefer_position", "below"),
            auto_sku_template=ocr_config.get("auto_sku_template", "{pdf_name}-P{page:03d}-S{swatch:02d}")
        )
