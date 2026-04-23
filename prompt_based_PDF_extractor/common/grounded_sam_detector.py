"""
Grounded SAM Detector for Prompt-Based PDF Extractor

Uses Grounding DINO for text-prompted object detection and optionally SAM for segmentation.
Detects laminate swatches/product samples in catalog pages.
"""

import io
import os
import base64
import tempfile
import logging
import requests
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

logger = logging.getLogger(__name__)


def _encode_image_for_vlm(image_path: str, max_size: int = 1024) -> tuple:
    """
    Resize image to max_size on the longest side and return (mime_type, base64_str).
    Keeps file size small to avoid API upload timeouts.
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        ext = Path(image_path).suffix.lower()
        fmt = "JPEG" if ext in [".jpg", ".jpeg"] else "PNG"
        mime = "image/jpeg" if fmt == "JPEG" else "image/png"
        buf = io.BytesIO()
        img.save(buf, format=fmt, quality=85)
        return mime, base64.b64encode(buf.getvalue()).decode("utf-8")


class GroundedSAMDetector:
    """
    Grounding DINO + SAM based detector for product swatches.
    
    Uses text prompts to detect laminate/panel samples with filtering.
    """
    
    def __init__(
        self,
        dino_weights: str = "models/groundingdino_swint_ogc.pth",
        sam_checkpoint: Optional[str] = "models/sam_vit_b_01ec64.pth",
        prompt: str = "laminate swatch . decorative panel sample . textured sheet",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        nms_threshold: float = 0.5,
        min_area_pct: float = 3.0,
        max_area_pct: float = 40.0,
        min_aspect: float = 0.4,
        use_sam: bool = False,
        device: str = "auto"
    ):
        """
        Initialize detector.
        
        Args:
            dino_weights: Path to Grounding DINO weights
            sam_checkpoint: Path to SAM checkpoint (optional)
            prompt: Detection prompt for Grounding DINO
            box_threshold: Confidence threshold for boxes
            text_threshold: Confidence threshold for text matching
            nms_threshold: IoU threshold for NMS
            min_area_pct: Minimum detection area as % of page
            max_area_pct: Maximum detection area as % of page
            min_aspect: Minimum aspect ratio (height/width or width/height)
            use_sam: Whether to use SAM for segmentation refinement
            device: Device to use (auto, cuda, cpu)
        """
        self.dino_weights = Path(dino_weights)
        self.sam_checkpoint = Path(sam_checkpoint) if sam_checkpoint else None
        self.prompt = prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_threshold = nms_threshold
        self.min_area_pct = min_area_pct
        self.max_area_pct = max_area_pct
        self.min_aspect = min_aspect
        self.use_sam = use_sam
        
        # Models (lazy loaded)
        self._dino_model = None
        self._sam_predictor = None
        self._device = None
        
        # Detect device
        if device == "auto":
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device
        
        logger.info(f"GroundedSAMDetector initialized")
        logger.info(f"  Device: {self._device}")
        logger.info(f"  Prompt: {self.prompt}")
        logger.info(f"  Area range: {self.min_area_pct}% - {self.max_area_pct}%")
    
    def _load_dino(self):
        """Load Grounding DINO model."""
        if self._dino_model is not None:
            return
        
        try:
            from groundingdino.util.inference import load_model
            import groundingdino
        except ImportError:
            raise ImportError(
                "GroundingDINO not installed. Install with:\n"
                "pip install groundingdino-py"
            )
        
        if not self.dino_weights.exists():
            raise FileNotFoundError(
                f"Grounding DINO weights not found: {self.dino_weights}\n"
                "Download from: https://github.com/IDEA-Research/GroundingDINO/releases"
            )
        
        # Find config file
        gd_path = Path(groundingdino.__file__).parent
        config_path = gd_path / "config" / "GroundingDINO_SwinT_OGC.py"
        
        logger.info("Loading Grounding DINO...")
        self._dino_model = load_model(str(config_path), str(self.dino_weights), device=self._device)
        logger.info("✓ Grounding DINO loaded")
    
    def _load_sam(self):
        """Load SAM predictor."""
        if self._sam_predictor is not None:
            return
        
        if not self.use_sam or self.sam_checkpoint is None:
            return
        
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            logger.warning("segment-anything not installed, SAM disabled")
            self.use_sam = False
            return
        
        if not self.sam_checkpoint.exists():
            logger.warning(f"SAM checkpoint not found: {self.sam_checkpoint}, SAM disabled")
            self.use_sam = False
            return
        
        logger.info("Loading SAM...")
        sam = sam_model_registry["vit_b"](checkpoint=str(self.sam_checkpoint))
        sam.to(device=self._device)
        self._sam_predictor = SamPredictor(sam)
        logger.info("✓ SAM loaded")
    
    def detect(self, image_path: str) -> Dict[str, Any]:
        """
        Detect product swatches in an image.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Dict with detection results
        """
        # Load models
        self._load_dino()
        if self.use_sam:
            self._load_sam()
        
        # Load image
        image = np.array(Image.open(image_path).convert("RGB"))
        h, w = image.shape[:2]
        page_area = h * w
        
        # Run Grounding DINO
        boxes, scores, phrases = self._detect_with_dino(image)
        
        # Filter by area and aspect ratio
        boxes, scores, phrases = self._filter_detections(boxes, scores, phrases, page_area)
        
        # Build results
        detections = []
        for i, (box, score, phrase) in enumerate(zip(boxes, scores, phrases)):
            detection = {
                "index": i,
                "bbox_pixels": box,
                "confidence": round(score, 3),
                "phrase": phrase,
                "area_pct": round((box[2]-box[0]) * (box[3]-box[1]) / page_area * 100, 2)
            }
            detections.append(detection)
        
        return {
            "image_path": str(image_path),
            "image_width": w,
            "image_height": h,
            "detections": detections,
            "detection_count": len(detections)
        }
    
    def _detect_with_dino(self, image: np.ndarray) -> Tuple[List, List, List]:
        """Run Grounding DINO detection."""
        from groundingdino.util.inference import load_image, predict
        
        # Save to temp file (DINO requires file path)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name
            Image.fromarray(image).save(tmp_path)
        
        try:
            image_source, image_transformed = load_image(tmp_path)
            
            boxes, logits, phrases = predict(
                model=self._dino_model,
                image=image_transformed,
                caption=self.prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self._device
            )
        finally:
            os.unlink(tmp_path)
        
        # Convert normalized boxes to pixel coordinates
        h, w = image.shape[:2]
        boxes_pixel = []
        for box in boxes:
            cx, cy, bw, bh = box.tolist()
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            boxes_pixel.append([x1, y1, x2, y2])
        
        # Apply NMS
        boxes_pixel, logits_list, phrases = self._apply_nms(
            boxes_pixel, logits.tolist(), list(phrases)
        )
        
        return boxes_pixel, logits_list, phrases
    
    def _apply_nms(self, boxes, scores, phrases) -> Tuple[List, List, List]:
        """Apply Non-Maximum Suppression."""
        if len(boxes) == 0:
            return boxes, scores, phrases
        
        boxes_arr = np.array(boxes)
        scores_arr = np.array(scores)
        
        x1 = boxes_arr[:, 0]
        y1 = boxes_arr[:, 1]
        x2 = boxes_arr[:, 2]
        y2 = boxes_arr[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        order = scores_arr.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h
            
            iou = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-6)
            inds = np.where(iou <= self.nms_threshold)[0]
            order = order[inds + 1]
        
        return (
            [boxes[i] for i in keep],
            [scores[i] for i in keep],
            [phrases[i] for i in keep]
        )
    
    def _filter_detections(self, boxes, scores, phrases, page_area: int) -> Tuple[List, List, List]:
        """Filter by area and aspect ratio."""
        filtered_boxes = []
        filtered_scores = []
        filtered_phrases = []
        
        for box, score, phrase in zip(boxes, scores, phrases):
            box_w = box[2] - box[0]
            box_h = box[3] - box[1]
            box_area = box_w * box_h
            
            # Check area
            area_pct = (box_area / page_area) * 100
            if area_pct < self.min_area_pct or area_pct > self.max_area_pct:
                continue
            
            # Check aspect ratio
            aspect = min(box_w, box_h) / max(box_w, box_h)
            if aspect < self.min_aspect:
                continue
            
            filtered_boxes.append(box)
            filtered_scores.append(score)
            filtered_phrases.append(phrase)
        
        return filtered_boxes, filtered_scores, filtered_phrases
    
    def crop_detections(
        self, 
        image_path: str, 
        detections: List[Dict], 
        output_dir: str,
        padding: int = 5
    ) -> List[Dict]:
        """
        Crop detected regions and save to files.
        
        Args:
            image_path: Path to source image
            detections: List of detection dicts with bbox_pixels
            output_dir: Directory to save cropped images
            padding: Padding around bounding box
        
        Returns:
            List of detections with added 'crop_path' field
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image = Image.open(image_path)
        w, h = image.size
        
        for i, det in enumerate(detections):
            bbox = det["bbox_pixels"]
            
            # Apply padding
            x1 = max(0, bbox[0] - padding)
            y1 = max(0, bbox[1] - padding)
            x2 = min(w, bbox[2] + padding)
            y2 = min(h, bbox[3] + padding)
            
            # Crop
            crop = image.crop((x1, y1, x2, y2))
            
            # Generate filename
            page_name = Path(image_path).stem
            crop_path = output_dir / f"{page_name}_crop_{i:02d}.png"
            crop.save(crop_path)
            
            det["crop_path"] = str(crop_path)
            det["crop_size"] = crop.size
        
        return detections
    
    def visualize_detections(
        self, 
        image_path: str, 
        detections: List[Dict], 
        output_path: Optional[str] = None
    ) -> Image.Image:
        """
        Draw bounding boxes on image for visualization.
        
        Args:
            image_path: Path to source image
            detections: List of detection dicts
            output_path: Optional path to save visualization
        
        Returns:
            PIL Image with drawn boxes
        """
        image = Image.open(image_path).copy()
        draw = ImageDraw.Draw(image)
        
        for det in detections:
            bbox = det["bbox_pixels"]
            conf = det.get("confidence", 0)
            
            # Draw box
            draw.rectangle(bbox, outline="red", width=3)
            
            # Draw label
            label = f"{conf:.2f}"
            draw.text((bbox[0], bbox[1] - 15), label, fill="red")
        
        if output_path:
            image.save(output_path)
        
        return image


class VLMPageClassifier:
    """
    VLM-based page classifier to filter out application/lifestyle pages.
    Also supports dynamic prompt generation for Grounding DINO.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: str = "https://openrouter.ai/api/v1/chat/completions",
        model: str = "meta-llama/llama-4-maverick"
    ):
        """
        Initialize page classifier.
        
        Args:
            api_key: OpenRouter API key
            api_url: API endpoint URL
            model: Model to use for classification
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.api_url = api_url
        self.model = model
        
        self._prompt = None
        self._dynamic_dino_prompt = None  # Stores the generated DINO prompt
    
    def set_prompt(self, prompt: str):
        """Set custom classification prompt."""
        self._prompt = prompt
    
    def get_dynamic_dino_prompt(self) -> Optional[str]:
        """Get the dynamically generated DINO prompt."""
        return self._dynamic_dino_prompt
    
    def analyze_sample_pages_for_prompt(
        self, 
        page_images: List[str], 
        num_product_pages: int = 5
    ) -> Optional[str]:
        """
        Analyze sample pages to generate an optimized Grounding DINO prompt.
        
        This method iterates through pages until it finds num_product_pages that
        contain actual products, describes what those products look like, then
        generates an optimal detection prompt.
        
        IMPORTANT: Only uses descriptions from ACTUAL PRODUCT PAGES, not covers
        or application/lifestyle photos.
        
        Args:
            page_images: List of page image paths
            num_product_pages: Number of PRODUCT pages to collect descriptions from
            
        Returns:
            Optimized detection prompt for Grounding DINO, or None if failed
        """
        if not self.api_key:
            logger.warning("No API key - cannot generate dynamic prompt")
            return None
        
        total_pages = len(page_images)
        if total_pages == 0:
            return None
        
        logger.info(f"Analyzing pages to find {num_product_pages} product pages for prompt generation...")
        
        # Iterate through pages until we have enough product descriptions
        product_descriptions = []
        pages_checked = 0
        max_pages_to_check = min(total_pages, 30)  # Don't check more than 30 pages
        
        for page_path in page_images:
            if len(product_descriptions) >= num_product_pages:
                break
            
            if pages_checked >= max_pages_to_check:
                logger.info(f"  Checked {max_pages_to_check} pages, stopping search")
                break
            
            pages_checked += 1
            page_name = Path(page_path).name
            
            logger.info(f"  Checking page {pages_checked}: {page_name}")
            
            # Ask VLM to describe products on this page
            # This returns None if it's NOT a product page
            description = self._describe_products_on_page(page_path)
            
            if description:
                product_descriptions.append(description)
                logger.info(f"    ✓ Product page! ({len(product_descriptions)}/{num_product_pages} collected)")
                logger.debug(f"    Description: {description[:100]}...")
            else:
                logger.info(f"    ✗ Not a product page (cover/application/text)")
        
        if not product_descriptions:
            logger.warning("Could not find any product pages in first 30 pages - will use default prompt")
            return None
        
        if len(product_descriptions) < num_product_pages:
            logger.warning(f"Only found {len(product_descriptions)} product pages (wanted {num_product_pages})")
        
        logger.info(f"\nGenerating DINO prompt from {len(product_descriptions)} product page descriptions...")
        
        # Generate optimized prompt based on observations
        self._dynamic_dino_prompt = self._generate_dino_prompt(product_descriptions)
        
        if self._dynamic_dino_prompt:
            logger.info(f"✓ Generated dynamic DINO prompt: \"{self._dynamic_dino_prompt}\"")
        
        return self._dynamic_dino_prompt
    
    def generate_page_specific_prompt(self, image_path: str) -> Optional[str]:
        """
        Generate a DINO prompt specifically for a single page that had 0 detections.
        Uses VLM to describe the page and DeepSeek to generate an optimized prompt.
        
        Args:
            image_path: Path to page image
            
        Returns:
            Page-specific detection prompt, or None if failed
        """
        logger.info(f"    Generating page-specific DINO prompt for: {Path(image_path).name}")
        
        # Step 1: Get VLM description of what's on this page
        description = self._describe_products_on_page(image_path)
        
        if not description:
            logger.warning(f"    Could not describe products on {Path(image_path).name}")
            return None
        
        logger.info(f"    VLM description: {description[:100]}...")
        
        # Step 2: Generate a simpler, more targeted prompt for this page
        try:
            deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
            if not deepseek_key:
                return self._generate_fallback_prompt([description])
            
            prompt = f"""A product detection model (Grounding DINO) found 0 products on a catalog page.
The VLM describes this page as: "{description}"

Generate a BETTER, more GENERIC detection prompt to find these products.

RULES:
1. Create 5-7 short phrases separated by " . "
2. Each phrase should be 2-3 words maximum  
3. Use SIMPLE, GENERIC object terms that Grounding DINO can understand
4. ALWAYS start with "product sample . panel"
5. Add material terms: "wood panel", "stone slab", "colored sheet" etc based on description
6. Add shape terms: "rectangular panel", "arch panel", "framed panel" etc
7. Keep phrases VERY SIMPLE - avoid compound descriptors

RESPOND WITH ONLY THE PROMPT STRING. No quotes, no explanation."""

            response = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers={
                    "Authorization": f"Bearer {deepseek_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 100,
                    "temperature": 0.3
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                generated = result["choices"][0]["message"]["content"].strip().strip('"\'')
                
                if " . " not in generated:
                    if ", " in generated:
                        generated = generated.replace(", ", " . ")
                    elif "," in generated:
                        generated = generated.replace(",", " . ")
                
                if len(generated) > 200:
                    parts = generated.split(" . ")[:6]
                    generated = " . ".join(parts)
                
                logger.info(f"    Page-specific prompt: \"{generated}\"")
                return generated
            else:
                return self._generate_fallback_prompt([description])
                
        except Exception as e:
            logger.warning(f"    Error generating page-specific prompt: {e}")
            return self._generate_fallback_prompt([description])
    
    def _describe_products_on_page(self, image_path: str) -> Optional[str]:
        """
        Ask VLM to describe what product samples look like on a page.
        
        Args:
            image_path: Path to page image
            
        Returns:
            Description of products, or None if not a product page
        """
        try:
            mime_type, image_data = _encode_image_for_vlm(image_path, max_size=1024)

            prompt = """Look at this catalog page. If it contains product samples or swatches, describe them briefly:

1. **Material type**: What material category? (wood grain, stone/marble, solid color, abstract/geometric pattern, fabric/leather texture, metallic, concrete)
2. **Shape**: What shape are the products? (rectangular, square, arch-topped with flat bottom, circular, irregular)
3. **Surface**: What does the surface look like? (natural grain, veined, grooved/fluted, textured, glossy, matte)
4. **Size on page**: How much of the page does each product take? (small swatch, medium panel, large display)
5. **Labels**: Are there product codes/SKU numbers printed near the products? (yes/no)

If this is NOT a product page (it's a room photo, cover, index, or text page), respond with exactly: NOT_PRODUCT_PAGE

Be very concise - 2-3 sentences maximum. Always mention the material type first."""

            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}}
                        ]
                    }],
                    "max_tokens": 150,
                    "temperature": 0.3
                },
                timeout=90
            )
            
            if response.status_code == 200:
                result = response.json()
                description = result["choices"][0]["message"]["content"].strip()
                
                # Check if it's a product page
                if "NOT_PRODUCT_PAGE" in description.upper():
                    return None
                
                return description
            else:
                logger.warning(f"VLM API error {response.status_code}: {response.text[:200]}")
                return None
            
        except Exception as e:
            logger.warning(f"Error describing page {Path(image_path).name}: {e}")
            return None
    
    def _generate_dino_prompt(self, descriptions: List[str]) -> Optional[str]:
        """
        Generate optimized Grounding DINO prompt based on VLM observations.
        
        Args:
            descriptions: List of product descriptions from sample pages
            
        Returns:
            Optimized detection prompt
        """
        try:
            # Get DeepSeek API key
            deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
            if not deepseek_key:
                logger.warning("No DeepSeek API key - using fallback prompt generation")
                return self._generate_fallback_prompt(descriptions)
            
            combined_descriptions = "\n".join([f"- Page {i+1}: {d}" for i, d in enumerate(descriptions)])
            
            prompt = f"""Based on these descriptions of product samples from a catalog PDF:

{combined_descriptions}

Generate an optimized object detection prompt for Grounding DINO model.

RULES:
1. Create a prompt with 5-7 alternative descriptions separated by " . "
2. Each phrase should be 2-4 words
3. ALWAYS include these GENERIC terms: "product sample" and "decorative panel" (these help detect ALL products)
4. Add MATERIAL-SPECIFIC terms based on the descriptions:
   - If wood grain/wood: include "wood panel" or "wood grain sheet"
   - If stone/marble: include "stone slab" or "marble panel"
   - If solid color: include "colored panel" or "solid surface"
   - If geometric/abstract: include "patterned panel" or "decorative sheet"
5. Add SHAPE terms based on what you see:
   - If arch-shaped: include "panel with arch top"
   - If rectangular: include "rectangular sheet"
   - If with border/frame: include "framed panel"
6. Keep prompts GENERIC enough to catch ALL variations — avoid overly specific phrases
7. Do NOT use compound phrases like "arch-shaped wood sample" — split into simpler terms

EXAMPLES of good prompts:
- "product sample . decorative panel . wood grain sheet . rectangular panel . framed sample . surface display"
- "product sample . decorative panel . grooved wall panel . fluted surface . vertical stripe panel . material sheet"
- "product sample . decorative panel . stone slab . marble surface . rectangular sheet . material sample"

Now generate the best prompt for detecting these products.
RESPOND WITH ONLY THE PROMPT STRING, nothing else. No quotes, no explanation."""

            response = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers={
                    "Authorization": f"Bearer {deepseek_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 100,
                    "temperature": 0.3
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_prompt = result["choices"][0]["message"]["content"].strip()
                
                # Clean up - remove quotes if present
                generated_prompt = generated_prompt.strip('"\'')
                
                # Validate: should contain " . " separators
                if " . " not in generated_prompt:
                    # Try to fix if it has other separators
                    if ", " in generated_prompt:
                        generated_prompt = generated_prompt.replace(", ", " . ")
                    elif "," in generated_prompt:
                        generated_prompt = generated_prompt.replace(",", " . ")
                
                # Ensure it's not too long (DINO has limits)
                if len(generated_prompt) > 200:
                    # Take first few phrases
                    parts = generated_prompt.split(" . ")[:5]
                    generated_prompt = " . ".join(parts)
                
                return generated_prompt
            else:
                logger.warning(f"DeepSeek API error {response.status_code}")
                return self._generate_fallback_prompt(descriptions)
            
        except Exception as e:
            logger.warning(f"Error generating DINO prompt: {e}")
            return self._generate_fallback_prompt(descriptions)
    
    def _generate_fallback_prompt(self, descriptions: List[str]) -> str:
        """
        Generate a fallback prompt by extracting keywords from descriptions.
        
        Args:
            descriptions: List of product descriptions
            
        Returns:
            Fallback detection prompt
        """
        # Combine all descriptions
        combined = " ".join(descriptions).lower()
        
        # Build prompt based on keywords found
        prompt_parts = ["product sample"]
        
        # Shape keywords
        if "arch" in combined:
            prompt_parts.append("arch-shaped panel")
        if "rectangular" in combined or "rectangle" in combined:
            prompt_parts.append("rectangular sample")
        if "square" in combined:
            prompt_parts.append("square swatch")
        
        # Surface keywords
        if "flute" in combined or "groove" in combined or "ribbed" in combined or "vertical" in combined:
            prompt_parts.append("fluted panel")
            prompt_parts.append("grooved surface")
        if "wood" in combined:
            prompt_parts.append("wood panel sample")
        if "marble" in combined or "stone" in combined:
            prompt_parts.append("marble surface sample")
        if "gloss" in combined or "shiny" in combined:
            prompt_parts.append("glossy panel")
        if "matte" in combined or "flat" in combined:
            prompt_parts.append("matte surface sample")
        
        # Generic fallback if nothing specific found
        if len(prompt_parts) < 3:
            prompt_parts.extend(["decorative panel", "material sample", "surface swatch"])
        
        # Deduplicate and limit
        prompt_parts = list(dict.fromkeys(prompt_parts))[:6]
        
        return " . ".join(prompt_parts)
    
    def classify(self, image_path: str) -> str:
        """
        Classify a page as product, application, mixed, or skip.
        
        Args:
            image_path: Path to page image
        
        Returns:
            Classification string
        """
        if not self.api_key:
            logger.warning("No VLM API key, defaulting to 'product'")
            return "product"
        
        mime_type, image_data = _encode_image_for_vlm(image_path, max_size=1024)

        # Default prompt
        prompt = self._prompt or """Analyze this PDF catalog page and classify it into ONE of these categories:

1. "product" - The page shows laminate/decorative panel/acrylic product samples:
   - Grid of product swatches with codes/labels
   - Single large product sample sheets
   - Room photo WITH a small product swatch visible with a product code

2. "application" - The page shows ONLY lifestyle/room photos with NO product codes visible

3. "mixed" - The page contains BOTH product swatches AND separate lifestyle images

4. "skip" - Cover page, table of contents, index, contact info, text-only, blank

Respond with ONLY one word: product, application, mixed, or skip"""

        try:
            response = requests.post(
                self.api_url,
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': self.model,
                    'messages': [{
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': prompt},
                            {'type': 'image_url', 'image_url': {
                                'url': f'data:{mime_type};base64,{image_data}'
                            }}
                        ]
                    }],
                    'max_tokens': 20,
                    'temperature': 0.1
                },
                timeout=90
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result['choices'][0]['message']['content'].strip().lower()
                
                # Parse response
                if 'skip' in answer:
                    return 'skip'
                elif 'application' in answer and 'product' not in answer:
                    return 'application'
                elif 'mixed' in answer:
                    return 'mixed'
                else:
                    return 'product'
            else:
                logger.warning(f"VLM API error: {response.status_code}")
                return 'product'
                
        except Exception as e:
            logger.warning(f"VLM classification failed: {e}")
            return 'product'

    def verify_crop(self, crop_path: str) -> bool:
        """
        Verify if a crop is a valid product swatch vs application/lifestyle/noise.
        
        Args:
            crop_path: Path to cropped image
            
        Returns:
            True if valid product swatch, False if application/noise
        """
        if not self.api_key:
            return True
            
        mime_type, image_data = _encode_image_for_vlm(crop_path, max_size=1024)

        prompt = """Look at this cropped image from a product catalog.

Is this a PRODUCT MATERIAL SAMPLE/SWATCH? Answer YES or NO.

A PRODUCT SWATCH (answer YES) is:
- A flat material sample shown as a single piece — can be rectangular, square, or even have a rounded/arch top with a flat bottom
- Can have ANY pattern: wood grain, marble, solid color, geometric, tribal, abstract art, floral, stripes, dots, mosaic, ethnic, or any decorative design
- Can have ANY color combination — bold, dark, colorful, muted, pastel, neon, etc.
- May show a surface texture (matte, gloss, embossed, rough, smooth)
- May have a product code/number/label near or on it
- The key feature: it shows a SURFACE MATERIAL/TEXTURE meant to be applied on furniture or walls

A NON-PRODUCT image (answer NO) is ONLY:
- A full furnished room scene (showing sofas, tables, beds, people, appliances in a room)
- A photograph of an installed wall/floor in a real room with furniture clearly visible
- A page with ONLY text, logos, or table of contents
- An extremely blurry or corrupted image

IMPORTANT RULES:
- If you see a single panel/sheet with ANY decorative pattern → YES
- If you see artistic/geometric/tribal/abstract designs on a flat surface → YES
- If you see a close-up of a textured or patterned surface → YES
- Colorful or bold artistic designs are STILL product swatches → YES
- Unusual shapes (arch top, rounded corners) are STILL product swatches → YES
- If unsure, default to YES (we prefer to keep borderline cases)

Respond with ONLY: YES or NO"""

        try:
            response = requests.post(
                self.api_url,
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': self.model,
                    'messages': [{
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': prompt},
                            {'type': 'image_url', 'image_url': {
                                'url': f'data:{mime_type};base64,{image_data}'
                            }}
                        ]
                    }],
                    'max_tokens': 10,
                    'temperature': 0.1
                },
                timeout=90
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result['choices'][0]['message']['content'].strip().upper()
                return "YES" in answer
            else:
                logger.warning(f"VLM API error during crop verification: {response.status_code}")
                return True # Default to keep on error
                
        except Exception as e:
            logger.warning(f"Crop verification failed: {e}")
            return True
