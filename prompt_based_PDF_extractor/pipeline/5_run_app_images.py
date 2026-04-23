#!/usr/bin/env python3
"""
Stage 5: Application Image Generation

Generates application/lifestyle images for each laminate product using:
1. DeepSeek LLM to create tailored image-generation prompts per product
2. Flux2 Klein 9B model on Hugging Face Spaces to generate room and furniture images
3. S3 upload of generated images

The generated image URLs are appended to each product's data so that
the Matrixify stage can include them in the Image Src column.

Usage:
    python 5_run_app_images.py <pdf_output_dir>
    python 5_run_app_images.py outputs/MY_CATALOG_PDF
"""

import argparse
import json
import os
import sys
import time
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from jinja2 import Template
from common.llm_client import get_vision_llm_client
from common.config_manager import get_config_manager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ApplicationImageGenerator:
    """
    Generates application/lifestyle images for laminate products.
    Uses DeepSeek for prompt generation and HF Spaces Flux model for image generation.
    """

    def __init__(self, config_manager, hf_space: str = "Automate-GPT/NLDApplicationImageGenerator"):
        self.config_manager = config_manager
        self.hf_space = hf_space
        self.api_delay = float(os.getenv("APP_IMAGE_DELAY", "2"))

        # Initialize Vision LLM for prompt generation
        self.vlm = get_vision_llm_client()
        logger.info(f"Vision LLM initialized for prompt generation")

        # Initialize Gradio client for HF Space
        hf_token = os.getenv("HF_API_KEY")
        if not hf_token:
            raise ValueError("HF_API_KEY not found in .env file. Add your Hugging Face token.")

        from gradio_client import Client
        # Set timeout to handle potentially longer generation times efficiently
        self.hf_client = Client(hf_space, token=hf_token)
        logger.info("HF Space connected successfully")

        # Initialize S3 uploader (reuse from stage 4)
        try:
            import importlib.util
            s3_module_path = Path(__file__).parent / "4_run_s3_upload.py"
            spec = importlib.util.spec_from_file_location("s3_upload", s3_module_path)
            s3_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(s3_module)
            self.s3_uploader = s3_module.S3Uploader()
        except Exception as e:
            logger.warning(f"S3 uploader not available: {e}. Generated images will be local only.")
            self.s3_uploader = None

    # ------------------------------------------------------------------
    # Prompt generation via DeepSeek
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Image generation via HF Space
    # ------------------------------------------------------------------

    def _prepare_reference_image(self, image_path: str) -> str:
        """
        Passes the original high-resolution image to preserve laminate texture details.
        """
        return image_path

    def _generate_images_batch(self, batch_info: List[Dict], max_retries: int = 2) -> List[str]:
        """
        Call the HF Space with a batch of up to 5 products (room+furn prompts / image).
        Returns a list of local paths to the generated images in order.
        """
        from gradio_client import handle_file

        # Prepare exactly 15 parameters (5 sets of room_prompt, furn_prompt, image)
        args = []
        for i in range(5):
            if i < len(batch_info):
                args.append(batch_info[i]["room_prompt"])
                args.append(batch_info[i]["furn_prompt"])
                compressed_path = self._prepare_reference_image(batch_info[i]["crop_path"])
                args.append(handle_file(compressed_path))
            else:
                args.append("")
                args.append("")
                args.append(None)

        for attempt in range(max_retries + 1):
            try:
                gallery = self.hf_client.predict(
                    *args,
                    api_name="/process_api_request"
                )
                
                result_paths = []
                if gallery:
                    for g_item in gallery:
                        if isinstance(g_item, dict) and "image" in g_item:
                            filepath = g_item["image"]
                        elif isinstance(g_item, dict) and "image" not in g_item and len(g_item) > 0:
                            filepath = g_item.get("path", str(g_item))
                        else:
                            filepath = str(g_item)
                        
                        if isinstance(filepath, dict):
                            filepath = filepath.get("path", filepath)
                            
                        result_paths.append(str(filepath))
                        
                return result_paths

            except Exception as e:
                logger.warning(f"HF Space batch call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                if attempt < max_retries:
                    wait = 10 * (attempt + 1)
                    logger.info(f"Retrying in {wait}s ...")
                    time.sleep(wait)

        return []

    def _generate_single_image(self, prompt: str, ref_image_path: str, max_retries: int = 2) -> object:
        """
        Call the HF Space for a SINGLE image generation.
        Returns the local path to the generated image, or None if failed.
        """
        from gradio_client import handle_file
        
        if not prompt or not ref_image_path:
            return None

        compressed_path = self._prepare_reference_image(ref_image_path)
        
        # Prepare 15 parameters: slot 1 room prompt is used, everything else empty.
        args = [
            prompt, "", handle_file(compressed_path),
            "", "", None,
            "", "", None,
            "", "", None,
            "", "", None
        ]

        for attempt in range(max_retries + 1):
            try:
                gallery = self.hf_client.predict(
                    *args,
                    api_name="/process_api_request"
                )
                
                # Extracted paths from gradio gallery output
                if gallery and len(gallery) > 0:
                    g_item = gallery[0]
                    if isinstance(g_item, dict) and "image" in g_item:
                        filepath = g_item["image"]
                    elif isinstance(g_item, dict) and "image" not in g_item and len(g_item) > 0:
                        filepath = g_item.get("path", str(g_item))
                    else:
                        filepath = str(g_item)
                    
                    if isinstance(filepath, dict):
                        filepath = filepath.get("path", filepath)
                        
                    return str(filepath)

            except Exception as e:
                logger.warning(f"HF Space call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                if attempt < max_retries:
                    wait = 10 * (attempt + 1)
                    logger.info(f"Retrying in {wait}s ...")
                    time.sleep(wait)

        return None

    def _upscale_local(self, image_path: str) -> str:
        """
        Upscales the generated 768px image to 1536px (or larger) using Real-ESRGAN.
        Uses realesrgan-x4plus model (standard realistic model included in release).
        """
        import subprocess
        try:
            upscaled_path = str(Path(image_path).parent / f"upscaled_{Path(image_path).name}")
            
            # Point directly to the local realesrgan folder inside the pipeline directory
            esrgan_exe = Path(__file__).parent / "realesrgan" / "realesrgan-ncnn-vulkan.exe"
            
            # Using Real-ESRGAN with the requested model
            cmd = [
                str(esrgan_exe), 
                "-i", str(image_path), 
                "-o", upscaled_path, 
                "-n", "realesrgan-x4plus",
                "-s", "2"  # 2x upscale (from 768 to 1536)
            ]
            
            # Only run if the executable actually exists in the folder
            if esrgan_exe.exists():
                # Run without blocking the main event loops or flooding standard out too much
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0 and Path(upscaled_path).exists():
                    return upscaled_path
                else:
                    logger.warning(f"Real-ESRGAN failed with return code {result.returncode}: {result.stderr}. Falling back to PIL.")
            else:
                logger.warning(f"Real-ESRGAN executable not found at {esrgan_exe}. Falling back to PIL.")
                
        except Exception as e:
            logger.warning(f"Error executing Real-ESRGAN: {e}. Falling back to PIL Lanczos.")

        # Fallback to PIL Lanczos if the command-line tool fails or is not available
        try:
            from PIL import Image
            img = Image.open(image_path)
            new_size = (img.width * 2, img.height * 2)
            img = img.resize(new_size, Image.LANCZOS)
            
            upscaled_path = str(Path(image_path).parent / f"upscaled_{Path(image_path).name}")
            img.save(upscaled_path, "PNG", quality=95)
            return upscaled_path
            
        except Exception as e:
            logger.warning(f"Local upscale fallback failed, using original: {e}")
            return image_path

    # ------------------------------------------------------------------
    # S3 upload
    # ------------------------------------------------------------------

    def _upload_to_s3(self, local_path: str, s3_key: str) -> Optional[str]:
        """Upload a generated image to S3 and return the public URL."""
        if not self.s3_uploader:
            return None
        return self.s3_uploader.upload_file(local_path, s3_key)

    # ------------------------------------------------------------------
    # Process a single product
    # ------------------------------------------------------------------

    def generate_prompts_for_product(
        self,
        product: Dict,
        product_index: int,
    ) -> Optional[Dict]:
        """
        Phase 1: Pre-generate all Flux prompts for a product using Vision LLM.
        Returns dict with scene info + prompts.
        """
        sku = product.get("sku", f"unknown_{product_index}")
        crop_path = product.get("crop_path", "")

        if not crop_path or not Path(crop_path).is_file():
            logger.warning(f"[{sku}] No valid crop image found, skipping")
            return None

        # Call Vision LLM for scene context
        sys_prompt = """You are an expert interior design AI. Analyze this laminate image and determine where it would be most suited. 
CRITICAL RULES:
- If the laminate is a solid color (e.g., green, blue, pink), DO NOT use material-biased words like "wooden", "timber", "rustic", or "natural". Use terms like "modern", "minimalist", "sleek", or "painted".
- Focus only on functional room types and surface locations.

Return ONLY valid JSON in this exact structure:
{
    "room_scene": {
        "room": "Identify the best room type (e.g., modern living room, sleek bedroom)",
        "surface": "Identify the best architectural surface (e.g., TV accent wall, headboard wall)"
    },
    "furniture_scene": {
        "furniture": "Identify the best furniture piece (e.g., sleek wardrobe, L-shaped kitchen)",
        "room": "Identify the room setting for this furniture (e.g., spacious dressing room, minimalist open kitchen)"
    }
}
Do not include markdown code block tags or any other text."""

        try:
            logger.info(f"  [{sku}] Analyzing image with Vision LLM ...")
            response = self.vlm.chat_with_image(
                prompt=sys_prompt,
                image_path=crop_path
            )
            content = getattr(response, "content", response) if hasattr(response, "content") else str(response)
            
            clean_json = content.strip()
            if clean_json.startswith('```json'):
                clean_json = clean_json[7:-3]
            elif clean_json.startswith('```'):
                clean_json = clean_json[3:-3]
                
            scene_data = json.loads(clean_json.strip())
            
            room_scene = scene_data.get('room_scene', {})
            furn_scene = scene_data.get('furniture_scene', {})
            
            room_template_str = "A {{ room_scene.room }} featuring a {{ room_scene.surface }} in the exact color, texture and finish of the provided laminate. Photorealistic, 8k, highly detailed, photorealism."
            furniture_template_str = "A {{ furniture_scene.furniture }} in a {{ furniture_scene.room }} featuring the provided laminate in its exact color, texture and finish. Photorealistic, 8k, highly detailed, photorealism."
            
            room_prompt = Template(room_template_str).render(room_scene=room_scene).strip()
            furn_prompt = Template(furniture_template_str).render(furniture_scene=furn_scene).strip()

            logger.info(f"  [{sku}] Flux prompts generated successfully.")

            return {
                "sku": sku,
                "product_index": product_index,
                "room_prompt": room_prompt,
                "furn_prompt": furn_prompt,
                "crop_path": crop_path,
            }

        except Exception as e:
            logger.warning(f"  [{sku}] Failed to generate prompts with Vision LLM: {e}")
            return None

    def generate_image_for_prompt(
        self,
        prompt_info: Dict,
        output_images_dir: Path,
        s3_base_prefix: str,
    ) -> Optional[str]:
        """
        Phase 2: Generate a single image from a pre-generated prompt, save + upload to S3.
        Returns S3 URL or None.
        """
        sku = prompt_info["sku"]
        label = prompt_info["label"]
        filename = prompt_info["filename"]
        flux_prompt = prompt_info["flux_prompt"]
        crop_path = prompt_info["crop_path"]

        logger.info(f"  [{sku}] Generating {label} image ...")
        generated_path = self._generate_image(flux_prompt, crop_path)
        if not generated_path:
            logger.warning(f"  [{sku}] Failed to generate {label} image")
            return None

        # Copy to our output directory with a clean name
        dest_path = output_images_dir / filename
        shutil.copy2(generated_path, dest_path)
        logger.info(f"  [{sku}] Saved {label} image: {dest_path}")

        # Return local path; S3 upload will be done in a batch later
        return str(dest_path)


def run_app_images(output_dir: str, config_dir: str = None) -> Dict[str, Any]:
    """
    Run the application image generation stage.

    Reads 4_s3_uploaded.json, generates application images for each product,
    uploads to S3, and saves 5_app_images.json.
    """
    output_dir = Path(output_dir)

    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    logger.info("=" * 60)
    logger.info("Stage 5: Application Image Generation")
    logger.info(f"Directory: {output_dir.name}")
    logger.info("=" * 60)

    # Load input data
    # First try to load existing stage 5 data to resume progress
    input_path = output_dir / "data" / "5_app_images.json"
    if not input_path.exists():
        input_path = output_dir / "data" / "4_s3_uploaded.json"
    if not input_path.exists():
        # Fallback to SEO results
        input_path = output_dir / "data" / "3_seo_results.json"
    if not input_path.exists():
        raise FileNotFoundError(f"Input data not found in {output_dir / 'data'}")

    logger.info(f"Loading data from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "products" in data:
        products = data["products"]
        wrapper = data
    else:
        products = data
        wrapper = {"products": products}

    logger.info(f"Found {len(products)} products")

    # Create output directory for application images
    app_images_dir = output_dir / "application_images"
    app_images_dir.mkdir(parents=True, exist_ok=True)

    # Determine S3 prefix
    pdf_dir_name = output_dir.name
    s3_folder = os.getenv("S3_FOLDER_OVERRIDE", pdf_dir_name.replace("outputs_", "").replace("_results", ""))
    s3_base_prefix = f"pdf_extractor/{s3_folder}"

    # Load config and initialize generator
    config_manager = get_config_manager(config_dir)
    generator = ApplicationImageGenerator(config_manager)

    success_count = 0
    fail_count = 0
    skipped_count = 0

    # ============================================================
    # PHASE 1: Pre-generate ALL prompts via DeepSeek
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: Pre-generating all prompts via DeepSeek ...")
    logger.info("=" * 60)

    all_prompts = []  # list of {sku, product_index, room_prompt, furn_prompt, crop_path}
    product_indices = {}  # product_index -> product reference

    for i, product in enumerate(products):
        sku = product.get("sku", f"unknown_{i}")

        # Resume support: skip products that already have application images
        if product.get("application_image_urls"):
            logger.info(f"[{i+1}/{len(products)}] {sku} — already has application images, skipping")
            skipped_count += 1
            continue

        logger.info(f"[{i+1}/{len(products)}] Generating prompts for {sku} ...")
        prompts = generator.generate_prompts_for_product(product, i)
        if prompts:
            all_prompts.append(prompts)
            product_indices[i] = product
        else:
            fail_count += 1

    # Save prompts to disk for resume/debugging
    prompts_cache_path = output_dir / "data" / "5_prompts_cache.json"
    with open(prompts_cache_path, "w", encoding="utf-8") as f:
        json.dump(all_prompts, f, indent=2, ensure_ascii=False)
    logger.info(f"\nPhase 1 complete: {len(all_prompts)} prompts generated, saved to {prompts_cache_path}")

    # ============================================================
    # PHASE 2: Generate ALL images with Optimistic Batching
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info(f"PHASE 2: Generating {len(all_prompts)} products (Room + Furniture) with optimistic batching ...")
    logger.info("=" * 60)

    # Track which products got which URLs
    product_urls = {}  # product_index -> [url, url, ...]

    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    total_chunks = (len(all_prompts) + 4) // 5

    for chunk_idx, batch_info in enumerate(chunker(all_prompts, 5)):
        logger.info(f"--- Processing Batch {chunk_idx+1}/{total_chunks} ---")
        
        expected_images = len(batch_info) * 2
        results = generator._generate_images_batch(batch_info)
        
        # Did Hugging Face return a perfect batch?
        if len(results) == expected_images:
            # BATCH SUCCEEDED - Safe to map incrementally
            logger.info(f"  Batch {chunk_idx+1} perfect match! ({len(results)} images returned)")
            for i, prompt_info in enumerate(batch_info):
                pi = prompt_info["product_index"]
                sku = prompt_info["sku"]

                img_index_room = i * 2
                img_index_furn = i * 2 + 1
                
                # Save room image
                generated_room = results[img_index_room]
                if os.path.exists(generated_room) and os.path.getsize(generated_room) < 1000:
                    logger.warning(f"  [{sku}] Received blank/fallback room image. Skipping.")
                else:
                    dest_path_room = app_images_dir / f"{sku}_room.png"
                    shutil.copy2(generated_room, dest_path_room)
                    logger.info(f"  [{sku}] Saved room image: {dest_path_room}")
                    product_urls.setdefault(pi, []).append(str(dest_path_room))
                
                # Save furniture image
                generated_furn = results[img_index_furn]
                if os.path.exists(generated_furn) and os.path.getsize(generated_furn) < 1000:
                    logger.warning(f"  [{sku}] Received blank/fallback furniture image. Skipping.")
                else:
                    dest_path_furn = app_images_dir / f"{sku}_furniture.png"
                    shutil.copy2(generated_furn, dest_path_furn)
                    logger.info(f"  [{sku}] Saved furniture image: {dest_path_furn}")
                    product_urls.setdefault(pi, []).append(str(dest_path_furn))

        else:
            # BATCH COLLAPSED - Mismatch detected! 
            logger.warning(f"  Batch {chunk_idx+1} COLLAPSED: returned {len(results)}/{expected_images} images. Falling back to sequential 1-by-1.")
            
            # Re-run ONLY this chunk one-by-one to guarantee correct names
            for prompt_info in batch_info:
                pi = prompt_info["product_index"]
                sku = prompt_info["sku"]
                room_prompt = prompt_info["room_prompt"]
                furn_prompt = prompt_info["furn_prompt"]
                crop_path = prompt_info["crop_path"]

                logger.info(f"  [{sku}] Generating strictly individually...")
                
                # Room Image
                room_path = generator._generate_single_image(room_prompt, crop_path)
                if room_path and os.path.exists(room_path):
                    if os.path.getsize(room_path) < 1000:
                        logger.warning(f"  [{sku}] Received blank room image. Skipping.")
                    else:
                        dest_path_room = app_images_dir / f"{sku}_room.png"
                        shutil.copy2(room_path, dest_path_room)
                        logger.info(f"  [{sku}] Saved room image: {dest_path_room}")
                        product_urls.setdefault(pi, []).append(str(dest_path_room))
                else:
                    logger.warning(f"  [{sku}] Failed to generate room image")
                
                # Furniture Image
                furn_path = generator._generate_single_image(furn_prompt, crop_path)
                if furn_path and os.path.exists(furn_path):
                    if os.path.getsize(furn_path) < 1000:
                        logger.warning(f"  [{sku}] Received blank furniture image. Skipping.")
                    else:
                        dest_path_furn = app_images_dir / f"{sku}_furniture.png"
                        shutil.copy2(furn_path, dest_path_furn)
                        logger.info(f"  [{sku}] Saved furniture image: {dest_path_furn}")
                        product_urls.setdefault(pi, []).append(str(dest_path_furn))
                else:
                    logger.warning(f"  [{sku}] Failed to generate furniture image")

        # Minimal delay to avoid rate limits
        if generator.api_delay > 0:
            time.sleep(generator.api_delay)

        # Save progress every chunk instead of every product
        _apply_urls_to_products(products, product_urls)
        _save_progress(wrapper, output_dir)
        logger.info(f"--- Progress saved (Batch {chunk_idx+1}) ---")

    # Apply all URLs to products
    _apply_urls_to_products(products, product_urls)

    # ============================================================
    # PHASE 2.5: Batch Local AI Upscaling (DISABLED)
    # ============================================================
    # User requested to skip upscaling because images are already good.
    # logger.info("\n" + "=" * 60)
    # logger.info("PHASE 2.5: Batch Local AI Upscaling ...")
    # logger.info("=" * 60)
    # 
    # for pi, urls in product_urls.items():
    #     upscaled_urls = []
    #     sku = products[pi].get("sku", "unknown") if pi < len(products) else "unknown"
    #     for url in urls:
    #         if not url.startswith("http") and Path(url).exists():
    #             logger.info(f"  [{sku}] Upscaling image locally...")
    #             upscaled_url = generator._upscale_local(url)
    #             upscaled_urls.append(upscaled_url)
    #         else:
    #             upscaled_urls.append(url)
    #     product_urls[pi] = upscaled_urls
    # 
    # # Apply updated upscaled URLs to products
    # _apply_urls_to_products(products, product_urls)
    # _save_progress(wrapper, output_dir)
    # logger.info("Local upscaling complete.")

    # ============================================================
    # PHASE 3: Batch Upload to S3
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: Batch uploading generated images to S3 ...")
    logger.info("=" * 60)

    for product in products:
        urls = product.get("application_image_urls")
        if not urls:
            continue
            
        new_urls = []
        for u in urls:
            # If it's a local path (not an S3 http URL), upload it now
            if not u.startswith("http") and generator.s3_uploader:
                local_path = u
                sku = product.get("sku", "unknown")
                if Path(local_path).exists():
                    filename = Path(local_path).name
                    s3_key = f"{s3_base_prefix}/application/{filename}"
                    logger.info(f"  [{sku}] Uploading {filename} to S3...")
                    s3_url = generator._upload_to_s3(local_path, s3_key)
                    if s3_url:
                        new_urls.append(s3_url)
                        logger.info(f"  [{sku}] S3 URL: {s3_url}")
                    else:
                        new_urls.append(local_path)
                else:
                    logger.warning(f"  [{sku}] Local file missing for upload: {local_path}")
                    new_urls.append(local_path)
            else:
                # Already an HTTP URL
                new_urls.append(u)
                
        product["application_image_urls"] = new_urls

    _save_progress(wrapper, output_dir)

    # Count successes/failures
    for pi in product_indices:
        if pi in product_urls and product_urls[pi]:
            success_count += 1
        else:
            fail_count += 1

    # Final save
    output_path = output_dir / "data" / "5_app_images.json"
    wrapper["app_image_stats"] = {
        "success_count": success_count,
        "fail_count": fail_count,
        "skipped_count": skipped_count,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(wrapper, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'=' * 60}")
    logger.info("Application Image Generation Complete!")
    logger.info(f"  Success: {success_count}")
    logger.info(f"  Failed:  {fail_count}")
    logger.info(f"  Skipped: {skipped_count}")
    logger.info(f"  Output:  {output_path}")
    logger.info(f"{'=' * 60}")

    return {
        "status": "success",
        "products_processed": len(products),
        "images_success": success_count,
        "images_failed": fail_count,
        "images_skipped": skipped_count,
        "output_file": str(output_path),
    }


def _save_progress(wrapper: Dict, output_dir: Path):
    """Save intermediate progress to disk."""
    progress_path = output_dir / "data" / "5_app_images.json"
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(wrapper, f, indent=2, ensure_ascii=False)


def _apply_urls_to_products(products: List[Dict], product_urls: Dict[int, List[str]]):
    """Apply collected application image URLs back to product dicts."""
    for pi, urls in product_urls.items():
        if urls and pi < len(products):
            products[pi]["application_image_urls"] = urls


def main():
    parser = argparse.ArgumentParser(
        description="Stage 5: Generate application images for laminate products"
    )
    parser.add_argument(
        "output_dir",
        help="Path to PDF output directory (e.g., outputs/MY_CATALOG_PDF)",
    )
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Path to config directory",
    )

    args = parser.parse_args()

    try:
        run_app_images(args.output_dir, args.config_dir)
    except Exception as e:
        logger.error(f"Application image generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
