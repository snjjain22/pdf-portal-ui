"""
LLM Client for Prompt-Based PDF Extractor
Unified interface for calling LLM APIs (DeepSeek, OpenRouter, etc.)
"""

import os
import io
import json
import time
import base64
import requests
from PIL import Image
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    # Look for .env in the prompt_based_PDF_extractor folder
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars


@dataclass
class LLMResponse:
    """Response from an LLM API call."""
    content: str
    model: str
    usage: Dict[str, int]
    raw_response: Dict[str, Any]


class LLMClient:
    """Unified client for LLM API calls."""
    
    def __init__(
        self,
        provider: str = "deepseek",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 120
    ):
        """
        Initialize LLM client.
        
        Args:
            provider: LLM provider (deepseek, openrouter, openai)
            api_key: API key (will read from env if not provided)
            base_url: Base URL for API (provider-specific default if not provided)
            model: Model name (provider-specific default if not provided)
            max_retries: Maximum retry attempts for failed requests
            timeout: Request timeout in seconds
        """
        self.provider = provider.lower()
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Configure provider-specific defaults
        if self.provider == "deepseek":
            self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
            self.base_url = base_url or "https://api.deepseek.com/v1"
            self.model = model or "deepseek-chat"
        
        elif self.provider == "openrouter":
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            self.base_url = base_url or "https://openrouter.ai/api/v1"
            self.model = model or "meta-llama/llama-4-maverick"
        
        elif self.provider == "openai":
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.base_url = base_url or "https://api.openai.com/v1"
            self.model = model or "gpt-4o-mini"
        
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        if not self.api_key:
            raise ValueError(f"API key not found for {provider}. Set {provider.upper()}_API_KEY environment variable.")
    
    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        response_format: Optional[str] = None
    ) -> LLMResponse:
        """
        Send a chat completion request.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            response_format: Optional response format ("json_object" for JSON mode)
        
        Returns:
            LLMResponse object
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        return self._make_request(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format
        )
    
    def chat_with_image(
        self,
        prompt: str,
        image_path: Union[str, Path],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> LLMResponse:
        """
        Send a chat completion request with an image.
        
        Args:
            prompt: User prompt
            image_path: Path to image file
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        
        Returns:
            LLMResponse object
        """
        # Encode image to base64 (resized in-memory for VLM, originals untouched)
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Resize proportionally to max 1024px on longest side (in-memory only)
        max_vlm_size = 1024
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img.thumbnail((max_vlm_size, max_vlm_size), Image.LANCZOS)
            suffix = image_path.suffix.lower()
            fmt = "JPEG" if suffix in [".jpg", ".jpeg"] else "PNG"
            buf = io.BytesIO()
            img.save(buf, format=fmt, quality=85)
            image_data = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        # Determine image type
        media_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }.get(suffix, "image/png")
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Create message with image
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{image_data}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        })
        
        return self._make_request(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def _make_request(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        response_format: Optional[str] = None
    ) -> LLMResponse:
        """Make the actual API request with retries."""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Add provider-specific headers
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://nextleveldecor.in"
            headers["X-Title"] = "Next Level Decor PDF Extractor"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
        
        url = f"{self.base_url}/chat/completions"
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return LLMResponse(
                        content=data["choices"][0]["message"]["content"],
                        model=data.get("model", self.model),
                        usage=data.get("usage", {}),
                        raw_response=data
                    )
                
                elif response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = min(2 ** attempt * 10, 60)
                    print(f"Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                
                else:
                    error_msg = f"API error {response.status_code}: {response.text}"
                    print(f"Attempt {attempt + 1} failed: {error_msg}")
                    last_error = Exception(error_msg)
                    
            except requests.exceptions.Timeout:
                print(f"Attempt {attempt + 1} timed out")
                last_error = TimeoutError(f"Request timed out after {self.timeout}s")
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                last_error = e
            
            # Wait before retry
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)
        
        raise last_error or Exception("All retry attempts failed")
    
    def parse_json_response(self, response: LLMResponse) -> Dict[str, Any]:
        """
        Parse JSON from LLM response, handling common issues.
        
        Args:
            response: LLMResponse object
        
        Returns:
            Parsed JSON dict
        """
        content = response.content.strip()
        
        # Remove markdown code fences if present
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        
        if content.endswith("```"):
            content = content[:-3]
        
        content = content.strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            # Try to extract JSON from the content
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            raise ValueError(f"Failed to parse JSON response: {e}\nContent: {content[:500]}")


def get_text_llm_client(**kwargs) -> LLMClient:
    """Get an LLM client configured for text tasks (DeepSeek)."""
    return LLMClient(provider="deepseek", **kwargs)


def get_vision_llm_client(**kwargs) -> LLMClient:
    """Get an LLM client configured for vision tasks (OpenRouter/Llama-4)."""
    return LLMClient(provider="openrouter", **kwargs)
