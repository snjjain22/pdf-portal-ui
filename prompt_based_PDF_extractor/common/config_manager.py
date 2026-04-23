"""
Configuration Manager for Prompt-Based PDF Extractor
Handles loading and managing YAML configurations, prompts, and taxonomies.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from jinja2 import Environment, FileSystemLoader


class ConfigManager:
    """Manages configuration files for the PDF extraction pipeline."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize ConfigManager.
        
        Args:
            config_dir: Path to config directory. Defaults to ../config relative to this file.
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "config"
        
        self.config_dir = Path(config_dir)
        self.prompts_dir = self.config_dir / "prompts"
        self.taxonomies_dir = self.config_dir / "taxonomies"
        self.pipeline_dir = self.config_dir / "pipeline"
        
        # Cache for loaded configs
        self._cache: Dict[str, Any] = {}
        
        # Jinja2 environment for prompt templates
        self._jinja_env = None
        if self.prompts_dir.exists():
            self._jinja_env = Environment(
                loader=FileSystemLoader(str(self.prompts_dir)),
                trim_blocks=True,
                lstrip_blocks=True
            )
    
    def load_website_config(self, pdf_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Load the main website.yaml configuration.
        
        Args:
            pdf_name: Optional PDF name to apply overrides for
            
        Returns:
            Configuration dict with applied overrides
        """
        config = self._load_yaml(self.config_dir / "website.yaml", "website")
        
        # Apply overrides if pdf_name is provided
        if pdf_name and "pdf_overrides" in config:
            overrides = config.pop("pdf_overrides")
            
            # Check for matches (partial match allowed)
            for key, override_config in overrides.items():
                if key in pdf_name:
                    # Recursive merge
                    self._deep_merge(config, override_config)
        
        return config
        
    def _deep_merge(self, base: Dict, update: Dict) -> None:
        """Recursive dict merge."""
        for k, v in update.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                self._deep_merge(base[k], v)
            else:
                base[k] = v
    
    def load_vlm_config(self) -> Dict[str, Any]:
        """Load VLM pipeline configuration."""
        return self._load_yaml(self.pipeline_dir / "vlm_config.yaml", "vlm")
    
    def load_seo_config(self) -> Dict[str, Any]:
        """Load SEO pipeline configuration."""
        return self._load_yaml(self.pipeline_dir / "seo_config.yaml", "seo")
    
    def load_matrixify_schema(self) -> Dict[str, Any]:
        """Load Matrixify schema configuration."""
        return self._load_yaml(self.pipeline_dir / "matrixify_schema.yaml", "matrixify")
    
    def load_taxonomy(self, name: str) -> Dict[str, Any]:
        """
        Load a taxonomy file.
        
        Args:
            name: Taxonomy name (colors, appearances, categories, sizes)
        
        Returns:
            Taxonomy configuration dict
        """
        return self._load_yaml(self.taxonomies_dir / f"{name}.yaml", f"taxonomy_{name}")
    
    def load_all_taxonomies(self) -> Dict[str, Any]:
        """Load all taxonomy files into a single dict."""
        return {
            "colors": self.load_taxonomy("colors"),
            "appearances": self.load_taxonomy("appearances"),
            "categories": self.load_taxonomy("categories"),
            "sizes": self.load_taxonomy("sizes")
        }
    
    def get_prompt_template(self, name: str) -> str:
        """
        Load a Jinja2 prompt template.
        
        Args:
            name: Template name (e.g., 'field_mapping_derivation')
        
        Returns:
            Template string (unrendered)
        """
        template_path = self.prompts_dir / f"{name}.j2"
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")
        
        return template_path.read_text(encoding="utf-8")
    
    def render_prompt(self, name: str, **kwargs) -> str:
        """
        Render a Jinja2 prompt template with variables.
        
        Args:
            name: Template name without .j2 extension
            **kwargs: Variables to pass to the template
        
        Returns:
            Rendered prompt string
        """
        if self._jinja_env is None:
            raise RuntimeError("Jinja2 environment not initialized - prompts directory may not exist")
        
        template = self._jinja_env.get_template(f"{name}.j2")
        return template.render(**kwargs)
    
    def _load_yaml(self, path: Path, cache_key: str) -> Dict[str, Any]:
        """Load a YAML file with caching."""
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        self._cache[cache_key] = data
        return data
    
    def get_product_defaults(self) -> Dict[str, Any]:
        """Get default product values from website config."""
        config = self.load_website_config()
        return config.get("product_defaults", {})
    
    def get_detection_settings(self) -> Dict[str, Any]:
        """Get detection settings from website config."""
        config = self.load_website_config()
        return config.get("detection", {})
    
    def get_llm_settings(self) -> Dict[str, Any]:
        """Get LLM settings from website config."""
        config = self.load_website_config()
        return config.get("llm", {})
    
    def get_output_settings(self) -> Dict[str, str]:
        """Get output directory settings."""
        config = self.load_website_config()
        return {
            "output_dir": config.get("output_dir", "outputs"),
            "pdfs_dir": config.get("pdfs_dir", "PDFs")
        }
    
    def normalize_color(self, color: str) -> str:
        """
        Normalize a color name using the colors taxonomy.
        
        Args:
            color: Raw color name
        
        Returns:
            Normalized color name
        """
        taxonomy = self.load_taxonomy("colors")
        color_lower = color.lower().strip()
        
        for item in taxonomy.get("items", []):
            # Check canonical name
            if item["name"].lower() == color_lower:
                return item["name"]
            
            # Check aliases
            for alias in item.get("aliases", []):
                if alias.lower() == color_lower:
                    return item["name"]
        
        # Return default if not found
        return taxonomy.get("default", "Multi Colour")
    
    def normalize_appearance(self, appearance: str) -> str:
        """
        Normalize an appearance using the appearances taxonomy.
        
        Args:
            appearance: Raw appearance name
        
        Returns:
            Normalized appearance name
        """
        taxonomy = self.load_taxonomy("appearances")
        appearance_lower = appearance.lower().strip()
        
        for item in taxonomy.get("items", []):
            # Check canonical name
            if item["name"].lower() == appearance_lower:
                return item["name"]
            
            # Check aliases
            for alias in item.get("aliases", []):
                if alias.lower() == appearance_lower:
                    return item["name"]
            
            # Check keywords
            for keyword in item.get("keywords", []):
                if keyword.lower() in appearance_lower:
                    return item["name"]
        
        return taxonomy.get("default", "Solid Colour")
    
    def clear_cache(self):
        """Clear the configuration cache."""
        self._cache.clear()


# Singleton instance for convenience
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_dir: Optional[str] = None) -> ConfigManager:
    """Get or create the ConfigManager singleton."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_dir)
    return _config_manager
