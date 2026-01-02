from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from typing import Literal, Optional
from pathlib import Path
from datetime import datetime


class GitHubConfig(BaseModel):
    """GitHub API configuration."""
    token: str = Field(description="GitHub Personal Access Token")
    username: str = Field(description="GitHub username to scrape")
    min_commits: int = Field(
        default=5,
        ge=1,
        description="Minimum commits to consider a repo significant"
    )
    exclude_forks: bool = Field(
        default=True,
        description="Whether to exclude forked repositories"
    )
    exclude_archived: bool = Field(
        default=True,
        description="Whether to exclude archived repositories"
    )
    min_updated_year: Optional[int] = Field(
        default=None,
        description="Minimum year for last update (e.g., 2022 to exclude older projects)"
    )
    min_readme_length: int = Field(
        default=0,
        description="Minimum README character length (e.g., 100 to filter sparse projects)"
    )


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    provider: Literal["openai", "ollama", "groq"] = Field(
        default="openai",
        description="Which LLM backend to use"
    )
    model: str = Field(
        default="gpt-4o-mini",
        description="Model identifier"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for cloud providers (not needed for Ollama)"
    )
    base_url: Optional[str] = Field(
        default=None,
        description="Custom API endpoint (e.g., for Ollama: http://localhost:11434)"
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Lower = more deterministic outputs"
    )


class BuildConfig(BaseModel):
    """LaTeX build configuration."""
    template_path: Path = Field(
        default=Path("templates/resume.tex.j2"),
        description="Path to Jinja2 LaTeX template"
    )
    output_dir: Path = Field(
        default=Path("output"),
        description="Directory for generated PDFs"
    )
    tectonic_path: Optional[str] = Field(
        default=None,
        description="Custom path to tectonic binary (uses PATH if None)"
    )


class AppConfig(BaseSettings):
    """Root application configuration.
    
    Loads from environment variables with the RESUME_ prefix.
    Example: RESUME_GITHUB__TOKEN for github.token
    """
    github: GitHubConfig
    llm: LLMConfig = Field(default_factory=LLMConfig)
    build: BuildConfig = Field(default_factory=BuildConfig)
    
    # Pipeline settings
    max_projects: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Maximum projects to include in resume"
    )
    debug: bool = Field(
        default=False,
        description="Enable verbose logging and state dumps"
    )

    class Config:
        env_prefix = "RESUME_"
        env_nested_delimiter = "__"
