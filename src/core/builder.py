"""Build stage - Jinja2 template rendering and Tectonic PDF compilation."""

import subprocess
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
import os

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..models import ResumeState, AppConfig

logger = logging.getLogger(__name__)


class BuildError(Exception):
    """Raised when resume build fails."""
    pass


def create_jinja_env(template_dir: Path) -> Environment:
    """Create a Jinja2 environment configured for LaTeX.
    
    Uses custom delimiters to avoid conflicts with LaTeX syntax:
    - Block: <% ... %>
    - Variable: << ... >>
    - Comment: <# ... #>
    """
    return Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=False,  # LaTeX handles its own escaping
        block_start_string="<%",
        block_end_string="%>",
        variable_start_string="<<",
        variable_end_string=">>",
        comment_start_string="<#",
        comment_end_string="#>",
        trim_blocks=True,
        lstrip_blocks=True,
    )


def escape_latex(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    
    # Map of characters that MUST be escaped for LaTeX
    mapping = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    
    # We use a regex to replace everything in one pass to avoid double-escaping
    import re
    regex = re.compile(r"([&%$#_{}~^\\])")
    return regex.sub(lambda match: mapping[match.group()], text)

def escape_latex_join(items: list, separator: str = ", ") -> str:
    """Escape and join a list of items."""
    return separator.join(escape_latex(str(item)) for item in items)


async def build_resume(
    state: ResumeState, 
    config: AppConfig
) -> ResumeState:
    """BUILD stage: Render template and compile PDF.
    
    Supports two modes:
    1. Single template mode: Renders one .tex.j2 file as complete resume
    2. Modular mode: Renders partial .j2 templates in src/ folder, then compiles main resume.tex
    
    Args:
        state: Current pipeline state with selected_projects
        config: Application configuration
        
    Returns:
        Same state (build is a terminal stage)
        
    Raises:
        BuildError: If template rendering or PDF compilation fails
    """
    template_path = config.build.template_path
    output_dir = config.build.output_dir
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not template_path.exists():
        raise BuildError(f"Template not found: {template_path}")
    
    template_dir = template_path.parent
    
    # Check if this is a modular template (has src/ folder with .j2 files)
    src_dir = template_dir / "src"
    is_modular = src_dir.exists() and any(src_dir.glob("*.j2"))
    
    # Create Jinja environment
    env = create_jinja_env(template_dir)
    env.filters["escape_latex"] = escape_latex
    env.filters["escape_latex_join"] = escape_latex_join
    
    # Template context
    context = {
        "skills": state.reranked_skills or state.current_skills,
        "projects": state.selected_projects,
        "keywords": state.extracted_keywords,
        "generated_at": datetime.now().isoformat(),
    }
    
    if is_modular:
        # Modular mode: render partial templates in-place
        logger.info(f"Modular template mode: rendering partials in {src_dir}")
        
        # Create env for src/ subdirectory
        src_env = create_jinja_env(src_dir)
        src_env.filters["escape_latex"] = escape_latex
        src_env.filters["escape_latex_join"] = escape_latex_join
        
        for j2_file in src_dir.glob("*.j2"):
            template = src_env.get_template(j2_file.name)
            rendered = template.render(**context)
            
            # Write to .tex file (remove .j2 extension)
            tex_name = j2_file.stem  # e.g., "projects.tex"
            tex_path = src_dir / tex_name
            tex_path.write_text(rendered, encoding="utf-8")
            logger.info(f"Rendered partial: {tex_name}")
        
        # Compile the main resume.tex
        main_tex = template_path
        logger.info(f"Compiling main template: {main_tex}")
        pdf_path = await compile_with_tectonic(main_tex, config.build.tectonic_path)
        
    else:
        # Single template mode: render complete resume
        logger.info(f"Single template mode: {template_path}")
        template = env.get_template(template_path.name)
        
        try:
            rendered = template.render(**context)
        except Exception as e:
            raise BuildError(f"Template rendering failed: {e}") from e
        
        # Write .tex file to output
        tex_path = output_dir / "resume.tex"
        tex_path.write_text(rendered, encoding="utf-8")
        logger.info(f"Wrote LaTeX source: {tex_path}")
        
        # Compile with Tectonic
        pdf_path = await compile_with_tectonic(tex_path, config.build.tectonic_path)
    
    logger.info(f"Resume generated: {pdf_path}")
    return state


async def compile_with_tectonic(
    tex_path: Path,
    tectonic_path: Optional[str] = None
) -> Path:
    """Compile a .tex file to PDF using Tectonic.
    
    Tectonic is a modern LaTeX engine that:
    - Automatically downloads required packages
    - Runs the correct number of passes
    - Produces reproducible builds
    
    Args:
        tex_path: Path to the .tex file
        tectonic_path: Custom path to tectonic binary
        
    Returns:
        Path to the generated PDF
        
    Raises:
        BuildError: If compilation fails
    """
    tectonic_cmd = tectonic_path or "tectonic"
    output_dir = tex_path.parent
    
    cmd = [
        tectonic_cmd,
        "-X", "compile",
        "--outdir", str(output_dir),
        str(tex_path),
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
            cwd=tex_path.parent,  # Run from template directory for correct relative paths
        )
        
        if result.returncode != 0:
            logger.error(f"Tectonic stderr: {result.stderr}")
            raise BuildError(f"Tectonic compilation failed: {result.stderr[:500]}")
        
        pdf_path = output_dir / "resume.pdf"
        
        if not pdf_path.exists():
            raise BuildError("PDF file not generated")
        
        return pdf_path
        
    except subprocess.TimeoutExpired:
        raise BuildError("Tectonic compilation timed out (>2 minutes)")
    except FileNotFoundError:
        raise BuildError(
            f"Tectonic not found. Install with: cargo install tectonic "
            f"or download from https://tectonic-typesetting.github.io/"
        )


def render_preview(state: ResumeState, config: AppConfig) -> str:
    """Render template without compilation (for previewing).
    
    Returns the raw LaTeX source for inspection.
    """
    template_path = config.build.template_path
    
    if not template_path.exists():
        raise BuildError(f"Template not found: {template_path}")
    
    env = create_jinja_env(template_path.parent)
    env.filters["escape_latex"] = escape_latex
    
    template = env.get_template(template_path.name)
    
    context = {
        "skills": state.reranked_skills or state.current_skills,
        "projects": state.selected_projects,
        "keywords": state.extracted_keywords,
        "generated_at": datetime.now().isoformat(),
    }
    
    return template.render(**context)
