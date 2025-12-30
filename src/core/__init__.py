"""Core pipeline logic for The Resume Orchestrator."""

from .pipeline import ResumeOrchestrator, Stage
from .ingestion import ingest_projects, load_local_projects
from .strategy import strategize, manual_select
from .builder import build_resume, render_preview, BuildError

__all__ = [
    "ResumeOrchestrator",
    "Stage",
    "ingest_projects",
    "load_local_projects",
    "strategize",
    "manual_select",
    "build_resume",
    "render_preview",
    "BuildError",
]
