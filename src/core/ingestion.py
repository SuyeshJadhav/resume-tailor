"""Ingestion stage - GitHub project scraping and filtering."""

import logging
from typing import List

from ..models import ResumeState, AppConfig, Project
from ..services import GitHubService

logger = logging.getLogger(__name__)


async def ingest_projects(
    state: ResumeState, 
    config: AppConfig
) -> ResumeState:
    """INGEST stage: Scrape GitHub and populate project_inventory.
    
    This stage:
    1. Connects to GitHub API
    2. Fetches all user repositories
    3. Filters by commit count, fork status, etc.
    4. Extracts project metadata (languages, description)
    5. Populates state.project_inventory
    
    Args:
        state: Current pipeline state
        config: Application configuration
        
    Returns:
        Updated state with project_inventory populated
    """
    logger.info(f"Ingesting projects for user: {config.github.username}")
    
    async with GitHubService(config.github) as gh:
        # Fetch all repositories
        raw_repos = await gh.fetch_user_repos()
        logger.info(f"Found {len(raw_repos)} total repositories")
        
        # Filter and extract project models
        projects = await gh.filter_and_extract(raw_repos)
        logger.info(
            f"Filtered to {len(projects)} projects "
            f"(min {config.github.min_commits} commits)"
        )
    
    # Sort by commit count descending (most active first)
    projects.sort(key=lambda p: p.commit_count, reverse=True)
    
    # Update state with new project inventory
    return state.model_copy(update={"project_inventory": projects})


async def load_local_projects(
    state: ResumeState,
    projects_file: str
) -> ResumeState:
    """Alternative ingestion from a local JSON file.
    
    Useful for:
    - Offline development
    - Testing without API calls
    - Manual project curation
    
    Args:
        state: Current pipeline state
        projects_file: Path to JSON file with project data
        
    Returns:
        Updated state with project_inventory populated
    """
    import json
    from pathlib import Path
    
    path = Path(projects_file)
    if not path.exists():
        raise FileNotFoundError(f"Projects file not found: {projects_file}")
    
    with path.open() as f:
        data = json.load(f)
    
    projects = [Project(**p) for p in data["projects"]]
    logger.info(f"Loaded {len(projects)} projects from {projects_file}")
    
    return state.model_copy(update={"project_inventory": projects})
