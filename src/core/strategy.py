"""Strategy stage - LLM-powered project matching and skill ranking."""

import asyncio
import logging
from typing import List

from ..models import ResumeState, AppConfig, Project
from ..services import LLMService

logger = logging.getLogger(__name__)


async def strategize(
    state: ResumeState, 
    config: AppConfig
) -> ResumeState:
    """STRATEGIZE stage: Match projects to JD and rank skills.
    
    This stage:
    1. Extracts keywords from the job description
    2. Scores each project against the JD
    3. Generates bullet points for top projects  
    4. Re-ranks skills by relevance
    5. Selects top N projects for the resume
    
    Args:
        state: Current pipeline state with project_inventory populated
        config: Application configuration
        
    Returns:
        Updated state with selected_projects and reranked_skills
    """
    llm = LLMService(config.llm)
    
    # Step 1: Extract keywords from JD
    logger.info("Extracting keywords from job description...")
    keyword_result = await llm.extract_keywords(state.job_description_text)
    
    extracted_keywords = keyword_result.keywords
    logger.info(f"Extracted {len(extracted_keywords)} keywords: {extracted_keywords[:5]}...")
    
    # Step 2: Score and generate bullets for each project
    logger.info(f"Matching {len(state.project_inventory)} projects against JD...")
    
    scored_projects: List[Project] = []
    
    # Process projects concurrently (with rate limiting)
    semaphore = asyncio.Semaphore(3)  # Max 3 concurrent LLM calls
    
    async def process_project(project: Project) -> Project:
        async with semaphore:
            try:
                match_result = await llm.match_project(
                    project, 
                    state.job_description_text,
                    extracted_keywords
                )
                
                return project.model_copy(update={
                    "relevance_score": match_result.relevance_score,
                    "relevance_reason": match_result.relevance_reason,
                    "generated_bullets": match_result.generated_bullets,
                })
            except Exception as e:
                logger.warning(f"Failed to process project {project.name}: {e}")
                return project
    
    tasks = [process_project(p) for p in state.project_inventory]
    scored_projects = await asyncio.gather(*tasks)
    
    # Step 3: Sort by relevance and select top N
    scored_projects.sort(key=lambda p: p.relevance_score, reverse=True)
    selected = scored_projects[:config.max_projects]
    
    logger.info(
        f"Selected top {len(selected)} projects: "
        f"{[p.name for p in selected]}"
    )
    
    # Step 4: Rerank skills
    logger.info("Re-ranking skills by JD relevance...")
    rerank_result = await llm.rerank_skills(
        state.current_skills,
        state.job_description_text,
        extracted_keywords
    )
    
    # Update state
    return state.model_copy(update={
        "extracted_keywords": extracted_keywords,
        "project_inventory": scored_projects,  # All projects now have scores
        "selected_projects": selected,
        "reranked_skills": rerank_result.reranked_skills,
    })


async def manual_select(
    state: ResumeState,
    project_names: List[str]
) -> ResumeState:
    """Manually select projects by name (bypasses LLM scoring).
    
    Useful for:
    - Overriding LLM selection
    - Testing specific projects
    - Fine-grained control
    
    Args:
        state: Current pipeline state
        project_names: List of project names to select
        
    Returns:
        Updated state with manually selected projects
    """
    selected = [
        p for p in state.project_inventory 
        if p.name in project_names
    ]
    
    if len(selected) != len(project_names):
        found_names = {p.name for p in selected}
        missing = set(project_names) - found_names
        logger.warning(f"Projects not found in inventory: {missing}")
    
    return state.model_copy(update={"selected_projects": selected})
