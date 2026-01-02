from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class Project(BaseModel):
    """Represents a GitHub project with relevance scoring.
    
    This model captures both raw GitHub metadata and LLM-generated
    content for resume presentation.
    """
    name: str
    url: str  # Essential for the PDF link
    stack: List[str]  # e.g. ["Python", "FastAPI"]
    description: str  # Raw summary from GitHub
    readme_content: str = Field(
        default="",
        description="First ~500 chars of README for context"
    )
    topics: List[str] = Field(
        default_factory=list,
        description="GitHub topics/tags for the repo"
    )
    generated_bullets: List[str] = Field(
        default_factory=list,
        description="The 3 specific bullets for the resume (Result, Action, Tech)"
    )
    relevance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Semantic match score against the JD (0-1)"
    )
    relevance_reason: Optional[str] = Field(
        default=None,
        description="Why did the AI pick this? (Good for debugging)"
    )
    commit_count: int = Field(
        default=0,
        description="Number of commits to gauge contribution depth"
    )


class ResumeState(BaseModel):
    """Central state object passed through the pipeline.
    
    This is the "traveling context" that accumulates data as it flows
    through: INGEST → STRATEGIZE → BUILD stages.
    """
    # === Input Stage ===
    job_description_text: str = Field(
        description="Raw JD text pasted by the user"
    )
    
    # === Analysis Stage (LLM Output) ===
    extracted_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords extracted from JD, e.g. ['Distributed Systems', 'Golang']"
    )
    
    # === User Data ===
    # Matches LaTeX sections: "Languages", "Web Development", "Databases & Tools"
    current_skills: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Skills grouped by category for LaTeX section mapping"
    )
    project_inventory: List[Project] = Field(
        default_factory=list,
        description="All scraped GitHub projects before filtering"
    )
    
    # === Final Selection ===
    selected_projects: List[Project] = Field(
        default_factory=list,
        description="Top N projects selected for the resume"
    )
    reranked_skills: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Skills re-ordered by JD relevance within each category"
    )


class PipelineStage(BaseModel):
    """Tracks the current stage of the pipeline for state machine logic."""
    current: str = "IDLE"
    completed: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
