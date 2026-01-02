"""Master Resume Service - Tag-based bullet selection system.

Loads master resume data and matches achievements to JD tags.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..models.bullet_schema import (
    Achievement, 
    ProjectData, 
    MasterResume
)

logger = logging.getLogger(__name__)


class MasterResumeService:
    """Service for tag-based bullet selection.
    
    Loads master resume YAML and provides:
    - Project lookup by name
    - Achievement matching by tag overlap
    - Variant selection based on JD focus
    """
    
    def __init__(self, yaml_path: Path = None):
        self.yaml_path = yaml_path or Path("data/master_resume.yaml")
        self._data: Optional[MasterResume] = None
        self._loaded = False
    
    def load(self) -> None:
        """Load master resume from YAML file."""
        if not self.yaml_path.exists():
            logger.warning(f"Master resume file not found: {self.yaml_path}")
            return
        
        try:
            with open(self.yaml_path, encoding="utf-8") as f:
                raw_data = yaml.safe_load(f)
            
            # Parse projects
            projects = {}
            for name, project_data in raw_data.get("projects", {}).items():
                # Parse achievements
                achievements = []
                for ach_data in project_data.get("achievements", []):
                    achievements.append(Achievement(**ach_data))
                
                projects[name.lower()] = ProjectData(
                    display_name=project_data.get("display_name", name),
                    github_url=project_data.get("github_url", ""),
                    tech_stack=project_data.get("tech_stack", {}),
                    achievements=achievements
                )
            
            self._data = MasterResume(
                tag_vocabulary=raw_data.get("tag_vocabulary", {}),
                projects=projects
            )
            
            logger.info(
                f"Loaded master resume: {len(projects)} projects, "
                f"{sum(len(p.achievements) for p in projects.values())} achievements"
            )
            self._loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load master resume: {e}")
            raise
    
    def get_project(self, name: str) -> Optional[ProjectData]:
        """Get project by name (case-insensitive matching)."""
        if not self._loaded:
            self.load()
        
        if not self._data:
            return None
        
        return self._data.get_project(name)
    
    def match_achievements(
        self,
        project_name: str,
        jd_tags: List[str],
        top_n: int = 3
    ) -> List[Tuple[Achievement, float]]:
        """Match project achievements to JD tags.
        
        Args:
            project_name: Name of the project
            jd_tags: Tags extracted from job description
            top_n: Number of top achievements to return
            
        Returns:
            List of (Achievement, score) tuples sorted by relevance
        """
        project = self.get_project(project_name)
        if not project:
            logger.warning(f"Project not found: {project_name}")
            return []
        
        return project.match_achievements(jd_tags, top_n)
    
    def get_bullets_for_jd(
        self,
        project_name: str,
        jd_tags: List[str],
        focus: str = "default",
        top_n: int = 3
    ) -> List[str]:
        """Get formatted bullets for a project based on JD tags.
        
        Args:
            project_name: Name of the project
            jd_tags: Tags extracted from job description
            focus: Variant focus (default, ml_focus, systems_focus, frontend_focus)
            top_n: Number of bullets to return
            
        Returns:
            List of bullet strings with appropriate variants
        """
        matched = self.match_achievements(project_name, jd_tags, top_n)
        
        bullets = []
        for achievement, score in matched:
            bullet = achievement.get_variant(focus)
            if bullet:
                bullets.append(bullet)
                logger.debug(
                    f"Selected achievement '{achievement.id}' "
                    f"(score={score:.2f}, focus={focus})"
                )
        
        return bullets
    
    def get_tech_stack(
        self,
        project_name: str,
        focus: str = "default"
    ) -> str:
        """Get tech stack string for a project with specific focus."""
        project = self.get_project(project_name)
        if not project:
            return ""
        
        return project.get_tech_stack(focus)
    
    def get_all_projects(self) -> Dict[str, ProjectData]:
        """Get all projects."""
        if not self._loaded:
            self.load()
        
        return self._data.projects if self._data else {}


# Global service instance
_service: Optional[MasterResumeService] = None


def get_master_service() -> MasterResumeService:
    """Get or create the master resume service instance."""
    global _service
    if _service is None:
        _service = MasterResumeService()
        _service.load()
    return _service


def determine_focus_from_tags(jd_tags: List[str]) -> str:
    """Determine the best variant focus based on JD tags.
    
    Args:
        jd_tags: Tags extracted from job description
        
    Returns:
        Focus string: 'ml_focus', 'systems_focus', 'frontend_focus', or 'default'
    """
    # Normalize tags
    tag_set = set(tag.lower().replace(" ", "-").replace("_", "-") for tag in jd_tags)
    
    # Expand common variations
    expanded_tags = set()
    for tag in tag_set:
        expanded_tags.add(tag)
        # Handle common variations
        if "api" in tag:
            expanded_tags.add("apis")
            expanded_tags.add("backend")
        if "rest" in tag:
            expanded_tags.add("apis")
            expanded_tags.add("backend")
        if "microservice" in tag:
            expanded_tags.add("microservices")
            expanded_tags.add("backend")
            expanded_tags.add("distributed")
        if "database" in tag or tag in {"postgresql", "mongodb", "redis", "mysql", "sqlite"}:
            expanded_tags.add("databases")
            expanded_tags.add("backend")
        if tag in {"aws", "gcp", "azure", "cloud"}:
            expanded_tags.add("backend")
            expanded_tags.add("distributed")
        if tag in {"python", "go", "java", "node.js", "nodejs"}:
            expanded_tags.add("backend")
        if "llm" in tag or "gpt" in tag or tag in {"langchain", "openai", "ollama"}:
            expanded_tags.add("llm")
            expanded_tags.add("ml")
        if tag in {"react", "vue", "angular", "svelte", "nextjs", "next.js"}:
            expanded_tags.add("frontend")
    
    # Define focus indicators
    ml_indicators = {"llm", "nlp", "ml", "machine-learning", "ai", "transformers", 
                     "embeddings", "inference", "pytorch", "tensorflow", "agents",
                     "deep-learning", "neural", "model", "training", "huggingface"}
    systems_indicators = {"backend", "distributed", "microservices", "apis", 
                          "databases", "caching", "latency", "throughput", "scalability",
                          "infrastructure", "devops", "kubernetes", "docker", "redis",
                          "postgresql", "mongodb", "rest-apis", "grpc"}
    frontend_indicators = {"frontend", "react", "ui", "visualization", "d3js", 
                           "nextjs", "components", "typescript", "css", "vue",
                           "angular", "svelte", "tailwind", "responsive"}
    
    # Count overlaps
    ml_score = len(expanded_tags & ml_indicators)
    systems_score = len(expanded_tags & systems_indicators)
    frontend_score = len(expanded_tags & frontend_indicators)
    
    logger.debug(f"Focus scores - ML: {ml_score}, Systems: {systems_score}, Frontend: {frontend_score}")
    
    # Determine focus
    max_score = max(ml_score, systems_score, frontend_score)
    
    if max_score == 0:
        return "default"
    elif systems_score == max_score:
        return "systems_focus"  # Prioritize systems for ties
    elif ml_score == max_score:
        return "ml_focus"
    else:
        return "frontend_focus"
