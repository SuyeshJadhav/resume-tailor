"""Pydantic models for tag-based bullet selection system."""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class Achievement(BaseModel):
    """A single achievement with tags and role-focused variants."""
    id: str = Field(description="Unique identifier for the achievement")
    metric: str = Field(description="The quantified metric (never changes)")
    tags: List[str] = Field(description="Tags for matching to JD keywords")
    variants: Dict[str, str] = Field(
        description="Role-focused bullet variants (default, ml_focus, systems_focus, etc.)"
    )
    
    def get_variant(self, focus: str = "default") -> str:
        """Get the bullet variant for a specific focus, falling back to default."""
        return self.variants.get(focus, self.variants.get("default", ""))
    
    def tag_overlap_score(self, jd_tags: List[str]) -> float:
        """Calculate overlap score between achievement tags and JD tags."""
        if not jd_tags or not self.tags:
            return 0.0
        
        jd_set = set(tag.lower() for tag in jd_tags)
        achievement_set = set(tag.lower() for tag in self.tags)
        
        overlap = jd_set & achievement_set
        # Jaccard-like scoring: overlap / union
        union = jd_set | achievement_set
        return len(overlap) / len(union) if union else 0.0


class ProjectData(BaseModel):
    """Project data with achievements."""
    display_name: str = Field(description="Display name for resume")
    github_url: str = Field(default="", description="GitHub repository URL")
    tech_stack: Dict[str, str] = Field(
        default_factory=dict,
        description="Tech stack variants (default, ml_focus, systems_focus)"
    )
    achievements: List[Achievement] = Field(
        default_factory=list,
        description="List of achievements with tags and variants"
    )
    
    def get_tech_stack(self, focus: str = "default") -> str:
        """Get tech stack for a specific focus."""
        return self.tech_stack.get(focus, self.tech_stack.get("default", ""))
    
    def match_achievements(
        self, 
        jd_tags: List[str], 
        top_n: int = 3
    ) -> List[tuple[Achievement, float]]:
        """Return top N achievements sorted by tag overlap score."""
        scored = [
            (achievement, achievement.tag_overlap_score(jd_tags))
            for achievement in self.achievements
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]


class MasterResume(BaseModel):
    """Complete master resume data."""
    tag_vocabulary: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Tag categories and their keywords"
    )
    projects: Dict[str, ProjectData] = Field(
        default_factory=dict,
        description="Projects keyed by internal name"
    )
    
    def get_project(self, name: str) -> Optional[ProjectData]:
        """Get project by name (case-insensitive, flexible matching)."""
        # Normalize key
        key = name.lower().replace("-", "_").replace(" ", "_")
        
        # Exact match
        if key in self.projects:
            return self.projects[key]
        
        # Try lowercase keys
        for stored_key, project in self.projects.items():
            if stored_key.lower() == key or key in stored_key.lower():
                return project
        
        return None


class JDTagResult(BaseModel):
    """Result of JD tagging by LLM."""
    primary_focus: str = Field(
        description="Primary role focus: 'ml', 'systems', 'frontend', or 'fullstack'"
    )
    tags: List[str] = Field(
        description="Extracted tags from job description"
    )
    priority_tags: List[str] = Field(
        description="Top 5 most important tags"
    )
