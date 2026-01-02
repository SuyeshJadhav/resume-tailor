"""LLM service abstraction for resume strategy operations.

Supports multiple backends: OpenAI, Ollama, Groq.
Uses langchain-core for unified interface.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Protocol
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from ..models import Project, LLMConfig

logger = logging.getLogger(__name__)


# === Output Schemas for Structured LLM Responses ===

class KeywordExtractionResult(BaseModel):
    """Schema for JD keyword extraction."""
    keywords: List[str] = Field(description="Key skills/requirements from the JD")
    priority_areas: List[str] = Field(description="Top 3 most important areas")


class ProjectMatchResult(BaseModel):
    """Schema for project-to-JD matching."""
    relevance_score: float = Field(ge=0.0, le=1.0, description="Match score 0-1")
    relevance_reason: str = Field(description="Why this project matches")
    generated_bullets: List[str] = Field(
        max_length=3,
        description="3 resume bullet points (Result, Action, Tech)"
    )


class SkillRankResult(BaseModel):
    """Schema for skill re-ranking."""
    reranked_skills: Dict[str, List[str]] = Field(
        description="Skills re-ordered by relevance within each category"
    )


class JDTagResult(BaseModel):
    """Schema for JD tagging."""
    primary_focus: str = Field(
        description="Primary role focus: 'ml', 'systems', 'frontend', or 'fullstack'"
    )
    tags: List[str] = Field(
        description="All relevant tags extracted from JD (lowercase, no spaces)"
    )
    priority_tags: List[str] = Field(
        description="Top 5 most important tags for matching"
    )


# === LLM Service Interface ===

class LLMServiceProtocol(Protocol):
    """Protocol defining the LLM service interface."""
    
    async def extract_keywords(self, job_description: str) -> KeywordExtractionResult:
        """Extract keywords and priority areas from a job description."""
        ...
    
    async def match_project(
        self, 
        project: Project, 
        job_description: str,
        keywords: List[str]
    ) -> ProjectMatchResult:
        """Score and generate bullets for a project against the JD."""
        ...
    
    async def rerank_skills(
        self,
        skills: Dict[str, List[str]],
        job_description: str,
        keywords: List[str]
    ) -> SkillRankResult:
        """Re-order skills by relevance to the JD."""
        ...


class LLMService:
    """LLM service implementation using langchain-core.
    
    Supports:
    - OpenAI (gpt-4o-mini, gpt-4o)
    - Ollama (local models)
    - Groq (fast inference)
    
    Usage:
        service = LLMService(config)
        keywords = await service.extract_keywords(jd_text)
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._llm: BaseChatModel | None = None
    
    def _get_llm(self) -> BaseChatModel:
        """Lazy-load the LLM based on configuration."""
        if self._llm is not None:
            return self._llm
        
        if self.config.provider == "openai":
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model=self.config.model,
                api_key=self.config.api_key,
                temperature=self.config.temperature,
            )
        elif self.config.provider == "ollama":
            from langchain_ollama import ChatOllama
            self._llm = ChatOllama(
                model=self.config.model,
                base_url=self.config.base_url or "http://localhost:11434",
                temperature=self.config.temperature,
            )
        elif self.config.provider == "groq":
            from langchain_groq import ChatGroq
            self._llm = ChatGroq(
                model=self.config.model,
                api_key=self.config.api_key,
                temperature=self.config.temperature,
            )
        else:
            raise ValueError(f"Unknown LLM provider: {self.config.provider}")
        
        return self._llm
    
    async def extract_keywords(self, job_description: str) -> KeywordExtractionResult:
        """Extract keywords and priority areas from a job description."""
        llm = self._get_llm()
        parser = JsonOutputParser(pydantic_object=KeywordExtractionResult)
        
        system_prompt = """You are an expert resume consultant. 
Extract the key technical skills, tools, and requirements from the job description.
Identify the 3 most important areas the employer is looking for.

Respond in JSON format matching this schema:
{format_instructions}"""
        
        messages = [
            SystemMessage(content=system_prompt.format(
                format_instructions=parser.get_format_instructions()
            )),
            HumanMessage(content=f"Job Description:\n\n{job_description}"),
        ]
        
        response = await llm.ainvoke(messages)
        result = parser.parse(response.content)
        
        logger.info(f"Extracted {len(result['keywords'])} keywords from JD")
        return KeywordExtractionResult(**result)
    
    async def match_project(
        self, 
        project: Project, 
        job_description: str,
        keywords: List[str],
        jd_tags: List[str] = None
    ) -> ProjectMatchResult:
        """Score and SELECT bullets for a project using tag-based matching.
        
        Uses master resume achievements matched by tag overlap.
        Selects appropriate variant based on JD focus (ml, systems, frontend).
        Falls back to generation for projects not in master resume.
        """
        from .master_resume_service import get_master_service, determine_focus_from_tags
        
        llm = self._get_llm()
        parser = JsonOutputParser(pydantic_object=ProjectMatchResult)
        
        # Use provided tags or extract from keywords
        jd_tags = jd_tags or [k.lower().replace(" ", "-") for k in keywords]
        
        # Try to get master resume data for this project
        master_service = get_master_service()
        project_data = master_service.get_project(project.name)
        
        if project_data and project_data.achievements:
            # === TAG-BASED SELECTION MODE ===
            logger.info(f"Using tag-based selection for {project.name}")
            
            # Determine focus from JD tags
            focus = determine_focus_from_tags(jd_tags)
            logger.info(f"Detected JD focus: {focus}")
            
            # Get matched bullets using tag overlap
            bullets = master_service.get_bullets_for_jd(
                project.name, 
                jd_tags, 
                focus=focus,
                top_n=3
            )
            
            # Get appropriate tech stack
            tech_stack = project_data.get_tech_stack(focus)
            
            # Calculate relevance score from tag overlap
            matched = master_service.match_achievements(project.name, jd_tags, top_n=3)
            avg_score = sum(score for _, score in matched) / len(matched) if matched else 0.0
            relevance_score = min(0.95, max(0.3, avg_score * 2))  # Scale to 0.3-0.95
            
            # Build reason
            matched_tags = set()
            for ach, _ in matched:
                matched_tags.update(tag for tag in ach.tags if tag in jd_tags)
            
            reason = (
                f"Selected for {focus.replace('_', ' ')}: "
                f"matched tags [{', '.join(list(matched_tags)[:5])}]"
            )
            
            return ProjectMatchResult(
                relevance_score=relevance_score,
                relevance_reason=reason,
                generated_bullets=bullets[:3]
            )
            
        else:
            # === FALLBACK GENERATION MODE ===
            logger.info(f"No master data for {project.name}, generating...")
            
            allowed_techs = project.stack + project.topics if project.stack else []
            readme_section = f"\nREADME:\n{project.readme_content}" if project.readme_content else ""
            
            system_prompt = f"""You are an expert resume writer. Be TRUTHFUL and ACCURATE.

ALLOWED TECHNOLOGIES: {', '.join(allowed_techs) if allowed_techs else 'Use only what appears in README/description'}

RULES:
1. Only mention technologies from the ALLOWED list or README.
2. Do NOT fabricate metrics or achievements.
3. Keep bullets factual based on README/description.

Respond in JSON format:
{{format_instructions}}"""
            
            user_prompt = f"""Project: {project.name}
Technologies: {', '.join(allowed_techs)}
Description: {project.description or 'No description'}{readme_section}

JD Keywords: {', '.join(keywords[:10])}

Generate 3 factual bullet points using ONLY the allowed technologies."""
        
            messages = [
                SystemMessage(content=system_prompt.format(
                    format_instructions=parser.get_format_instructions()
                )),
                HumanMessage(content=user_prompt),
            ]
            
            response = await llm.ainvoke(messages)
            result = parser.parse(response.content)
            
            logger.debug(f"Project {project.name} scored {result['relevance_score']:.2f}")
            return ProjectMatchResult(**result)
    
    async def rerank_skills(
        self,
        skills: Dict[str, List[str]],
        job_description: str,
        keywords: List[str]
    ) -> SkillRankResult:
        """Re-order skills within each category by relevance to the JD."""
        llm = self._get_llm()
        parser = JsonOutputParser(pydantic_object=SkillRankResult)
        
        # Get exact category names for enforcement
        category_names = list(skills.keys())
        
        system_prompt = f"""You are an expert resume optimizer.
Given a candidate's skills (grouped by category) and a job description,
re-order the skills WITHIN each category so the most relevant appear first.

CRITICAL RULES:
1. You MUST use EXACTLY these category names: {category_names}
2. Do NOT add new categories.
3. Do NOT remove categories.
4. Do NOT add new skills that aren't in the original list.
5. Do NOT remove skills from the original list.
6. ONLY reorder skills within each category.

Respond in JSON format:
{{format_instructions}}"""
        
        user_prompt = f"""Job Keywords: {', '.join(keywords)}

Current Skills (ordered by category):
{skills}

Job Description:
{job_description}"""
        
        messages = [
            SystemMessage(content=system_prompt.format(
                format_instructions=parser.get_format_instructions()
            )),
            HumanMessage(content=user_prompt),
        ]
        
        response = await llm.ainvoke(messages)
        result = parser.parse(response.content)
        
        # Post-process: enforce original categories only
        validated_skills = {}
        for cat in category_names:
            if cat in result['reranked_skills']:
                # Only keep skills that were in original
                original_set = set(skills[cat])
                validated_skills[cat] = [
                    s for s in result['reranked_skills'][cat] 
                    if s in original_set
                ]
                # Add any missing skills at the end
                for s in skills[cat]:
                    if s not in validated_skills[cat]:
                        validated_skills[cat].append(s)
            else:
                # Category missing from LLM output, keep original order
                validated_skills[cat] = skills[cat]
        
        logger.info("Skills reranked by JD relevance (validated)")
        return SkillRankResult(reranked_skills=validated_skills)
