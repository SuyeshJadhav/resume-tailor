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
        keywords: List[str]
    ) -> ProjectMatchResult:
        """Score and generate bullets for a project against the JD."""
        llm = self._get_llm()
        parser = JsonOutputParser(pydantic_object=ProjectMatchResult)
        
        system_prompt = """You are an expert resume writer.
Given a GitHub project and a job description, determine how relevant the project is.

Score from 0.0 (no match) to 1.0 (perfect match).
Generate exactly 3 resume bullet points following the pattern:
1. RESULT: What was achieved (with metrics if possible)
2. ACTION: What you built/implemented
3. TECH: Key technologies used

Respond in JSON format:
{format_instructions}"""
        
        user_prompt = f"""Job Description Keywords: {', '.join(keywords)}

Project: {project.name}
URL: {project.url}
Tech Stack: {', '.join(project.stack)}
Description: {project.description}

Full Job Description:
{job_description}"""
        
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
        
        system_prompt = """You are an expert resume optimizer.
Given a candidate's skills (grouped by category) and a job description,
re-order the skills WITHIN each category so the most relevant appear first.

Do NOT add or remove skills. Only reorder within each category.

Respond in JSON format:
{format_instructions}"""
        
        user_prompt = f"""Job Keywords: {', '.join(keywords)}

Current Skills:
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
        
        logger.info("Skills reranked by JD relevance")
        return SkillRankResult(**result)
