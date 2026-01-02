"""Test INGEST + STRATEGIZE stages without PDF build."""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from src.models import ResumeState, GitHubConfig, LLMConfig
from src.services import GitHubService, LLMService


SAMPLE_JD = """
Software Engineer - Backend

We're looking for a skilled backend engineer to join our team.

Requirements:
- 2+ years of experience with Python or Go
- Experience with REST APIs and microservices architecture
- Familiarity with databases (PostgreSQL, MongoDB, Redis)
- Experience with cloud platforms (AWS, GCP, or Azure)
- Knowledge of containerization (Docker, Kubernetes)
- Strong understanding of software design patterns

Nice to have:
- Experience with Machine Learning or NLP
- Open source contributions
- Experience with real-time systems
"""

CURRENT_SKILLS = {
    "Languages": ["Python", "JavaScript", "TypeScript", "Java", "Dart", "C"],
    "Web Development": ["FastAPI", "React", "Next.js", "Flask", "Node.js"],
    "Databases & Tools": ["PostgreSQL", "MongoDB", "Redis", "Docker", "Git"],
}


async def test_strategy():
    """Test ingestion and LLM strategy without PDF build."""
    
    github_config = GitHubConfig(
        token=os.getenv("RESUME_GITHUB__TOKEN", ""),
        username=os.getenv("RESUME_GITHUB__USERNAME", ""),
        min_commits=int(os.getenv("RESUME_GITHUB__MIN_COMMITS", "5")),
        exclude_forks=os.getenv("RESUME_GITHUB__EXCLUDE_FORKS", "false").lower() == "true",
    )
    
    llm_config = LLMConfig(
        provider=os.getenv("RESUME_LLM__PROVIDER", "openai"),
        model=os.getenv("RESUME_LLM__MODEL", "gpt-4o-mini"),
        api_key=os.getenv("RESUME_LLM__API_KEY"),
        base_url=os.getenv("RESUME_LLM__BASE_URL"),
        temperature=0.3,
    )
    
    print("=" * 60)
    print("STRATEGY TEST (No PDF)")
    print("=" * 60)
    print(f"LLM: {llm_config.provider} / {llm_config.model}")
    print("=" * 60)
    
    # INGEST
    print("\n[1/3] INGESTING projects from GitHub...")
    async with GitHubService(github_config) as gh:
        repos = await gh.fetch_user_repos()
        projects = await gh.filter_and_extract(repos)
    print(f"      Found {len(projects)} projects")
    
    # STRATEGIZE - Extract keywords
    print("\n[2/3] EXTRACTING keywords from JD...")
    llm = LLMService(llm_config)
    keyword_result = await llm.extract_keywords(SAMPLE_JD)
    print(f"      Keywords: {keyword_result.keywords[:8]}")
    
    # STRATEGIZE - Match projects
    print("\n[3/3] MATCHING projects to JD...")
    scored_projects = []
    for project in projects:
        try:
            match = await llm.match_project(project, SAMPLE_JD, keyword_result.keywords)
            project_copy = project.model_copy(update={
                "relevance_score": match.relevance_score,
                "relevance_reason": match.relevance_reason,
                "generated_bullets": match.generated_bullets,
            })
            scored_projects.append(project_copy)
            print(f"      {project.name}: {match.relevance_score:.2f}")
        except Exception as e:
            print(f"      {project.name}: ERROR - {e}")
    
    # Sort and select top 4
    scored_projects.sort(key=lambda p: p.relevance_score, reverse=True)
    selected = scored_projects[:4]
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    for i, p in enumerate(selected, 1):
        print(f"\n{i}. {p.name} (Score: {p.relevance_score:.2f})")
        print(f"   Stack: {', '.join(p.stack[:4])}")
        print(f"   Reason: {p.relevance_reason}")
        print("   Bullets:")
        for bullet in p.generated_bullets:
            print(f"     - {bullet}")
    
    # Rerank skills
    print("\n" + "=" * 60)
    print("RERANKED SKILLS")
    print("=" * 60)
    rerank = await llm.rerank_skills(CURRENT_SKILLS, SAMPLE_JD, keyword_result.keywords)
    for cat, skills in rerank.reranked_skills.items():
        print(f"{cat}: {', '.join(skills)}")


if __name__ == "__main__":
    asyncio.run(test_strategy())
