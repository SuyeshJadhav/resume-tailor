"""Test the full resume pipeline with a sample job description."""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from src.models import ResumeState, AppConfig, GitHubConfig, LLMConfig, BuildConfig
from src.core import ResumeOrchestrator
from pathlib import Path


# Sample JD - replace with a real one for better results
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

You'll be working on:
- Building scalable data pipelines
- Designing and implementing APIs
- Optimizing database performance
- Collaborating with frontend and ML teams
"""

# Your current skills (matches LaTeX sections)
CURRENT_SKILLS = {
    "Languages": ["Python", "JavaScript", "TypeScript", "Java", "Dart", "C"],
    "Web Development": ["FastAPI", "React", "Next.js", "Flask", "Node.js"],
    "Databases and Tools": ["PostgreSQL", "MongoDB", "Redis", "Docker", "Git"],
}


async def run_pipeline():
    """Run the full pipeline and display results."""
    
    # Build config from environment
    config = AppConfig(
        github=GitHubConfig(
            token=os.getenv("RESUME_GITHUB__TOKEN", ""),
            username=os.getenv("RESUME_GITHUB__USERNAME", ""),
            min_commits=int(os.getenv("RESUME_GITHUB__MIN_COMMITS", "5")),
            exclude_forks=os.getenv("RESUME_GITHUB__EXCLUDE_FORKS", "false").lower() == "true",
            min_updated_year=2024,  # Only projects updated in 2024 or later
            min_readme_length=100,  # Only projects with substantial README
        ),
        llm=LLMConfig(
            provider=os.getenv("RESUME_LLM__PROVIDER", "openai"),
            model=os.getenv("RESUME_LLM__MODEL", "gpt-4o-mini"),
            api_key=os.getenv("RESUME_LLM__API_KEY"),
            temperature=0.3,
        ),
        build=BuildConfig(
            template_path=Path("templates/Resume_Template/resume.tex").resolve(),
            output_dir=Path("output").resolve(),
            tectonic_path=r"C:\Users\SuyeshJadhav\Documents\Apps\tectonic.exe",
        ),
        max_projects=4,
    )
    
    # Initialize state
    state = ResumeState(
        job_description_text=SAMPLE_JD,
        current_skills=CURRENT_SKILLS,
    )
    
    print("=" * 60)
    print("RESUME TAILOR - Pipeline Test")
    print("=" * 60)
    print(f"\n📋 JD Preview: {SAMPLE_JD[:100].strip()}...")
    print(f"🔧 LLM Provider: {config.llm.provider} / {config.llm.model}")
    print(f"📊 Max Projects: {config.max_projects}")
    print("\n" + "=" * 60)
    
    # Run orchestrator
    orchestrator = ResumeOrchestrator(config)
    
    try:
        final_state = await orchestrator.run(state)
        
        # Display results
        print("\n✅ PIPELINE COMPLETED\n")
        
        print("📌 EXTRACTED KEYWORDS:")
        for kw in final_state.extracted_keywords[:10]:
            print(f"   • {kw}")
        
        print(f"\n🎯 SELECTED PROJECTS ({len(final_state.selected_projects)}):\n")
        
        for i, project in enumerate(final_state.selected_projects, 1):
            print(f"{i}. {project.name}")
            print(f"   Score: {project.relevance_score:.2f}")
            print(f"   Reason: {project.relevance_reason}")
            print(f"   Stack: {', '.join(project.stack[:5])}")
            print(f"   Bullets:")
            for bullet in project.generated_bullets:
                print(f"     • {bullet}")
            print()
        
        print("📝 RERANKED SKILLS:")
        for category, skills in final_state.reranked_skills.items():
            print(f"   {category}: {', '.join(skills[:5])}...")
        
        print("\n" + "=" * 60)
        print("📄 Output: output/resume.tex and output/resume.pdf")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_pipeline())
