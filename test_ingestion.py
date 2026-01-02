"""Quick test script to verify GitHub ingestion works."""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.models import GitHubConfig, Project
from src.services import GitHubService


async def test_ingestion():
    """Test GitHub API connection and project scraping."""
    
    # Build config from environment
    config = GitHubConfig(
        token=os.getenv("RESUME_GITHUB__TOKEN", ""),
        username=os.getenv("RESUME_GITHUB__USERNAME", ""),
        min_commits=int(os.getenv("RESUME_GITHUB__MIN_COMMITS", "5")),
    )
    
    print(f"📡 Testing GitHub ingestion for: {config.username}")
    print(f"   Min commits threshold: {config.min_commits}")
    print("-" * 50)
    
    async with GitHubService(config) as gh:
        # Fetch repos
        print("🔍 Fetching repositories...")
        repos = await gh.fetch_user_repos()
        print(f"   Found {len(repos)} total repositories")
        
        # Filter and extract
        print("\n🔬 Filtering and extracting project metadata...")
        projects = await gh.filter_and_extract(repos)
        
        print(f"\n✅ Extracted {len(projects)} qualifying projects:\n")
        
        for i, p in enumerate(projects[:10], 1):  # Show top 10
            print(f"{i:2}. {p.name}")
            print(f"    📊 {p.commit_count} commits | Stack: {', '.join(p.stack[:4])}")
            print(f"    🔗 {p.url}")
            if p.description:
                print(f"    📝 {p.description[:80]}...")
            print()
        
        if len(projects) > 10:
            print(f"   ... and {len(projects) - 10} more projects")
        
        return projects


if __name__ == "__main__":
    projects = asyncio.run(test_ingestion())
    print(f"\n🎯 Total projects ready for LLM matching: {len(projects)}")
