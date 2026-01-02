"""List all extracted projects with their details."""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from src.models import GitHubConfig
from src.services import GitHubService


async def list_projects():
    config = GitHubConfig(
        token=os.getenv("RESUME_GITHUB__TOKEN", ""),
        username=os.getenv("RESUME_GITHUB__USERNAME", ""),
        min_commits=int(os.getenv("RESUME_GITHUB__MIN_COMMITS", "5")),
        exclude_forks=os.getenv("RESUME_GITHUB__EXCLUDE_FORKS", "false").lower() == "true",
    )
    
    async with GitHubService(config) as gh:
        repos = await gh.fetch_user_repos()
        projects = await gh.filter_and_extract(repos)
        
        # Sort by commits
        projects.sort(key=lambda p: p.commit_count, reverse=True)
        
        for p in projects:
            print(f"{p.name}")
            print(f"  Commits: {p.commit_count}")
            print(f"  Stack: {p.stack}")
            print(f"  URL: {p.url}")
            print()


if __name__ == "__main__":
    asyncio.run(list_projects())
