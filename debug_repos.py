"""Debug script to see ALL repos and why they're filtered."""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from src.models import GitHubConfig
from src.services import GitHubService

async def debug_repos():
    config = GitHubConfig(
        token=os.getenv("RESUME_GITHUB__TOKEN", ""),
        username=os.getenv("RESUME_GITHUB__USERNAME", ""),
        min_commits=1,  # Show everything
        exclude_forks=False,
        exclude_archived=False,
    )
    
    print(f"Fetching ALL repos for {config.username}...\n")
    
    async with GitHubService(config) as gh:
        repos = await gh.fetch_user_repos()
        
        print(f"{'Repo':<30} {'Fork?':<6} {'Commits':<8} {'Languages'}")
        print("-" * 70)
        
        for repo in repos:
            name = repo["name"][:29]
            is_fork = "Yes" if repo.get("fork") else "No"
            
            # Get commit count
            commits = await gh.get_commit_count(repo["name"])
            
            # Get languages
            try:
                langs = await gh.get_languages(repo["name"])
                lang_str = ", ".join(langs[:3]) if langs else "None"
            except:
                lang_str = "Error"
            
            print(f"{name:<30} {is_fork:<6} {commits:<8} {lang_str}")


if __name__ == "__main__":
    asyncio.run(debug_repos())
