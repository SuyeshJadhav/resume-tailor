"""GitHub API service for repository scraping and analysis.

Uses httpx for async HTTP requests and tenacity for robust retry logic.
"""

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from typing import List, Dict, Any, Optional
import logging

from ..models import Project, GitHubConfig

logger = logging.getLogger(__name__)


class GitHubServiceError(Exception):
    """Base exception for GitHub service errors."""
    pass


class RateLimitError(GitHubServiceError):
    """Raised when GitHub API rate limit is exceeded."""
    pass


class GitHubService:
    """Async GitHub API client for repository scraping.
    
    Responsibilities:
    - Fetch all repositories for a user
    - Filter by commit count and other criteria
    - Extract project metadata (languages, README, topics)
    
    Usage:
        async with GitHubService(config) as gh:
            repos = await gh.fetch_user_repos()
            projects = await gh.filter_and_extract(repos)
    """
    
    BASE_URL = "https://api.github.com"
    
    def __init__(self, config: GitHubConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self) -> "GitHubService":
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                "Authorization": f"Bearer {self.config.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=30.0,
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(httpx.HTTPStatusError),
    )
    async def fetch_user_repos(self) -> List[Dict[str, Any]]:
        """Fetch all repositories for the configured user.
        
        Returns:
            List of raw repository dictionaries from GitHub API.
            
        Raises:
            RateLimitError: If rate limit is exceeded.
            GitHubServiceError: For other API errors.
        """
        if not self._client:
            raise GitHubServiceError("Service not initialized. Use async context manager.")
        
        repos: List[Dict[str, Any]] = []
        page = 1
        
        while True:
            response = await self._client.get(
                f"/users/{self.config.username}/repos",
                params={"per_page": 100, "page": page, "sort": "updated"},
            )
            
            if response.status_code == 403:
                raise RateLimitError("GitHub API rate limit exceeded")
            
            response.raise_for_status()
            page_repos = response.json()
            
            if not page_repos:
                break
                
            repos.extend(page_repos)
            page += 1
            
            logger.debug(f"Fetched page {page - 1}, total repos: {len(repos)}")
        
        return repos
    
    async def get_commit_count(self, repo_name: str) -> int:
        """Get the commit count for a repository by the configured user.
        
        Uses the commits API with author filter to correctly count
        commits in forks where the user has contributed.
        """
        if not self._client:
            raise GitHubServiceError("Service not initialized.")
        
        try:
            # Use commits API with author filter - works for forks too
            response = await self._client.get(
                f"/repos/{self.config.username}/{repo_name}/commits",
                params={
                    "author": self.config.username,
                    "per_page": 1,  # We just need the count from headers
                },
            )
            
            if response.status_code == 409:
                # Empty repository
                return 0
            
            response.raise_for_status()
            
            # GitHub returns pagination info in Link header
            # Format: <url>; rel="last", page=N
            link_header = response.headers.get("Link", "")
            
            if "last" in link_header:
                # Extract page number from last link
                import re
                match = re.search(r'page=(\d+)>; rel="last"', link_header)
                if match:
                    return int(match.group(1))
            
            # If no "last" link, count is <= per_page
            commits = response.json()
            return len(commits) if isinstance(commits, list) else 0
            
        except httpx.HTTPStatusError as e:
            logger.warning(f"Could not get commits for {repo_name}: {e}")
            return 0
    
    async def get_languages(self, repo_name: str) -> List[str]:
        """Get the programming languages used in a repository."""
        if not self._client:
            raise GitHubServiceError("Service not initialized.")
        
        response = await self._client.get(
            f"/repos/{self.config.username}/{repo_name}/languages"
        )
        response.raise_for_status()
        
        languages = response.json()
        # Return languages sorted by bytes of code (descending)
        return list(languages.keys())
    
    async def get_readme(self, repo_name: str, max_chars: int = 500) -> str:
        """Get the README content for a repository (truncated).
        
        Returns first ~500 chars to provide context without overwhelming LLM.
        """
        if not self._client:
            raise GitHubServiceError("Service not initialized.")
        
        try:
            response = await self._client.get(
                f"/repos/{self.config.username}/{repo_name}/readme",
                headers={"Accept": "application/vnd.github.raw+json"},
            )
            
            if response.status_code == 404:
                return ""
            
            response.raise_for_status()
            content = response.text
            
            # Truncate and clean up
            if len(content) > max_chars:
                content = content[:max_chars] + "..."
            
            return content
            
        except httpx.HTTPStatusError:
            logger.debug(f"No README for {repo_name}")
            return ""
    
    async def get_topics(self, repo_name: str) -> List[str]:
        """Get GitHub topics/tags for a repository."""
        if not self._client:
            raise GitHubServiceError("Service not initialized.")
        
        try:
            response = await self._client.get(
                f"/repos/{self.config.username}/{repo_name}/topics",
                headers={"Accept": "application/vnd.github+json"},
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("names", [])
            
        except httpx.HTTPStatusError:
            return []
    
    async def filter_and_extract(
        self, 
        repos: List[Dict[str, Any]]
    ) -> List[Project]:
        """Filter repositories and extract Project models.
        
        Applies configured filters:
        - Profile README repos (name == username)
        - Minimum commit threshold
        - Fork exclusion
        - Archive exclusion
        
        Args:
            repos: Raw repository data from fetch_user_repos()
            
        Returns:
            List of Project models ready for LLM processing.
        """
        projects: List[Project] = []
        
        for repo in repos:
            repo_name = repo["name"]
            
            # Skip profile README repos (username/username)
            if repo_name.lower() == self.config.username.lower():
                logger.debug(f"Skipping profile README: {repo_name}")
                continue
            
            # Apply exclusion filters
            if self.config.exclude_forks and repo.get("fork", False):
                logger.debug(f"Skipping fork: {repo_name}")
                continue
                
            if self.config.exclude_archived and repo.get("archived", False):
                logger.debug(f"Skipping archived: {repo_name}")
                continue
            
            # Date filter - skip old repos
            if self.config.min_updated_year:
                updated_at = repo.get("pushed_at", repo.get("updated_at", ""))
                if updated_at:
                    try:
                        from datetime import datetime
                        update_year = datetime.fromisoformat(updated_at.replace("Z", "+00:00")).year
                        if update_year < self.config.min_updated_year:
                            logger.debug(f"Skipping old repo: {repo_name} (last updated {update_year})")
                            continue
                    except (ValueError, AttributeError):
                        pass
            
            # Get commit count
            commit_count = await self.get_commit_count(repo_name)
            
            if commit_count < self.config.min_commits:
                logger.debug(
                    f"Skipping {repo_name}: {commit_count} commits "
                    f"< {self.config.min_commits} minimum"
                )
                continue
            
            # Get enriched metadata
            languages = await self.get_languages(repo_name)
            readme = await self.get_readme(repo_name)
            topics = await self.get_topics(repo_name)
            
            # README quality filter
            if self.config.min_readme_length > 0:
                if len(readme) < self.config.min_readme_length:
                    logger.debug(
                        f"Skipping {repo_name}: README too short "
                        f"({len(readme)} < {self.config.min_readme_length} chars)"
                    )
                    continue
            
            project = Project(
                name=repo_name,
                url=repo["html_url"],
                stack=languages,
                description=repo.get("description") or "",
                readme_content=readme,
                topics=topics,
                commit_count=commit_count,
            )
            
            projects.append(project)
            logger.info(f"Extracted project: {project.name} ({commit_count} commits)")
        
        return projects
