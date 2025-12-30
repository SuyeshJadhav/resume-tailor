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
        """Get the commit count for a repository.
        
        Uses the contributors endpoint which includes commit counts.
        Falls back to listing commits if needed.
        """
        if not self._client:
            raise GitHubServiceError("Service not initialized.")
        
        try:
            response = await self._client.get(
                f"/repos/{self.config.username}/{repo_name}/contributors",
                params={"per_page": 1},
            )
            response.raise_for_status()
            
            contributors = response.json()
            if contributors:
                # Return commits by the repo owner
                owner_commits = next(
                    (c["contributions"] for c in contributors 
                     if c["login"].lower() == self.config.username.lower()),
                    0
                )
                return owner_commits
        except httpx.HTTPStatusError:
            logger.warning(f"Could not get contributors for {repo_name}")
        
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
    
    async def filter_and_extract(
        self, 
        repos: List[Dict[str, Any]]
    ) -> List[Project]:
        """Filter repositories and extract Project models.
        
        Applies configured filters:
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
            # Apply exclusion filters
            if self.config.exclude_forks and repo.get("fork", False):
                logger.debug(f"Skipping fork: {repo['name']}")
                continue
                
            if self.config.exclude_archived and repo.get("archived", False):
                logger.debug(f"Skipping archived: {repo['name']}")
                continue
            
            # Get commit count
            commit_count = await self.get_commit_count(repo["name"])
            
            if commit_count < self.config.min_commits:
                logger.debug(
                    f"Skipping {repo['name']}: {commit_count} commits "
                    f"< {self.config.min_commits} minimum"
                )
                continue
            
            # Get languages for stack
            languages = await self.get_languages(repo["name"])
            
            project = Project(
                name=repo["name"],
                url=repo["html_url"],
                stack=languages,
                description=repo.get("description") or "",
                commit_count=commit_count,
            )
            
            projects.append(project)
            logger.info(f"Extracted project: {project.name} ({commit_count} commits)")
        
        return projects
