# Resume Tailor

A local-first, privacy-focused tool that tailors your resume for a specific Job Description.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    INGESTION    │────▶│    STRATEGY     │────▶│      BUILD      │
│                 │     │                 │     │                 │
│  • GitHub API   │     │  • LLM Match    │     │  • Jinja2/TeX   │
│  • Repo Filter  │     │  • Skill Rank   │     │  • Tectonic PDF │
│  • Commit Count │     │  • Bullet Gen   │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        └───────────────────────┴───────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │     ResumeState       │
                    │   (Pydantic Model)    │
                    └───────────────────────┘
```

## How It Works

1. **Ingest**: Scrapes your GitHub repos, filtering for projects with >5 commits
2. **Strategize**: Uses an LLM to match projects to the JD, generate bullet points, and re-rank skills
3. **Build**: Renders a Jinja2 LaTeX template and compiles to PDF with Tectonic

## Project Structure

```
resume-tailor/
├── src/
│   ├── core/           # Pipeline logic
│   │   ├── pipeline.py     # State machine orchestrator
│   │   ├── ingestion.py    # GitHub scraping
│   │   ├── strategy.py     # LLM matching/ranking
│   │   └── builder.py      # LaTeX rendering + compilation
│   ├── models/         # Pydantic schemas
│   │   ├── state.py        # Project, ResumeState
│   │   └── config.py       # AppConfig, GitHubConfig, etc.
│   └── services/       # External API clients
│       ├── github_service.py   # Async GitHub API
│       └── llm_service.py      # OpenAI/Ollama/Groq abstraction
├── templates/          # Jinja2 LaTeX templates
│   └── resume.tex.j2
├── output/             # Generated PDFs (gitignored)
├── requirements.txt
├── pyproject.toml
└── .env.example
```

## Quick Start

### Prerequisites

- Python 3.11+
- [Tectonic](https://tectonic-typesetting.github.io/) for LaTeX compilation
- GitHub Personal Access Token
- OpenAI API key (or Ollama for local LLM)

### Installation

```bash
# Clone the repo
git clone https://github.com/SuyeshJadhav/resume-tailor.git
cd resume-tailor

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -e ".[dev]"

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Usage

```python
import asyncio
from src.core import ResumeOrchestrator
from src.models import ResumeState, AppConfig, GitHubConfig

async def main():
    # Load config from environment
    config = AppConfig(
        github=GitHubConfig(
            token="ghp_...",
            username="your-github-username"
        )
    )
    
    # Initialize state with your JD and current skills
    state = ResumeState(
        job_description_text="We are looking for a Python developer...",
        current_skills={
            "Languages": ["Python", "TypeScript", "Go"],
            "Web": ["FastAPI", "React", "PostgreSQL"],
            "Tools": ["Docker", "Kubernetes", "AWS"],
        }
    )
    
    # Run the pipeline
    orchestrator = ResumeOrchestrator(config)
    final_state = await orchestrator.run(state)
    
    # PDF is generated at output/resume.pdf
    print(f"Selected projects: {[p.name for p in final_state.selected_projects]}")

asyncio.run(main())
```

## Configuration

All configuration is via environment variables (see `.env.example`):

| Variable | Description |
|----------|-------------|
| `RESUME_GITHUB__TOKEN` | GitHub Personal Access Token |
| `RESUME_GITHUB__USERNAME` | Your GitHub username |
| `RESUME_LLM__PROVIDER` | `openai`, `ollama`, or `groq` |
| `RESUME_LLM__API_KEY` | API key for cloud providers |
| `RESUME_MAX_PROJECTS` | Max projects to include (default: 4) |

## Design Principles

- **Type Safety**: All data flows through Pydantic models
- **State Machine**: Deterministic pipeline with clear stage transitions
- **Separation of Concerns**: Services, core logic, and models are isolated
- **No String Concatenation**: Jinja2 templates for all LaTeX generation
- **Retry Logic**: Tenacity-based retries for external API calls

## License

MIT
