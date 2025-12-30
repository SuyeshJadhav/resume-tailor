"""Resume Orchestrator - Main pipeline orchestration.

Implements a state machine pattern for deterministic pipeline execution.
Prepares for future migration to LangGraph.
"""

from enum import Enum, auto
from typing import Callable, Awaitable, Dict, Optional
import logging

from ..models import ResumeState, PipelineStage, AppConfig
from .ingestion import ingest_projects
from .strategy import strategize
from .builder import build_resume

logger = logging.getLogger(__name__)


class Stage(Enum):
    """Pipeline stages in execution order."""
    IDLE = auto()
    INGEST = auto()
    STRATEGIZE = auto()
    BUILD = auto()
    COMPLETE = auto()
    ERROR = auto()


# Type alias for stage handlers
StageHandler = Callable[[ResumeState, AppConfig], Awaitable[ResumeState]]


class ResumeOrchestrator:
    """State machine orchestrator for the resume pipeline.
    
    Manages the flow: IDLE → INGEST → STRATEGIZE → BUILD → COMPLETE
    
    Each stage:
    1. Validates preconditions
    2. Executes the stage handler
    3. Updates pipeline state
    4. Transitions to the next stage
    
    Usage:
        orchestrator = ResumeOrchestrator(config)
        final_state = await orchestrator.run(initial_state)
    """
    
    # Stage transition map: current -> next
    TRANSITIONS: Dict[Stage, Stage] = {
        Stage.IDLE: Stage.INGEST,
        Stage.INGEST: Stage.STRATEGIZE,
        Stage.STRATEGIZE: Stage.BUILD,
        Stage.BUILD: Stage.COMPLETE,
    }
    
    # Stage handlers
    HANDLERS: Dict[Stage, StageHandler] = {
        Stage.INGEST: ingest_projects,
        Stage.STRATEGIZE: strategize,
        Stage.BUILD: build_resume,
    }
    
    def __init__(self, config: AppConfig):
        self.config = config
        self._current_stage = Stage.IDLE
        self._pipeline_state = PipelineStage()
    
    @property
    def current_stage(self) -> Stage:
        return self._current_stage
    
    def _validate_preconditions(self, state: ResumeState) -> None:
        """Validate that the state is valid for current stage."""
        if self._current_stage == Stage.INGEST:
            if not state.job_description_text:
                raise ValueError("Job description is required for INGEST stage")
        
        elif self._current_stage == Stage.STRATEGIZE:
            if not state.project_inventory:
                raise ValueError("No projects found. INGEST stage may have failed.")
        
        elif self._current_stage == Stage.BUILD:
            if not state.selected_projects:
                raise ValueError("No projects selected. STRATEGIZE stage may have failed.")
    
    async def _execute_stage(self, state: ResumeState) -> ResumeState:
        """Execute the handler for the current stage."""
        handler = self.HANDLERS.get(self._current_stage)
        
        if handler is None:
            logger.debug(f"No handler for stage {self._current_stage.name}, skipping")
            return state
        
        logger.info(f"Executing stage: {self._current_stage.name}")
        
        try:
            self._validate_preconditions(state)
            updated_state = await handler(state, self.config)
            self._pipeline_state.completed.append(self._current_stage.name)
            return updated_state
            
        except Exception as e:
            logger.error(f"Stage {self._current_stage.name} failed: {e}")
            self._pipeline_state.errors.append(f"{self._current_stage.name}: {str(e)}")
            self._current_stage = Stage.ERROR
            raise
    
    def _transition(self) -> None:
        """Transition to the next stage."""
        next_stage = self.TRANSITIONS.get(self._current_stage)
        
        if next_stage is None:
            logger.warning(f"No transition defined from {self._current_stage.name}")
            return
        
        logger.debug(f"Transitioning: {self._current_stage.name} → {next_stage.name}")
        self._current_stage = next_stage
        self._pipeline_state.current = next_stage.name
    
    async def run(self, initial_state: ResumeState) -> ResumeState:
        """Execute the full pipeline from IDLE to COMPLETE.
        
        Args:
            initial_state: ResumeState with job_description_text populated
            
        Returns:
            Final ResumeState with selected_projects and generated content
            
        Raises:
            ValueError: If preconditions fail
            Exception: If any stage handler fails
        """
        state = initial_state
        self._pipeline_state = PipelineStage(current="IDLE")
        
        logger.info("Starting Resume Orchestrator pipeline")
        
        while self._current_stage not in (Stage.COMPLETE, Stage.ERROR):
            self._transition()
            state = await self._execute_stage(state)
        
        if self._current_stage == Stage.ERROR:
            logger.error(f"Pipeline failed. Errors: {self._pipeline_state.errors}")
        else:
            logger.info("Pipeline completed successfully")
        
        return state
    
    async def run_stage(
        self, 
        stage: Stage, 
        state: ResumeState
    ) -> ResumeState:
        """Execute a single stage (for testing or manual control).
        
        Args:
            stage: The stage to execute
            state: Current ResumeState
            
        Returns:
            Updated ResumeState
        """
        self._current_stage = stage
        return await self._execute_stage(state)
