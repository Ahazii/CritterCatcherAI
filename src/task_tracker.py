"""
Task Tracker for Async Background Operations
Manages progress tracking for long-running tasks like face extraction, video confirmation, etc.
"""

import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskProgress:
    """Progress information for a background task"""
    task_id: str
    status: TaskStatus
    current: int = 0
    total: int = 0
    message: str = ""
    details: str = ""
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage"""
        if self.total == 0:
            return 0.0
        return (self.current / self.total) * 100.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API responses"""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "current": self.current,
            "total": self.total,
            "progress_percentage": round(self.progress_percentage, 2),
            "message": self.message,
            "details": self.details,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


class TaskTracker:
    """Global task tracker for managing background operations"""
    
    def __init__(self, max_completed_tasks: int = 100):
        """
        Initialize task tracker
        
        Args:
            max_completed_tasks: Maximum number of completed tasks to keep in memory
        """
        self.tasks: Dict[str, TaskProgress] = {}
        self.max_completed_tasks = max_completed_tasks
        logger.info(f"TaskTracker initialized (max_completed_tasks={max_completed_tasks})")
    
    def create_task(self, total: int = 0, message: str = "Starting task...") -> str:
        """
        Create a new task and return task ID
        
        Args:
            total: Total number of items to process
            message: Initial status message
            
        Returns:
            task_id: Unique task identifier
        """
        task_id = str(uuid.uuid4())
        task = TaskProgress(
            task_id=task_id,
            status=TaskStatus.PENDING,
            total=total,
            message=message
        )
        self.tasks[task_id] = task
        logger.info(f"Task created: {task_id} (total={total})")
        return task_id
    
    def update_task(
        self,
        task_id: str,
        status: Optional[TaskStatus] = None,
        current: Optional[int] = None,
        total: Optional[int] = None,
        message: Optional[str] = None,
        details: Optional[str] = None,
        error: Optional[str] = None
    ) -> bool:
        """
        Update task progress
        
        Args:
            task_id: Task identifier
            status: New status
            current: Current progress
            total: Total items
            message: Status message
            details: Detailed status message
            error: Error message if failed
            
        Returns:
            True if task was updated, False if task not found
        """
        if task_id not in self.tasks:
            logger.warning(f"Task not found: {task_id}")
            return False
        
        task = self.tasks[task_id]
        
        if status is not None:
            task.status = status
        if current is not None:
            task.current = current
        if total is not None:
            task.total = total
        if message is not None:
            task.message = message
        if details is not None:
            task.details = details
        if error is not None:
            task.error = error
        
        task.updated_at = datetime.now()
        
        # Mark completion time if completed or failed
        if status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            task.completed_at = datetime.now()
            logger.info(f"Task {status.value}: {task_id} ({task.message})")
        
        return True
    
    def start_task(self, task_id: str, message: str = "Processing...") -> bool:
        """
        Mark task as running
        
        Args:
            task_id: Task identifier
            message: Status message
            
        Returns:
            True if task was started, False if task not found
        """
        return self.update_task(task_id, status=TaskStatus.RUNNING, message=message)
    
    def complete_task(self, task_id: str, message: str = "Completed successfully") -> bool:
        """
        Mark task as completed
        
        Args:
            task_id: Task identifier
            message: Completion message
            
        Returns:
            True if task was completed, False if task not found
        """
        success = self.update_task(task_id, status=TaskStatus.COMPLETED, message=message)
        if success:
            self._cleanup_old_tasks()
        return success
    
    def fail_task(self, task_id: str, error: str) -> bool:
        """
        Mark task as failed
        
        Args:
            task_id: Task identifier
            error: Error message
            
        Returns:
            True if task was marked failed, False if task not found
        """
        success = self.update_task(
            task_id,
            status=TaskStatus.FAILED,
            message="Task failed",
            error=error
        )
        if success:
            self._cleanup_old_tasks()
        return success
    
    def get_task(self, task_id: str) -> Optional[TaskProgress]:
        """
        Get task progress
        
        Args:
            task_id: Task identifier
            
        Returns:
            TaskProgress object or None if not found
        """
        return self.tasks.get(task_id)
    
    def get_active_tasks(self) -> Dict[str, TaskProgress]:
        """Get all active (pending or running) tasks"""
        return {
            task_id: task
            for task_id, task in self.tasks.items()
            if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]
        }
    
    def _cleanup_old_tasks(self):
        """Remove old completed tasks to prevent memory growth"""
        completed_tasks = [
            (task_id, task)
            for task_id, task in self.tasks.items()
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
        ]
        
        if len(completed_tasks) > self.max_completed_tasks:
            # Sort by completion time and remove oldest
            completed_tasks.sort(key=lambda x: x[1].completed_at or datetime.min)
            tasks_to_remove = len(completed_tasks) - self.max_completed_tasks
            
            for task_id, _ in completed_tasks[:tasks_to_remove]:
                del self.tasks[task_id]
                logger.debug(f"Cleaned up old task: {task_id}")


# Global task tracker instance
task_tracker = TaskTracker(max_completed_tasks=100)
