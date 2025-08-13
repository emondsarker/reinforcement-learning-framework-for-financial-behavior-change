import logging
import json
import os
import shutil
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.models.database import (
    TrainingDataset, ModelTrainingEvent, ModelVersion, DataQualityMetrics
)
from app.models.ml_models import ModelTrainingEventInfo, ModelVersionInfo
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)

class TrainingJob:
    """Represents a training job"""
    
    def __init__(self, dataset_id: uuid.UUID, model_type: str, priority: int = 1):
        self.id = uuid.uuid4()
        self.dataset_id = dataset_id
        self.model_type = model_type
        self.priority = priority
        self.status = 'queued'
        self.created_at = datetime.utcnow()
        self.started_at = None
        self.completed_at = None
        self.error_message = None
        self.training_event_id = None

class TrainingOrchestrator:
    """Orchestrates model training jobs"""
    
    def __init__(self):
        self.job_queue = []
        self.active_job = None
        self.job_history = []
        self.executor = ThreadPoolExecutor(max_workers=1)  # Single worker for simplicity
        self.lock = threading.Lock()
        
    def queue_training_job(self, dataset_id: uuid.UUID, model_type: str, priority: int = 1) -> TrainingJob:
        """Queue a new training job"""
        with self.lock:
            # Check if there's already a job for this model type
            existing_job = next(
                (job for job in self.job_queue if job.model_type == model_type and job.status == 'queued'),
                None
            )
            
            if existing_job:
                logger.info(f"Training job for {model_type} already queued")
                return existing_job
            
            # Check if there's an active job for this model type
            if self.active_job and self.active_job.model_type == model_type:
                logger.info(f"Training job for {model_type} already active")
                return self.active_job
            
            job = TrainingJob(dataset_id, model_type, priority)
            self.job_queue.append(job)
            
            # Sort queue by priority (higher priority first)
            self.job_queue.sort(key=lambda x: x.priority, reverse=True)
            
            logger.info(f"Queued training job for {model_type} with priority {priority}")
            return job
    
    def get_next_job(self) -> Optional[TrainingJob]:
        """Get the next job from the queue"""
        with self.lock:
            if not self.job_queue or self.active_job is not None:
                return None
            
            return self.job_queue.pop(0)
    
    def start_training_job(self, job: TrainingJob, db: Session) -> bool:
        """Start a training job"""
        try:
            with self.lock:
                if self.active_job is not None:
                    logger.warning("Cannot start job - another job is already active")
                    return False
                
                self.active_job = job
                job.status = 'in_progress'
                job.started_at = datetime.utcnow()
            
            # Create training event record
            training_event = ModelTrainingEvent(
                id=uuid.uuid4(),
                model_type=job.model_type,
                training_dataset_id=job.dataset_id,
                start_time=job.started_at,
                status='in_progress',
                training_data_size=0,
                created_at=datetime.utcnow()
            )
            
            db.add(training_event)
            db.commit()
            
            job.training_event_id = training_event.id
            
            logger.info(f"Started training job {job.id} for {job.model_type}")
            
            # Submit job to executor
            future = self.executor.submit(self._execute_training_job, job, db)
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting training job: {e}")
            with self.lock:
                if self.active_job and self.active_job.id == job.id:
                    self.active_job = None
                job.status = 'failed'
                job.error_message = str(e)
            return False
    
    def _execute_training_job(self, job: TrainingJob, db: Session):
        """Execute the actual training job (mock implementation)"""
        try:
            from app.services.model_trainers import ModelTrainerFactory
            
            logger.info(f"Starting MOCK training job {job.id} for {job.model_type}")
            
            # Get the appropriate trainer
            trainer = ModelTrainerFactory.get_trainer(job.model_type)
            
            if not trainer:
                raise ValueError(f"No trainer available for model type: {job.model_type}")
            
            # Get training dataset
            dataset = db.query(TrainingDataset).filter(TrainingDataset.id == job.dataset_id).first()
            if not dataset:
                raise ValueError(f"Training dataset not found: {job.dataset_id}")
            
            # Prepare mock training data (simplified for demo)
            mock_prepared_data = {
                'success': True,
                'dataset_type': job.model_type,
                'data_count': dataset.data_count,
                'prepared_data': {
                    'feedback_counts': {'helpful': 50, 'not_helpful': 20} if job.model_type == 'recommendation' else {},
                    'user_count': 100 if job.model_type == 'segmentation' else 0,
                    'transaction_count': 500 if job.model_type == 'spending' else 0,
                    'event_count': 75 if job.model_type == 'goal' else 0
                }
            }
            
            # Execute mock training
            training_result = trainer.train(
                training_data=mock_prepared_data['prepared_data'],
                model_type=job.model_type
            )
            
            # Update training event
            training_event = db.query(ModelTrainingEvent).filter(
                ModelTrainingEvent.id == job.training_event_id
            ).first()
            
            if training_event:
                training_event.end_time = datetime.utcnow()
                training_event.status = 'completed' if training_result['success'] else 'failed'
                training_event.performance_metrics = json.dumps(training_result.get('metrics', {}))
                training_event.model_path = training_result.get('model_path')
                training_event.validation_score = training_result.get('validation_score', 0.0)
                training_event.training_data_size = mock_prepared_data.get('data_count', 0)
                
                # Create model version if training was successful
                if training_result['success']:
                    model_version = self._create_model_version(
                        job.model_type, 
                        training_result, 
                        training_event.id, 
                        db
                    )
                    
                    # Reset dataset counters (for mock, just reduce by half)
                    dataset.data_count = max(0, dataset.data_count // 2)
                    dataset.is_ready_for_training = False
                    dataset.threshold_reached = False
                    dataset.last_training_date = datetime.utcnow()
                    dataset.updated_at = datetime.utcnow()
                
                db.commit()
            
            # Update job status
            with self.lock:
                job.status = 'completed' if training_result['success'] else 'failed'
                job.completed_at = datetime.utcnow()
                job.error_message = training_result.get('error') if not training_result['success'] else None
                
                # Move to history and clear active job
                self.job_history.append(job)
                self.active_job = None
            
            logger.info(f"MOCK training job {job.id} completed with status: {job.status}")
            
            # Try to start next job
            self._process_queue(db)
            
        except Exception as e:
            logger.error(f"Error executing mock training job {job.id}: {e}")
            
            # Update job status
            with self.lock:
                job.status = 'failed'
                job.completed_at = datetime.utcnow()
                job.error_message = str(e)
                
                # Move to history and clear active job
                self.job_history.append(job)
                self.active_job = None
            
            # Update training event
            try:
                training_event = db.query(ModelTrainingEvent).filter(
                    ModelTrainingEvent.id == job.training_event_id
                ).first()
                
                if training_event:
                    training_event.end_time = datetime.utcnow()
                    training_event.status = 'failed'
                    training_event.performance_metrics = json.dumps({'error': str(e)})
                    db.commit()
            except Exception as db_error:
                logger.error(f"Error updating training event: {db_error}")
    
    def _create_model_version(self, model_type: str, training_result: Dict, training_event_id: uuid.UUID, db: Session) -> ModelVersion:
        """Create a new model version record"""
        try:
            # Generate version number (simple incrementing)
            latest_version = db.query(ModelVersion).filter(
                ModelVersion.model_type == model_type
            ).order_by(ModelVersion.created_at.desc()).first()
            
            if latest_version:
                # Parse version and increment
                try:
                    major, minor, patch = map(int, latest_version.version.split('.'))
                    new_version = f"{major}.{minor}.{patch + 1}"
                except:
                    new_version = "1.0.1"
            else:
                new_version = "1.0.0"
            
            model_version = ModelVersion(
                id=uuid.uuid4(),
                model_type=model_type,
                version=new_version,
                model_path=training_result.get('model_path', ''),
                training_event_id=training_event_id,
                is_active=False,  # Manual deployment for now
                performance_baseline=training_result.get('validation_score', 0.0),
                performance_current=training_result.get('validation_score', 0.0),
                model_metadata=json.dumps(training_result.get('metadata', {})),
                created_at=datetime.utcnow()
            )
            
            db.add(model_version)
            db.commit()
            
            logger.info(f"Created model version {new_version} for {model_type}")
            return model_version
            
        except Exception as e:
            logger.error(f"Error creating model version: {e}")
            raise
    
    def _process_queue(self, db: Session):
        """Process the job queue"""
        next_job = self.get_next_job()
        if next_job:
            self.start_training_job(next_job, db)
    
    def get_job_status(self, job_id: uuid.UUID) -> Optional[Dict]:
        """Get status of a specific job"""
        with self.lock:
            # Check active job
            if self.active_job and self.active_job.id == job_id:
                return self._job_to_dict(self.active_job)
            
            # Check queue
            for job in self.job_queue:
                if job.id == job_id:
                    return self._job_to_dict(job)
            
            # Check history
            for job in self.job_history:
                if job.id == job_id:
                    return self._job_to_dict(job)
        
        return None
    
    def get_all_jobs(self) -> Dict[str, List[Dict]]:
        """Get all jobs (active, queued, completed)"""
        with self.lock:
            return {
                'active': [self._job_to_dict(self.active_job)] if self.active_job else [],
                'queued': [self._job_to_dict(job) for job in self.job_queue],
                'completed': [self._job_to_dict(job) for job in self.job_history[-10:]]  # Last 10
            }
    
    def _job_to_dict(self, job: TrainingJob) -> Dict:
        """Convert job to dictionary"""
        return {
            'id': str(job.id),
            'dataset_id': str(job.dataset_id),
            'model_type': job.model_type,
            'priority': job.priority,
            'status': job.status,
            'created_at': job.created_at.isoformat(),
            'started_at': job.started_at.isoformat() if job.started_at else None,
            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            'error_message': job.error_message,
            'training_event_id': str(job.training_event_id) if job.training_event_id else None
        }
    
    def trigger_training_if_ready(self, db: Session) -> List[TrainingJob]:
        """Check for ready datasets and trigger training"""
        from app.services.continuous_learning_service import ContinuousLearningService
        
        cl_service = ContinuousLearningService()
        ready_datasets = cl_service.check_training_readiness(db)
        
        triggered_jobs = []
        
        for dataset in ready_datasets:
            try:
                dataset_uuid = uuid.UUID(dataset.id)
                job = self.queue_training_job(dataset_uuid, dataset.dataset_type, priority=2)
                triggered_jobs.append(job)
                
                logger.info(f"Auto-triggered training for {dataset.dataset_type}")
                
            except Exception as e:
                logger.error(f"Error auto-triggering training for {dataset.dataset_type}: {e}")
        
        # Start processing if we have jobs
        if triggered_jobs:
            self._process_queue(db)
        
        return triggered_jobs
    
    def shutdown(self):
        """Shutdown the orchestrator"""
        logger.info("Shutting down training orchestrator")
        self.executor.shutdown(wait=True)

# Global orchestrator instance
training_orchestrator = TrainingOrchestrator()
