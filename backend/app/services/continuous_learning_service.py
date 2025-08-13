import json
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from app.models.database import (
    TrainingDataset, ModelTrainingEvent, ModelVersion, DataQualityMetrics,
    BehavioralEvent, RecommendationFeedback, Transaction, UserSegment
)
from app.models.ml_models import (
    TrainingDatasetInfo, ModelTrainingEventInfo, ModelVersionInfo, DataQualityMetricsInfo
)
import uuid

logger = logging.getLogger(__name__)

class ContinuousLearningService:
    """Service for managing continuous learning operations"""
    
    def __init__(self):
        self.model_types = ['segmentation', 'recommendation', 'spending', 'goal']
        self.thresholds = {
            'segmentation': 100,  # 100 for demo (was 10k)
            'recommendation': 100,  # 100 for demo (was 10k)
            'spending': 100,  # 100 for demo (was 10k)
            'goal': 100  # 100 for demo (was 10k)
        }
        self.quality_assessor = DataQualityAssessor()
        
    def initialize_datasets(self, db: Session) -> None:
        """Initialize training datasets if they don't exist"""
        for model_type in self.model_types:
            existing = db.query(TrainingDataset).filter(
                TrainingDataset.dataset_type == model_type
            ).first()
            
            if not existing:
                new_dataset = TrainingDataset(
                    id=uuid.uuid4(),
                    dataset_type=model_type,
                    data_count=0,
                    threshold=self.thresholds[model_type],
                    is_ready_for_training=False,
                    threshold_reached=False,
                    data_quality_score=0.0,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                db.add(new_dataset)
                
        db.commit()
        logger.info("Training datasets initialized")
    
    def process_event_for_training(self, event_type: str, event_data: Dict, db: Session) -> None:
        """Process an event for potential inclusion in training datasets"""
        # Map event types to model types
        event_model_mapping = {
            'recommendation_interaction': 'recommendation',
            'transaction_creation': 'spending',
            'goal_modification': 'goal',
            'page_view': None  # Not directly used for training
        }
        
        # Get model type for this event
        model_type = event_model_mapping.get(event_type)
        if not model_type:
            return  # Event not relevant for training
            
        # Update training dataset
        self.update_training_dataset(model_type, 1, db)
        
        # For certain events, we might want to update multiple datasets
        if event_type == 'recommendation_interaction' and event_data.get('interaction_type') in ['helpful', 'not_helpful']:
            # Feedback events are also relevant for segmentation
            self.update_training_dataset('segmentation', 1, db)
    
    def update_training_dataset(self, model_type: str, count: int, db: Session) -> None:
        """Update training dataset status based on new data"""
        dataset = db.query(TrainingDataset).filter(
            TrainingDataset.dataset_type == model_type
        ).first()
        
        if not dataset:
            logger.warning(f"No training dataset found for model type: {model_type}")
            return
            
        # Update data count
        dataset.data_count += count
        dataset.updated_at = datetime.utcnow()
        
        # Check if threshold reached
        if dataset.data_count >= dataset.threshold and not dataset.threshold_reached:
            dataset.threshold_reached = True
            logger.info(f"Threshold reached for {model_type} dataset: {dataset.data_count}/{dataset.threshold}")
            
            # Assess data quality
            quality_metrics = self.assess_data_quality(dataset.id, db)
            
            # Update data quality score
            dataset.data_quality_score = quality_metrics.get('overall_score', 0.0)
            
            # Check if ready for training (threshold reached and quality acceptable)
            if dataset.data_quality_score >= 0.7:  # Minimum quality threshold
                dataset.is_ready_for_training = True
                logger.info(f"{model_type} dataset ready for training with quality score: {dataset.data_quality_score}")
            else:
                logger.warning(f"{model_type} dataset quality below threshold: {dataset.data_quality_score}")
        
        db.commit()
    
    def assess_data_quality(self, dataset_id: uuid.UUID, db: Session) -> Dict:
        """Assess data quality for a training dataset"""
        dataset = db.query(TrainingDataset).filter(TrainingDataset.id == dataset_id).first()
        
        if not dataset:
            logger.error(f"Dataset not found: {dataset_id}")
            return {'overall_score': 0.0}
            
        # Get data for quality assessment based on model type
        data = self._get_model_data(dataset.dataset_type, db)
        
        # Calculate quality metrics
        completeness = self.quality_assessor.calculate_completeness(data)
        consistency = self.quality_assessor.calculate_consistency(data)
        validity = self.quality_assessor.calculate_validity(data)
        anomaly_count = self.quality_assessor.detect_anomalies(data)
        duplicate_count = self.quality_assessor.count_duplicates(data)
        missing_values_count = self.quality_assessor.count_missing_values(data)
        
        # Calculate overall score (weighted average)
        overall_score = (completeness * 0.4 + consistency * 0.3 + validity * 0.3)
        
        # Store quality metrics
        quality_metrics = DataQualityMetrics(
            id=uuid.uuid4(),
            training_dataset_id=dataset_id,
            completeness_score=completeness,
            consistency_score=consistency,
            validity_score=validity,
            anomaly_count=anomaly_count,
            duplicate_count=duplicate_count,
            missing_values_count=missing_values_count,
            assessment_date=datetime.utcnow(),
            created_at=datetime.utcnow()
        )
        
        db.add(quality_metrics)
        db.commit()
        
        return {
            'completeness': completeness,
            'consistency': consistency,
            'validity': validity,
            'anomaly_count': anomaly_count,
            'duplicate_count': duplicate_count,
            'missing_values_count': missing_values_count,
            'overall_score': overall_score
        }
    
    def _get_model_data(self, model_type: str, db: Session) -> List[Dict]:
        """Get data for a specific model type for quality assessment"""
        if model_type == 'recommendation':
            # Get recommendation feedback data
            feedback = db.query(RecommendationFeedback).order_by(
                RecommendationFeedback.created_at.desc()
            ).limit(1000).all()
            
            return [
                {
                    'recommendation_id': str(f.recommendation_id),
                    'user_id': str(f.user_id),
                    'feedback_type': f.feedback_type,
                    'feedback_text': f.feedback_text,
                    'created_at': f.created_at.isoformat()
                }
                for f in feedback
            ]
            
        elif model_type == 'spending':
            # Get transaction data
            transactions = db.query(Transaction).order_by(
                Transaction.transaction_date.desc()
            ).limit(1000).all()
            
            return [
                {
                    'transaction_id': str(t.id),
                    'user_id': str(t.user_id),
                    'amount': float(t.amount),
                    'category': t.category,
                    'transaction_type': t.transaction_type,
                    'transaction_date': t.transaction_date.isoformat()
                }
                for t in transactions
            ]
            
        elif model_type == 'segmentation':
            # Get user segment data
            segments = db.query(UserSegment).all()
            
            return [
                {
                    'segment_id': s.segment_id,
                    'user_id': str(s.user_id),
                    'segment_name': s.segment_name,
                    'confidence': float(s.confidence),
                    'updated_at': s.updated_at.isoformat()
                }
                for s in segments
            ]
            
        elif model_type == 'goal':
            # Get goal-related events
            goal_events = db.query(BehavioralEvent).filter(
                BehavioralEvent.event_type == 'goal_modification'
            ).order_by(BehavioralEvent.timestamp.desc()).limit(1000).all()
            
            return [
                {
                    'event_id': str(e.id),
                    'user_id': str(e.user_id),
                    'event_data': json.loads(e.event_data) if e.event_data else {},
                    'timestamp': e.timestamp.isoformat()
                }
                for e in goal_events
            ]
            
        return []
    
    def check_training_readiness(self, db: Session) -> List[TrainingDatasetInfo]:
        """Check if any datasets are ready for training"""
        ready_datasets = db.query(TrainingDataset).filter(
            TrainingDataset.is_ready_for_training == True
        ).all()
        
        return [
            TrainingDatasetInfo(
                id=str(d.id),
                dataset_type=d.dataset_type,
                data_count=d.data_count,
                threshold=d.threshold,
                last_training_date=d.last_training_date,
                is_ready_for_training=d.is_ready_for_training,
                threshold_reached=d.threshold_reached,
                data_quality_score=float(d.data_quality_score) if d.data_quality_score else None,
                created_at=d.created_at,
                updated_at=d.updated_at
            )
            for d in ready_datasets
        ]
    
    def prepare_training_data(self, dataset_id: uuid.UUID, db: Session) -> Dict:
        """Prepare data for model training"""
        dataset = db.query(TrainingDataset).filter(TrainingDataset.id == dataset_id).first()
        
        if not dataset:
            logger.error(f"Dataset not found: {dataset_id}")
            return {'success': False, 'error': 'Dataset not found'}
            
        # Get raw data
        raw_data = self._get_model_data(dataset.dataset_type, db)
        
        # Prepare data based on model type
        if dataset.dataset_type == 'recommendation':
            prepared_data = self._prepare_recommendation_data(raw_data)
        elif dataset.dataset_type == 'spending':
            prepared_data = self._prepare_spending_data(raw_data)
        elif dataset.dataset_type == 'segmentation':
            prepared_data = self._prepare_segmentation_data(raw_data)
        elif dataset.dataset_type == 'goal':
            prepared_data = self._prepare_goal_data(raw_data)
        else:
            return {'success': False, 'error': 'Unknown model type'}
            
        return {
            'success': True,
            'dataset_type': dataset.dataset_type,
            'data_count': len(raw_data),
            'prepared_data': prepared_data
        }
    
    def _prepare_recommendation_data(self, raw_data: List[Dict]) -> Dict:
        """Prepare recommendation feedback data for training"""
        # Extract state-action-reward tuples for CQL training
        training_data = {
            'feedback_counts': {},
            'state_action_rewards': []
        }
        
        for item in raw_data:
            feedback_type = item.get('feedback_type')
            if feedback_type:
                training_data['feedback_counts'][feedback_type] = training_data['feedback_counts'].get(feedback_type, 0) + 1
        
        return training_data
    
    def _prepare_spending_data(self, raw_data: List[Dict]) -> Dict:
        """Prepare transaction data for spending prediction training"""
        # Group transactions by user for time series data
        user_transactions = {}
        
        for item in raw_data:
            user_id = item.get('user_id')
            if user_id:
                if user_id not in user_transactions:
                    user_transactions[user_id] = []
                user_transactions[user_id].append(item)
        
        return {
            'user_count': len(user_transactions),
            'transaction_count': len(raw_data),
            'user_transactions': user_transactions
        }
    
    def _prepare_segmentation_data(self, raw_data: List[Dict]) -> Dict:
        """Prepare user segment data for segmentation model training"""
        # Extract features for clustering
        segment_distribution = {}
        
        for item in raw_data:
            segment_name = item.get('segment_name')
            if segment_name:
                segment_distribution[segment_name] = segment_distribution.get(segment_name, 0) + 1
        
        return {
            'user_count': len(raw_data),
            'segment_distribution': segment_distribution
        }
    
    def _prepare_goal_data(self, raw_data: List[Dict]) -> Dict:
        """Prepare goal event data for goal achievement prediction training"""
        # Extract goal features and outcomes
        interaction_types = {}
        
        for item in raw_data:
            event_data = item.get('event_data', {})
            interaction_type = event_data.get('interaction_type')
            if interaction_type:
                interaction_types[interaction_type] = interaction_types.get(interaction_type, 0) + 1
        
        return {
            'event_count': len(raw_data),
            'interaction_types': interaction_types
        }
    
    def get_training_datasets(self, db: Session) -> List[TrainingDatasetInfo]:
        """Get all training datasets"""
        datasets = db.query(TrainingDataset).all()
        
        return [
            TrainingDatasetInfo(
                id=str(d.id),
                dataset_type=d.dataset_type,
                data_count=d.data_count,
                threshold=d.threshold,
                last_training_date=d.last_training_date,
                is_ready_for_training=d.is_ready_for_training,
                threshold_reached=d.threshold_reached,
                data_quality_score=float(d.data_quality_score) if d.data_quality_score else None,
                created_at=d.created_at,
                updated_at=d.updated_at
            )
            for d in datasets
        ]
    
    def get_data_quality_metrics(self, dataset_id: uuid.UUID, db: Session) -> List[DataQualityMetricsInfo]:
        """Get data quality metrics for a dataset"""
        metrics = db.query(DataQualityMetrics).filter(
            DataQualityMetrics.training_dataset_id == dataset_id
        ).order_by(DataQualityMetrics.assessment_date.desc()).all()
        
        return [
            DataQualityMetricsInfo(
                id=str(m.id),
                training_dataset_id=str(m.training_dataset_id),
                completeness_score=float(m.completeness_score) if m.completeness_score else None,
                consistency_score=float(m.consistency_score) if m.consistency_score else None,
                validity_score=float(m.validity_score) if m.validity_score else None,
                anomaly_count=m.anomaly_count,
                duplicate_count=m.duplicate_count,
                missing_values_count=m.missing_values_count,
                assessment_date=m.assessment_date,
                created_at=m.created_at
            )
            for m in metrics
        ]
    
    def reset_dataset_counters(self, dataset_id: uuid.UUID, db: Session) -> bool:
        """Reset dataset counters after training"""
        dataset = db.query(TrainingDataset).filter(TrainingDataset.id == dataset_id).first()
        
        if not dataset:
            logger.error(f"Dataset not found: {dataset_id}")
            return False
            
        dataset.data_count = 0
        dataset.is_ready_for_training = False
        dataset.threshold_reached = False
        dataset.last_training_date = datetime.utcnow()
        dataset.updated_at = datetime.utcnow()
        
        db.commit()
        logger.info(f"Reset counters for dataset {dataset.dataset_type}")
        
        return True

    def get_continuous_learning_status(self, db: Session) -> Dict:
        """Get overall continuous learning system status"""
        try:
            datasets = self.get_training_datasets(db)
            ready_datasets = [d for d in datasets if d.is_ready_for_training]
            
            return {
                'status': 'healthy',
                'total_datasets': len(datasets),
                'ready_for_training': len(ready_datasets),
                'datasets': [
                    {
                        'type': d.dataset_type,
                        'data_count': d.data_count,
                        'threshold': d.threshold,
                        'ready': d.is_ready_for_training,
                        'quality_score': d.data_quality_score
                    }
                    for d in datasets
                ],
                'last_updated': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting continuous learning status: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'last_updated': datetime.utcnow().isoformat()
            }


class DataQualityAssessor:
    """Assess data quality for training datasets"""
    
    def calculate_completeness(self, data: List[Dict]) -> float:
        """Calculate data completeness percentage"""
        if not data:
            return 0.0
            
        total_fields = 0
        populated_fields = 0
        
        for item in data:
            for key, value in item.items():
                total_fields += 1
                if value is not None and value != "":
                    populated_fields += 1
        
        return populated_fields / max(total_fields, 1)
    
    def calculate_consistency(self, data: List[Dict]) -> float:
        """Calculate data consistency score"""
        if not data or len(data) < 2:
            return 1.0  # Single item is consistent with itself
            
        # Check for consistent data types across records
        field_types = {}
        inconsistencies = 0
        
        for item in data:
            for key, value in item.items():
                if key not in field_types:
                    field_types[key] = type(value)
                elif value is not None and not isinstance(value, field_types[key]):
                    inconsistencies += 1
        
        total_checks = len(field_types) * len(data)
        return 1.0 - (inconsistencies / max(total_checks, 1))
    
    def calculate_validity(self, data: List[Dict]) -> float:
        """Calculate data validity score"""
        if not data:
            return 0.0
            
        # Simple validity checks
        valid_fields = 0
        total_fields = 0
        
        for item in data:
            for key, value in item.items():
                total_fields += 1
                
                # Basic validity checks
                if value is None:
                    continue
                    
                if isinstance(value, (int, float)):
                    # Numeric fields should be reasonable
                    if -1e9 <= value <= 1e9:  # Reasonable range
                        valid_fields += 1
                elif isinstance(value, str):
                    # String fields should not be too long or too short
                    if 0 < len(value) < 10000:
                        valid_fields += 1
                else:
                    # Other types are considered valid
                    valid_fields += 1
        
        return valid_fields / max(total_fields, 1)
    
    def detect_anomalies(self, data: List[Dict]) -> int:
        """Detect anomalies in the dataset"""
        if not data:
            return 0
            
        # Simple anomaly detection for numeric fields
        anomalies = 0
        numeric_fields = {}
        
        # First pass: collect numeric fields and calculate mean/std
        for item in data:
            for key, value in item.items():
                if isinstance(value, (int, float)):
                    if key not in numeric_fields:
                        numeric_fields[key] = []
                    numeric_fields[key].append(value)
        
        # Calculate stats
        field_stats = {}
        for key, values in numeric_fields.items():
            if len(values) > 1:
                mean = sum(values) / len(values)
                std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
                field_stats[key] = {'mean': mean, 'std': std}
        
        # Second pass: detect anomalies (values more than 3 std from mean)
        for item in data:
            for key, value in item.items():
                if key in field_stats and isinstance(value, (int, float)):
                    stats = field_stats[key]
                    if stats['std'] > 0 and abs(value - stats['mean']) > 3 * stats['std']:
                        anomalies += 1
        
        return anomalies
    
    def count_duplicates(self, data: List[Dict]) -> int:
        """Count duplicate records"""
        if not data:
            return 0
            
        # Convert dicts to frozensets of items for hashability
        seen = set()
        duplicates = 0
        
        for item in data:
            # Create a hashable representation
            item_hash = frozenset(item.items())
            if item_hash in seen:
                duplicates += 1
            else:
                seen.add(item_hash)
        
        return duplicates
    
    def count_missing_values(self, data: List[Dict]) -> int:
        """Count missing values"""
        if not data:
            return 0
            
        missing = 0
        
        for item in data:
            for value in item.values():
                if value is None or (isinstance(value, str) and value.strip() == ""):
                    missing += 1
        
        return missing
