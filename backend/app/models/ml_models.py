import torch
import torch.nn as nn
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime

class QNetwork(nn.Module):
    """Q-Network architecture matching the training notebook"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.network(state)

class AIRecommendation(BaseModel):
    action_id: int
    action_type: str
    recommendation: str
    confidence: float
    state_summary: Dict
    generated_at: datetime

class FinancialState(BaseModel):
    current_balance: float
    weekly_spending: float
    weekly_income: float
    transaction_count: int
    savings_rate: float
    daily_spending_avg: float
    category_spending: Dict[str, float]

class CoachingAction(BaseModel):
    action_id: int
    action_name: str
    description: str
    priority: str  # low, medium, high
    category: str  # spending, saving, budgeting, general

class ModelMetrics(BaseModel):
    version: str
    last_updated: datetime
    inference_count: int
    average_confidence: float
    size_mb: float
    inference_time_ms: float

class RecommendationHistory(BaseModel):
    user_id: str
    recommendations: List[AIRecommendation]
    total_count: int
    date_range: Dict[str, datetime]

# Enhanced ML Model Data Structures for Phase 1 Integration

class UserSegmentInfo(BaseModel):
    """User behavioral segment information"""
    segment_id: int
    segment_name: str
    confidence: float
    characteristics: Dict[str, float]
    peer_comparison: Optional[Dict[str, float]] = None

class EnhancedFinancialState(FinancialState):
    """Extended financial state with behavioral features"""
    behavioral_metrics: Dict[str, float]
    segment_info: Optional[UserSegmentInfo] = None
    temporal_patterns: Optional[Dict[str, List[float]]] = None
    financial_discipline: Optional[float] = None

class SpendingPrediction(BaseModel):
    """Weekly spending forecasts"""
    user_id: str
    prediction_date: datetime
    total_predicted: float
    category_predictions: Dict[str, float]
    confidence_intervals: Dict[str, List[float]]
    previous_accuracy: Optional[float] = None

class GoalAchievementProbability(BaseModel):
    """Goal completion likelihood"""
    goal_id: str
    probability: float
    expected_completion_date: Optional[datetime] = None
    risk_factors: List[Dict[str, str]]
    improvement_suggestions: List[str]

class BehavioralInsight(BaseModel):
    """User behavior analysis results"""
    insight_type: str
    title: str
    description: str
    priority: str  # low, medium, high
    related_metrics: Dict[str, float]
    generated_at: datetime

class RecommendationStrategy(BaseModel):
    """Multi-armed bandit strategy selection"""
    strategy_id: int
    strategy_name: str
    parameters: Dict[str, float]
    performance_metrics: Dict[str, float]
    selection_count: int
    last_updated: datetime

class BehavioralEventData(BaseModel):
    """Structured data for behavioral events"""
    event_type: str
    page_url: Optional[str] = None
    recommendation_id: Optional[str] = None
    product_id: Optional[str] = None
    goal_id: Optional[str] = None
    interaction_type: Optional[str] = None  # click, dismiss, implement
    session_id: Optional[str] = None
    additional_data: Optional[Dict] = None

class PredictionCacheEntry(BaseModel):
    """Cached prediction entry"""
    user_id: str
    prediction_type: str
    prediction_data: Dict
    created_at: datetime
    expires_at: datetime
    cache_hit: bool = False

class ModelPerformanceMetrics(BaseModel):
    """Model performance tracking"""
    name: str
    accuracy_metric: float
    sample_count: int
    feature_importance: Dict[str, float]
    created_at: datetime
    version: Optional[str] = None

class EnhancedAIRecommendation(AIRecommendation):
    """Enhanced recommendation with segment and strategy info"""
    segment_context: Optional[UserSegmentInfo] = None
    strategy_used: Optional[RecommendationStrategy] = None
    behavioral_triggers: Optional[List[str]] = None
    personalization_score: Optional[float] = None

# Continuous Learning Module Data Structures for Phase 1

class TrainingDatasetInfo(BaseModel):
    """Training dataset information"""
    id: Optional[str] = None
    dataset_type: str  # 'segmentation', 'recommendation', 'spending', 'goal'
    data_count: int = 0
    threshold: int
    last_training_date: Optional[datetime] = None
    is_ready_for_training: bool = False
    threshold_reached: bool = False
    data_quality_score: Optional[float] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class ModelTrainingEventInfo(BaseModel):
    """Model training event information"""
    id: Optional[str] = None
    model_type: str
    training_dataset_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = 'pending'  # 'pending', 'in_progress', 'completed', 'failed', 'deployed'
    performance_metrics: Optional[Dict] = None
    model_path: Optional[str] = None
    previous_model_path: Optional[str] = None
    training_data_size: int = 0
    validation_score: Optional[float] = None
    deployment_approved: bool = False
    created_at: Optional[datetime] = None

class ModelVersionInfo(BaseModel):
    """Model version information"""
    id: Optional[str] = None
    model_type: str
    version: str  # Semantic version (e.g., "2.1.3")
    model_path: str
    training_event_id: str
    is_active: bool = False
    performance_baseline: Optional[float] = None
    performance_current: Optional[float] = None
    deployment_date: Optional[datetime] = None
    rollback_count: int = 0
    model_metadata: Optional[Dict] = None
    created_at: Optional[datetime] = None

class DataQualityMetricsInfo(BaseModel):
    """Data quality metrics information"""
    id: Optional[str] = None
    training_dataset_id: str
    completeness_score: Optional[float] = None
    consistency_score: Optional[float] = None
    validity_score: Optional[float] = None
    anomaly_count: int = 0
    duplicate_count: int = 0
    missing_values_count: int = 0
    assessment_date: Optional[datetime] = None
    created_at: Optional[datetime] = None

class ContinuousLearningStatus(BaseModel):
    """Overall continuous learning system status"""
    datasets: List[TrainingDatasetInfo]
    active_training_events: List[ModelTrainingEventInfo]
    model_versions: List[ModelVersionInfo]
    system_health: Dict[str, str]
    last_updated: datetime

class TrainingTriggerInfo(BaseModel):
    """Information about training trigger conditions"""
    dataset_type: str
    current_data_count: int
    threshold: int
    threshold_reached: bool
    data_quality_score: float
    ready_for_training: bool
    estimated_training_time: Optional[str] = None
    last_training_date: Optional[datetime] = None
