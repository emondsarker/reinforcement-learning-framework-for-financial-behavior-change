from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import desc
from app.models.ml_models import (
    AIRecommendation, FinancialState, EnhancedAIRecommendation, 
    UserSegmentInfo, SpendingPrediction, GoalAchievementProbability,
    BehavioralInsight
)
from app.services.ml_service import (
    CQLModelService, EnhancedCQLModelService, UserSegmentationService,
    SpendingPredictionService, GoalPredictionService
)
from app.services.behavioral_event_service import BehavioralEventService
from app.middleware.auth_middleware import get_current_active_user
from app.database import get_db
from app.models.database import User, RecommendationHistory, RecommendationFeedback, ModelTrainingEvent, ModelVersion, TrainingDataset
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import json

router = APIRouter(prefix="/coaching", tags=["ai-coaching"])

# Initialize ML service (in production, this would be a singleton)
ml_service = CQLModelService()

@router.get("/recommendation", response_model=AIRecommendation)
async def get_coaching_recommendation(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get personalized AI coaching recommendation"""
    try:
        recommendation = ml_service.get_recommendation(str(current_user.id), db)
        return recommendation
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate coaching recommendation"
        )

@router.get("/financial-state", response_model=FinancialState)
async def get_financial_state(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get detailed financial state analysis"""
    try:
        financial_state = ml_service.state_generator.get_financial_state_summary(
            str(current_user.id), db
        )
        return financial_state
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze financial state"
        )

@router.get("/model-info")
async def get_model_info():
    """Get information about the AI model"""
    try:
        metrics = ml_service.get_model_metrics()
        return metrics
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model information"
        )

@router.get("/health")
async def check_model_health():
    """Check AI model health status"""
    try:
        health_status = ml_service.validate_model_health()
        return health_status
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check model health"
        )

@router.get("/actions")
async def get_available_actions():
    """Get available coaching actions and their descriptions"""
    actions = [
        {
            "id": 0,
            "type": "continue_current_behavior",
            "name": "Continue Current Behavior",
            "description": "User is maintaining good financial habits",
            "priority": "low",
            "category": "general"
        },
        {
            "id": 1,
            "type": "spending_alert",
            "name": "Spending Alert",
            "description": "User needs to reduce spending",
            "priority": "high",
            "category": "spending"
        },
        {
            "id": 2,
            "type": "budget_suggestion",
            "name": "Budget Suggestion",
            "description": "User should create or adjust budget",
            "priority": "medium",
            "category": "budgeting"
        },
        {
            "id": 3,
            "type": "savings_nudge",
            "name": "Savings Nudge",
            "description": "User has opportunity to save more",
            "priority": "medium",
            "category": "saving"
        },
        {
            "id": 4,
            "type": "positive_reinforcement",
            "name": "Positive Reinforcement",
            "description": "User is doing well financially",
            "priority": "low",
            "category": "general"
        }
    ]
    
    return {"actions": actions}

# Pydantic models for request/response
class FeedbackRequest(BaseModel):
    recommendation_id: str
    helpful: bool
    feedback_text: Optional[str] = None

class RecommendationHistoryResponse(BaseModel):
    id: str
    recommendation_text: str
    action_type: str
    confidence_score: float
    created_at: str
    feedback_status: str

class HistoryResponse(BaseModel):
    recommendations: List[RecommendationHistoryResponse]
    total: int
    page: int
    limit: int

@router.get("/history", response_model=HistoryResponse)
async def get_recommendation_history(
    limit: int = 20,
    offset: int = 0,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get paginated recommendation history for the current user"""
    try:
        # Get total count
        total = db.query(RecommendationHistory).filter(
            RecommendationHistory.user_id == current_user.id
        ).count()
        
        # Get paginated recommendations
        recommendations = db.query(RecommendationHistory).filter(
            RecommendationHistory.user_id == current_user.id
        ).order_by(desc(RecommendationHistory.created_at)).offset(offset).limit(limit).all()
        
        # Convert to response format
        recommendation_responses = []
        for rec in recommendations:
            recommendation_responses.append(RecommendationHistoryResponse(
                id=str(rec.id),
                recommendation_text=rec.recommendation_text,
                action_type=rec.action_type,
                confidence_score=float(rec.confidence_score),
                created_at=rec.created_at.isoformat(),
                feedback_status=rec.feedback_status
            ))
        
        return HistoryResponse(
            recommendations=recommendation_responses,
            total=total,
            page=(offset // limit) + 1,
            limit=limit
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve recommendation history"
        )

@router.post("/feedback")
async def submit_recommendation_feedback(
    feedback_request: FeedbackRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Submit feedback on a coaching recommendation (for future model improvement)"""
    try:
        # Map frontend feedback to backend format
        feedback_type = "helpful" if feedback_request.helpful else "not_helpful"
        
        # Check if recommendation exists
        recommendation = db.query(RecommendationHistory).filter(
            RecommendationHistory.id == feedback_request.recommendation_id,
            RecommendationHistory.user_id == current_user.id
        ).first()
        
        if not recommendation:
            # For backwards compatibility, create a temporary recommendation entry
            # This handles cases where frontend sends feedback for recommendations
            # that weren't stored in the database yet
            recommendation = RecommendationHistory(
                user_id=current_user.id,
                recommendation_text="Legacy recommendation",
                action_type="unknown",
                confidence_score=0.5,
                feedback_status=feedback_type
            )
            db.add(recommendation)
            db.flush()  # Get the ID without committing
        
        # Create or update feedback
        existing_feedback = db.query(RecommendationFeedback).filter(
            RecommendationFeedback.recommendation_id == recommendation.id,
            RecommendationFeedback.user_id == current_user.id
        ).first()
        
        if existing_feedback:
            # Update existing feedback
            existing_feedback.feedback_type = feedback_type
            existing_feedback.feedback_text = feedback_request.feedback_text
        else:
            # Create new feedback
            new_feedback = RecommendationFeedback(
                recommendation_id=recommendation.id,
                user_id=current_user.id,
                feedback_type=feedback_type,
                feedback_text=feedback_request.feedback_text
            )
            db.add(new_feedback)
        
        # Update recommendation feedback status
        recommendation.feedback_status = feedback_type
        
        db.commit()
        
        return {
            "message": "Feedback submitted successfully",
            "feedback_id": str(recommendation.id)
        }
    
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit feedback"
        )

@router.get("/insights")
async def get_financial_insights(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get additional financial insights and tips"""
    try:
        # Get current financial state
        financial_state = ml_service.state_generator.get_financial_state_summary(
            str(current_user.id), db
        )
        
        # Generate insights based on financial state
        insights = []
        
        # Spending velocity insight
        if financial_state.daily_spending_avg > 50:
            insights.append({
                "type": "spending_velocity",
                "title": "High Daily Spending",
                "message": f"Your daily spending average is ${financial_state.daily_spending_avg:.2f}. Consider tracking your daily expenses more closely.",
                "priority": "medium"
            })
        
        # Savings rate insight
        if financial_state.savings_rate < 0.1:
            insights.append({
                "type": "savings_rate",
                "title": "Low Savings Rate",
                "message": f"Your savings rate is {financial_state.savings_rate*100:.1f}%. Financial experts recommend saving at least 20% of income.",
                "priority": "high"
            })
        elif financial_state.savings_rate > 0.3:
            insights.append({
                "type": "savings_rate",
                "title": "Excellent Savings Rate",
                "message": f"Your savings rate of {financial_state.savings_rate*100:.1f}% is excellent! Consider investing your surplus.",
                "priority": "low"
            })
        
        # Category spending insights
        if financial_state.category_spending:
            top_category = max(financial_state.category_spending.items(), key=lambda x: x[1])
            if top_category[1] > financial_state.weekly_spending * 0.4:
                insights.append({
                    "type": "category_concentration",
                    "title": "High Category Spending",
                    "message": f"You're spending ${top_category[1]:.2f} on {top_category[0]}, which is {(top_category[1]/financial_state.weekly_spending)*100:.1f}% of your weekly spending.",
                    "priority": "medium"
                })
        
        return {
            "insights": insights,
            "generated_at": financial_state.current_balance  # Placeholder timestamp
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate financial insights"
        )

# Enhanced ML endpoints for Phase 3

# Initialize enhanced services
enhanced_ml_service = EnhancedCQLModelService()
segmentation_service = UserSegmentationService()
spending_prediction_service = SpendingPredictionService()
goal_prediction_service = GoalPredictionService()
behavioral_service = BehavioralEventService()

@router.get("/enhanced-recommendation", response_model=EnhancedAIRecommendation)
async def get_enhanced_coaching_recommendation(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get enhanced AI coaching recommendation with behavioral segmentation"""
    try:
        # Track recommendation view event
        background_tasks.add_task(
            behavioral_service.track_recommendation_interaction,
            str(current_user.id), "enhanced_recommendation", "view", None, db
        )
        
        recommendation = enhanced_ml_service.get_enhanced_recommendation(str(current_user.id), db)
        return recommendation
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate enhanced coaching recommendation"
        )

@router.get("/user-segment", response_model=UserSegmentInfo)
async def get_user_segment(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get user's behavioral segment information"""
    try:
        segment_info = segmentation_service.classify_user_segment(str(current_user.id), db)
        return segment_info
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user segment information"
        )

@router.get("/segment-insights")
async def get_segment_insights(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get segment-specific insights and characteristics"""
    try:
        # Get user segment
        segment_info = segmentation_service.classify_user_segment(str(current_user.id), db)
        
        # Get segment characteristics
        characteristics = segmentation_service.get_segment_characteristics(segment_info.segment_id)
        
        # Get peer comparison
        peers = segmentation_service.get_segment_peers(str(current_user.id), db, limit=5)
        
        return {
            "segment_info": segment_info,
            "characteristics": characteristics,
            "peer_comparison": peers,
            "generated_at": segment_info.assigned_at
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve segment insights"
        )

@router.get("/segment-comparison")
async def get_segment_comparison(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Compare user with segment peers"""
    try:
        peers = segmentation_service.get_segment_peers(str(current_user.id), db, limit=10)
        
        if not peers:
            return {
                "message": "No peer data available for comparison",
                "peers": []
            }
        
        # Calculate peer averages
        peer_avg_balance = sum(peer['current_balance'] for peer in peers) / len(peers)
        peer_avg_savings_rate = sum(peer['savings_rate'] for peer in peers) / len(peers)
        peer_avg_volatility = sum(peer['spending_volatility'] for peer in peers) / len(peers)
        
        return {
            "peer_count": len(peers),
            "peer_averages": {
                "current_balance": peer_avg_balance,
                "savings_rate": peer_avg_savings_rate,
                "spending_volatility": peer_avg_volatility
            },
            "peers": peers
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve segment comparison"
        )

@router.post("/refresh-segment")
async def refresh_user_segment(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Trigger user segment recalculation"""
    try:
        updated_segment = segmentation_service.update_user_segment(str(current_user.id), db)
        
        return {
            "message": "User segment updated successfully",
            "segment_info": updated_segment
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to refresh user segment"
        )

@router.get("/spending-prediction", response_model=SpendingPrediction)
async def get_spending_prediction(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get weekly spending forecasts"""
    try:
        prediction = await spending_prediction_service.predict_weekly_spending(str(current_user.id), db)
        return prediction
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate spending prediction"
        )

class GoalPredictionRequest(BaseModel):
    goal_amount: float
    timeline_days: int

@router.post("/goal-probability", response_model=GoalAchievementProbability)
async def get_goal_achievement_probability(
    goal_request: GoalPredictionRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get goal achievement probability"""
    try:
        prediction = await goal_prediction_service.predict_goal_achievement(
            str(current_user.id), 
            goal_request.goal_amount, 
            goal_request.timeline_days, 
            db
        )
        return prediction
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate goal achievement prediction"
        )

@router.get("/financial-forecast")
async def get_financial_forecast(
    days: int = 30,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get financial health projections"""
    try:
        # Get spending prediction
        spending_prediction = await spending_prediction_service.predict_weekly_spending(str(current_user.id), db)
        
        # Get current financial state
        enhanced_state_generator = enhanced_ml_service.enhanced_state_generator
        current_state = enhanced_state_generator.get_enhanced_financial_state_summary(str(current_user.id), db)
        
        # Project future balance
        weekly_predicted_spending = spending_prediction.total_predicted_spending
        weekly_income = current_state.weekly_income
        weekly_net = weekly_income - weekly_predicted_spending
        
        projected_balance = current_state.current_balance + (weekly_net * (days / 7))
        
        return {
            "current_balance": current_state.current_balance,
            "projected_balance": projected_balance,
            "projection_days": days,
            "weekly_predicted_spending": weekly_predicted_spending,
            "weekly_income": weekly_income,
            "weekly_net": weekly_net,
            "spending_prediction": spending_prediction,
            "generated_at": spending_prediction.generated_at
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate financial forecast"
        )

@router.get("/behavioral-insights")
async def get_behavioral_insights(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get user behavior analysis"""
    try:
        behavior_summary = await behavioral_service.get_user_behavior_summary(str(current_user.id), db)
        return behavior_summary
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve behavioral insights"
        )

@router.get("/behavioral-events")
async def get_behavioral_events(
    days: int = 7,
    event_type: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get user's recent behavioral events"""
    try:
        behavior_data = await behavioral_service.aggregate_user_behavior(str(current_user.id), days, db)
        
        # Filter by event type if specified
        if event_type and behavior_data:
            event_counts = behavior_data.get('event_counts', {})
            interaction_patterns = behavior_data.get('interaction_patterns', {})
            
            filtered_counts = {k: v for k, v in event_counts.items() if k == event_type}
            filtered_patterns = {k: v for k, v in interaction_patterns.items() if k == event_type}
            
            behavior_data['event_counts'] = filtered_counts
            behavior_data['interaction_patterns'] = filtered_patterns
        
        return behavior_data
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve behavioral events"
        )

@router.get("/behavior-changes")
async def get_behavior_changes(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get detected behavioral changes"""
    try:
        changes = await behavioral_service.detect_behavior_changes(str(current_user.id), db)
        
        return {
            "user_id": str(current_user.id),
            "behavior_changes": changes,
            "change_count": len(changes),
            "analysis_date": behavioral_service.validate_service_health()['last_check']
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to detect behavior changes"
        )

@router.get("/enhanced-health")
async def check_enhanced_model_health():
    """Check enhanced AI model health status"""
    try:
        health_status = {
            "enhanced_cql_service": enhanced_ml_service.validate_model_health() if hasattr(enhanced_ml_service, 'validate_model_health') else {"status": "unknown"},
            "segmentation_service": segmentation_service.validate_segmentation_health(),
            "behavioral_service": behavioral_service.validate_service_health(),
            "spending_prediction_service": {"status": "healthy", "model_loaded": spending_prediction_service.spending_model is not None},
            "goal_prediction_service": {"status": "healthy", "model_loaded": goal_prediction_service.goal_model is not None}
        }
        
        # Overall health status
        all_healthy = all(
            service_health.get("status") == "healthy" 
            for service_health in health_status.values()
        )
        
        health_status["overall_status"] = "healthy" if all_healthy else "degraded"
        
        return health_status
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check enhanced model health"
        )

# Continuous Learning endpoints

@router.get("/continuous-learning/status")
async def get_continuous_learning_status(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get continuous learning system status"""
    try:
        from app.services.continuous_learning_service import ContinuousLearningService
        
        cl_service = ContinuousLearningService()
        status = cl_service.get_continuous_learning_status(db)
        
        return status
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve continuous learning status"
        )

@router.get("/continuous-learning/datasets")
async def get_training_datasets(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all training datasets information"""
    try:
        from app.services.continuous_learning_service import ContinuousLearningService
        
        cl_service = ContinuousLearningService()
        datasets = cl_service.get_training_datasets(db)
        
        return {
            "datasets": datasets,
            "total_count": len(datasets),
            "ready_for_training": len([d for d in datasets if d.is_ready_for_training])
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve training datasets"
        )

@router.get("/continuous-learning/ready-datasets")
async def get_ready_datasets(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get datasets ready for training"""
    try:
        from app.services.continuous_learning_service import ContinuousLearningService
        
        cl_service = ContinuousLearningService()
        ready_datasets = cl_service.check_training_readiness(db)
        
        return {
            "ready_datasets": ready_datasets,
            "count": len(ready_datasets)
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve ready datasets"
        )

@router.get("/continuous-learning/data-aggregation")
async def get_data_aggregation_summary(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get data aggregation summary for all model types"""
    try:
        # Aggregate data for all model types
        recommendation_data = await behavioral_service.aggregate_recommendation_feedback(db)
        transaction_data = await behavioral_service.aggregate_user_transactions(db)
        segment_data = await behavioral_service.aggregate_user_segments(db)
        goal_data = await behavioral_service.aggregate_goal_events(db)
        
        return {
            "recommendation_feedback": recommendation_data,
            "user_transactions": transaction_data,
            "user_segments": segment_data,
            "goal_events": goal_data,
            "aggregation_timestamp": behavioral_service.validate_service_health()['last_check']
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve data aggregation summary"
        )

class DataQualityRequest(BaseModel):
    dataset_id: str

@router.post("/continuous-learning/assess-quality")
async def assess_data_quality(
    quality_request: DataQualityRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Assess data quality for a specific dataset"""
    try:
        from app.services.continuous_learning_service import ContinuousLearningService
        import uuid
        
        cl_service = ContinuousLearningService()
        dataset_uuid = uuid.UUID(quality_request.dataset_id)
        
        quality_metrics = cl_service.assess_data_quality(dataset_uuid, db)
        
        return {
            "dataset_id": quality_request.dataset_id,
            "quality_metrics": quality_metrics,
            "assessment_timestamp": quality_metrics.get('assessment_date', 'unknown')
        }
    
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid dataset ID format"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to assess data quality"
        )

@router.get("/continuous-learning/quality-metrics/{dataset_id}")
async def get_quality_metrics(
    dataset_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get historical data quality metrics for a dataset"""
    try:
        from app.services.continuous_learning_service import ContinuousLearningService
        import uuid
        
        cl_service = ContinuousLearningService()
        dataset_uuid = uuid.UUID(dataset_id)
        
        metrics = cl_service.get_data_quality_metrics(dataset_uuid, db)
        
        return {
            "dataset_id": dataset_id,
            "quality_metrics": metrics,
            "metrics_count": len(metrics)
        }
    
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid dataset ID format"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve quality metrics"
        )

class PrepareDataRequest(BaseModel):
    dataset_id: str

@router.post("/continuous-learning/prepare-data")
async def prepare_training_data(
    prepare_request: PrepareDataRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Prepare data for model training"""
    try:
        from app.services.continuous_learning_service import ContinuousLearningService
        import uuid
        
        cl_service = ContinuousLearningService()
        dataset_uuid = uuid.UUID(prepare_request.dataset_id)
        
        prepared_data = cl_service.prepare_training_data(dataset_uuid, db)
        
        return prepared_data
    
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid dataset ID format"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to prepare training data"
        )

# Phase 3: Model Training endpoints

class TriggerTrainingRequest(BaseModel):
    priority: int = 1

@router.post("/continuous-learning/trigger-training/{model_type}")
async def trigger_model_training(
    model_type: str,
    training_request: TriggerTrainingRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Manually trigger training for a specific model type"""
    try:
        from app.services.training_orchestrator import training_orchestrator
        from app.services.continuous_learning_service import ContinuousLearningService
        import uuid
        
        # Validate model type
        valid_model_types = ['recommendation', 'segmentation', 'spending', 'goal']
        if model_type not in valid_model_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model type. Must be one of: {valid_model_types}"
            )
        
        # Get the dataset for this model type
        cl_service = ContinuousLearningService()
        datasets = cl_service.get_training_datasets(db)
        
        target_dataset = next((d for d in datasets if d.dataset_type == model_type), None)
        if not target_dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No dataset found for model type: {model_type}"
            )
        
        # Queue training job
        dataset_uuid = uuid.UUID(target_dataset.id)
        job = training_orchestrator.queue_training_job(
            dataset_uuid, 
            model_type, 
            training_request.priority
        )
        
        # Try to start the job if no other job is active
        if not training_orchestrator.active_job:
            training_orchestrator.start_training_job(job, db)
        
        return {
            "message": f"Training job queued for {model_type}",
            "job_id": str(job.id),
            "status": job.status,
            "priority": job.priority,
            "dataset_id": target_dataset.id,
            "queue_position": len(training_orchestrator.job_queue) if job.status == 'queued' else 0
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger training: {str(e)}"
        )

@router.get("/continuous-learning/training-status")
async def get_training_status(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get status of all training jobs"""
    try:
        from app.services.training_orchestrator import training_orchestrator
        
        all_jobs = training_orchestrator.get_all_jobs()
        
        return {
            "training_status": all_jobs,
            "queue_length": len(all_jobs['queued']),
            "active_job_count": len(all_jobs['active']),
            "completed_job_count": len(all_jobs['completed']),
            "last_updated": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve training status"
        )

@router.get("/continuous-learning/training-job/{job_id}")
async def get_training_job_status(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get status of a specific training job"""
    try:
        from app.services.training_orchestrator import training_orchestrator
        import uuid
        
        job_uuid = uuid.UUID(job_id)
        job_status = training_orchestrator.get_job_status(job_uuid)
        
        if not job_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Training job not found"
            )
        
        return {
            "job_status": job_status,
            "retrieved_at": datetime.utcnow().isoformat()
        }
    
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid job ID format"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve job status"
        )

@router.get("/continuous-learning/training-history")
async def get_training_history(
    limit: int = 20,
    model_type: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get history of training events"""
    try:
        query = db.query(ModelTrainingEvent)
        
        if model_type:
            query = query.filter(ModelTrainingEvent.model_type == model_type)
        
        training_events = query.order_by(
            ModelTrainingEvent.created_at.desc()
        ).limit(limit).all()
        
        history = []
        for event in training_events:
            history.append({
                "id": str(event.id),
                "model_type": event.model_type,
                "status": event.status,
                "start_time": event.start_time.isoformat(),
                "end_time": event.end_time.isoformat() if event.end_time else None,
                "validation_score": float(event.validation_score) if event.validation_score else None,
                "training_data_size": event.training_data_size,
                "model_path": event.model_path,
                "performance_metrics": json.loads(event.performance_metrics) if event.performance_metrics else {}
            })
        
        return {
            "training_history": history,
            "total_events": len(history),
            "filter_applied": {"model_type": model_type} if model_type else None
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve training history"
        )

@router.get("/continuous-learning/model-versions")
async def get_model_versions(
    model_type: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get information about model versions"""
    try:
        query = db.query(ModelVersion)
        
        if model_type:
            query = query.filter(ModelVersion.model_type == model_type)
        
        model_versions = query.order_by(
            ModelVersion.model_type,
            ModelVersion.created_at.desc()
        ).all()
        
        versions = []
        for version in model_versions:
            versions.append({
                "id": str(version.id),
                "model_type": version.model_type,
                "version": version.version,
                "is_active": version.is_active,
                "performance_baseline": float(version.performance_baseline) if version.performance_baseline else None,
                "performance_current": float(version.performance_current) if version.performance_current else None,
                "deployment_date": version.deployment_date.isoformat() if version.deployment_date else None,
                "rollback_count": version.rollback_count,
                "model_path": version.model_path,
                "created_at": version.created_at.isoformat(),
                "metadata": json.loads(version.model_metadata) if version.model_metadata else {}
            })
        
        return {
            "model_versions": versions,
            "total_versions": len(versions),
            "filter_applied": {"model_type": model_type} if model_type else None
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model versions"
        )

@router.get("/continuous-learning/validation-results/{training_event_id}")
async def get_validation_results(
    training_event_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get validation results for a training event"""
    try:
        import uuid
        
        event_uuid = uuid.UUID(training_event_id)
        training_event = db.query(ModelTrainingEvent).filter(
            ModelTrainingEvent.id == event_uuid
        ).first()
        
        if not training_event:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Training event not found"
            )
        
        # Get associated model version
        model_version = db.query(ModelVersion).filter(
            ModelVersion.training_event_id == event_uuid
        ).first()
        
        validation_results = {
            "training_event_id": training_event_id,
            "model_type": training_event.model_type,
            "status": training_event.status,
            "validation_score": float(training_event.validation_score) if training_event.validation_score else None,
            "performance_metrics": json.loads(training_event.performance_metrics) if training_event.performance_metrics else {},
            "training_duration": None,
            "model_version": None
        }
        
        # Calculate training duration
        if training_event.start_time and training_event.end_time:
            duration = training_event.end_time - training_event.start_time
            validation_results["training_duration"] = str(duration)
        
        # Add model version info
        if model_version:
            validation_results["model_version"] = {
                "version": model_version.version,
                "is_active": model_version.is_active,
                "model_path": model_version.model_path,
                "metadata": json.loads(model_version.model_metadata) if model_version.model_metadata else {}
            }
        
        return validation_results
    
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid training event ID format"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve validation results"
        )

@router.post("/continuous-learning/auto-trigger-ready")
async def auto_trigger_ready_training(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Automatically trigger training for all ready datasets"""
    try:
        from app.services.training_orchestrator import training_orchestrator
        
        triggered_jobs = training_orchestrator.trigger_training_if_ready(db)
        
        return {
            "message": f"Auto-triggered {len(triggered_jobs)} training jobs",
            "triggered_jobs": [
                {
                    "job_id": str(job.id),
                    "model_type": job.model_type,
                    "status": job.status,
                    "priority": job.priority
                }
                for job in triggered_jobs
            ],
            "triggered_at": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to auto-trigger training"
        )

# Demo/Testing endpoints

class SetDataCountRequest(BaseModel):
    model_type: str
    count: int

@router.post("/continuous-learning/demo/set-data-count")
async def set_dataset_count_for_demo(
    request: SetDataCountRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Set dataset count for demo purposes"""
    try:
        from app.services.continuous_learning_service import ContinuousLearningService
        
        # Validate model type
        valid_model_types = ['recommendation', 'segmentation', 'spending', 'goal']
        if request.model_type not in valid_model_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model type. Must be one of: {valid_model_types}"
            )
        
        # Get dataset
        dataset = db.query(TrainingDataset).filter(
            TrainingDataset.dataset_type == request.model_type
        ).first()
        
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset not found for model type: {request.model_type}"
            )
        
        # Update count
        old_count = dataset.data_count
        dataset.data_count = request.count
        dataset.updated_at = datetime.utcnow()
        
        # Check if threshold reached
        cl_service = ContinuousLearningService()
        if dataset.data_count >= dataset.threshold and not dataset.threshold_reached:
            dataset.threshold_reached = True
            dataset.data_quality_score = 0.85  # Mock good quality score
            dataset.is_ready_for_training = True
        elif dataset.data_count < dataset.threshold:
            dataset.threshold_reached = False
            dataset.is_ready_for_training = False
        
        db.commit()
        
        return {
            "message": f"Updated {request.model_type} dataset count from {old_count} to {request.count}",
            "dataset": {
                "model_type": dataset.dataset_type,
                "data_count": dataset.data_count,
                "threshold": dataset.threshold,
                "threshold_reached": dataset.threshold_reached,
                "is_ready_for_training": dataset.is_ready_for_training,
                "data_quality_score": float(dataset.data_quality_score) if dataset.data_quality_score else None
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to set dataset count: {str(e)}"
        )

@router.post("/continuous-learning/demo/simulate-data-collection")
async def simulate_data_collection(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Simulate data collection to reach thresholds for demo"""
    try:
        from app.services.continuous_learning_service import ContinuousLearningService
        
        cl_service = ContinuousLearningService()
        datasets = cl_service.get_training_datasets(db)
        
        updated_datasets = []
        
        for dataset_info in datasets:
            dataset = db.query(TrainingDataset).filter(
                TrainingDataset.id == dataset_info.id
            ).first()
            
            if dataset and not dataset.is_ready_for_training:
                # Set count to just above threshold
                new_count = dataset.threshold + 10
                old_count = dataset.data_count
                
                dataset.data_count = new_count
                dataset.threshold_reached = True
                dataset.data_quality_score = 0.85  # Mock good quality
                dataset.is_ready_for_training = True
                dataset.updated_at = datetime.utcnow()
                
                updated_datasets.append({
                    "model_type": dataset.dataset_type,
                    "old_count": old_count,
                    "new_count": new_count,
                    "threshold": dataset.threshold,
                    "ready_for_training": True
                })
        
        db.commit()
        
        return {
            "message": f"Simulated data collection for {len(updated_datasets)} datasets",
            "updated_datasets": updated_datasets,
            "simulation_time": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to simulate data collection: {str(e)}"
        )
