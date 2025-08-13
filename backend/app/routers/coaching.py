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
from app.models.database import User, RecommendationHistory, RecommendationFeedback
from typing import List, Optional
from pydantic import BaseModel
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
