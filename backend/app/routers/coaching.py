from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import desc
from app.models.ml_models import AIRecommendation, FinancialState
from app.services.ml_service import CQLModelService
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
