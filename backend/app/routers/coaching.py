from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.models.ml_models import AIRecommendation, FinancialState
from app.services.ml_service import CQLModelService
from app.middleware.auth_middleware import get_current_active_user
from app.database import get_db
from app.models.database import User

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

@router.post("/feedback")
async def submit_recommendation_feedback(
    recommendation_id: str,
    helpful: bool,
    feedback_text: str = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Submit feedback on a coaching recommendation (for future model improvement)"""
    try:
        # In a full implementation, this would store feedback in the database
        # for model retraining and improvement
        feedback_data = {
            "user_id": str(current_user.id),
            "recommendation_id": recommendation_id,
            "helpful": helpful,
            "feedback_text": feedback_text,
            "timestamp": "2024-01-01T00:00:00"  # Would use actual timestamp
        }
        
        # For now, just return success
        return {
            "message": "Feedback submitted successfully",
            "feedback_id": "temp-feedback-id"
        }
    
    except Exception as e:
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
