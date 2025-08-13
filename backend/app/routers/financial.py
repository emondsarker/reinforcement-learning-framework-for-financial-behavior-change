from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
from app.models.financial import (
    TransactionCreate, TransactionResponse, WalletResponse, 
    SpendingAnalytics, FinancialHealthSummary, TransactionFilter
)
from app.services.financial_service import FinancialService
from app.services.behavioral_event_service import BehavioralEventService
from app.middleware.auth_middleware import get_current_active_user
from app.database import get_db
from app.models.database import User

router = APIRouter(prefix="/financial", tags=["financial"])

@router.get("/wallet", response_model=WalletResponse)
async def get_wallet(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get user's wallet information"""
    try:
        service = FinancialService(db)
        wallet = service.get_user_wallet(str(current_user.id))
        return WalletResponse(
            id=str(wallet.id),
            balance=wallet.balance,
            currency=wallet.currency,
            updated_at=wallet.updated_at
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve wallet information"
        )

@router.post("/transactions", response_model=TransactionResponse, status_code=status.HTTP_201_CREATED)
async def create_transaction(
    transaction_data: TransactionCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create a new transaction"""
    try:
        service = FinancialService(db)
        transaction = service.process_transaction(str(current_user.id), transaction_data)
        
        return TransactionResponse(
            id=str(transaction.id),
            transaction_type=transaction.transaction_type,
            amount=transaction.amount,
            category=transaction.category,
            description=transaction.description,
            merchant_name=transaction.merchant_name,
            location_city=transaction.location_city,
            location_country=transaction.location_country,
            balance_after=transaction.balance_after,
            transaction_date=transaction.transaction_date
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create transaction"
        )

@router.get("/transactions", response_model=List[TransactionResponse])
async def get_transactions(
    limit: int = Query(50, le=100, description="Maximum number of transactions to return"),
    days: Optional[int] = Query(None, description="Filter by days back from today"),
    category: Optional[str] = Query(None, description="Filter by transaction category"),
    transaction_type: Optional[str] = Query(None, description="Filter by transaction type (debit/credit)"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get user's transaction history with filters"""
    try:
        service = FinancialService(db)
        transactions = service.get_transaction_history(
            str(current_user.id), 
            limit=limit,
            days=days,
            category=category,
            transaction_type=transaction_type
        )
        
        return [
            TransactionResponse(
                id=str(t.id),
                transaction_type=t.transaction_type,
                amount=t.amount,
                category=t.category,
                description=t.description,
                merchant_name=t.merchant_name,
                location_city=t.location_city,
                location_country=t.location_country,
                balance_after=t.balance_after,
                transaction_date=t.transaction_date
            )
            for t in transactions
        ]
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve transactions"
        )

@router.get("/analytics/spending-by-category", response_model=List[SpendingAnalytics])
async def get_spending_analytics(
    days: int = Query(30, description="Number of days to analyze"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get spending analytics by category"""
    try:
        service = FinancialService(db)
        analytics = service.get_spending_analytics(str(current_user.id), days)
        return analytics
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve spending analytics"
        )

@router.get("/health-summary", response_model=FinancialHealthSummary)
async def get_financial_health_summary(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get comprehensive financial health summary"""
    try:
        service = FinancialService(db)
        summary = service.get_financial_health_summary(str(current_user.id))
        return summary
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve financial health summary"
        )

@router.get("/balance-history")
async def get_balance_history(
    days: int = Query(30, description="Number of days of history to return"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get balance history over time"""
    try:
        service = FinancialService(db)
        history = service.get_balance_history(str(current_user.id), days)
        return {"balance_history": history}
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve balance history"
        )

@router.get("/monthly-summary")
async def get_monthly_summary(
    year: int = Query(..., description="Year for the summary"),
    month: int = Query(..., ge=1, le=12, description="Month for the summary (1-12)"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get monthly financial summary"""
    try:
        service = FinancialService(db)
        summary = service.get_monthly_summary(str(current_user.id), year, month)
        return summary
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve monthly summary"
        )

@router.get("/transaction/{transaction_id}", response_model=TransactionResponse)
async def get_transaction_by_id(
    transaction_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get a specific transaction by ID"""
    try:
        import uuid
        from app.models.database import Transaction
        
        transaction_uuid = uuid.UUID(transaction_id)
        transaction = db.query(Transaction).filter(
            Transaction.id == transaction_uuid,
            Transaction.user_id == current_user.id
        ).first()
        
        if not transaction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Transaction not found"
            )
        
        return TransactionResponse(
            id=str(transaction.id),
            transaction_type=transaction.transaction_type,
            amount=transaction.amount,
            category=transaction.category,
            description=transaction.description,
            merchant_name=transaction.merchant_name,
            location_city=transaction.location_city,
            location_country=transaction.location_country,
            balance_after=transaction.balance_after,
            transaction_date=transaction.transaction_date
        )
    
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid transaction ID format"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve transaction"
        )

@router.get("/categories")
async def get_transaction_categories():
    """Get available transaction categories"""
    from app.models.financial import TransactionCategory
    
    categories = [
        {
            "value": category.value,
            "label": category.value.replace('_', ' ').title(),
            "description": f"Transactions related to {category.value.replace('_', ' ')}"
        }
        for category in TransactionCategory
    ]
    
    return {"categories": categories}

# Enhanced behavioral analytics endpoints for Phase 3

# Initialize behavioral service
behavioral_service = BehavioralEventService()

@router.get("/spending-patterns")
async def get_spending_patterns(
    days: int = Query(30, description="Number of days to analyze"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get behavioral spending analysis"""
    try:
        # Get behavioral data
        behavior_data = await behavioral_service.aggregate_user_behavior(str(current_user.id), days, db)
        
        # Get financial service data
        service = FinancialService(db)
        analytics = service.get_spending_analytics(str(current_user.id), days)
        
        # Combine behavioral and financial data
        spending_patterns = {
            "period_days": days,
            "behavioral_data": behavior_data,
            "spending_analytics": [analytics_item.dict() for analytics_item in analytics],
            "engagement_score": behavior_data.get('engagement_score', 0) if behavior_data else 0,
            "behavior_diversity": behavior_data.get('behavior_diversity', 0) if behavior_data else 0,
            "generated_at": behavior_data.get('generated_at') if behavior_data else None
        }
        
        return spending_patterns
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve spending patterns"
        )

@router.get("/category-preferences")
async def get_category_preferences(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get category behavior insights"""
    try:
        # Get spending analytics
        service = FinancialService(db)
        analytics = service.get_spending_analytics(str(current_user.id), 30)
        
        # Get behavioral events related to transactions
        behavior_data = await behavioral_service.aggregate_user_behavior(str(current_user.id), 30, db)
        
        # Analyze category preferences
        category_insights = []
        for analytics_item in analytics:
            category_insight = {
                "category": analytics_item.category,
                "total_amount": analytics_item.total_amount,
                "transaction_count": analytics_item.transaction_count,
                "average_amount": analytics_item.average_amount,
                "percentage_of_spending": analytics_item.percentage_of_spending,
                "behavioral_score": 0  # Default
            }
            
            # Add behavioral scoring if available
            if behavior_data and 'interaction_patterns' in behavior_data:
                transaction_interactions = behavior_data['interaction_patterns'].get('transaction_creation', {})
                category_insight['behavioral_score'] = transaction_interactions.get(analytics_item.category, 0)
            
            category_insights.append(category_insight)
        
        # Sort by total amount descending
        category_insights.sort(key=lambda x: x['total_amount'], reverse=True)
        
        return {
            "category_preferences": category_insights,
            "top_category": category_insights[0] if category_insights else None,
            "behavioral_data": behavior_data,
            "analysis_period_days": 30
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve category preferences"
        )

@router.get("/financial-personality")
async def get_financial_personality(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get user financial personality profile"""
    try:
        # Get enhanced financial state
        from app.services.ml_service import EnhancedFinancialStateVector
        enhanced_state_generator = EnhancedFinancialStateVector()
        enhanced_state = enhanced_state_generator.get_enhanced_financial_state_summary(str(current_user.id), db)
        
        # Get behavioral data
        behavior_data = await behavioral_service.aggregate_user_behavior(str(current_user.id), 30, db)
        
        # Get user segment
        from app.services.ml_service import UserSegmentationService
        segmentation_service = UserSegmentationService()
        segment_info = segmentation_service.classify_user_segment(str(current_user.id), db)
        
        # Create personality profile
        personality_traits = []
        
        # Spending behavior traits
        if enhanced_state.spending_volatility > 200:
            personality_traits.append({
                "trait": "Variable Spender",
                "description": "Your spending varies significantly from week to week",
                "score": min(1.0, enhanced_state.spending_volatility / 500)
            })
        else:
            personality_traits.append({
                "trait": "Consistent Spender",
                "description": "You maintain steady spending patterns",
                "score": max(0.3, 1.0 - (enhanced_state.spending_volatility / 200))
            })
        
        # Impulse buying trait
        if enhanced_state.impulse_buying_score > 0.3:
            personality_traits.append({
                "trait": "Impulse Buyer",
                "description": "You tend to make quick purchasing decisions",
                "score": enhanced_state.impulse_buying_score
            })
        
        # Category diversity trait
        if enhanced_state.category_diversity > 0.7:
            personality_traits.append({
                "trait": "Diverse Spender",
                "description": "You spend across many different categories",
                "score": enhanced_state.category_diversity
            })
        
        # Financial discipline trait
        if enhanced_state.savings_rate > 0.2:
            personality_traits.append({
                "trait": "Disciplined Saver",
                "description": "You consistently save a good portion of your income",
                "score": min(1.0, enhanced_state.savings_rate * 2)
            })
        
        return {
            "user_id": str(current_user.id),
            "segment_info": segment_info,
            "personality_traits": personality_traits,
            "financial_metrics": {
                "spending_volatility": enhanced_state.spending_volatility,
                "impulse_buying_score": enhanced_state.impulse_buying_score,
                "category_diversity": enhanced_state.category_diversity,
                "savings_rate": enhanced_state.savings_rate,
                "financial_stress_indicator": enhanced_state.financial_stress_indicator
            },
            "behavioral_summary": behavior_data,
            "generated_at": enhanced_state.dict().get('generated_at', 'unknown')
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate financial personality profile"
        )

@router.get("/peer-comparison")
async def get_peer_comparison(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Compare financial behavior with similar users"""
    try:
        # Get user segment and peers
        from app.services.ml_service import UserSegmentationService
        segmentation_service = UserSegmentationService()
        
        segment_info = segmentation_service.classify_user_segment(str(current_user.id), db)
        peers = segmentation_service.get_segment_peers(str(current_user.id), db, limit=10)
        
        if not peers:
            return {
                "message": "No peer data available for comparison",
                "segment_info": segment_info,
                "comparison": None
            }
        
        # Get current user's financial data
        service = FinancialService(db)
        user_summary = service.get_financial_health_summary(str(current_user.id))
        
        # Calculate peer averages
        peer_avg_balance = sum(peer['current_balance'] for peer in peers) / len(peers)
        peer_avg_savings_rate = sum(peer['savings_rate'] for peer in peers) / len(peers)
        peer_avg_volatility = sum(peer['spending_volatility'] for peer in peers) / len(peers)
        
        # Create comparison
        comparison = {
            "user_metrics": {
                "current_balance": float(user_summary.current_balance),
                "savings_rate": float(user_summary.savings_rate),
                "spending_volatility": 0  # Would need to calculate from enhanced state
            },
            "peer_averages": {
                "current_balance": peer_avg_balance,
                "savings_rate": peer_avg_savings_rate,
                "spending_volatility": peer_avg_volatility
            },
            "comparison_scores": {
                "balance_percentile": 0.5,  # Simplified calculation
                "savings_percentile": 0.5,
                "volatility_percentile": 0.5
            },
            "peer_count": len(peers)
        }
        
        # Calculate percentiles (simplified)
        balance_better_count = sum(1 for peer in peers if peer['current_balance'] < float(user_summary.current_balance))
        savings_better_count = sum(1 for peer in peers if peer['savings_rate'] < float(user_summary.savings_rate))
        
        comparison["comparison_scores"]["balance_percentile"] = balance_better_count / len(peers)
        comparison["comparison_scores"]["savings_percentile"] = savings_better_count / len(peers)
        
        return {
            "segment_info": segment_info,
            "comparison": comparison,
            "insights": [
                f"You're in the {segment_info.segment_name} segment",
                f"Your balance is {'above' if comparison['comparison_scores']['balance_percentile'] > 0.5 else 'below'} average for your segment",
                f"Your savings rate is {'above' if comparison['comparison_scores']['savings_percentile'] > 0.5 else 'below'} average for your segment"
            ]
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate peer comparison"
        )

@router.post("/track-transaction-event")
async def track_transaction_event(
    background_tasks: BackgroundTasks,
    transaction_id: str,
    event_type: str = Query(..., description="Event type: 'create', 'view', 'analyze'"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Track transaction-related behavioral events"""
    try:
        # Add background task to track the event
        background_tasks.add_task(
            behavioral_service.track_event,
            str(current_user.id),
            'transaction_interaction',
            {
                'transaction_id': transaction_id,
                'interaction_type': event_type,
                'additional_data': {'endpoint': '/financial/track-transaction-event'}
            },
            db
        )
        
        return {
            "message": "Transaction event tracked successfully",
            "transaction_id": transaction_id,
            "event_type": event_type
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to track transaction event"
        )
