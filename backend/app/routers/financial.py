from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from app.models.financial import (
    TransactionCreate, TransactionResponse, WalletResponse, 
    SpendingAnalytics, FinancialHealthSummary, TransactionFilter
)
from app.services.financial_service import FinancialService
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
