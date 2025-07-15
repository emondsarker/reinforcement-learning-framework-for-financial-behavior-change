from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from app.models.database import Wallet, Transaction, User
from app.models.financial import TransactionCreate, TransactionType, SpendingAnalytics, FinancialHealthSummary
from decimal import Decimal
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import uuid

class FinancialService:
    def __init__(self, db: Session):
        self.db = db

    def get_user_wallet(self, user_id: str) -> Wallet:
        """Get or create user's wallet"""
        user_uuid = uuid.UUID(user_id)
        wallet = self.db.query(Wallet).filter(Wallet.user_id == user_uuid).first()
        
        if not wallet:
            # Create wallet if doesn't exist
            wallet = Wallet(user_id=user_uuid, balance=Decimal('1000.00'))
            self.db.add(wallet)
            self.db.commit()
            self.db.refresh(wallet)
        
        return wallet

    def process_transaction(self, user_id: str, transaction_data: TransactionCreate) -> Transaction:
        """Process a financial transaction"""
        user_uuid = uuid.UUID(user_id)
        wallet = self.get_user_wallet(user_id)

        # Determine transaction type and validate balance
        if transaction_data.amount > 0:
            transaction_type = TransactionType.CREDIT
            new_balance = wallet.balance + transaction_data.amount
        else:
            transaction_type = TransactionType.DEBIT
            if wallet.balance + transaction_data.amount < 0:
                raise ValueError("Insufficient funds")
            new_balance = wallet.balance + transaction_data.amount

        # Create transaction record
        transaction = Transaction(
            user_id=user_uuid,
            wallet_id=wallet.id,
            transaction_type=transaction_type.value,
            amount=abs(transaction_data.amount),
            category=transaction_data.category.value,
            description=transaction_data.description,
            merchant_name=transaction_data.merchant_name,
            location_city=transaction_data.location_city,
            location_country=transaction_data.location_country,
            balance_after=new_balance
        )

        # Update wallet balance
        wallet.balance = new_balance
        wallet.updated_at = datetime.utcnow()

        self.db.add(transaction)
        self.db.commit()
        self.db.refresh(transaction)
        
        return transaction

    def get_transaction_history(
        self, 
        user_id: str, 
        limit: int = 50, 
        days: Optional[int] = None,
        category: Optional[str] = None,
        transaction_type: Optional[str] = None
    ) -> List[Transaction]:
        """Get user's transaction history with filters"""
        user_uuid = uuid.UUID(user_id)
        query = self.db.query(Transaction).filter(Transaction.user_id == user_uuid)

        # Apply filters
        if days:
            start_date = datetime.utcnow() - timedelta(days=days)
            query = query.filter(Transaction.transaction_date >= start_date)
        
        if category:
            query = query.filter(Transaction.category == category)
        
        if transaction_type:
            query = query.filter(Transaction.transaction_type == transaction_type)

        return query.order_by(Transaction.transaction_date.desc()).limit(limit).all()

    def get_spending_analytics(self, user_id: str, days: int = 30) -> List[SpendingAnalytics]:
        """Get spending analytics by category"""
        user_uuid = uuid.UUID(user_id)
        start_date = datetime.utcnow() - timedelta(days=days)

        # Query spending by category
        results = self.db.query(
            Transaction.category,
            func.sum(Transaction.amount).label('total_amount'),
            func.count(Transaction.id).label('transaction_count')
        ).filter(
            and_(
                Transaction.user_id == user_uuid,
                Transaction.transaction_type == 'debit',
                Transaction.transaction_date >= start_date
            )
        ).group_by(Transaction.category).all()

        # Calculate total spending for percentages
        total_spending = sum(result.total_amount for result in results)
        
        analytics = []
        for result in results:
            percentage = float(result.total_amount / total_spending * 100) if total_spending > 0 else 0
            analytics.append(SpendingAnalytics(
                category=result.category,
                total_amount=result.total_amount,
                transaction_count=result.transaction_count,
                percentage=percentage
            ))

        return sorted(analytics, key=lambda x: x.total_amount, reverse=True)

    def get_financial_health_summary(self, user_id: str) -> FinancialHealthSummary:
        """Get comprehensive financial health summary"""
        user_uuid = uuid.UUID(user_id)
        wallet = self.get_user_wallet(user_id)
        
        # Get last 7 days of transactions
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)

        transactions = self.db.query(Transaction).filter(
            and_(
                Transaction.user_id == user_uuid,
                Transaction.transaction_date >= start_date,
                Transaction.transaction_date <= end_date
            )
        ).all()

        # Calculate metrics
        weekly_spending = sum(
            t.amount for t in transactions 
            if t.transaction_type == 'debit'
        )
        weekly_income = sum(
            t.amount for t in transactions 
            if t.transaction_type == 'credit'
        )
        transaction_count = len(transactions)
        
        # Calculate savings rate
        savings_rate = float((weekly_income - weekly_spending) / max(weekly_income, 1))
        daily_spending_avg = weekly_spending / 7

        # Get top spending categories
        top_categories = self.get_spending_analytics(user_id, 7)[:5]

        return FinancialHealthSummary(
            current_balance=wallet.balance,
            weekly_spending=weekly_spending,
            weekly_income=weekly_income,
            transaction_count=transaction_count,
            savings_rate=savings_rate,
            daily_spending_avg=daily_spending_avg,
            top_spending_categories=top_categories
        )

    def get_balance_history(self, user_id: str, days: int = 30) -> List[Dict]:
        """Get balance history over time"""
        user_uuid = uuid.UUID(user_id)
        start_date = datetime.utcnow() - timedelta(days=days)

        transactions = self.db.query(Transaction).filter(
            and_(
                Transaction.user_id == user_uuid,
                Transaction.transaction_date >= start_date
            )
        ).order_by(Transaction.transaction_date.asc()).all()

        balance_history = []
        for transaction in transactions:
            balance_history.append({
                'date': transaction.transaction_date.isoformat(),
                'balance': float(transaction.balance_after),
                'transaction_type': transaction.transaction_type,
                'amount': float(transaction.amount)
            })

        return balance_history

    def get_monthly_summary(self, user_id: str, year: int, month: int) -> Dict:
        """Get monthly financial summary"""
        user_uuid = uuid.UUID(user_id)
        
        # Calculate date range for the month
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1)
        else:
            end_date = datetime(year, month + 1, 1)

        transactions = self.db.query(Transaction).filter(
            and_(
                Transaction.user_id == user_uuid,
                Transaction.transaction_date >= start_date,
                Transaction.transaction_date < end_date
            )
        ).all()

        total_income = sum(
            t.amount for t in transactions 
            if t.transaction_type == 'credit'
        )
        total_spending = sum(
            t.amount for t in transactions 
            if t.transaction_type == 'debit'
        )
        net_change = total_income - total_spending

        return {
            'month': month,
            'year': year,
            'total_income': float(total_income),
            'total_spending': float(total_spending),
            'net_change': float(net_change),
            'transaction_count': len(transactions),
            'average_transaction': float(sum(t.amount for t in transactions) / len(transactions)) if transactions else 0
        }
