from pydantic import BaseModel, validator
from decimal import Decimal
from datetime import datetime
from typing import Optional, List
from enum import Enum

class TransactionType(str, Enum):
    DEBIT = "debit"
    CREDIT = "credit"

class TransactionCategory(str, Enum):
    GROCERIES = "groceries"
    DINE_OUT = "dine_out"
    ENTERTAINMENT = "entertainment"
    BILLS = "bills"
    TRANSPORT = "transport"
    SHOPPING = "shopping"
    HEALTH = "health"
    FITNESS = "fitness"
    SAVINGS = "savings"
    INCOME = "income"
    OTHER = "other"

class TransactionCreate(BaseModel):
    amount: Decimal
    category: TransactionCategory
    description: str
    merchant_name: Optional[str] = None
    location_city: Optional[str] = "Unknown"
    location_country: str = "US"

    @validator('amount')
    def validate_amount(cls, v):
        if v == 0:
            raise ValueError('Transaction amount cannot be zero')
        return v

class TransactionResponse(BaseModel):
    id: str
    transaction_type: TransactionType
    amount: Decimal
    category: str
    description: str
    merchant_name: Optional[str]
    location_city: Optional[str]
    location_country: str
    balance_after: Decimal
    transaction_date: datetime

    class Config:
        from_attributes = True

class WalletResponse(BaseModel):
    id: str
    balance: Decimal
    currency: str
    updated_at: datetime

    class Config:
        from_attributes = True

class SpendingAnalytics(BaseModel):
    category: str
    total_amount: Decimal
    transaction_count: int
    percentage: float

class FinancialHealthSummary(BaseModel):
    current_balance: Decimal
    weekly_spending: Decimal
    weekly_income: Decimal
    transaction_count: int
    savings_rate: float
    daily_spending_avg: Decimal
    top_spending_categories: List[SpendingAnalytics]

class TransactionFilter(BaseModel):
    limit: Optional[int] = 50
    days: Optional[int] = None
    category: Optional[str] = None
    transaction_type: Optional[TransactionType] = None
    min_amount: Optional[Decimal] = None
    max_amount: Optional[Decimal] = None

    @validator('limit')
    def validate_limit(cls, v):
        if v and (v < 1 or v > 100):
            raise ValueError('Limit must be between 1 and 100')
        return v
