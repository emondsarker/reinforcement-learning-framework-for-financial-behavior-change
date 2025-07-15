from pydantic import BaseModel, validator
from decimal import Decimal
from typing import Optional, List
from datetime import datetime

class ProductCategoryResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    transaction_category: str

    class Config:
        from_attributes = True

class ProductResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    price: Decimal
    category_id: str
    merchant_name: str
    is_available: bool
    stock_quantity: int
    image_url: Optional[str]

    class Config:
        from_attributes = True

class ProductCreate(BaseModel):
    name: str
    description: Optional[str] = None
    price: Decimal
    category_id: str
    merchant_name: str
    stock_quantity: int = 100
    image_url: Optional[str] = None

    @validator('price')
    def validate_price(cls, v):
        if v <= 0:
            raise ValueError('Price must be greater than zero')
        return v

    @validator('stock_quantity')
    def validate_stock(cls, v):
        if v < 0:
            raise ValueError('Stock quantity cannot be negative')
        return v

class ProductPurchase(BaseModel):
    product_id: str
    quantity: int = 1

    @validator('quantity')
    def validate_quantity(cls, v):
        if v < 1:
            raise ValueError('Quantity must be at least 1')
        if v > 10:
            raise ValueError('Quantity cannot exceed 10 per purchase')
        return v

class PurchaseResponse(BaseModel):
    id: str
    product: ProductResponse
    quantity: int
    total_amount: Decimal
    transaction_id: str
    purchase_date: datetime

    class Config:
        from_attributes = True

class ProductFilter(BaseModel):
    category_id: Optional[str] = None
    search: Optional[str] = None
    min_price: Optional[Decimal] = None
    max_price: Optional[Decimal] = None
    limit: Optional[int] = 20
    available_only: bool = True

    @validator('limit')
    def validate_limit(cls, v):
        if v and (v < 1 or v > 100):
            raise ValueError('Limit must be between 1 and 100')
        return v

class CategoryCreate(BaseModel):
    name: str
    description: Optional[str] = None
    transaction_category: str

    @validator('transaction_category')
    def validate_transaction_category(cls, v):
        valid_categories = [
            'groceries', 'dine_out', 'entertainment', 'bills',
            'transport', 'shopping', 'health', 'fitness',
            'savings', 'income', 'other'
        ]
        if v not in valid_categories:
            raise ValueError(f'Transaction category must be one of: {", ".join(valid_categories)}')
        return v
