from sqlalchemy import Column, String, DateTime, Boolean, DECIMAL, Integer, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base
import uuid

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    date_of_birth = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    is_active = Column(Boolean, default=True)

    # Relationships
    wallet = relationship("Wallet", back_populates="user", uselist=False)
    transactions = relationship("Transaction", back_populates="user")
    purchases = relationship("UserPurchase", back_populates="user")

class UserProfile(Base):
    __tablename__ = "user_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    monthly_income = Column(DECIMAL(10, 2), default=0)
    savings_goal = Column(DECIMAL(10, 2), default=0)
    risk_tolerance = Column(String(20), default='medium')  # low, medium, high
    financial_goals = Column(Text)  # JSON string for array storage
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    user = relationship("User")

class Wallet(Base):
    __tablename__ = "wallets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    balance = Column(DECIMAL(12, 2), default=1000.00)
    currency = Column(String(3), default="USD")
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="wallet")
    transactions = relationship("Transaction", back_populates="wallet")

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    wallet_id = Column(UUID(as_uuid=True), ForeignKey("wallets.id"), nullable=False)
    transaction_type = Column(String(20), nullable=False)  # debit, credit
    amount = Column(DECIMAL(12, 2), nullable=False)
    category = Column(String(50), nullable=False)
    description = Column(Text, nullable=False)
    merchant_name = Column(String(255))
    location_city = Column(String(100))
    location_country = Column(String(100), default="US")
    balance_after = Column(DECIMAL(12, 2), nullable=False)
    transaction_date = Column(DateTime, server_default=func.now())
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="transactions")
    wallet = relationship("Wallet", back_populates="transactions")

class ProductCategory(Base):
    __tablename__ = "product_categories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    transaction_category = Column(String(50), nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    products = relationship("Product", back_populates="category")

class Product(Base):
    __tablename__ = "products"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    price = Column(DECIMAL(10, 2), nullable=False)
    category_id = Column(UUID(as_uuid=True), ForeignKey("product_categories.id"))
    merchant_name = Column(String(255), nullable=False)
    is_available = Column(Boolean, default=True)
    stock_quantity = Column(Integer, default=100)
    image_url = Column(String(500))
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    category = relationship("ProductCategory", back_populates="products")
    purchases = relationship("UserPurchase", back_populates="product")

class UserPurchase(Base):
    __tablename__ = "user_purchases"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    product_id = Column(UUID(as_uuid=True), ForeignKey("products.id"))
    transaction_id = Column(UUID(as_uuid=True), ForeignKey("transactions.id"))
    quantity = Column(Integer, default=1)
    total_amount = Column(DECIMAL(10, 2), nullable=False)
    purchase_date = Column(DateTime, server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="purchases")
    product = relationship("Product", back_populates="purchases")
    transaction = relationship("Transaction")

class RecommendationHistory(Base):
    __tablename__ = "recommendation_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    recommendation_text = Column(Text, nullable=False)
    action_type = Column(String(50), nullable=False)
    confidence_score = Column(DECIMAL(5, 4), nullable=False)  # 0.0000 to 1.0000
    financial_state_snapshot = Column(Text)  # JSON string of financial state
    created_at = Column(DateTime, server_default=func.now())
    feedback_status = Column(String(20), default="pending")  # pending, helpful, not_helpful, implemented

    # Relationships
    user = relationship("User")
    feedback = relationship("RecommendationFeedback", back_populates="recommendation", uselist=False)

class RecommendationFeedback(Base):
    __tablename__ = "recommendation_feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    recommendation_id = Column(UUID(as_uuid=True), ForeignKey("recommendation_history.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    feedback_type = Column(String(20), nullable=False)  # helpful, not_helpful, implemented
    feedback_text = Column(Text)
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    user = relationship("User")
    recommendation = relationship("RecommendationHistory", back_populates="feedback")
