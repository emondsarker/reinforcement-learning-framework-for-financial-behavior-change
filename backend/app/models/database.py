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

class BehavioralEvent(Base):
    __tablename__ = "behavioral_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    event_type = Column(String(50), nullable=False)  # page_view, recommendation_interaction, purchase, goal_modification
    event_data = Column(Text)  # JSON string for flexible event data storage
    timestamp = Column(DateTime, server_default=func.now())
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    user = relationship("User")

class UserSegment(Base):
    __tablename__ = "user_segments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, unique=True)
    segment_id = Column(Integer)  # Dynamically determined (1-5)
    segment_name = Column(String(100))
    confidence = Column(DECIMAL(5, 4))  # 0.0000 to 1.0000
    features = Column(Text)  # JSON string of segmentation features
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    user = relationship("User")

class PredictionCache(Base):
    __tablename__ = "prediction_cache"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    prediction_type = Column(String(50), nullable=False)  # spending, goal, recommendation
    prediction_data = Column(Text)  # JSON string of prediction results
    created_at = Column(DateTime, server_default=func.now())
    expires_at = Column(DateTime, nullable=False)  # TTL-based expiration

    # Relationships
    user = relationship("User")

class ModelPerformance(Base):
    __tablename__ = "model_performance"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(100), nullable=False)  # segmentation, recommendation, spending, goal
    accuracy_metric = Column(DECIMAL(6, 4))
    sample_count = Column(Integer, default=0)
    feature_importance = Column(Text)  # JSON string
    created_at = Column(DateTime, server_default=func.now())

class TrainingDataset(Base):
    __tablename__ = "training_datasets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_type = Column(String(50), nullable=False)  # 'segmentation', 'recommendation', 'spending', 'goal'
    data_count = Column(Integer, default=0)
    threshold = Column(Integer, nullable=False)
    last_training_date = Column(DateTime)
    is_ready_for_training = Column(Boolean, default=False)
    threshold_reached = Column(Boolean, default=False)
    data_quality_score = Column(DECIMAL(5, 4))  # 0.0000 to 1.0000
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    training_events = relationship("ModelTrainingEvent", back_populates="training_dataset")
    quality_metrics = relationship("DataQualityMetrics", back_populates="training_dataset")

class ModelTrainingEvent(Base):
    __tablename__ = "model_training_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_type = Column(String(50), nullable=False)  # Type of model being trained
    training_dataset_id = Column(UUID(as_uuid=True), ForeignKey("training_datasets.id"), nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    status = Column(String(20), default='pending')  # 'pending', 'in_progress', 'completed', 'failed', 'deployed'
    performance_metrics = Column(Text)  # JSON string of training results and validation metrics
    model_path = Column(String(500))  # Path to trained model file
    previous_model_path = Column(String(500))  # Path to previous model for rollback
    training_data_size = Column(Integer, default=0)
    validation_score = Column(DECIMAL(6, 4))
    deployment_approved = Column(Boolean, default=False)
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    training_dataset = relationship("TrainingDataset", back_populates="training_events")
    model_versions = relationship("ModelVersion", back_populates="training_event")

class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_type = Column(String(50), nullable=False)
    version = Column(String(20), nullable=False)  # Semantic version (e.g., "2.1.3")
    model_path = Column(String(500), nullable=False)
    training_event_id = Column(UUID(as_uuid=True), ForeignKey("model_training_events.id"), nullable=False)
    is_active = Column(Boolean, default=False)
    performance_baseline = Column(DECIMAL(6, 4))
    performance_current = Column(DECIMAL(6, 4))
    deployment_date = Column(DateTime)
    rollback_count = Column(Integer, default=0)
    model_metadata = Column(Text)  # JSON string for additional model metadata
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    training_event = relationship("ModelTrainingEvent", back_populates="model_versions")

class DataQualityMetrics(Base):
    __tablename__ = "data_quality_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    training_dataset_id = Column(UUID(as_uuid=True), ForeignKey("training_datasets.id"), nullable=False)
    completeness_score = Column(DECIMAL(5, 4))  # Data completeness percentage
    consistency_score = Column(DECIMAL(5, 4))  # Data consistency score
    validity_score = Column(DECIMAL(5, 4))  # Data validity score
    anomaly_count = Column(Integer, default=0)
    duplicate_count = Column(Integer, default=0)
    missing_values_count = Column(Integer, default=0)
    assessment_date = Column(DateTime, server_default=func.now())
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    training_dataset = relationship("TrainingDataset", back_populates="quality_metrics")
