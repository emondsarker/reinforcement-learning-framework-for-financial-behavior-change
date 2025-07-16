# FinCoach Full-Stack Implementation Epic

**AI-Powered Financial Wellness Platform with Reinforcement Learning**

---

## 🚀 **IMPLEMENTATION STATUS UPDATE**

### **Phase 3: Backend Development** ✅ **COMPLETED (December 2024)**

**🎯 Current Status:** **FULLY FUNCTIONAL BACKEND DEPLOYED**

- ✅ **Authentication System**: Complete JWT-based auth with registration, login, middleware
- ✅ **Financial Engine**: Virtual wallets, transaction processing, balance management
- ✅ **Product Marketplace**: 8 categories, 40+ products, purchase system
- ✅ **AI Model Integration**: CQL model service with fallback recommendations
- ✅ **Database**: PostgreSQL with all tables, relationships, and seed data
- ✅ **API Documentation**: Swagger UI available at `/docs`
- ✅ **Testing**: 13/13 tests passing (authentication + financial)
- ✅ **Containerization**: Docker + Docker Compose setup
- ✅ **Production Ready**: Environment configs, health checks, monitoring

**🔧 Technical Implementation:**

- **Framework**: FastAPI 0.104.1 with Python 3.9
- **Database**: PostgreSQL 14 with SQLAlchemy ORM
- **Authentication**: JWT tokens with bcrypt password hashing
- **ML Integration**: PyTorch CQL model with 17-dimensional state vectors
- **API Endpoints**: 25+ endpoints across auth, financial, products, coaching
- **Deployment**: Docker containers with health checks

**📊 Test Results:**

```
13 passed, 18 warnings in 6.22s
- Authentication tests: 6/6 passing
- Financial tests: 7/7 passing
- API health checks: All systems operational
```

**🌐 Running Services:**

- Backend API: `http://localhost:8000` (Docker)
- Database: PostgreSQL on port 5432
- API Documentation: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

**📁 Implemented Files:**

```
backend/
├── app/
│   ├── models/ (5 files) - Pydantic & SQLAlchemy models
│   ├── services/ (4 files) - Business logic layer
│   ├── routers/ (4 files) - API endpoints
│   ├── middleware/ (1 file) - JWT authentication
│   ├── main.py - FastAPI application
│   ├── database.py - DB configuration
│   └── seed_data.py - Database seeding
├── tests/ (2 files) - Unit & integration tests
├── Dockerfile - Container configuration
├── requirements.txt - Python dependencies
├── .env.example - Environment template
├── README.md - Comprehensive documentation
└── start.sh - Development startup script
docker-compose.yml - Multi-service orchestration
```

### **Next Phase: Frontend Development** 🔄 **READY TO START**

**🎯 Next Steps for Implementation:**

1. **React Application Setup** - TypeScript, Tailwind CSS, routing
2. **Authentication UI** - Login/register forms with JWT integration
3. **Financial Dashboard** - Wallet display, transaction history, analytics
4. **Product Marketplace** - Browse/purchase interface with real-time updates
5. **AI Coaching Interface** - Recommendation display and interaction
6. **Responsive Design** - Mobile-first approach with modern UI/UX

**📋 Ready-to-Use Backend APIs:**

- `POST /auth/register` - User registration
- `POST /auth/login` - User authentication
- `GET /financial/wallet` - Get user balance
- `POST /financial/transactions` - Create transactions
- `GET /products/categories` - Browse product categories
- `POST /products/purchase` - Purchase products
- `GET /coaching/recommendation` - Get AI recommendations

---

## Epic Overview

Transform FinCoach from a research prototype into a production-ready full-stack application that combines financial simulation, AI coaching, and user engagement through a comprehensive web platform.

### Vision Statement

Build a complete financial coaching platform where users can simulate real financial behaviors through a marketplace system, generate authentic transaction data, and receive personalized AI-driven financial coaching recommendations.

### Success Criteria

- ✅ Users can register, authenticate, and manage their profiles
- ✅ Complete financial simulation system with wallets and transactions
- ✅ Product marketplace for realistic spending simulation
- ✅ AI model serving infrastructure for real-time coaching
- ✅ Responsive React dashboard with financial insights
- ✅ Dockerized deployment on Google Cloud Platform

---

## System Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend │    │  FastAPI Backend │    │  PostgreSQL DB  │
│                 │    │                 │    │                 │
│ • Dashboard     │◄──►│ • Auth APIs     │◄──►│ • Users         │
│ • Marketplace   │    │ • Transaction   │    │ • Transactions  │
│ • Profile       │    │ • AI Coaching   │    │ • Products      │
│ • Auth          │    │ • Model Serving │    │ • Coaching      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  CQL ML Model   │
                       │                 │
                       │ • State Vector  │
                       │ • Q-Learning    │
                       │ • Recommendations│
                       └─────────────────┘
```

### Technology Stack

| Layer              | Technology                         | Purpose                       |
| ------------------ | ---------------------------------- | ----------------------------- |
| **Frontend**       | React 18, TypeScript, Tailwind CSS | User interface and experience |
| **Backend**        | FastAPI, Python 3.9+, Pydantic     | API server and business logic |
| **Database**       | PostgreSQL 14+                     | Data persistence              |
| **ML Serving**     | PyTorch, NumPy                     | AI model inference            |
| **Authentication** | JWT, bcrypt                        | User security                 |
| **Deployment**     | Docker, GCP Cloud Run, Cloud SQL   | Production hosting            |
| **Development**    | Docker Compose                     | Local development             |

---

## Phase 3: Backend Development ✅ **COMPLETED**

### Epic 3.1: Authentication & User Management ✅ **COMPLETED**

#### Story 3.1.1: User Registration System ✅ **COMPLETED**

**As a new user, I want to create an account so that I can access the FinCoach platform.**

**Acceptance Criteria:** ✅ **ALL COMPLETED**

- ✅ User can register with email, password, and basic profile info
- ✅ Email validation and password strength requirements
- ✅ Duplicate email prevention
- ✅ Password hashing with bcrypt
- ✅ JWT token generation upon successful registration

**Implementation Status:** ✅ **FULLY IMPLEMENTED**

- All authentication models created in `backend/app/models/auth.py`
- Authentication service implemented in `backend/app/services/auth_service.py`
- JWT middleware implemented in `backend/app/middleware/auth_middleware.py`
- Authentication endpoints implemented in `backend/app/routers/auth.py`
- Database models created in `backend/app/models/database.py`
- All tests passing (6/6 authentication tests)

**Technical Tasks:**

**Task 3.1.1.1: Database Schema Design**

```sql
-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    date_of_birth DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- User profiles table
CREATE TABLE user_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    monthly_income DECIMAL(10,2) DEFAULT 0,
    savings_goal DECIMAL(10,2) DEFAULT 0,
    risk_tolerance VARCHAR(20) DEFAULT 'medium', -- low, medium, high
    financial_goals TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Task 3.1.1.2: FastAPI Authentication Models**

```python
# File: backend/app/models/auth.py
from pydantic import BaseModel, EmailStr, validator
from typing import Optional
from datetime import date

class UserRegistration(BaseModel):
    email: EmailStr
    password: str
    first_name: str
    last_name: str
    date_of_birth: Optional[date] = None

    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    first_name: str
    last_name: str
    created_at: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse
```

**Task 3.1.1.3: Authentication Service**

```python
# File: backend/app/services/auth_service.py
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import os

class AuthService:
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.secret_key = os.getenv("JWT_SECRET_KEY")
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30

    def hash_password(self, password: str) -> str:
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return self.pwd_context.verify(plain_password, hashed_password)

    def create_access_token(self, data: dict) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
```

**Task 3.1.1.4: Authentication Endpoints**

```python
# File: backend/app/routers/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.models.auth import UserRegistration, UserLogin, TokenResponse
from app.services.auth_service import AuthService
from app.database import get_db

router = APIRouter(prefix="/auth", tags=["authentication"])
auth_service = AuthService()

@router.post("/register", response_model=TokenResponse)
async def register(user_data: UserRegistration, db: Session = Depends(get_db)):
    # Implementation details for user registration
    pass

@router.post("/login", response_model=TokenResponse)
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    # Implementation details for user login
    pass

@router.post("/logout")
async def logout():
    # Implementation details for logout
    pass
```

#### Story 3.1.2: JWT Authentication Middleware

**As a system, I need to protect API endpoints with JWT authentication.**

**Technical Tasks:**

**Task 3.1.2.1: JWT Middleware Implementation**

```python
# File: backend/app/middleware/auth_middleware.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.database import User

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    # JWT token validation and user retrieval
    pass
```

### Epic 3.2: Financial Simulation Engine

#### Story 3.2.1: Wallet Management System

**As a user, I want to manage my virtual wallet so that I can simulate financial transactions.**

**Technical Tasks:**

**Task 3.2.1.1: Wallet Database Schema**

```sql
-- Wallets table
CREATE TABLE wallets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    balance DECIMAL(12,2) DEFAULT 1000.00, -- Starting balance
    currency VARCHAR(3) DEFAULT 'USD',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Transactions table
CREATE TABLE transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    wallet_id UUID REFERENCES wallets(id) ON DELETE CASCADE,
    transaction_type VARCHAR(20) NOT NULL, -- debit, credit
    amount DECIMAL(12,2) NOT NULL,
    category VARCHAR(50) NOT NULL,
    description TEXT NOT NULL,
    merchant_name VARCHAR(255),
    location_city VARCHAR(100),
    location_country VARCHAR(100) DEFAULT 'US',
    balance_after DECIMAL(12,2) NOT NULL,
    transaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_transactions_user_date ON transactions(user_id, transaction_date);
CREATE INDEX idx_transactions_category ON transactions(category);
```

**Task 3.2.1.2: Transaction Models**

```python
# File: backend/app/models/financial.py
from pydantic import BaseModel, validator
from decimal import Decimal
from datetime import datetime
from typing import Optional
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

class TransactionResponse(BaseModel):
    id: str
    transaction_type: TransactionType
    amount: Decimal
    category: str
    description: str
    merchant_name: Optional[str]
    balance_after: Decimal
    transaction_date: datetime

class WalletResponse(BaseModel):
    id: str
    balance: Decimal
    currency: str
    updated_at: datetime
```

**Task 3.2.1.3: Financial Service**

```python
# File: backend/app/services/financial_service.py
from sqlalchemy.orm import Session
from app.models.database import Wallet, Transaction, User
from app.models.financial import TransactionCreate, TransactionType
from decimal import Decimal
from datetime import datetime

class FinancialService:
    def __init__(self, db: Session):
        self.db = db

    def get_user_wallet(self, user_id: str) -> Wallet:
        wallet = self.db.query(Wallet).filter(Wallet.user_id == user_id).first()
        if not wallet:
            # Create wallet if doesn't exist
            wallet = Wallet(user_id=user_id, balance=Decimal('1000.00'))
            self.db.add(wallet)
            self.db.commit()
        return wallet

    def process_transaction(self, user_id: str, transaction_data: TransactionCreate) -> Transaction:
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
            user_id=user_id,
            wallet_id=wallet.id,
            transaction_type=transaction_type,
            amount=abs(transaction_data.amount),
            category=transaction_data.category,
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
        return transaction

    def get_transaction_history(self, user_id: str, limit: int = 50) -> list[Transaction]:
        return self.db.query(Transaction)\
            .filter(Transaction.user_id == user_id)\
            .order_by(Transaction.transaction_date.desc())\
            .limit(limit)\
            .all()
```

#### Story 3.2.2: Product Marketplace Backend

**As a user, I want to browse and purchase products to generate realistic transaction data.**

**Technical Tasks:**

**Task 3.2.2.1: Product Database Schema**

```sql
-- Product categories table
CREATE TABLE product_categories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    transaction_category VARCHAR(50) NOT NULL, -- Maps to transaction categories
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Products table
CREATE TABLE products (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    price DECIMAL(10,2) NOT NULL,
    category_id UUID REFERENCES product_categories(id),
    merchant_name VARCHAR(255) NOT NULL,
    is_available BOOLEAN DEFAULT TRUE,
    stock_quantity INTEGER DEFAULT 100,
    image_url VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User purchases table
CREATE TABLE user_purchases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    product_id UUID REFERENCES products(id),
    transaction_id UUID REFERENCES transactions(id),
    quantity INTEGER DEFAULT 1,
    total_amount DECIMAL(10,2) NOT NULL,
    purchase_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample product categories
INSERT INTO product_categories (name, description, transaction_category) VALUES
('Grocery Items', 'Food and household essentials', 'groceries'),
('Restaurant Meals', 'Dining and takeout options', 'dine_out'),
('Entertainment', 'Movies, games, and fun activities', 'entertainment'),
('Clothing', 'Apparel and accessories', 'shopping'),
('Electronics', 'Tech gadgets and devices', 'shopping'),
('Health & Wellness', 'Medical and fitness products', 'health'),
('Transportation', 'Travel and commute options', 'transport'),
('Utilities', 'Bills and services', 'bills');
```

**Task 3.2.2.2: Product Models**

```python
# File: backend/app/models/products.py
from pydantic import BaseModel
from decimal import Decimal
from typing import Optional, List
from datetime import datetime

class ProductCategory(BaseModel):
    id: str
    name: str
    description: Optional[str]
    transaction_category: str

class Product(BaseModel):
    id: str
    name: str
    description: Optional[str]
    price: Decimal
    category_id: str
    merchant_name: str
    is_available: bool
    stock_quantity: int
    image_url: Optional[str]

class ProductPurchase(BaseModel):
    product_id: str
    quantity: int = 1

class PurchaseResponse(BaseModel):
    id: str
    product: Product
    quantity: int
    total_amount: Decimal
    transaction_id: str
    purchase_date: datetime
```

**Task 3.2.2.3: Product Service**

```python
# File: backend/app/services/product_service.py
from sqlalchemy.orm import Session
from app.models.database import Product, ProductCategory, UserPurchase
from app.services.financial_service import FinancialService
from app.models.financial import TransactionCreate
from decimal import Decimal

class ProductService:
    def __init__(self, db: Session):
        self.db = db
        self.financial_service = FinancialService(db)

    def get_products_by_category(self, category_id: str) -> List[Product]:
        return self.db.query(Product)\
            .filter(Product.category_id == category_id, Product.is_available == True)\
            .all()

    def purchase_product(self, user_id: str, product_id: str, quantity: int = 1) -> UserPurchase:
        product = self.db.query(Product).filter(Product.id == product_id).first()
        if not product or not product.is_available:
            raise ValueError("Product not available")

        if product.stock_quantity < quantity:
            raise ValueError("Insufficient stock")

        total_amount = product.price * quantity

        # Create transaction (negative amount for purchase)
        transaction_data = TransactionCreate(
            amount=-total_amount,
            category=self._get_transaction_category(product.category_id),
            description=f"Purchase: {product.name} x{quantity}",
            merchant_name=product.merchant_name
        )

        transaction = self.financial_service.process_transaction(user_id, transaction_data)

        # Record purchase
        purchase = UserPurchase(
            user_id=user_id,
            product_id=product_id,
            transaction_id=transaction.id,
            quantity=quantity,
            total_amount=total_amount
        )

        # Update stock
        product.stock_quantity -= quantity

        self.db.add(purchase)
        self.db.commit()
        return purchase
```

### Epic 3.3: AI Model Serving Infrastructure

#### Story 3.3.1: CQL Model Loading and Inference

**As a system, I need to load the trained CQL model and provide inference capabilities.**

**Technical Tasks:**

**Task 3.3.1.1: Model Loading Service**

```python
# File: backend/app/services/ml_service.py
import torch
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.models.database import Transaction, User
import pandas as pd

class FinancialStateVector:
    """Generates state vectors for the CQL model"""

    def __init__(self):
        self.categories = [
            'groceries', 'dine_out', 'entertainment', 'bills',
            'transport', 'shopping', 'health', 'fitness',
            'savings', 'income', 'other'
        ]

    def generate_weekly_state(self, user_id: str, db: Session) -> np.ndarray:
        """Generate weekly financial state vector for a user"""
        # Get last 7 days of transactions
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)

        transactions = db.query(Transaction)\
            .filter(
                Transaction.user_id == user_id,
                Transaction.transaction_date >= start_date,
                Transaction.transaction_date <= end_date
            )\
            .all()

        # Calculate base metrics
        total_spending = sum(t.amount for t in transactions if t.transaction_type == 'debit')
        total_income = sum(t.amount for t in transactions if t.transaction_type == 'credit')
        current_balance = transactions[-1].balance_after if transactions else 0
        transaction_count = len(transactions)

        # Calculate spending by category
        category_spending = {cat: 0 for cat in self.categories}
        for transaction in transactions:
            if transaction.transaction_type == 'debit':
                category = transaction.category.lower().replace(' ', '_')
                if category in category_spending:
                    category_spending[category] += float(transaction.amount)

        # Create state vector
        state_vector = [
            float(current_balance),
            float(total_spending),
            float(total_income),
            float(transaction_count)
        ]

        # Add category spending
        state_vector.extend([category_spending[cat] for cat in self.categories])

        # Add derived metrics
        savings_rate = (total_income - total_spending) / max(total_income, 1)
        spending_velocity = total_spending / 7  # Daily average

        state_vector.extend([savings_rate, spending_velocity])

        return np.array(state_vector, dtype=np.float32)

class CQLModelService:
    """Service for loading and running CQL model inference"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.state_generator = FinancialStateVector()
        self.action_meanings = {
            0: "continue_current_behavior",
            1: "spending_alert",
            2: "budget_suggestion",
            3: "savings_nudge",
            4: "positive_reinforcement"
        }
        self.load_model()

    def load_model(self):
        """Load the trained CQL model"""
        try:
            # Recreate model architecture (must match training)
            from app.models.ml_models import QNetwork

            # Determine dimensions (should match training data)
            state_dim = 17  # Based on FinancialStateVector output
            action_dim = 5   # Number of coaching actions

            self.model = QNetwork(state_dim, action_dim)
            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
            self.model.eval()

        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def get_recommendation(self, user_id: str, db: Session) -> Dict:
        """Get AI coaching recommendation for a user"""
        if not self.model:
            return {"error": "Model not loaded"}

        try:
            # Generate state vector
            state_vector = self.state_generator.generate_weekly_state(user_id, db)

            # Run inference
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
                q_values = self.model(state_tensor)
                recommended_action = torch.argmax(q_values, dim=1).item()

            # Generate human-readable recommendation
            recommendation = self._generate_recommendation_text(
                recommended_action, state_vector
            )

            return {
                "action_id": recommended_action,
                "action_type": self.action_meanings[recommended_action],
                "recommendation": recommendation,
                "confidence": float(torch.max(q_values).item()),
                "state_summary": self._summarize_state(state_vector)
            }

        except Exception as e:
            return {"error": f"Inference error: {str(e)}"}

    def _generate_recommendation_text(self, action: int, state: np.ndarray) -> str:
        """Generate human-readable recommendation text"""
        current_balance = state[0]
        total_spending = state[1]
        total_income = state[2]

        recommendations = {
            0: "Keep up your current financial habits! You're doing well.",
            1: f"Consider reducing your spending this week. You've spent ${total_spending:.2f} recently.",
            2: f"Based on your income of ${total_income:.2f}, consider setting a weekly budget of ${total_income * 0.7:.2f}.",
            3: f"Great opportunity to save! With a balance of ${current_balance:.2f}, consider setting aside ${current_balance * 0.1:.2f} for savings.",
            4: f"Excellent financial discipline! Your balance of ${current_balance:.2f} shows good money management."
        }

        return recommendations.get(action, "Continue monitoring your financial health.")

    def _summarize_state(self, state: np.ndarray) -> Dict:
        """Summarize the current financial state"""
        return {
            "current_balance": float(state[0]),
            "weekly_spending": float(state[1]),
            "weekly_income": float(state[2]),
            "transaction_count": int(state[3]),
            "savings_rate": float(state[-2]),
            "daily_spending_avg": float(state[-1])
        }
```

**Task 3.3.1.2: ML Model Architecture**

```python
# File: backend/app/models/ml_models.py
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    """Q-Network architecture matching the training notebook"""

    def __init__(self, state_dim: int, action_dim: int):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        return self.network(state)
```

**Task 3.3.1.3: AI Coaching Endpoints**

```python
# File: backend/app/routers/coaching.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.services.ml_service import CQLModelService
from app.middleware.auth_middleware import get_current_user
from app.database import get_db
from app.models.database import User

router = APIRouter(prefix="/coaching", tags=["ai-coaching"])

# Initialize ML service (in production, this would be a singleton)
ml_service = CQLModelService("models/cql_fincoach_model.pth")

@router.get("/recommendation")
async def get_coaching_recommendation(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get personalized AI coaching recommendation"""
    recommendation = ml_service.get_recommendation(str(current_user.id), db)
    return recommendation

@router.get("/financial-health")
async def get_financial_health_summary(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get comprehensive financial health analysis"""
    state_vector = ml_service.state_generator.generate_weekly_state(
        str(current_user.id), db
    )
    summary = ml_service._summarize_state(state_vector)
    return summary
```

### Epic 3.4: API Integration and Documentation

#### Story 3.4.1: Complete API Endpoints

**As a frontend developer, I need comprehensive API endpoints to build the user interface.**

**Technical Tasks:**

**Task 3.4.1.1: Financial Endpoints**

```python
# File: backend/app/routers/financial.py
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
from app.services.financial_service import FinancialService
from app.models.financial import TransactionCreate, TransactionResponse, WalletResponse
from app.middleware.auth_middleware import get_current_user
from app.database import get_db

router = APIRouter(prefix="/financial", tags=["financial"])

@router.get("/wallet", response_model=WalletResponse)
async def get_wallet(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's wallet information"""
    service = FinancialService(db)
    wallet = service.get_user_wallet(str(current_user.id))
    return wallet

@router.post("/transactions", response_model=TransactionResponse)
async def create_transaction(
    transaction_data: TransactionCreate,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new transaction"""
    service = FinancialService(db)
    transaction = service.process_transaction(str(current_user.id), transaction_data)
    return transaction

@router.get("/transactions", response_model=List[TransactionResponse])
async def get_transactions(
    limit: int = Query(50, le=100),
    days: Optional[int] = Query(None, description="Filter by days back"),
    category: Optional[str] = Query(None),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's transaction history with filters"""
    service = FinancialService(db)
    transactions = service.get_transaction_history(
        str(current_user.id), limit, days, category
    )
    return transactions

@router.get("/analytics/spending-by-category")
async def get_spending_analytics(
    days: int = Query(30, description="Days to analyze"),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get spending analytics by category"""
    service = FinancialService(db)
    analytics = service.get_spending_analytics(str(current_user.id), days)
    return analytics
```

**Task 3.4.1.2: Product Endpoints**

```python
# File: backend/app/routers/products.py
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from app.services.product_service import ProductService
from app.models.products import Product, ProductCategory, ProductPurchase, PurchaseResponse
from app.middleware.auth_middleware import get_current_user
from app.database import get_db

router = APIRouter(prefix="/products", tags=["products"])

@router.get("/categories", response_model=List[ProductCategory])
async def get_product_categories(db: Session = Depends(get_db)):
    """Get all product categories"""
    service = ProductService(db)
    return service.get_all_categories()

@router.get("/", response_model=List[Product])
async def get_products(
    category_id: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(20, le=100),
    db: Session = Depends(get_db)
):
    """Get products with optional filtering"""
    service = ProductService(db)
    return service.get_products(category_id, search, limit)

@router.post("/purchase", response_model=PurchaseResponse)
async def purchase_product(
    purchase_data: ProductPurchase,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Purchase a product"""
    service = ProductService(db)
    purchase = service.purchase_product(
        str(current_user.id),
        purchase_data.product_id,
        purchase_data.quantity
    )
    return purchase

@router.get("/purchases", response_model=List[PurchaseResponse])
async def get_user_purchases(
    limit: int = Query(20, le=100),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's purchase history"""
    service = ProductService(db)
    return service.get_user_purchases(str(current_user.id), limit)
```

**Task 3.4.1.3: FastAPI Application Setup**

```python
# File: backend/app/main.py
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from app.routers import auth, financial, products, coaching
from app.database import engine, Base
import os

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="FinCoach API",
    description="AI-Powered Financial Wellness Platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(financial.router)
app.include_router(products.router)
app.include_router(coaching.router)

@app.get("/")
async def root():
    return {"message": "FinCoach API is running!", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "fincoach-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Task 3.4.1.4: Database Configuration**

```python
# File: backend/app/database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://fincoach:password@localhost:5432/fincoach_db"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

**Task 3.4.1.5: Database Models**

```python
# File: backend/app/models/database.py
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
    transaction_type = Column(String(20), nullable=False)
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
```

---

## Phase 4: Frontend Development

### Epic 4.1: React Application Setup

#### Story 4.1.1: Project Initialization and Configuration

**As a developer, I need a properly configured React application with TypeScript and Tailwind CSS.**

**Technical Tasks:**

**Task 4.1.1.1: Project Setup**

```bash
# Create React app with TypeScript
npx create-react-app frontend --template typescript
cd frontend

# Install additional dependencies
npm install axios react-router-dom @types/react-router-dom
npm install @headlessui/react @heroicons/react
npm install recharts date-fns
npm install react-hook-form @hookform/resolvers yup
npm install react-query

# Install Tailwind CSS
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

**Task 4.1.1.2: Tailwind Configuration**

```javascript
// File: frontend/tailwind.config.js
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: {
          50: "#eff6ff",
          500: "#3b82f6",
          600: "#2563eb",
          700: "#1d4ed8",
        },
        success: {
          50: "#f0fdf4",
          500: "#22c55e",
          600: "#16a34a",
        },
        warning: {
          50: "#fffbeb",
          500: "#f59e0b",
          600: "#d97706",
        },
        danger: {
          50: "#fef2f2",
          500: "#ef4444",
          600: "#dc2626",
        },
      },
    },
  },
  plugins: [],
};
```

**Task 4.1.1.3: API Client Setup**

```typescript
// File: frontend/src/services/api.ts
import axios, { AxiosInstance, AxiosRequestConfig } from "axios";

const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        "Content-Type": "application/json",
      },
    });

    // Request interceptor to add auth token
    this.client.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem("access_token");
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          localStorage.removeItem("access_token");
          window.location.href = "/login";
        }
        return Promise.reject(error);
      }
    );
  }

  async get<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.get<T>(url, config);
    return response.data;
  }

  async post<T>(
    url: string,
    data?: any,
    config?: AxiosRequestConfig
  ): Promise<T> {
    const response = await this.client.post<T>(url, data, config);
    return response.data;
  }

  async put<T>(
    url: string,
    data?: any,
    config?: AxiosRequestConfig
  ): Promise<T> {
    const response = await this.client.put<T>(url, data, config);
    return response.data;
  }

  async delete<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.delete<T>(url, config);
    return response.data;
  }
}

export const apiClient = new ApiClient();
```

### Epic 4.2: Authentication System

#### Story 4.2.1: User Registration and Login

**As a user, I want to register and login to access the FinCoach platform.**

**Technical Tasks:**

**Task 4.2.1.1: Authentication Types**

```typescript
// File: frontend/src/types/auth.ts
export interface User {
  id: string;
  email: string;
  first_name: string;
  last_name: string;
  created_at: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  password: string;
  first_name: string;
  last_name: string;
  date_of_birth?: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface AuthContextType {
  user: User | null;
  login: (credentials: LoginRequest) => Promise<void>;
  register: (userData: RegisterRequest) => Promise<void>;
  logout: () => void;
  isLoading: boolean;
  isAuthenticated: boolean;
}
```

**Task 4.2.1.2: Authentication Service**

```typescript
// File: frontend/src/services/authService.ts
import { apiClient } from "./api";
import { LoginRequest, RegisterRequest, AuthResponse } from "../types/auth";

export class AuthService {
  async login(credentials: LoginRequest): Promise<AuthResponse> {
    return apiClient.post<AuthResponse>("/auth/login", credentials);
  }

  async register(userData: RegisterRequest): Promise<AuthResponse> {
    return apiClient.post<AuthResponse>("/auth/register", userData);
  }

  async logout(): Promise<void> {
    return apiClient.post("/auth/logout");
  }

  getStoredToken(): string | null {
    return localStorage.getItem("access_token");
  }

  setToken(token: string): void {
    localStorage.setItem("access_token", token);
  }

  removeToken(): void {
    localStorage.removeItem("access_token");
  }
}

export const authService = new AuthService();
```

**Task 4.2.1.3: Authentication Context**

```typescript
// File: frontend/src/contexts/AuthContext.tsx
import React, {
  createContext,
  useContext,
  useEffect,
  useState,
  ReactNode,
} from "react";
import {
  User,
  AuthContextType,
  LoginRequest,
  RegisterRequest,
} from "../types/auth";
import { authService } from "../services/authService";

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const token = authService.getStoredToken();
    if (token) {
      // Validate token and get user info
      // This would typically involve a /me endpoint
      setIsLoading(false);
    } else {
      setIsLoading(false);
    }
  }, []);

  const login = async (credentials: LoginRequest): Promise<void> => {
    setIsLoading(true);
    try {
      const response = await authService.login(credentials);
      authService.setToken(response.access_token);
      setUser(response.user);
    } catch (error) {
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const register = async (userData: RegisterRequest): Promise<void> => {
    setIsLoading(true);
    try {
      const response = await authService.register(userData);
      authService.setToken(response.access_token);
      setUser(response.user);
    } catch (error) {
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const logout = (): void => {
    authService.removeToken();
    setUser(null);
  };

  const value: AuthContextType = {
    user,
    login,
    register,
    logout,
    isLoading,
    isAuthenticated: !!user,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
};
```

**Task 4.2.1.4: Login Component**

```typescript
// File: frontend/src/components/auth/LoginForm.tsx
import React, { useState } from "react";
import { useForm } from "react-hook-form";
import { yupResolver } from "@hookform/resolvers/yup";
import * as yup from "yup";
import { useAuth } from "../../contexts/AuthContext";
import { LoginRequest } from "../../types/auth";
import { Link, useNavigate } from "react-router-dom";

const schema = yup.object({
  email: yup.string().email("Invalid email").required("Email is required"),
  password: yup.string().required("Password is required"),
});

export const LoginForm: React.FC = () => {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [error, setError] = useState<string>("");

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<LoginRequest>({
    resolver: yupResolver(schema),
  });

  const onSubmit = async (data: LoginRequest) => {
    try {
      setError("");
      await login(data);
      navigate("/dashboard");
    } catch (err: any) {
      setError(err.response?.data?.detail || "Login failed");
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">
            Sign in to FinCoach
          </h2>
        </div>
        <form className="mt-8 space-y-6" onSubmit={handleSubmit(onSubmit)}>
          {error && (
            <div className="bg-red-50 border border-red-200 text-red-600 px-4 py-3 rounded">
              {error}
            </div>
          )}

          <div>
            <label
              htmlFor="email"
              className="block text-sm font-medium text-gray-700"
            >
              Email address
            </label>
            <input
              {...register("email")}
              type="email"
              className="mt-1 appearance-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-md focus:outline-none focus:ring-primary-500 focus:border-primary-500"
              placeholder="Enter your email"
            />
            {errors.email && (
              <p className="mt-1 text-sm text-red-600">
                {errors.email.message}
              </p>
            )}
          </div>

          <div>
            <label
              htmlFor="password"
              className="block text-sm font-medium text-gray-700"
            >
              Password
            </label>
            <input
              {...register("password")}
              type="password"
              className="mt-1 appearance-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-md focus:outline-none focus:ring-primary-500 focus:border-primary-500"
              placeholder="Enter your password"
            />
            {errors.password && (
              <p className="mt-1 text-sm text-red-600">
                {errors.password.message}
              </p>
            )}
          </div>

          <div>
            <button
              type="submit"
              disabled={isSubmitting}
              className="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50"
            >
              {isSubmitting ? "Signing in..." : "Sign in"}
            </button>
          </div>

          <div className="text-center">
            <span className="text-sm text-gray-600">
              Don't have an account?{" "}
              <Link
                to="/register"
                className="font-medium text-primary-600 hover:text-primary-500"
              >
                Sign up
              </Link>
            </span>
          </div>
        </form>
      </div>
    </div>
  );
};
```

### Epic 4.3: Financial Dashboard

#### Story 4.3.1: Main Dashboard with Financial Overview

**As a user, I want to see my financial overview and AI coaching recommendations on a dashboard.**

**Technical Tasks:**

**Task 4.3.1.1: Financial Types**

```typescript
// File: frontend/src/types/financial.ts
export interface Wallet {
  id: string;
  balance: number;
  currency: string;
  updated_at: string;
}

export interface Transaction {
  id: string;
  transaction_type: "debit" | "credit";
  amount: number;
  category: string;
  description: string;
  merchant_name?: string;
  balance_after: number;
  transaction_date: string;
}

export interface SpendingAnalytics {
  category: string;
  total_amount: number;
  transaction_count: number;
  percentage: number;
}

export interface AIRecommendation {
  action_id: number;
  action_type: string;
  recommendation: string;
  confidence: number;
  state_summary: {
    current_balance: number;
    weekly_spending: number;
    weekly_income: number;
    transaction_count: number;
    savings_rate: number;
    daily_spending_avg: number;
  };
}
```

**Task 4.3.1.2: Financial Service**

```typescript
// File: frontend/src/services/financialService.ts
import { apiClient } from "./api";
import {
  Wallet,
  Transaction,
  SpendingAnalytics,
  AIRecommendation,
} from "../types/financial";

export class FinancialService {
  async getWallet(): Promise<Wallet> {
    return apiClient.get<Wallet>("/financial/wallet");
  }

  async getTransactions(params?: {
    limit?: number;
    days?: number;
    category?: string;
  }): Promise<Transaction[]> {
    return apiClient.get<Transaction[]>("/financial/transactions", { params });
  }

  async getSpendingAnalytics(days: number = 30): Promise<SpendingAnalytics[]> {
    return apiClient.get<SpendingAnalytics[]>(
      `/financial/analytics/spending-by-category?days=${days}`
    );
  }

  async getAIRecommendation(): Promise<AIRecommendation> {
    return apiClient.get<AIRecommendation>("/coaching/recommendation");
  }

  async createTransaction(data: {
    amount: number;
    category: string;
    description: string;
    merchant_name?: string;
  }): Promise<Transaction> {
    return apiClient.post<Transaction>("/financial/transactions", data);
  }
}

export const financialService = new FinancialService();
```

**Task 4.3.1.3: Dashboard Component**

```typescript
// File: frontend/src/components/dashboard/Dashboard.tsx
import React, { useEffect, useState } from "react";
import { useQuery } from "react-query";
import { financialService } from "../../services/financialService";
import { WalletCard } from "./WalletCard";
import { SpendingChart } from "./SpendingChart";
import { RecentTransactions } from "./RecentTransactions";
import { AICoachingCard } from "./AICoachingCard";
import { QuickActions } from "./QuickActions";

export const Dashboard: React.FC = () => {
  const { data: wallet, isLoading: walletLoading } = useQuery(
    "wallet",
    financialService.getWallet
  );

  const { data: transactions, isLoading: transactionsLoading } = useQuery(
    "transactions",
    () => financialService.getTransactions({ limit: 10 })
  );

  const { data: spendingAnalytics, isLoading: analyticsLoading } = useQuery(
    "spending-analytics",
    () => financialService.getSpendingAnalytics(30)
  );

  const { data: aiRecommendation, isLoading: recommendationLoading } = useQuery(
    "ai-recommendation",
    financialService.getAIRecommendation
  );

  if (walletLoading || transactionsLoading || analyticsLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="px-4 py-6 sm:px-0">
          <h1 className="text-3xl font-bold text-gray-900 mb-8">
            Financial Dashboard
          </h1>

          {/* Top Row - Wallet and AI Coaching */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
            <div className="lg:col-span-2">
              <WalletCard wallet={wallet} />
            </div>
            <div>
              <AICoachingCard
                recommendation={aiRecommendation}
                isLoading={recommendationLoading}
              />
            </div>
          </div>

          {/* Middle Row - Spending Chart and Quick Actions */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
            <div className="lg:col-span-2">
              <SpendingChart
                data={spendingAnalytics}
                isLoading={analyticsLoading}
              />
            </div>
            <div>
              <QuickActions />
            </div>
          </div>

          {/* Bottom Row - Recent Transactions */}
          <div className="grid grid-cols-1 gap-6">
            <RecentTransactions
              transactions={transactions}
              isLoading={transactionsLoading}
            />
          </div>
        </div>
      </div>
    </div>
  );
};
```

**Task 4.3.1.4: Wallet Card Component**

```typescript
// File: frontend/src/components/dashboard/WalletCard.tsx
import React from "react";
import { Wallet } from "../../types/financial";
import {
  CurrencyDollarIcon,
  TrendingUpIcon,
  TrendingDownIcon,
} from "@heroicons/react/24/outline";

interface WalletCardProps {
  wallet?: Wallet;
}

export const WalletCard: React.FC<WalletCardProps> = ({ wallet }) => {
  if (!wallet) {
    return (
      <div className="bg-white overflow-hidden shadow rounded-lg animate-pulse">
        <div className="p-6">
          <div className="h-4 bg-gray-200 rounded w-1/4 mb-4"></div>
          <div className="h-8 bg-gray-200 rounded w-1/2"></div>
        </div>
      </div>
    );
  }

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: wallet.currency,
    }).format(amount);
  };

  return (
    <div className="bg-white overflow-hidden shadow rounded-lg">
      <div className="p-6">
        <div className="flex items-center">
          <div className="flex-shrink-0">
            <CurrencyDollarIcon className="h-8 w-8 text-primary-600" />
          </div>
          <div className="ml-5 w-0 flex-1">
            <dl>
              <dt className="text-sm font-medium text-gray-500 truncate">
                Current Balance
              </dt>
              <dd className="text-3xl font-bold text-gray-900">
                {formatCurrency(wallet.balance)}
              </dd>
            </dl>
          </div>
        </div>

        <div className="mt-6 grid grid-cols-2 gap-4">
          <div className="bg-green-50 p-4 rounded-lg">
            <div className="flex items-center">
              <TrendingUpIcon className="h-5 w-5 text-green-600" />
              <span className="ml-2 text-sm font-medium text-green-600">
                This Week
              </span>
            </div>
            <p className="mt-1 text-lg font-semibold text-green-900">
              +$234.50
            </p>
          </div>

          <div className="bg-red-50 p-4 rounded-lg">
            <div className="flex items-center">
              <TrendingDownIcon className="h-5 w-5 text-red-600" />
              <span className="ml-2 text-sm font-medium text-red-600">
                Spent
              </span>
            </div>
            <p className="mt-1 text-lg font-semibold text-red-900">-$156.30</p>
          </div>
        </div>
      </div>
    </div>
  );
};
```

### Epic 4.4: Product Marketplace Frontend

#### Story 4.4.1: Product Browsing and Purchase Interface

**As a user, I want to browse and purchase products to simulate realistic spending.**

**Technical Tasks:**

**Task 4.4.1.1: Product Types**

```typescript
// File: frontend/src/types/products.ts
export interface ProductCategory {
  id: string;
  name: string;
  description?: string;
  transaction_category: string;
}

export interface Product {
  id: string;
  name: string;
  description?: string;
  price: number;
  category_id: string;
  merchant_name: string;
  is_available: boolean;
  stock_quantity: number;
  image_url?: string;
}

export interface Purchase {
  id: string;
  product: Product;
  quantity: number;
  total_amount: number;
  transaction_id: string;
  purchase_date: string;
}
```

**Task 4.4.1.2: Product Service**

```typescript
// File: frontend/src/services/productService.ts
import { apiClient } from "./api";
import { Product, ProductCategory, Purchase } from "../types/products";

export class ProductService {
  async getCategories(): Promise<ProductCategory[]> {
    return apiClient.get<ProductCategory[]>("/products/categories");
  }

  async getProducts(params?: {
    category_id?: string;
    search?: string;
    limit?: number;
  }): Promise<Product[]> {
    return apiClient.get<Product[]>("/products/", { params });
  }

  async purchaseProduct(
    productId: string,
    quantity: number = 1
  ): Promise<Purchase> {
    return apiClient.post<Purchase>("/products/purchase", {
      product_id: productId,
      quantity,
    });
  }

  async getPurchaseHistory(limit: number = 20): Promise<Purchase[]> {
    return apiClient.get<Purchase[]>(`/products/purchases?limit=${limit}`);
  }
}

export const productService = new ProductService();
```

**Task 4.4.1.3: Marketplace Component**

```typescript
// File: frontend/src/components/marketplace/Marketplace.tsx
import React, { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "react-query";
import { productService } from "../../services/productService";
import { ProductGrid } from "./ProductGrid";
import { CategoryFilter } from "./CategoryFilter";
import { SearchBar } from "./SearchBar";
import { PurchaseModal } from "./PurchaseModal";
import { Product } from "../../types/products";

export const Marketplace: React.FC = () => {
  const [selectedCategory, setSelectedCategory] = useState<string>("");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [selectedProduct, setSelectedProduct] = useState<Product | null>(null);
  const [showPurchaseModal, setShowPurchaseModal] = useState(false);

  const queryClient = useQueryClient();

  const { data: categories } = useQuery(
    "categories",
    productService.getCategories
  );

  const { data: products, isLoading } = useQuery(
    ["products", selectedCategory, searchQuery],
    () =>
      productService.getProducts({
        category_id: selectedCategory || undefined,
        search: searchQuery || undefined,
        limit: 50,
      })
  );

  const purchaseMutation = useMutation(
    ({ productId, quantity }: { productId: string; quantity: number }) =>
      productService.purchaseProduct(productId, quantity),
    {
      onSuccess: () => {
        queryClient.invalidateQueries("wallet");
        queryClient.invalidateQueries("transactions");
        setShowPurchaseModal(false);
        setSelectedProduct(null);
      },
    }
  );

  const handlePurchase = (product: Product) => {
    setSelectedProduct(product);
    setShowPurchaseModal(true);
  };

  const confirmPurchase = (quantity: number) => {
    if (selectedProduct) {
      purchaseMutation.mutate({
        productId: selectedProduct.id,
        quantity,
      });
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="px-4 py-6 sm:px-0">
          <h1 className="text-3xl font-bold text-gray-900 mb-8">Marketplace</h1>

          {/* Filters and Search */}
          <div className="mb-8 space-y-4 lg:space-y-0 lg:flex lg:space-x-4">
            <div className="flex-1">
              <SearchBar value={searchQuery} onChange={setSearchQuery} />
            </div>
            <div className="lg:w-64">
              <CategoryFilter
                categories={categories || []}
                selectedCategory={selectedCategory}
                onCategoryChange={setSelectedCategory}
              />
            </div>
          </div>

          {/* Product Grid */}
          <ProductGrid
            products={products || []}
            isLoading={isLoading}
            onPurchase={handlePurchase}
          />

          {/* Purchase Modal */}
          {showPurchaseModal && selectedProduct && (
            <PurchaseModal
              product={selectedProduct}
              onConfirm={confirmPurchase}
              onCancel={() => {
                setShowPurchaseModal(false);
                setSelectedProduct(null);
              }}
              isLoading={purchaseMutation.isLoading}
            />
          )}
        </div>
      </div>
    </div>
  );
};
```

---

## Phase 5: Deployment & DevOps

### Epic 5.1: Containerization

#### Story 5.1.1: Docker Configuration

**As a developer, I need Docker containers for consistent deployment across environments.**

**Technical Tasks:**

**Task 5.1.1.1: Backend Dockerfile**

```dockerfile
# File: backend/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p models

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Task 5.1.1.2: Frontend Dockerfile**

```dockerfile
# File: frontend/Dockerfile
# Build stage
FROM node:18-alpine as build

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci --only=production

# Copy source code and build
COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built app to nginx
COPY --from=build /app/build /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

**Task 5.1.1.3: Docker Compose for Development**

```yaml
# File: docker-compose.yml
version: "3.8"

services:
  db:
    image: postgres:14
    environment:
      POSTGRES_DB: fincoach_db
      POSTGRES_USER: fincoach
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://fincoach:password@db:5432/fincoach_db
      JWT_SECRET_KEY: your-secret-key-here
    depends_on:
      - db
    volumes:
      - ./backend:/app
      - ./models:/app/models

  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    environment:
      REACT_APP_API_URL: http://localhost:8000
    depends_on:
      - backend

volumes:
  postgres_data:
```

### Epic 5.2: Google Cloud Platform Deployment

#### Story 5.2.1: GCP Infrastructure Setup

**As a DevOps engineer, I need to deploy the application on Google Cloud Platform.**

**Technical Tasks:**

**Task 5.2.1.1: Cloud Run Deployment Configuration**

```yaml
# File: .github/workflows/deploy.yml
name: Deploy to GCP

on:
  push:
    branches: [main]

env:
  PROJECT_ID: fincoach-production
  REGION: us-central1

jobs:
  deploy-backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Cloud SDK
        uses: google-github-actions/setup-gcloud@v0
        with:
          project_id: ${{ env.PROJECT_ID }}
          service_account_key: ${{ secrets.GCP_SA_KEY }}
          export_default_credentials: true

      - name: Configure Docker
        run: gcloud auth configure-docker

      - name: Build and Push Backend
        run: |
          docker build -t gcr.io/$PROJECT_ID/fincoach-backend ./backend
          docker push gcr.io/$PROJECT_ID/fincoach-backend

      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy fincoach-backend \
            --image gcr.io/$PROJECT_ID/fincoach-backend \
            --platform managed \
            --region $REGION \
            --allow-unauthenticated \
            --set-env-vars DATABASE_URL=${{ secrets.DATABASE_URL }} \
            --set-env-vars JWT_SECRET_KEY=${{ secrets.JWT_SECRET_KEY }}

  deploy-frontend:
    runs-on: ubuntu-latest
    needs: deploy-backend
    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: "18"

      - name: Install and Build
        run: |
          cd frontend
          npm ci
          REACT_APP_API_URL=${{ secrets.BACKEND_URL }} npm run build

      - name: Deploy to Firebase Hosting
        uses: FirebaseExtended/action-hosting-deploy@v0
        with:
          repoToken: "${{ secrets.GITHUB_TOKEN }}"
          firebaseServiceAccount: "${{ secrets.FIREBASE_SERVICE_ACCOUNT }}"
          projectId: fincoach-production
```

**Task 5.2.1.2: Terraform Infrastructure**

```hcl
# File: infrastructure/main.tf
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Cloud SQL Database
resource "google_sql_database_instance" "fincoach_db" {
  name             = "fincoach-db-instance"
  database_version = "POSTGRES_14"
  region           = var.region

  settings {
    tier = "db-f1-micro"

    database_flags {
      name  = "log_checkpoints"
      value = "on"
    }
  }

  deletion_protection = false
}

resource "google_sql_database" "database" {
  name     = "fincoach_db"
  instance = google_sql_database_instance.fincoach_db.name
}

resource "google_sql_user" "user" {
  name     = "fincoach"
  instance = google_sql_database_instance.fincoach_db.name
  password = var.db_password
}

# Cloud Run Service
resource "google_cloud_run_service" "backend" {
  name     = "fincoach-backend"
  location = var.region

  template {
    spec {
      containers {
        image = "gcr.io/${var.project_id}/fincoach-backend"

        env {
          name  = "DATABASE_URL"
          value = "postgresql://${google_sql_user.user.name}:${var.db_password}@${google_sql_database_instance.fincoach_db.connection_name}/${google_sql_database.database.name}"
        }

        env {
          name  = "JWT_SECRET_KEY"
          value = var.jwt_secret
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

# IAM
resource "google_cloud_run_service_iam_member" "public" {
  service  = google_cloud_run_service.backend.name
  location = google_cloud_run_service.backend.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Variables
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

variable "jwt_secret" {
  description = "JWT Secret Key"
  type        = string
  sensitive   = true
}
```

---

## Project Management & Implementation Guide

### Implementation Timeline

#### Phase 3: Backend Development (4-6 weeks)

**Week 1-2: Core Infrastructure**

- Database setup and models
- Authentication system
- Basic API structure

**Week 3-4: Financial Engine**

- Wallet management
- Transaction processing
- Product marketplace backend

**Week 5-6: AI Integration**

- Model loading service
- Inference endpoints
- API documentation

#### Phase 4: Frontend Development (4-5 weeks)

**Week 1-2: Foundation**

- React app setup
- Authentication UI
- API integration

**Week 3-4: Core Features**

- Dashboard components
- Marketplace interface
- Transaction management

**Week 5: Polish & Testing**

- UI/UX improvements
- Error handling
- Performance optimization

#### Phase 5: Deployment (1-2 weeks)

**Week 1: Containerization**

- Docker configuration
- Local testing
- CI/CD pipeline

**Week 2: Production Deployment**

- GCP infrastructure
- Monitoring setup
- Performance testing

### Development Best Practices

#### Code Quality Standards

1. **TypeScript/Python Type Safety**

   - Strict type checking enabled
   - Comprehensive interface definitions
   - Runtime validation with Pydantic

2. **Testing Strategy**

   - Unit tests for all services
   - Integration tests for API endpoints
   - E2E tests for critical user flows

3. **Code Review Process**
   - All changes require PR review
   - Automated testing in CI
   - Code coverage minimum 80%

#### Security Considerations

1. **Authentication & Authorization**

   - JWT token expiration
   - Password hashing with bcrypt
   - API rate limiting

2. **Data Protection**

   - Input validation and sanitization
   - SQL injection prevention
   - CORS configuration

3. **Infrastructure Security**
   - Environment variable management
   - Database connection encryption
   - HTTPS enforcement

### Monitoring & Maintenance

#### Application Monitoring

```python
# File: backend/app/monitoring.py
from prometheus_client import Counter, Histogram, generate_latest
import time
import logging

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')

class MonitoringMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = time.time()

            # Process request
            await self.app(scope, receive, send)

            # Record metrics
            duration = time.time() - start_time
            REQUEST_DURATION.observe(duration)
            REQUEST_COUNT.labels(
                method=scope["method"],
                endpoint=scope["path"]
            ).inc()
```

#### Health Check Endpoints

```python
# File: backend/app/routers/health.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.ml_service import CQLModelService

router = APIRouter(prefix="/health", tags=["health"])

@router.get("/")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@router.get("/database")
async def database_health(db: Session = Depends(get_db)):
    try:
        db.execute("SELECT 1")
        return {"status": "healthy", "service": "database"}
    except Exception as e:
        return {"status": "unhealthy", "service": "database", "error": str(e)}

@router.get("/ml-model")
async def model_health():
    try:
        # Check if model is loaded
        ml_service = CQLModelService("models/cql_fincoach_model.pth")
        if ml_service.model is not None:
            return {"status": "healthy", "service": "ml-model"}
        else:
            return {"status": "unhealthy", "service": "ml-model", "error": "Model not loaded"}
    except Exception as e:
        return {"status": "unhealthy", "service": "ml-model", "error": str(e)}
```

### Success Metrics & KPIs

#### Technical Metrics

- **API Response Time**: < 200ms for 95% of requests
- **System Uptime**: > 99.5%
- **Database Query Performance**: < 100ms average
- **Model Inference Time**: < 50ms per recommendation

#### User Engagement Metrics

- **Daily Active Users**: Track user login frequency
- **Transaction Volume**: Monitor simulated transactions
- **Feature Adoption**: Dashboard usage, marketplace purchases
- **AI Recommendation Interaction**: Click-through rates

#### Business Metrics

- **User Retention**: 7-day and 30-day retention rates
- **Session Duration**: Average time spent in application
- **Feature Completion**: Percentage of users completing key flows
- **Error Rates**: Application and API error frequencies

---

## Conclusion

This comprehensive epic provides a complete roadmap for transforming FinCoach from a research prototype into a production-ready full-stack application. The implementation plan covers:

✅ **Backend Development**: Complete FastAPI server with authentication, financial simulation, AI model serving, and comprehensive APIs

✅ **Frontend Development**: Modern React application with TypeScript, responsive design, and intuitive user experience

✅ **AI Integration**: Seamless integration of the trained CQL model for real-time financial coaching recommendations

✅ **Deployment Strategy**: Containerized deployment on Google Cloud Platform with CI/CD automation

✅ **Quality Assurance**: Testing strategies, monitoring, and maintenance procedures

### Next Steps for Implementation

1. **Environment Setup**: Configure development environment with required tools and dependencies
2. **Database Initialization**: Set up PostgreSQL database and run initial migrations
3. **Backend Development**: Follow the detailed task breakdown for Phase 3
4. **Frontend Development**: Implement React components as outlined in Phase 4
5. **Integration Testing**: Ensure seamless communication between all system components
6. **Deployment**: Deploy to GCP following the infrastructure configuration
7. **Monitoring**: Implement health checks and performance monitoring
8. **User Testing**: Conduct thorough testing of all user flows

This epic serves as a comprehensive guide that any development team can follow to successfully implement the FinCoach platform. Each task includes detailed code examples, architectural decisions, and best practices to ensure a robust, scalable, and maintainable application.

**Total Estimated Timeline**: 10-13 weeks for complete implementation
**Team Size**: 2-3 developers (1 backend, 1 frontend, 1 DevOps/full-stack)
**Budget Considerations**: GCP costs, development tools, and potential third-party services

The resulting application will provide users with an engaging financial simulation platform powered by cutting-edge AI technology, delivering personalized coaching recommendations based on reinforcement learning principles.
