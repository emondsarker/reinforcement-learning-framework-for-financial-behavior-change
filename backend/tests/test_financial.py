import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.database import get_db, Base
from decimal import Decimal

# Create test database - use PostgreSQL for testing
SQLALCHEMY_DATABASE_URL = "postgresql://fincoach:password@db:5432/fincoach_db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(scope="module")
def client():
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as c:
        yield c
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def authenticated_user(client):
    """Create and return authenticated user token"""
    import uuid
    unique_email = f"financial-{uuid.uuid4().hex[:8]}@example.com"
    response = client.post("/auth/register", json={
        "email": unique_email,
        "password": "TestPass123",
        "first_name": "Financial",
        "last_name": "User"
    })
    if response.status_code != 201:
        print(f"Registration failed: {response.json()}")
        raise Exception(f"Failed to register user: {response.status_code}")
    return response.json()["access_token"]

def test_get_wallet(client, authenticated_user):
    """Test getting user wallet"""
    response = client.get(
        "/financial/wallet",
        headers={"Authorization": f"Bearer {authenticated_user}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "balance" in data
    assert "currency" in data
    assert data["currency"] == "USD"
    assert float(data["balance"]) == 1000.0  # Default starting balance

def test_create_transaction(client, authenticated_user):
    """Test creating a transaction"""
    transaction_data = {
        "amount": -50.0,
        "category": "groceries",
        "description": "Weekly grocery shopping",
        "merchant_name": "Local Grocery Store"
    }
    
    response = client.post(
        "/financial/transactions",
        json=transaction_data,
        headers={"Authorization": f"Bearer {authenticated_user}"}
    )
    assert response.status_code == 201
    data = response.json()
    assert float(data["amount"]) == 50.0  # Amount is stored as positive
    assert data["transaction_type"] == "debit"
    assert data["category"] == "groceries"
    assert float(data["balance_after"]) == 950.0  # 1000 - 50

def test_insufficient_funds(client, authenticated_user):
    """Test transaction with insufficient funds"""
    transaction_data = {
        "amount": -2000.0,  # More than the 1000 starting balance
        "category": "shopping",
        "description": "Expensive purchase"
    }
    
    response = client.post(
        "/financial/transactions",
        json=transaction_data,
        headers={"Authorization": f"Bearer {authenticated_user}"}
    )
    assert response.status_code == 400

def test_get_transactions(client, authenticated_user):
    """Test getting transaction history"""
    # Create a few transactions first
    transactions = [
        {"amount": -25.0, "category": "groceries", "description": "Groceries 1"},
        {"amount": -15.0, "category": "dine_out", "description": "Lunch"},
        {"amount": 100.0, "category": "income", "description": "Freelance work"}
    ]
    
    for transaction in transactions:
        client.post(
            "/financial/transactions",
            json=transaction,
            headers={"Authorization": f"Bearer {authenticated_user}"}
        )
    
    # Get transaction history
    response = client.get(
        "/financial/transactions",
        headers={"Authorization": f"Bearer {authenticated_user}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    assert data[0]["description"] == "Freelance work"  # Most recent first

def test_spending_analytics(client, authenticated_user):
    """Test spending analytics"""
    # Create some transactions
    transactions = [
        {"amount": -30.0, "category": "groceries", "description": "Groceries 1"},
        {"amount": -20.0, "category": "groceries", "description": "Groceries 2"},
        {"amount": -15.0, "category": "dine_out", "description": "Restaurant"},
    ]
    
    for transaction in transactions:
        client.post(
            "/financial/transactions",
            json=transaction,
            headers={"Authorization": f"Bearer {authenticated_user}"}
        )
    
    # Get analytics
    response = client.get(
        "/financial/analytics/spending-by-category",
        headers={"Authorization": f"Bearer {authenticated_user}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 2  # At least groceries and dine_out
    
    # Find groceries category
    groceries_data = next((item for item in data if item["category"] == "groceries"), None)
    assert groceries_data is not None
    assert float(groceries_data["total_amount"]) == 50.0  # 30 + 20
    assert groceries_data["transaction_count"] == 2

def test_financial_health_summary(client, authenticated_user):
    """Test financial health summary"""
    response = client.get(
        "/financial/health-summary",
        headers={"Authorization": f"Bearer {authenticated_user}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "current_balance" in data
    assert "weekly_spending" in data
    assert "weekly_income" in data
    assert "savings_rate" in data
    assert "daily_spending_avg" in data

def test_transaction_categories(client):
    """Test getting available transaction categories"""
    response = client.get("/financial/categories")
    assert response.status_code == 200
    data = response.json()
    assert "categories" in data
    assert len(data["categories"]) > 0
    
    # Check that groceries category exists
    category_values = [cat["value"] for cat in data["categories"]]
    assert "groceries" in category_values
