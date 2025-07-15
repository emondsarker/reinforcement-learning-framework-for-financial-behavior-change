import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.database import get_db, Base
import tempfile
import os

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

def test_register_user(client):
    """Test user registration"""
    response = client.post("/auth/register", json={
        "email": "test@example.com",
        "password": "TestPass123",
        "first_name": "Test",
        "last_name": "User"
    })
    assert response.status_code == 201
    data = response.json()
    assert "access_token" in data
    assert data["user"]["email"] == "test@example.com"

def test_register_duplicate_email(client):
    """Test registration with duplicate email"""
    # First registration
    client.post("/auth/register", json={
        "email": "duplicate@example.com",
        "password": "TestPass123",
        "first_name": "Test",
        "last_name": "User"
    })
    
    # Second registration with same email
    response = client.post("/auth/register", json={
        "email": "duplicate@example.com",
        "password": "TestPass123",
        "first_name": "Test2",
        "last_name": "User2"
    })
    assert response.status_code == 400

def test_login_user(client):
    """Test user login"""
    # Register user first
    client.post("/auth/register", json={
        "email": "login@example.com",
        "password": "TestPass123",
        "first_name": "Login",
        "last_name": "User"
    })
    
    # Login
    response = client.post("/auth/login", json={
        "email": "login@example.com",
        "password": "TestPass123"
    })
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["user"]["email"] == "login@example.com"

def test_login_invalid_credentials(client):
    """Test login with invalid credentials"""
    response = client.post("/auth/login", json={
        "email": "nonexistent@example.com",
        "password": "wrongpassword"
    })
    assert response.status_code == 401

def test_get_current_user(client):
    """Test getting current user info"""
    # Register and login
    register_response = client.post("/auth/register", json={
        "email": "current@example.com",
        "password": "TestPass123",
        "first_name": "Current",
        "last_name": "User"
    })
    token = register_response.json()["access_token"]
    
    # Get current user
    response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "current@example.com"

def test_unauthorized_access(client):
    """Test accessing protected endpoint without token"""
    response = client.get("/auth/me")
    assert response.status_code == 403  # Forbidden due to missing token
