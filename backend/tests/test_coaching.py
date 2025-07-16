import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from app.main import app
from app.database import get_db
from app.models.database import User, RecommendationHistory, RecommendationFeedback
from tests.conftest import TestingSessionLocal, override_get_db
import uuid

# Override the database dependency
app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

class TestCoachingEndpoints:
    """Test suite for AI coaching endpoints"""
    
    def setup_method(self):
        """Set up test data before each test"""
        self.db = TestingSessionLocal()
        
        # Create test user
        self.test_user = User(
            email="test@coaching.com",
            password_hash="hashed_password",
            first_name="Test",
            last_name="User"
        )
        self.db.add(self.test_user)
        self.db.commit()
        self.db.refresh(self.test_user)
        
        # Create test recommendation history
        self.test_recommendation = RecommendationHistory(
            user_id=self.test_user.id,
            recommendation_text="Test recommendation for saving more money",
            action_type="savings_nudge",
            confidence_score=0.85,
            financial_state_snapshot='{"balance": 1000}',
            feedback_status="pending"
        )
        self.db.add(self.test_recommendation)
        self.db.commit()
        self.db.refresh(self.test_recommendation)
    
    def teardown_method(self):
        """Clean up after each test"""
        self.db.close()
    
    def get_auth_headers(self):
        """Get authentication headers for test user"""
        # Login to get token
        login_response = client.post("/auth/login", json={
            "email": "test@coaching.com",
            "password": "password123"
        })
        
        if login_response.status_code == 200:
            token = login_response.json()["access_token"]
            return {"Authorization": f"Bearer {token}"}
        else:
            # Create user first if login fails
            client.post("/auth/register", json={
                "email": "test@coaching.com",
                "password": "password123",
                "first_name": "Test",
                "last_name": "User"
            })
            
            login_response = client.post("/auth/login", json={
                "email": "test@coaching.com",
                "password": "password123"
            })
            
            token = login_response.json()["access_token"]
            return {"Authorization": f"Bearer {token}"}
    
    def test_get_recommendation(self):
        """Test getting AI coaching recommendation"""
        headers = self.get_auth_headers()
        
        response = client.get("/coaching/recommendation", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "id" in data
        assert "recommendation_text" in data
        assert "action_type" in data
        assert "confidence_score" in data
        assert "created_at" in data
        
        # Check data types
        assert isinstance(data["confidence_score"], float)
        assert 0 <= data["confidence_score"] <= 1
    
    def test_get_financial_state(self):
        """Test getting financial state analysis"""
        headers = self.get_auth_headers()
        
        response = client.get("/coaching/financial-state", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "current_balance" in data
        assert "weekly_spending" in data
        assert "savings_rate" in data
        assert "financial_health_score" in data
        
        # Check data types
        assert isinstance(data["current_balance"], (int, float))
        assert isinstance(data["financial_health_score"], (int, float))
        assert 0 <= data["financial_health_score"] <= 100
    
    def test_get_recommendation_history_empty(self):
        """Test getting recommendation history when empty"""
        headers = self.get_auth_headers()
        
        response = client.get("/coaching/history", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "recommendations" in data
        assert "total" in data
        assert "page" in data
        assert "limit" in data
        
        assert isinstance(data["recommendations"], list)
        assert data["total"] >= 0
        assert data["page"] == 1
        assert data["limit"] == 20
    
    def test_get_recommendation_history_with_pagination(self):
        """Test getting recommendation history with pagination"""
        headers = self.get_auth_headers()
        
        # Test with custom limit and offset
        response = client.get("/coaching/history?limit=5&offset=0", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["limit"] == 5
        assert data["page"] == 1
        assert len(data["recommendations"]) <= 5
    
    def test_submit_feedback_helpful(self):
        """Test submitting helpful feedback"""
        headers = self.get_auth_headers()
        
        feedback_data = {
            "recommendation_id": str(self.test_recommendation.id),
            "helpful": True,
            "feedback_text": "This recommendation was very helpful!"
        }
        
        response = client.post("/coaching/feedback", json=feedback_data, headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "feedback_id" in data
        assert data["message"] == "Feedback submitted successfully"
    
    def test_submit_feedback_not_helpful(self):
        """Test submitting not helpful feedback"""
        headers = self.get_auth_headers()
        
        feedback_data = {
            "recommendation_id": str(self.test_recommendation.id),
            "helpful": False,
            "feedback_text": "This recommendation was not relevant to me."
        }
        
        response = client.post("/coaching/feedback", json=feedback_data, headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["message"] == "Feedback submitted successfully"
    
    def test_submit_feedback_legacy_recommendation(self):
        """Test submitting feedback for non-existent recommendation (legacy support)"""
        headers = self.get_auth_headers()
        
        feedback_data = {
            "recommendation_id": str(uuid.uuid4()),  # Random UUID
            "helpful": True,
            "feedback_text": "Legacy recommendation feedback"
        }
        
        response = client.post("/coaching/feedback", json=feedback_data, headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["message"] == "Feedback submitted successfully"
    
    def test_get_available_actions(self):
        """Test getting available coaching actions"""
        response = client.get("/coaching/actions")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "actions" in data
        assert isinstance(data["actions"], list)
        assert len(data["actions"]) > 0
        
        # Check action structure
        action = data["actions"][0]
        assert "id" in action
        assert "type" in action
        assert "name" in action
        assert "description" in action
        assert "priority" in action
        assert "category" in action
    
    def test_get_model_info(self):
        """Test getting model information"""
        response = client.get("/coaching/model-info")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return model metrics
        assert isinstance(data, dict)
    
    def test_check_model_health(self):
        """Test checking model health"""
        response = client.get("/coaching/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return health status
        assert isinstance(data, dict)
    
    def test_get_financial_insights(self):
        """Test getting financial insights"""
        headers = self.get_auth_headers()
        
        response = client.get("/coaching/insights", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "insights" in data
        assert "generated_at" in data
        assert isinstance(data["insights"], list)
    
    def test_unauthorized_access(self):
        """Test that endpoints require authentication"""
        # Test without authentication headers
        response = client.get("/coaching/recommendation")
        assert response.status_code == 401
        
        response = client.get("/coaching/financial-state")
        assert response.status_code == 401
        
        response = client.get("/coaching/history")
        assert response.status_code == 401
        
        response = client.post("/coaching/feedback", json={
            "recommendation_id": "test-id",
            "helpful": True
        })
        assert response.status_code == 401

if __name__ == "__main__":
    pytest.main([__file__])
