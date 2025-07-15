import torch
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.models.database import Transaction, User
from app.models.ml_models import QNetwork, AIRecommendation, FinancialState
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

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
        import uuid
        
        # Get last 7 days of transactions
        user_uuid = uuid.UUID(user_id)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)

        transactions = db.query(Transaction)\
            .filter(
                Transaction.user_id == user_uuid,
                Transaction.transaction_date >= start_date,
                Transaction.transaction_date <= end_date
            )\
            .all()

        # Calculate base metrics
        total_spending = sum(float(t.amount) for t in transactions if t.transaction_type == 'debit')
        total_income = sum(float(t.amount) for t in transactions if t.transaction_type == 'credit')
        current_balance = float(transactions[-1].balance_after) if transactions else 1000.0
        transaction_count = len(transactions)

        # Calculate spending by category
        category_spending = {cat: 0.0 for cat in self.categories}
        for transaction in transactions:
            if transaction.transaction_type == 'debit':
                category = transaction.category.lower().replace(' ', '_')
                if category in category_spending:
                    category_spending[category] += float(transaction.amount)

        # Create state vector
        state_vector = [
            current_balance,
            total_spending,
            total_income,
            float(transaction_count)
        ]

        # Add category spending
        state_vector.extend([category_spending[cat] for cat in self.categories])

        # Add derived metrics
        savings_rate = (total_income - total_spending) / max(total_income, 1)
        spending_velocity = total_spending / 7  # Daily average

        state_vector.extend([savings_rate, spending_velocity])

        return np.array(state_vector, dtype=np.float32)

    def get_financial_state_summary(self, user_id: str, db: Session) -> FinancialState:
        """Get detailed financial state summary"""
        import uuid
        
        user_uuid = uuid.UUID(user_id)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)

        transactions = db.query(Transaction)\
            .filter(
                Transaction.user_id == user_uuid,
                Transaction.transaction_date >= start_date,
                Transaction.transaction_date <= end_date
            )\
            .all()

        # Calculate metrics
        total_spending = sum(float(t.amount) for t in transactions if t.transaction_type == 'debit')
        total_income = sum(float(t.amount) for t in transactions if t.transaction_type == 'credit')
        current_balance = float(transactions[-1].balance_after) if transactions else 1000.0
        transaction_count = len(transactions)
        
        # Calculate category spending
        category_spending = {}
        for transaction in transactions:
            if transaction.transaction_type == 'debit':
                category = transaction.category
                category_spending[category] = category_spending.get(category, 0) + float(transaction.amount)

        savings_rate = (total_income - total_spending) / max(total_income, 1)
        daily_spending_avg = total_spending / 7

        return FinancialState(
            current_balance=current_balance,
            weekly_spending=total_spending,
            weekly_income=total_income,
            transaction_count=transaction_count,
            savings_rate=savings_rate,
            daily_spending_avg=daily_spending_avg,
            category_spending=category_spending
        )

class CQLModelService:
    """Service for loading and running CQL model inference"""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or "models/cql_fincoach_model.pth"
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
            if not os.path.exists(self.model_path):
                logger.warning(f"Model file not found at {self.model_path}. Using fallback recommendations.")
                self.model = None
                return

            # Determine dimensions (should match training data)
            state_dim = 17  # Based on FinancialStateVector output
            action_dim = 5   # Number of coaching actions

            self.model = QNetwork(state_dim, action_dim)
            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
            self.model.eval()
            logger.info("CQL model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None

    def get_recommendation(self, user_id: str, db: Session) -> AIRecommendation:
        """Get AI coaching recommendation for a user"""
        try:
            # Generate state vector
            state_vector = self.state_generator.generate_weekly_state(user_id, db)
            financial_state = self.state_generator.get_financial_state_summary(user_id, db)

            if self.model is not None:
                # Run inference with trained model
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
                    q_values = self.model(state_tensor)
                    recommended_action = torch.argmax(q_values, dim=1).item()
                    confidence = float(torch.max(q_values).item())
            else:
                # Fallback rule-based recommendations
                recommended_action, confidence = self._get_fallback_recommendation(financial_state)

            # Generate human-readable recommendation
            recommendation_text = self._generate_recommendation_text(
                recommended_action, financial_state
            )

            return AIRecommendation(
                action_id=recommended_action,
                action_type=self.action_meanings[recommended_action],
                recommendation=recommendation_text,
                confidence=confidence,
                state_summary=financial_state.dict(),
                generated_at=datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"Inference error: {str(e)}")
            # Return default recommendation on error
            return AIRecommendation(
                action_id=0,
                action_type="continue_current_behavior",
                recommendation="Keep monitoring your financial health. We're analyzing your spending patterns.",
                confidence=0.5,
                state_summary={},
                generated_at=datetime.utcnow()
            )

    def _get_fallback_recommendation(self, state: FinancialState) -> tuple[int, float]:
        """Generate rule-based recommendation when model is not available"""
        # Simple rule-based logic
        if state.current_balance < 100:
            return 1, 0.8  # spending_alert
        elif state.savings_rate < 0:
            return 2, 0.7  # budget_suggestion
        elif state.savings_rate > 0.2:
            return 4, 0.9  # positive_reinforcement
        elif state.daily_spending_avg > state.weekly_income / 7:
            return 1, 0.6  # spending_alert
        else:
            return 3, 0.6  # savings_nudge

    def _generate_recommendation_text(self, action: int, state: FinancialState) -> str:
        """Generate human-readable recommendation text"""
        current_balance = state.current_balance
        total_spending = state.weekly_spending
        total_income = state.weekly_income
        savings_rate = state.savings_rate

        recommendations = {
            0: f"You're maintaining good financial habits! Your current balance is ${current_balance:.2f}. Keep up the balanced approach to spending and saving.",
            
            1: f"⚠️ Spending Alert: You've spent ${total_spending:.2f} this week, which is quite high relative to your income of ${total_income:.2f}. Consider reviewing your recent purchases and identifying areas to cut back.",
            
            2: f"💡 Budget Suggestion: Based on your weekly income of ${total_income:.2f}, consider setting a spending limit of ${total_income * 0.7:.2f} per week. This would help you save ${total_income * 0.3:.2f} weekly.",
            
            3: f"🎯 Savings Opportunity: Great job managing your finances! With your current balance of ${current_balance:.2f}, consider setting aside ${current_balance * 0.1:.2f} for your emergency fund or long-term goals.",
            
            4: f"🌟 Excellent Work: Your savings rate of {savings_rate*100:.1f}% shows outstanding financial discipline! You're successfully balancing spending and saving. Consider exploring investment opportunities for your surplus."
        }

        return recommendations.get(action, "Continue monitoring your financial health and making mindful spending decisions.")

    def get_model_metrics(self) -> Dict:
        """Get model performance metrics"""
        return {
            "model_loaded": self.model is not None,
            "model_path": self.model_path,
            "state_dimensions": 17,
            "action_dimensions": 5,
            "available_actions": list(self.action_meanings.values())
        }

    def validate_model_health(self) -> Dict:
        """Validate model health and readiness"""
        health_status = {
            "status": "healthy" if self.model is not None else "degraded",
            "model_loaded": self.model is not None,
            "fallback_available": True,
            "last_check": datetime.utcnow().isoformat()
        }

        if self.model is not None:
            try:
                # Test inference with dummy data
                dummy_state = torch.randn(1, 17)
                with torch.no_grad():
                    output = self.model(dummy_state)
                    health_status["inference_test"] = "passed"
                    health_status["output_shape"] = list(output.shape)
            except Exception as e:
                health_status["inference_test"] = "failed"
                health_status["error"] = str(e)
                health_status["status"] = "unhealthy"

        return health_status
