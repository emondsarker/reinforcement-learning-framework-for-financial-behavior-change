import torch
import numpy as np
import joblib
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.models.database import Transaction, User, UserSegment, PredictionCache, BehavioralEvent
from app.models.ml_models import (
    QNetwork, AIRecommendation, FinancialState, EnhancedFinancialState,
    UserSegmentInfo, SpendingPrediction, GoalAchievementProbability,
    BehavioralInsight, EnhancedAIRecommendation
)
import pandas as pd
import os
import logging
from sklearn.preprocessing import StandardScaler

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


class EnhancedFinancialStateVector:
    """Generates enhanced 35-dimensional state vectors for advanced ML models"""

    def __init__(self):
        self.categories = [
            'groceries', 'dine_out', 'entertainment', 'bills',
            'transport', 'shopping', 'health', 'fitness',
            'savings', 'income', 'other'
        ]
        # Load feature configuration
        try:
            with open('models/enhanced/feature_config.json', 'r') as f:
                self.feature_config = json.load(f)
                self.feature_names = self.feature_config['feature_names']
        except Exception as e:
            logger.warning(f"Could not load feature config: {e}")
            self.feature_names = []

    def generate_enhanced_state(self, user_id: str, db: Session) -> np.ndarray:
        """Generate enhanced 35-dimensional financial state vector"""
        import uuid
        
        user_uuid = uuid.UUID(user_id)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)  # Extended period for behavioral analysis

        transactions = db.query(Transaction)\
            .filter(
                Transaction.user_id == user_uuid,
                Transaction.transaction_date >= start_date,
                Transaction.transaction_date <= end_date
            )\
            .all()

        if not transactions:
            # Return default state vector if no transactions
            return np.zeros(35, dtype=np.float32)

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([{
            'amount': float(t.amount),
            'transaction_type': t.transaction_type,
            'category': t.category.lower().replace(' ', '_'),
            'timestamp': t.transaction_date,
            'balance_after': float(t.balance_after),
            'merchant': getattr(t, 'merchant', 'unknown')
        } for t in transactions])

        # Calculate base features (first 17 features from original model)
        base_features = self._calculate_base_features(df)
        
        # Calculate enhanced behavioral features
        behavioral_features = self._calculate_behavioral_features(df)
        
        # Combine all features
        enhanced_vector = np.concatenate([base_features, behavioral_features])
        
        # Ensure exactly 35 dimensions
        if len(enhanced_vector) < 35:
            enhanced_vector = np.pad(enhanced_vector, (0, 35 - len(enhanced_vector)), 'constant')
        elif len(enhanced_vector) > 35:
            enhanced_vector = enhanced_vector[:35]
            
        return enhanced_vector.astype(np.float32)

    def _calculate_base_features(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate the original 17 base features"""
        # Filter last 7 days for weekly metrics
        week_ago = datetime.utcnow() - timedelta(days=7)
        weekly_df = df[df['timestamp'] >= week_ago]
        
        current_balance = df['balance_after'].iloc[-1] if len(df) > 0 else 1000.0
        total_spending = weekly_df[weekly_df['transaction_type'] == 'debit']['amount'].sum()
        total_income = weekly_df[weekly_df['transaction_type'] == 'credit']['amount'].sum()
        transaction_count = len(weekly_df)

        # Category spending
        category_spending = {cat: 0.0 for cat in self.categories}
        for _, row in weekly_df[weekly_df['transaction_type'] == 'debit'].iterrows():
            category = row['category']
            if category in category_spending:
                category_spending[category] += row['amount']

        # Base metrics
        savings_rate = (total_income - total_spending) / max(total_income, 1)
        spending_velocity = total_spending / 7

        base_features = [
            current_balance, total_spending, total_income, transaction_count
        ]
        base_features.extend([category_spending[cat] for cat in self.categories])
        base_features.extend([savings_rate, spending_velocity])

        return np.array(base_features, dtype=np.float32)

    def _calculate_behavioral_features(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate enhanced behavioral features (features 18-35)"""
        behavioral_features = []
        
        # Time-based spending patterns
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['weekday'] = pd.to_datetime(df['timestamp']).dt.weekday
        
        spending_df = df[df['transaction_type'] == 'debit']
        
        # Weekday vs weekend spending ratio
        weekday_spending = spending_df[spending_df['weekday'] < 5]['amount'].sum()
        weekend_spending = spending_df[spending_df['weekday'] >= 5]['amount'].sum()
        total_spending = spending_df['amount'].sum()
        
        weekday_ratio = weekday_spending / max(total_spending, 1)
        weekend_ratio = weekend_spending / max(total_spending, 1)
        
        # Morning vs evening spending
        morning_spending = spending_df[spending_df['hour'] < 12]['amount'].sum()
        evening_spending = spending_df[spending_df['hour'] >= 18]['amount'].sum()
        
        morning_ratio = morning_spending / max(total_spending, 1)
        evening_ratio = evening_spending / max(total_spending, 1)
        
        # Spending volatility
        daily_spending = spending_df.groupby(spending_df['timestamp'].dt.date)['amount'].sum()
        spending_volatility = daily_spending.std() if len(daily_spending) > 1 else 0
        
        # Large transaction ratio
        median_amount = spending_df['amount'].median() if len(spending_df) > 0 else 0
        large_transactions = spending_df[spending_df['amount'] > median_amount * 2]
        large_transaction_ratio = len(large_transactions) / max(len(spending_df), 1)
        
        # Impulse buying score (transactions in quick succession)
        spending_df_sorted = spending_df.sort_values('timestamp')
        time_diffs = spending_df_sorted['timestamp'].diff().dt.total_seconds() / 3600  # hours
        quick_transactions = (time_diffs < 1).sum() if len(time_diffs) > 1 else 0
        impulse_score = quick_transactions / max(len(spending_df), 1)
        
        # Category diversity
        unique_categories = spending_df['category'].nunique()
        category_diversity = unique_categories / len(self.categories)
        
        # Spending consistency (coefficient of variation)
        spending_consistency = 1 - (daily_spending.std() / max(daily_spending.mean(), 1)) if len(daily_spending) > 1 else 1
        
        # Budget adherence score (simplified)
        weekly_income = df[df['transaction_type'] == 'credit']['amount'].sum() / 4  # Monthly to weekly
        weekly_spending = total_spending / 4
        budget_adherence = max(0, 1 - (weekly_spending / max(weekly_income * 0.8, 1)))
        
        # Savings consistency
        weekly_savings = []
        for week in range(4):
            week_start = datetime.utcnow() - timedelta(days=(week+1)*7)
            week_end = datetime.utcnow() - timedelta(days=week*7)
            week_df = df[(df['timestamp'] >= week_start) & (df['timestamp'] < week_end)]
            week_income = week_df[week_df['transaction_type'] == 'credit']['amount'].sum()
            week_spend = week_df[week_df['transaction_type'] == 'debit']['amount'].sum()
            weekly_savings.append(week_income - week_spend)
        
        savings_consistency = 1 - (np.std(weekly_savings) / max(np.mean(weekly_savings), 1)) if len(weekly_savings) > 1 else 1
        
        # Emergency fund ratio
        current_balance = df['balance_after'].iloc[-1] if len(df) > 0 else 1000.0
        monthly_expenses = spending_df['amount'].sum()
        emergency_fund_ratio = current_balance / max(monthly_expenses, 1)
        
        # Debt indicator (negative balance periods)
        negative_balance_count = (df['balance_after'] < 0).sum()
        debt_indicator = negative_balance_count / max(len(df), 1)
        
        # Merchant loyalty score
        merchant_counts = spending_df['merchant'].value_counts()
        top_merchant_ratio = merchant_counts.iloc[0] / max(len(spending_df), 1) if len(merchant_counts) > 0 else 0
        
        # Location diversity (simplified as merchant diversity)
        location_diversity = spending_df['merchant'].nunique() / max(len(spending_df), 1)
        
        # Seasonal spending factor (simplified)
        current_month = datetime.utcnow().month
        seasonal_factor = 1.2 if current_month in [11, 12, 1] else 1.0  # Holiday season
        
        # Balance volatility
        balance_volatility = df['balance_after'].std() if len(df) > 1 else 0
        
        # Financial stress indicator
        low_balance_periods = (df['balance_after'] < df['balance_after'].mean() * 0.2).sum()
        financial_stress = low_balance_periods / max(len(df), 1)
        
        behavioral_features = [
            weekday_ratio, weekend_ratio, morning_ratio, evening_ratio,
            spending_volatility, large_transaction_ratio, impulse_score,
            category_diversity, spending_consistency, budget_adherence,
            savings_consistency, emergency_fund_ratio, debt_indicator,
            top_merchant_ratio, location_diversity, seasonal_factor,
            balance_volatility, financial_stress
        ]
        
        return np.array(behavioral_features, dtype=np.float32)

    def get_enhanced_financial_state_summary(self, user_id: str, db: Session) -> EnhancedFinancialState:
        """Get enhanced financial state summary with behavioral insights"""
        import uuid
        
        user_uuid = uuid.UUID(user_id)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)

        transactions = db.query(Transaction)\
            .filter(
                Transaction.user_id == user_uuid,
                Transaction.transaction_date >= start_date,
                Transaction.transaction_date <= end_date
            )\
            .all()

        if not transactions:
            return EnhancedFinancialState(
                current_balance=1000.0,
                weekly_spending=0.0,
                weekly_income=0.0,
                transaction_count=0,
                savings_rate=0.0,
                daily_spending_avg=0.0,
                category_spending={},
                spending_volatility=0.0,
                impulse_buying_score=0.0,
                category_diversity=0.0,
                financial_stress_indicator=0.0
            )

        # Calculate enhanced metrics
        df = pd.DataFrame([{
            'amount': float(t.amount),
            'transaction_type': t.transaction_type,
            'category': t.category,
            'timestamp': t.transaction_date,
            'balance_after': float(t.balance_after)
        } for t in transactions])

        # Base metrics (last 7 days)
        week_ago = datetime.utcnow() - timedelta(days=7)
        weekly_df = df[df['timestamp'] >= week_ago]
        
        current_balance = df['balance_after'].iloc[-1]
        total_spending = weekly_df[weekly_df['transaction_type'] == 'debit']['amount'].sum()
        total_income = weekly_df[weekly_df['transaction_type'] == 'credit']['amount'].sum()
        transaction_count = len(weekly_df)
        
        category_spending = {}
        for _, row in weekly_df[weekly_df['transaction_type'] == 'debit'].iterrows():
            category = row['category']
            category_spending[category] = category_spending.get(category, 0) + row['amount']

        savings_rate = (total_income - total_spending) / max(total_income, 1)
        daily_spending_avg = total_spending / 7

        # Enhanced behavioral metrics
        spending_df = df[df['transaction_type'] == 'debit']
        daily_spending = spending_df.groupby(spending_df['timestamp'].dt.date)['amount'].sum()
        spending_volatility = daily_spending.std() if len(daily_spending) > 1 else 0

        # Impulse buying score
        spending_df_sorted = spending_df.sort_values('timestamp')
        time_diffs = spending_df_sorted['timestamp'].diff().dt.total_seconds() / 3600
        quick_transactions = (time_diffs < 1).sum() if len(time_diffs) > 1 else 0
        impulse_buying_score = quick_transactions / max(len(spending_df), 1)

        # Category diversity
        unique_categories = spending_df['category'].nunique()
        category_diversity = unique_categories / 11  # Total categories

        # Financial stress indicator
        low_balance_periods = (df['balance_after'] < df['balance_after'].mean() * 0.2).sum()
        financial_stress_indicator = low_balance_periods / max(len(df), 1)

        return EnhancedFinancialState(
            current_balance=current_balance,
            weekly_spending=total_spending,
            weekly_income=total_income,
            transaction_count=transaction_count,
            savings_rate=savings_rate,
            daily_spending_avg=daily_spending_avg,
            category_spending=category_spending,
            spending_volatility=spending_volatility,
            impulse_buying_score=impulse_buying_score,
            category_diversity=category_diversity,
            financial_stress_indicator=financial_stress_indicator
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


class EnhancedCQLModelService:
    """Enhanced CQL Model Service with segment-aware recommendations and multi-armed bandit"""

    def __init__(self, models_path: str = "models/enhanced"):
        self.models_path = models_path
        self.enhanced_model = None
        self.bandit_state = None
        self.enhanced_state_generator = EnhancedFinancialStateVector()
        self.segmentation_service = UserSegmentationService(models_path)
        self.action_meanings = {
            0: "continue_current_behavior",
            1: "spending_alert", 
            2: "budget_suggestion",
            3: "savings_nudge",
            4: "positive_reinforcement"
        }
        self.load_enhanced_models()

    def load_enhanced_models(self):
        """Load enhanced CQL model and bandit state"""
        try:
            # Load enhanced CQL model
            enhanced_model_path = os.path.join(self.models_path, "enhanced_cql_model.pth")
            if os.path.exists(enhanced_model_path):
                # Enhanced model uses 35-dimensional state vectors
                state_dim = 35
                action_dim = 5
                self.enhanced_model = QNetwork(state_dim, action_dim)
                self.enhanced_model.load_state_dict(torch.load(enhanced_model_path, map_location='cpu'))
                self.enhanced_model.eval()
                logger.info("Enhanced CQL model loaded successfully")
            else:
                logger.warning(f"Enhanced CQL model not found at {enhanced_model_path}")

            # Load bandit state
            bandit_state_path = os.path.join(self.models_path, "bandit_state.json")
            if os.path.exists(bandit_state_path):
                with open(bandit_state_path, 'r') as f:
                    self.bandit_state = json.load(f)
                logger.info("Multi-armed bandit state loaded successfully")

        except Exception as e:
            logger.error(f"Error loading enhanced models: {e}")
            self.enhanced_model = None
            self.bandit_state = None

    def get_enhanced_recommendation(self, user_id: str, db: Session) -> EnhancedAIRecommendation:
        """Get enhanced AI recommendation with segment awareness"""
        try:
            # Get user segment
            user_segment = self.segmentation_service.classify_user_segment(user_id, db)
            
            # Generate enhanced state vector
            enhanced_state_vector = self.enhanced_state_generator.generate_enhanced_state(user_id, db)
            enhanced_financial_state = self.enhanced_state_generator.get_enhanced_financial_state_summary(user_id, db)

            # Select recommendation strategy using multi-armed bandit
            strategy = self._select_recommendation_strategy(user_segment.segment_id)

            if self.enhanced_model is not None:
                # Run inference with enhanced model
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(enhanced_state_vector).unsqueeze(0)
                    q_values = self.enhanced_model(state_tensor)
                    recommended_action = torch.argmax(q_values, dim=1).item()
                    confidence = float(torch.max(q_values).item())
            else:
                # Fallback to segment-aware rule-based recommendations
                recommended_action, confidence = self._get_segment_aware_fallback(enhanced_financial_state, user_segment)

            # Generate segment-aware recommendation text
            recommendation_text = self._generate_segment_aware_recommendation(
                recommended_action, enhanced_financial_state, user_segment
            )

            return EnhancedAIRecommendation(
                action_id=recommended_action,
                action_type=self.action_meanings[recommended_action],
                recommendation=recommendation_text,
                confidence=confidence,
                state_summary=enhanced_financial_state.dict(),
                user_segment=user_segment,
                strategy_used=strategy,
                generated_at=datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"Error generating enhanced recommendation: {e}")
            # Return basic recommendation on error
            basic_service = CQLModelService()
            basic_rec = basic_service.get_recommendation(user_id, db)
            
            return EnhancedAIRecommendation(
                action_id=basic_rec.action_id,
                action_type=basic_rec.action_type,
                recommendation=basic_rec.recommendation,
                confidence=basic_rec.confidence,
                state_summary=basic_rec.state_summary,
                user_segment=UserSegmentInfo(
                    segment_id=0,
                    segment_name="Unknown",
                    confidence=0.5,
                    characteristics={},
                    assigned_at=datetime.utcnow()
                ),
                strategy_used="fallback",
                generated_at=datetime.utcnow()
            )

    def _select_recommendation_strategy(self, segment_id: int) -> str:
        """Select recommendation strategy using multi-armed bandit"""
        if not self.bandit_state:
            return "default"

        segment_key = f"segment_{segment_id}"
        if segment_key not in self.bandit_state:
            return "default"

        strategies = self.bandit_state[segment_key]
        
        # Epsilon-greedy strategy selection
        epsilon = 0.1
        if np.random.random() < epsilon:
            # Explore: random strategy
            return np.random.choice(list(strategies.keys()))
        else:
            # Exploit: best performing strategy
            best_strategy = max(strategies.items(), key=lambda x: x[1].get('success_rate', 0))
            return best_strategy[0]

    def _get_segment_aware_fallback(self, state: EnhancedFinancialState, segment: UserSegmentInfo) -> Tuple[int, float]:
        """Generate segment-aware fallback recommendations"""
        # Segment-specific recommendation logic
        if segment.segment_name == "Conservative Savers":
            if state.savings_rate > 0.3:
                return 4, 0.9  # positive_reinforcement
            else:
                return 3, 0.8  # savings_nudge
                
        elif segment.segment_name == "Impulse Buyers":
            if state.impulse_buying_score > 0.4:
                return 1, 0.9  # spending_alert
            else:
                return 2, 0.8  # budget_suggestion
                
        elif segment.segment_name == "Goal-Oriented":
            if state.current_balance < 500:
                return 2, 0.8  # budget_suggestion
            else:
                return 3, 0.7  # savings_nudge
                
        elif segment.segment_name == "Budget Conscious":
            if state.spending_volatility > 200:
                return 2, 0.8  # budget_suggestion
            else:
                return 4, 0.7  # positive_reinforcement
                
        else:  # High Spenders
            if state.financial_stress_indicator > 0.3:
                return 1, 0.9  # spending_alert
            else:
                return 2, 0.7  # budget_suggestion

        # Default fallback
        return 0, 0.6

    def _generate_segment_aware_recommendation(self, action: int, state: EnhancedFinancialState, segment: UserSegmentInfo) -> str:
        """Generate segment-aware recommendation text"""
        segment_context = f"As a {segment.segment_name}, "
        
        base_recommendations = {
            0: f"{segment_context}you're maintaining good financial habits! Your current balance is ${state.current_balance:.2f}.",
            
            1: f"{segment_context}⚠️ Spending Alert: Your recent spending pattern shows ${state.weekly_spending:.2f} this week. Consider reviewing your purchases, especially given your tendency towards {segment.segment_name.lower()} behavior.",
            
            2: f"{segment_context}💡 Budget Suggestion: Based on your behavioral profile and weekly income of ${state.weekly_income:.2f}, consider setting a spending limit that aligns with your {segment.segment_name.lower()} tendencies.",
            
            3: f"{segment_context}🎯 Savings Opportunity: Your profile suggests you could benefit from structured savings. Consider setting aside ${state.current_balance * 0.1:.2f} for your goals.",
            
            4: f"{segment_context}🌟 Excellent Work: Your financial discipline aligns perfectly with your behavioral profile! Your savings rate of {state.savings_rate*100:.1f}% is impressive."
        }

        return base_recommendations.get(action, f"{segment_context}continue monitoring your financial health.")

    def update_strategy_performance(self, user_id: str, strategy: str, success: bool, db: Session):
        """Update multi-armed bandit strategy performance"""
        try:
            # Get user segment
            user_segment = self.segmentation_service.classify_user_segment(user_id, db)
            segment_key = f"segment_{user_segment.segment_id}"
            
            if not self.bandit_state:
                self.bandit_state = {}
            
            if segment_key not in self.bandit_state:
                self.bandit_state[segment_key] = {}
            
            if strategy not in self.bandit_state[segment_key]:
                self.bandit_state[segment_key][strategy] = {
                    'total_attempts': 0,
                    'successes': 0,
                    'success_rate': 0.0
                }
            
            # Update strategy performance
            strategy_stats = self.bandit_state[segment_key][strategy]
            strategy_stats['total_attempts'] += 1
            if success:
                strategy_stats['successes'] += 1
            
            strategy_stats['success_rate'] = strategy_stats['successes'] / strategy_stats['total_attempts']
            
            # Save updated bandit state
            bandit_state_path = os.path.join(self.models_path, "bandit_state.json")
            with open(bandit_state_path, 'w') as f:
                json.dump(self.bandit_state, f, indent=2)
                
            logger.info(f"Updated strategy performance: {strategy} for {segment_key}")

        except Exception as e:
            logger.error(f"Error updating strategy performance: {e}")


class SpendingPredictionService:
    """Service for spending prediction using LSTM models"""

    def __init__(self, models_path: str = "models/enhanced"):
        self.models_path = models_path
        self.spending_model = None
        self.spending_scaler = None
        self.enhanced_state_generator = EnhancedFinancialStateVector()
        self.load_models()

    def load_models(self):
        """Load spending prediction models"""
        try:
            # Load spending prediction model
            model_path = os.path.join(self.models_path, "spending_predictor.pth")
            if os.path.exists(model_path):
                self.spending_model = torch.load(model_path, map_location='cpu')
                self.spending_model.eval()
                logger.info("Spending prediction model loaded successfully")

            # Load scaler
            scaler_path = os.path.join(self.models_path, "spending_scaler.pkl")
            if os.path.exists(scaler_path):
                self.spending_scaler = joblib.load(scaler_path)
                logger.info("Spending prediction scaler loaded successfully")

        except Exception as e:
            logger.error(f"Error loading spending prediction models: {e}")
            self.spending_model = None
            self.spending_scaler = None

    def predict_weekly_spending(self, user_id: str, db: Session) -> SpendingPrediction:
        """Predict weekly spending by category"""
        try:
            # Check cache first
            cached_prediction = self._get_cached_prediction(user_id, "spending", db)
            if cached_prediction:
                return cached_prediction

            if self.spending_model is None or self.spending_scaler is None:
                return self._get_fallback_spending_prediction(user_id, db)

            # Generate features for prediction
            features = self._generate_spending_features(user_id, db)
            
            # Scale features
            scaled_features = self.spending_scaler.transform(features.reshape(1, -1))
            
            # Make prediction
            with torch.no_grad():
                features_tensor = torch.FloatTensor(scaled_features)
                prediction = self.spending_model(features_tensor)
                predicted_amounts = prediction.numpy().flatten()

            # Map predictions to categories
            category_predictions = {}
            categories = ['groceries', 'dine_out', 'entertainment', 'bills', 'transport', 'shopping', 'health', 'fitness', 'other']
            
            for i, category in enumerate(categories):
                if i < len(predicted_amounts):
                    category_predictions[category] = max(0, float(predicted_amounts[i]))

            total_predicted = sum(category_predictions.values())
            
            # Calculate confidence based on historical accuracy
            confidence = self._calculate_prediction_confidence(user_id, db)

            prediction_result = SpendingPrediction(
                total_predicted_spending=total_predicted,
                category_breakdown=category_predictions,
                confidence=confidence,
                prediction_period="weekly",
                generated_at=datetime.utcnow()
            )

            # Cache the prediction
            self._cache_prediction(user_id, "spending", prediction_result.dict(), db)

            return prediction_result

        except Exception as e:
            logger.error(f"Error predicting spending: {e}")
            return self._get_fallback_spending_prediction(user_id, db)

    def _generate_spending_features(self, user_id: str, db: Session) -> np.ndarray:
        """Generate features for spending prediction"""
        # Use enhanced state vector as base features
        enhanced_features = self.enhanced_state_generator.generate_enhanced_state(user_id, db)
        
        # Add time-based features
        current_time = datetime.utcnow()
        time_features = [
            current_time.weekday(),  # Day of week
            current_time.month,      # Month
            current_time.day,        # Day of month
        ]
        
        # Combine features
        all_features = np.concatenate([enhanced_features, time_features])
        return all_features

    def _get_fallback_spending_prediction(self, user_id: str, db: Session) -> SpendingPrediction:
        """Generate rule-based spending prediction"""
        try:
            enhanced_state = self.enhanced_state_generator.get_enhanced_financial_state_summary(user_id, db)
            
            # Simple rule-based prediction based on historical averages
            base_spending = enhanced_state.weekly_spending
            
            # Adjust based on behavioral indicators
            if enhanced_state.impulse_buying_score > 0.3:
                base_spending *= 1.2  # Impulse buyers tend to spend more
            
            if enhanced_state.financial_stress_indicator > 0.3:
                base_spending *= 0.8  # Financial stress leads to reduced spending
            
            # Distribute across categories based on historical patterns
            category_breakdown = {}
            for category, amount in enhanced_state.category_spending.items():
                if enhanced_state.weekly_spending > 0:
                    ratio = amount / enhanced_state.weekly_spending
                    category_breakdown[category] = base_spending * ratio

            return SpendingPrediction(
                total_predicted_spending=base_spending,
                category_breakdown=category_breakdown,
                confidence=0.6,  # Lower confidence for rule-based
                prediction_period="weekly",
                generated_at=datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"Error in fallback spending prediction: {e}")
            return SpendingPrediction(
                total_predicted_spending=0.0,
                category_breakdown={},
                confidence=0.5,
                prediction_period="weekly",
                generated_at=datetime.utcnow()
            )

    def _calculate_prediction_confidence(self, user_id: str, db: Session) -> float:
        """Calculate prediction confidence based on historical accuracy"""
        # Simplified confidence calculation
        # In a real implementation, this would compare past predictions with actual spending
        enhanced_state = self.enhanced_state_generator.get_enhanced_financial_state_summary(user_id, db)
        
        # Higher confidence for users with consistent spending patterns
        if enhanced_state.spending_volatility < 100:
            return 0.9
        elif enhanced_state.spending_volatility < 200:
            return 0.7
        else:
            return 0.6

    def _get_cached_prediction(self, user_id: str, prediction_type: str, db: Session):
        """Get cached prediction if available and not expired"""
        try:
            import uuid
            user_uuid = uuid.UUID(user_id)
            
            cached = db.query(PredictionCache)\
                .filter(
                    PredictionCache.user_id == user_uuid,
                    PredictionCache.prediction_type == prediction_type,
                    PredictionCache.expires_at > datetime.utcnow()
                )\
                .first()
            
            if cached:
                prediction_data = json.loads(cached.prediction_data)
                return SpendingPrediction(**prediction_data)
            
            return None

        except Exception as e:
            logger.error(f"Error retrieving cached prediction: {e}")
            return None

    def _cache_prediction(self, user_id: str, prediction_type: str, prediction_data: Dict, db: Session):
        """Cache prediction result"""
        try:
            import uuid
            user_uuid = uuid.UUID(user_id)
            
            # Set expiration (24 hours for spending predictions)
            expires_at = datetime.utcnow() + timedelta(hours=24)
            
            # Check if cache entry exists
            existing_cache = db.query(PredictionCache)\
                .filter(
                    PredictionCache.user_id == user_uuid,
                    PredictionCache.prediction_type == prediction_type
                )\
                .first()
            
            if existing_cache:
                existing_cache.prediction_data = json.dumps(prediction_data)
                existing_cache.expires_at = expires_at
                existing_cache.created_at = datetime.utcnow()
            else:
                new_cache = PredictionCache(
                    user_id=user_uuid,
                    prediction_type=prediction_type,
                    prediction_data=json.dumps(prediction_data),
                    created_at=datetime.utcnow(),
                    expires_at=expires_at
                )
                db.add(new_cache)
            
            db.commit()

        except Exception as e:
            logger.error(f"Error caching prediction: {e}")
            db.rollback()


class GoalPredictionService:
    """Service for goal achievement prediction"""

    def __init__(self, models_path: str = "models/enhanced"):
        self.models_path = models_path
        self.goal_model = None
        self.goal_scaler = None
        self.enhanced_state_generator = EnhancedFinancialStateVector()
        self.load_models()

    def load_models(self):
        """Load goal prediction models"""
        try:
            # Load goal achievement model
            model_path = os.path.join(self.models_path, "goal_achievement_model.pkl")
            if os.path.exists(model_path):
                self.goal_model = joblib.load(model_path)
                logger.info("Goal achievement model loaded successfully")

            # Load scaler
            scaler_path = os.path.join(self.models_path, "goal_scaler.pkl")
            if os.path.exists(scaler_path):
                self.goal_scaler = joblib.load(scaler_path)
                logger.info("Goal achievement scaler loaded successfully")

        except Exception as e:
            logger.error(f"Error loading goal prediction models: {e}")
            self.goal_model = None
            self.goal_scaler = None

    def predict_goal_achievement(self, user_id: str, goal_amount: float, goal_timeline_days: int, db: Session) -> GoalAchievementProbability:
        """Predict probability of achieving a financial goal"""
        try:
            # Check cache first
            cache_key = f"goal_{goal_amount}_{goal_timeline_days}"
            cached_prediction = self._get_cached_prediction(user_id, cache_key, db)
            if cached_prediction:
                return cached_prediction

            if self.goal_model is None or self.goal_scaler is None:
                return self._get_fallback_goal_prediction(user_id, goal_amount, goal_timeline_days, db)

            # Generate features for goal prediction
            features = self._generate_goal_features(user_id, goal_amount, goal_timeline_days, db)
            
            # Scale features
            scaled_features = self.goal_scaler.transform(features.reshape(1, -1))
            
            # Make prediction
            if hasattr(self.goal_model, 'predict_proba'):
                probabilities = self.goal_model.predict_proba(scaled_features)[0]
                achievement_probability = float(probabilities[1])  # Probability of success
            else:
                # For regression models
                prediction = self.goal_model.predict(scaled_features)[0]
                achievement_probability = max(0, min(1, float(prediction)))

            # Identify risk factors
            risk_factors = self._identify_risk_factors(user_id, goal_amount, goal_timeline_days, db)

            prediction_result = GoalAchievementProbability(
                goal_amount=goal_amount,
                timeline_days=goal_timeline_days,
                achievement_probability=achievement_probability,
                risk_factors=risk_factors,
                generated_at=datetime.utcnow()
            )

            # Cache the prediction
            self._cache_prediction(user_id, cache_key, prediction_result.dict(), db)

            return prediction_result

        except Exception as e:
            logger.error(f"Error predicting goal achievement: {e}")
            return self._get_fallback_goal_prediction(user_id, goal_amount, goal_timeline_days, db)

    def _generate_goal_features(self, user_id: str, goal_amount: float, timeline_days: int, db: Session) -> np.ndarray:
        """Generate features for goal achievement prediction"""
        # Get enhanced financial state
        enhanced_features = self.enhanced_state_generator.generate_enhanced_state(user_id, db)
        enhanced_state = self.enhanced_state_generator.get_enhanced_financial_state_summary(user_id, db)
        
        # Goal-specific features
        required_daily_savings = goal_amount / max(timeline_days, 1)
        current_daily_savings = enhanced_state.savings_rate * enhanced_state.weekly_income / 7
        savings_gap = required_daily_savings - current_daily_savings
        
        goal_features = [
            goal_amount,
            timeline_days,
            required_daily_savings,
            savings_gap,
            goal_amount / max(enhanced_state.current_balance, 1),  # Goal to balance ratio
        ]
        
        # Combine all features
        all_features = np.concatenate([enhanced_features, goal_features])
        return all_features

    def _identify_risk_factors(self, user_id: str, goal_amount: float, timeline_days: int, db: Session) -> List[str]:
        """Identify risk factors for goal achievement"""
        risk_factors = []
        enhanced_state = self.enhanced_state_generator.get_enhanced_financial_state_summary(user_id, db)
        
        required_daily_savings = goal_amount / max(timeline_days, 1)
        current_daily_savings = enhanced_state.savings_rate * enhanced_state.weekly_income / 7
        
        if current_daily_savings < required_daily_savings:
            risk_factors.append("Insufficient current savings rate")
        
        if enhanced_state.spending_volatility > 200:
            risk_factors.append("High spending volatility")
        
        if enhanced_state.impulse_buying_score > 0.3:
            risk_factors.append("High impulse buying tendency")
        
        if enhanced_state.financial_stress_indicator > 0.3:
            risk_factors.append("Financial stress indicators present")
        
        if timeline_days < 30:
            risk_factors.append("Very short timeline")
        
        return risk_factors

    def _get_fallback_goal_prediction(self, user_id: str, goal_amount: float, timeline_days: int, db: Session) -> GoalAchievementProbability:
        """Generate rule-based goal achievement prediction"""
        try:
            enhanced_state = self.enhanced_state_generator.get_enhanced_financial_state_summary(user_id, db)
            
            # Simple rule-based prediction
            required_daily_savings = goal_amount / max(timeline_days, 1)
            current_daily_savings = enhanced_state.savings_rate * enhanced_state.weekly_income / 7
            
            if current_daily_savings >= required_daily_savings:
                base_probability = 0.8
            else:
                savings_ratio = current_daily_savings / max(required_daily_savings, 1)
                base_probability = min(0.7, savings_ratio)
            
            # Adjust based on behavioral factors
            if enhanced_state.spending_volatility > 200:
                base_probability *= 0.8
            
            if enhanced_state.impulse_buying_score > 0.3:
                base_probability *= 0.7
            
            if enhanced_state.financial_stress_indicator > 0.3:
                base_probability *= 0.6
            
            risk_factors = self._identify_risk_factors(user_id, goal_amount, timeline_days, db)
            
            return GoalAchievementProbability(
                goal_amount=goal_amount,
                timeline_days=timeline_days,
                achievement_probability=max(0.1, min(0.9, base_probability)),
                risk_factors=risk_factors,
                generated_at=datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"Error in fallback goal prediction: {e}")
            return GoalAchievementProbability(
                goal_amount=goal_amount,
                timeline_days=timeline_days,
                achievement_probability=0.5,
                risk_factors=["Unable to analyze risk factors"],
                generated_at=datetime.utcnow()
            )

    def _get_cached_prediction(self, user_id: str, prediction_type: str, db: Session):
        """Get cached goal prediction if available"""
        try:
            import uuid
            user_uuid = uuid.UUID(user_id)
            
            cached = db.query(PredictionCache)\
                .filter(
                    PredictionCache.user_id == user_uuid,
                    PredictionCache.prediction_type == prediction_type,
                    PredictionCache.expires_at > datetime.utcnow()
                )\
                .first()
            
            if cached:
                prediction_data = json.loads(cached.prediction_data)
                return GoalAchievementProbability(**prediction_data)
            
            return None

        except Exception as e:
            logger.error(f"Error retrieving cached goal prediction: {e}")
            return None

    def _cache_prediction(self, user_id: str, prediction_type: str, prediction_data: Dict, db: Session):
        """Cache goal prediction result"""
        try:
            import uuid
            user_uuid = uuid.UUID(user_id)
            
            # Set expiration (7 days for goal predictions)
            expires_at = datetime.utcnow() + timedelta(days=7)
            
            # Check if cache entry exists
            existing_cache = db.query(PredictionCache)\
                .filter(
                    PredictionCache.user_id == user_uuid,
                    PredictionCache.prediction_type == prediction_type
                )\
                .first()
            
            if existing_cache:
                existing_cache.prediction_data = json.dumps(prediction_data)
                existing_cache.expires_at = expires_at
                existing_cache.created_at = datetime.utcnow()
            else:
                new_cache = PredictionCache(
                    user_id=user_uuid,
                    prediction_type=prediction_type,
                    prediction_data=json.dumps(prediction_data),
                    created_at=datetime.utcnow(),
                    expires_at=expires_at
                )
                db.add(new_cache)
            
            db.commit()

        except Exception as e:
            logger.error(f"Error caching goal prediction: {e}")
            db.rollback()


class UserSegmentationService:
    """Service for user behavioral segmentation using enhanced ML models"""

    def __init__(self, models_path: str = "models/enhanced"):
        self.models_path = models_path
        self.segmentation_model = None
        self.segmentation_scaler = None
        self.segment_profiles = None
        self.enhanced_state_generator = EnhancedFinancialStateVector()
        self.load_models()

    def load_models(self):
        """Load segmentation models and configuration"""
        try:
            # Load segmentation model
            segmentation_model_path = os.path.join(self.models_path, "user_segmentation_model.pkl")
            if os.path.exists(segmentation_model_path):
                self.segmentation_model = joblib.load(segmentation_model_path)
                logger.info("User segmentation model loaded successfully")
            else:
                logger.warning(f"Segmentation model not found at {segmentation_model_path}")

            # Load scaler
            scaler_path = os.path.join(self.models_path, "segmentation_scaler.pkl")
            if os.path.exists(scaler_path):
                self.segmentation_scaler = joblib.load(scaler_path)
                logger.info("Segmentation scaler loaded successfully")

            # Load segment profiles
            profiles_path = os.path.join(self.models_path, "segment_profiles.json")
            if os.path.exists(profiles_path):
                with open(profiles_path, 'r') as f:
                    self.segment_profiles = json.load(f)
                logger.info("Segment profiles loaded successfully")

            # Load model metadata
            metadata_path = os.path.join(self.models_path, "model_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {}

        except Exception as e:
            logger.error(f"Error loading segmentation models: {e}")
            self.segmentation_model = None
            self.segmentation_scaler = None
            self.segment_profiles = None

    def classify_user_segment(self, user_id: str, db: Session) -> UserSegmentInfo:
        """Classify user into behavioral segment"""
        try:
            # Generate enhanced state vector
            state_vector = self.enhanced_state_generator.generate_enhanced_state(user_id, db)
            
            if self.segmentation_model is None or self.segmentation_scaler is None:
                # Fallback to rule-based segmentation
                return self._get_fallback_segment(user_id, db)

            # Scale features
            scaled_features = self.segmentation_scaler.transform(state_vector.reshape(1, -1))
            
            # Predict segment
            segment_id = self.segmentation_model.predict(scaled_features)[0]
            
            # Get prediction probabilities for confidence
            if hasattr(self.segmentation_model, 'predict_proba'):
                probabilities = self.segmentation_model.predict_proba(scaled_features)[0]
                confidence = float(max(probabilities))
            else:
                # For clustering models, calculate distance-based confidence
                cluster_centers = self.segmentation_model.cluster_centers_
                distances = np.linalg.norm(cluster_centers - scaled_features, axis=1)
                confidence = 1.0 / (1.0 + min(distances))

            # Get segment characteristics
            segment_name = self.metadata.get('segment_names', {}).get(str(segment_id), f"Segment {segment_id}")
            characteristics = self.segment_profiles.get(str(segment_id), {}).get('characteristics', {}) if self.segment_profiles else {}

            # Update database with segment assignment
            self._update_user_segment_in_db(user_id, segment_id, segment_name, confidence, db)

            return UserSegmentInfo(
                segment_id=segment_id,
                segment_name=segment_name,
                confidence=confidence,
                characteristics=characteristics,
                assigned_at=datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"Error classifying user segment: {e}")
            return self._get_fallback_segment(user_id, db)

    def get_segment_characteristics(self, segment_id: int) -> Dict:
        """Get characteristics for a specific segment"""
        if self.segment_profiles and str(segment_id) in self.segment_profiles:
            return self.segment_profiles[str(segment_id)]
        return {}

    def get_segment_peers(self, user_id: str, db: Session, limit: int = 10) -> List[Dict]:
        """Find users in the same segment for peer comparison"""
        try:
            import uuid
            user_uuid = uuid.UUID(user_id)
            
            # Get user's current segment
            user_segment = db.query(UserSegment).filter(UserSegment.user_id == user_uuid).first()
            if not user_segment:
                return []

            # Find other users in the same segment
            peer_segments = db.query(UserSegment)\
                .filter(
                    UserSegment.segment_id == user_segment.segment_id,
                    UserSegment.user_id != user_uuid
                )\
                .limit(limit)\
                .all()

            peers = []
            for peer_segment in peer_segments:
                # Get basic financial metrics for peer
                peer_state = self.enhanced_state_generator.get_enhanced_financial_state_summary(
                    str(peer_segment.user_id), db
                )
                
                peers.append({
                    'user_id': str(peer_segment.user_id),
                    'segment_confidence': peer_segment.confidence,
                    'current_balance': peer_state.current_balance,
                    'savings_rate': peer_state.savings_rate,
                    'spending_volatility': peer_state.spending_volatility
                })

            return peers

        except Exception as e:
            logger.error(f"Error finding segment peers: {e}")
            return []

    def update_user_segment(self, user_id: str, db: Session) -> UserSegmentInfo:
        """Refresh user segment assignment based on recent data"""
        return self.classify_user_segment(user_id, db)

    def _get_fallback_segment(self, user_id: str, db: Session) -> UserSegmentInfo:
        """Generate rule-based segment when model is not available"""
        try:
            enhanced_state = self.enhanced_state_generator.get_enhanced_financial_state_summary(user_id, db)
            
            # Simple rule-based segmentation
            if enhanced_state.savings_rate > 0.2:
                segment_id, segment_name = 0, "Conservative Savers"
            elif enhanced_state.impulse_buying_score > 0.3:
                segment_id, segment_name = 1, "Impulse Buyers"
            elif enhanced_state.category_diversity > 0.7:
                segment_id, segment_name = 2, "Goal-Oriented"
            elif enhanced_state.spending_volatility < 100:
                segment_id, segment_name = 3, "Budget Conscious"
            else:
                segment_id, segment_name = 4, "High Spenders"

            confidence = 0.6  # Lower confidence for rule-based classification

            # Update database
            self._update_user_segment_in_db(user_id, segment_id, segment_name, confidence, db)

            return UserSegmentInfo(
                segment_id=segment_id,
                segment_name=segment_name,
                confidence=confidence,
                characteristics={},
                assigned_at=datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"Error in fallback segmentation: {e}")
            return UserSegmentInfo(
                segment_id=0,
                segment_name="Unknown",
                confidence=0.5,
                characteristics={},
                assigned_at=datetime.utcnow()
            )

    def _update_user_segment_in_db(self, user_id: str, segment_id: int, segment_name: str, confidence: float, db: Session):
        """Update user segment assignment in database"""
        try:
            import uuid
            user_uuid = uuid.UUID(user_id)
            
            # Check if user segment already exists
            existing_segment = db.query(UserSegment).filter(UserSegment.user_id == user_uuid).first()
            
            if existing_segment:
                # Update existing segment
                existing_segment.segment_id = segment_id
                existing_segment.segment_name = segment_name
                existing_segment.confidence = confidence
                existing_segment.updated_at = datetime.utcnow()
            else:
                # Create new segment assignment
                new_segment = UserSegment(
                    user_id=user_uuid,
                    segment_id=segment_id,
                    segment_name=segment_name,
                    confidence=confidence,
                    features=json.dumps({}),  # Could store feature vector here
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                db.add(new_segment)
            
            db.commit()
            logger.info(f"Updated segment for user {user_id}: {segment_name} (confidence: {confidence:.2f})")

        except Exception as e:
            logger.error(f"Error updating user segment in database: {e}")
            db.rollback()

    def get_segment_distribution(self, db: Session) -> Dict:
        """Get distribution of users across segments"""
        try:
            segment_counts = db.query(UserSegment.segment_name, db.func.count(UserSegment.id))\
                .group_by(UserSegment.segment_name)\
                .all()
            
            total_users = sum(count for _, count in segment_counts)
            
            distribution = {}
            for segment_name, count in segment_counts:
                distribution[segment_name] = {
                    'count': count,
                    'percentage': (count / max(total_users, 1)) * 100
                }
            
            return distribution

        except Exception as e:
            logger.error(f"Error getting segment distribution: {e}")
            return {}

    def validate_segmentation_health(self) -> Dict:
        """Validate segmentation model health"""
        health_status = {
            "status": "healthy" if self.segmentation_model is not None else "degraded",
            "model_loaded": self.segmentation_model is not None,
            "scaler_loaded": self.segmentation_scaler is not None,
            "profiles_loaded": self.segment_profiles is not None,
            "fallback_available": True,
            "last_check": datetime.utcnow().isoformat()
        }

        if self.segmentation_model is not None:
            try:
                # Test segmentation with dummy data
                dummy_features = np.random.randn(1, 35)
                if self.segmentation_scaler:
                    dummy_features = self.segmentation_scaler.transform(dummy_features)
                
                segment_prediction = self.segmentation_model.predict(dummy_features)
                health_status["inference_test"] = "passed"
                health_status["available_segments"] = len(set(segment_prediction)) if hasattr(self.segmentation_model, 'n_clusters') else "unknown"
                
            except Exception as e:
                health_status["inference_test"] = "failed"
                health_status["error"] = str(e)
                health_status["status"] = "unhealthy"

        return health_status
