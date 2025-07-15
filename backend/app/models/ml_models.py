import torch
import torch.nn as nn
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime

class QNetwork(nn.Module):
    """Q-Network architecture matching the training notebook"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.network(state)

class AIRecommendation(BaseModel):
    action_id: int
    action_type: str
    recommendation: str
    confidence: float
    state_summary: Dict
    generated_at: datetime

class FinancialState(BaseModel):
    current_balance: float
    weekly_spending: float
    weekly_income: float
    transaction_count: int
    savings_rate: float
    daily_spending_avg: float
    category_spending: Dict[str, float]

class CoachingAction(BaseModel):
    action_id: int
    action_name: str
    description: str
    priority: str  # low, medium, high
    category: str  # spending, saving, budgeting, general

class ModelMetrics(BaseModel):
    model_version: str
    last_updated: datetime
    inference_count: int
    average_confidence: float
    model_size_mb: float
    inference_time_ms: float

class RecommendationHistory(BaseModel):
    user_id: str
    recommendations: List[AIRecommendation]
    total_count: int
    date_range: Dict[str, datetime]
