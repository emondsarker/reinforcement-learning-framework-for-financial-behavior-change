// AI coaching and ML-related types matching backend models

export interface AIRecommendation {
  action_id: number;
  action_type: string;
  recommendation: string;
  confidence: number;
  state_summary: Record<string, unknown>;
  generated_at: Date;
}

export interface FinancialState {
  current_balance: number;
  weekly_spending: number;
  weekly_income: number;
  transaction_count: number;
  savings_rate: number;
  daily_spending_avg: number;
  category_spending: Record<string, number>;
}

export interface CoachingAction {
  action_id: number;
  action_name: string;
  description: string;
  priority: "low" | "medium" | "high";
  category: "spending" | "saving" | "budgeting" | "general";
}

export interface ModelMetrics {
  model_version: string;
  last_updated: Date;
  inference_count: number;
  average_confidence: number;
  model_size_mb: number;
  inference_time_ms: number;
}

export interface RecommendationHistory {
  user_id: string;
  recommendations: AIRecommendation[];
  total_count: number;
  date_range: {
    start: Date;
    end: Date;
  };
}

// Frontend-specific coaching types
export interface CoachingSession {
  id: string;
  user_id: string;
  session_date: Date;
  recommendations: AIRecommendation[];
  user_feedback: CoachingFeedback[];
  session_summary: string;
  next_session_date?: Date;
}

export interface CoachingFeedback {
  recommendation_id: string;
  feedback_type: "helpful" | "not_helpful" | "implemented" | "ignored";
  user_comment?: string;
  submitted_at: Date;
}

export interface CoachingGoal {
  id: string;
  title: string;
  description: string;
  target_amount?: number;
  target_date?: Date;
  category: "saving" | "spending" | "budgeting" | "investment";
  status: "active" | "completed" | "paused" | "cancelled";
  progress_percentage: number;
  created_at: Date;
  updated_at: Date;
}

export interface CoachingInsight {
  type: "achievement" | "warning" | "suggestion" | "milestone";
  title: string;
  message: string;
  icon?: string;
  action_required: boolean;
  related_goal_id?: string;
  generated_at: Date;
}

export interface CoachingDashboard {
  active_goals: CoachingGoal[];
  recent_recommendations: AIRecommendation[];
  insights: CoachingInsight[];
  progress_summary: {
    goals_completed: number;
    recommendations_followed: number;
    improvement_score: number;
  };
  financial_health_score: number;
}

export interface RecommendationCard {
  recommendation: AIRecommendation;
  isExpanded: boolean;
  userFeedback?: CoachingFeedback;
  relatedGoals: CoachingGoal[];
}

export interface CoachingPreferences {
  notification_frequency: "daily" | "weekly" | "monthly";
  focus_areas: ("spending" | "saving" | "budgeting" | "investment")[];
  risk_tolerance: "conservative" | "moderate" | "aggressive";
  communication_style: "formal" | "casual" | "motivational";
  reminder_enabled: boolean;
}

export interface FinancialHealthScore {
  overall_score: number;
  category_scores: {
    spending_control: number;
    savings_rate: number;
    budget_adherence: number;
    financial_stability: number;
  };
  trend: "improving" | "stable" | "declining";
  last_calculated: Date;
}

export interface CoachingAnalytics {
  recommendations_generated: number;
  recommendations_followed: number;
  average_confidence: number;
  most_common_action_types: string[];
  user_engagement_score: number;
  improvement_metrics: {
    savings_increase: number;
    spending_reduction: number;
    budget_accuracy: number;
  };
}
