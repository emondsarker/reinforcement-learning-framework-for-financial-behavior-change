import api from "./api";
import type {
  AIRecommendation,
  FinancialState,
  CoachingFeedback,
  RecommendationHistory,
  FinancialHealthScore,
  CoachingAnalytics,
} from "../types";

export class CoachingService {
  /**
   * Get current AI recommendation for the user
   */
  async getRecommendation(): Promise<AIRecommendation> {
    const response = await api.get<{
      action_id: number;
      action_type: string;
      recommendation: string;
      confidence: number;
      state_summary: Record<string, unknown>;
    }>("/coaching/recommendation");

    return {
      ...response.data,
      generated_at: new Date(),
    };
  }

  /**
   * Get financial health summary
   */
  async getFinancialHealthSummary(): Promise<FinancialState> {
    const response = await api.get<FinancialState>("/coaching/financial-state");
    return response.data;
  }

  /**
   * Submit feedback for a recommendation
   */
  async submitRecommendationFeedback(
    recommendationId: string,
    feedback: Omit<CoachingFeedback, "submitted_at">
  ): Promise<void> {
    // Map frontend feedback types to backend boolean
    const helpful =
      feedback.feedback_type === "helpful" ||
      feedback.feedback_type === "implemented";

    await api.post("/coaching/feedback", null, {
      params: {
        recommendation_id: recommendationId,
        helpful: helpful,
        feedback_text: feedback.user_comment || null,
      },
    });
  }

  /**
   * Get recommendation history
   * Note: This endpoint doesn't exist in the backend yet, so we'll simulate it
   */
  async getRecommendationHistory(
    limit: number = 20,
    offset: number = 0
  ): Promise<RecommendationHistory> {
    // For now, simulate history with the current recommendation
    try {
      const currentRec = await this.getRecommendation();

      // Create some mock historical data based on the current recommendation
      const mockHistory = Array.from(
        { length: Math.min(limit, 5) },
        (_, i) => ({
          ...currentRec,
          action_id: currentRec.action_id + i,
          generated_at: new Date(Date.now() - i * 24 * 60 * 60 * 1000), // i days ago
          confidence: Math.max(0.5, currentRec.confidence - i * 0.1),
        })
      );

      return {
        user_id: "", // Will be filled by the hook
        recommendations: mockHistory,
        total_count: 5,
        date_range: {
          start: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), // 30 days ago
          end: new Date(),
        },
      };
    } catch (error) {
      // If we can't get current recommendation, return empty history
      return {
        user_id: "",
        recommendations: [],
        total_count: 0,
        date_range: {
          start: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
          end: new Date(),
        },
      };
    }
  }

  /**
   * Get financial health score with breakdown
   */
  async getFinancialHealthScore(): Promise<FinancialHealthScore> {
    const state = await this.getFinancialHealthSummary();

    // Calculate health score based on financial state
    const spendingControlScore = Math.min(
      100,
      Math.max(
        0,
        100 - (state.weekly_spending / Math.max(state.weekly_income, 1)) * 100
      )
    );

    const savingsRateScore = Math.min(
      100,
      Math.max(0, state.savings_rate * 100)
    );

    const stabilityScore = Math.min(
      100,
      Math.max(
        0,
        (state.current_balance / Math.max(state.weekly_spending, 1)) * 10
      )
    );

    const budgetScore = Math.min(
      100,
      Math.max(
        0,
        100 -
          (Math.abs(state.daily_spending_avg - state.weekly_income / 7) /
            (state.weekly_income / 7)) *
            100
      )
    );

    const overallScore = Math.round(
      (spendingControlScore + savingsRateScore + stabilityScore + budgetScore) /
        4
    );

    return {
      overall_score: overallScore,
      category_scores: {
        spending_control: Math.round(spendingControlScore),
        savings_rate: Math.round(savingsRateScore),
        budget_adherence: Math.round(budgetScore),
        financial_stability: Math.round(stabilityScore),
      },
      trend:
        overallScore >= 70
          ? "improving"
          : overallScore >= 50
            ? "stable"
            : "declining",
      last_calculated: new Date(),
    };
  }

  /**
   * Get coaching analytics
   */
  async getCoachingAnalytics(): Promise<CoachingAnalytics> {
    // This would typically come from the backend, but we'll simulate it for now
    const history = await this.getRecommendationHistory(100);

    return {
      recommendations_generated: history.total_count,
      recommendations_followed: Math.round(history.total_count * 0.65), // Simulated
      average_confidence:
        history.recommendations.reduce((sum, rec) => sum + rec.confidence, 0) /
          history.recommendations.length || 0,
      most_common_action_types: this.getMostCommonActionTypes(
        history.recommendations
      ),
      user_engagement_score: 75, // Simulated
      improvement_metrics: {
        savings_increase: 12.5, // Simulated percentage
        spending_reduction: 8.3, // Simulated percentage
        budget_accuracy: 85.2, // Simulated percentage
      },
    };
  }

  /**
   * Helper method to get most common action types
   */
  private getMostCommonActionTypes(
    recommendations: AIRecommendation[]
  ): string[] {
    const actionCounts = recommendations.reduce(
      (counts, rec) => {
        counts[rec.action_type] = (counts[rec.action_type] || 0) + 1;
        return counts;
      },
      {} as Record<string, number>
    );

    return Object.entries(actionCounts)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 3)
      .map(([action]) => action);
  }

  /**
   * Get action type display information
   */
  getActionTypeInfo(actionType: string): {
    label: string;
    color: string;
    icon: string;
    priority: "low" | "medium" | "high";
  } {
    const actionTypeMap = {
      continue_current_behavior: {
        label: "Keep Going",
        color: "green",
        icon: "✅",
        priority: "low" as const,
      },
      spending_alert: {
        label: "Spending Alert",
        color: "red",
        icon: "⚠️",
        priority: "high" as const,
      },
      budget_suggestion: {
        label: "Budget Tip",
        color: "blue",
        icon: "💡",
        priority: "medium" as const,
      },
      savings_nudge: {
        label: "Save More",
        color: "purple",
        icon: "💰",
        priority: "medium" as const,
      },
      positive_reinforcement: {
        label: "Great Job!",
        color: "green",
        icon: "🎉",
        priority: "low" as const,
      },
    };

    return (
      actionTypeMap[actionType as keyof typeof actionTypeMap] || {
        label: "General Advice",
        color: "gray",
        icon: "💬",
        priority: "low" as const,
      }
    );
  }
}

export const coachingService = new CoachingService();
