import React from "react";
import { useAuth } from "../hooks/useAuth";
import { RecommendationCard } from "../components/coaching/RecommendationCard";
import { FinancialHealthScore } from "../components/coaching/FinancialHealthScore";
import { useRecommendationHistory } from "../hooks/useCoaching";
import {
  SparklesIcon,
  ClockIcon,
  ChartBarIcon,
} from "@heroicons/react/24/outline";

export const CoachingPage: React.FC = () => {
  const { user } = useAuth();
  const { data: history, isLoading: historyLoading } =
    useRecommendationHistory(10);

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="px-4 py-6 sm:px-0">
          {/* Header */}
          <div className="mb-8">
            <div className="flex items-center space-x-3 mb-2">
              <SparklesIcon className="h-8 w-8 text-purple-600" />
              <h1 className="text-3xl font-bold text-gray-900">
                AI Financial Coach
              </h1>
            </div>
            <p className="text-lg text-gray-600">
              Welcome back, {user?.first_name}! Here's your personalized
              financial guidance.
            </p>
          </div>

          {/* Main Content Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Left Column - Current Recommendation */}
            <div className="lg:col-span-2 space-y-8">
              {/* Current Recommendation */}
              <div>
                <h2 className="text-xl font-semibold text-gray-900 mb-4">
                  Current Recommendation
                </h2>
                <RecommendationCard />
              </div>

              {/* Recommendation History */}
              <div>
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xl font-semibold text-gray-900">
                    Recent Recommendations
                  </h2>
                  <div className="flex items-center text-sm text-gray-500">
                    <ClockIcon className="h-4 w-4 mr-1" />
                    Last 10 recommendations
                  </div>
                </div>

                {historyLoading ? (
                  <div className="space-y-4">
                    {[...Array(3)].map((_, i) => (
                      <div
                        key={i}
                        className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 animate-pulse"
                      >
                        <div className="flex items-center space-x-3 mb-3">
                          <div className="w-16 h-6 bg-gray-200 rounded-full"></div>
                          <div className="w-20 h-4 bg-gray-200 rounded"></div>
                        </div>
                        <div className="space-y-2">
                          <div className="h-4 bg-gray-200 rounded w-full"></div>
                          <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : history && history.recommendations.length > 0 ? (
                  <div className="space-y-4">
                    {history.recommendations
                      .slice(1)
                      .map((recommendation, index) => (
                        <div
                          key={`${recommendation.action_id}-${index}`}
                          className="bg-white rounded-lg shadow-sm border border-gray-200 p-4"
                        >
                          <div className="flex items-start justify-between mb-3">
                            <div className="flex items-center space-x-2">
                              <span className="text-lg">
                                {recommendation.action_type === "spending_alert"
                                  ? "⚠️"
                                  : recommendation.action_type ===
                                      "budget_suggestion"
                                    ? "💡"
                                    : recommendation.action_type ===
                                        "savings_nudge"
                                      ? "💰"
                                      : recommendation.action_type ===
                                          "positive_reinforcement"
                                        ? "🎉"
                                        : "✅"}
                              </span>
                              <span className="text-sm font-medium text-gray-700">
                                {recommendation.action_type
                                  .replace(/_/g, " ")
                                  .replace(/\b\w/g, (l) => l.toUpperCase())}
                              </span>
                            </div>
                            <span className="text-xs text-gray-500">
                              {new Date(
                                recommendation.generated_at
                              ).toLocaleDateString()}
                            </span>
                          </div>
                          <p className="text-sm text-gray-900 leading-relaxed">
                            {recommendation.recommendation}
                          </p>
                          <div className="mt-2 flex justify-between items-center">
                            <span className="text-xs text-gray-500">
                              Confidence:{" "}
                              {Math.round(recommendation.confidence * 100)}%
                            </span>
                          </div>
                        </div>
                      ))}
                  </div>
                ) : (
                  <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 text-center">
                    <ClockIcon className="mx-auto h-12 w-12 text-gray-400 mb-4" />
                    <h3 className="text-lg font-medium text-gray-900 mb-2">
                      No History Yet
                    </h3>
                    <p className="text-gray-500">
                      Your recommendation history will appear here as you use
                      the app.
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* Right Column - Health Score and Analytics */}
            <div className="space-y-8">
              {/* Financial Health Score */}
              <div>
                <h2 className="text-xl font-semibold text-gray-900 mb-4">
                  Financial Health
                </h2>
                <FinancialHealthScore />
              </div>

              {/* Quick Stats */}
              <div>
                <div className="flex items-center space-x-2 mb-4">
                  <ChartBarIcon className="h-5 w-5 text-gray-600" />
                  <h2 className="text-xl font-semibold text-gray-900">
                    Quick Stats
                  </h2>
                </div>
                <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                  <div className="space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600">
                        Total Recommendations
                      </span>
                      <span className="text-lg font-semibold text-gray-900">
                        {history?.total_count || 0}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600">This Week</span>
                      <span className="text-lg font-semibold text-purple-600">
                        {history?.recommendations.filter(
                          (r) =>
                            new Date(r.generated_at) >
                            new Date(Date.now() - 7 * 24 * 60 * 60 * 1000)
                        ).length || 0}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600">Most Common</span>
                      <span className="text-sm font-medium text-gray-900">
                        {(() => {
                          if (!history?.recommendations.length) return "None";

                          const counts = history.recommendations.reduce(
                            (acc, rec) => {
                              acc[rec.action_type] =
                                (acc[rec.action_type] || 0) + 1;
                              return acc;
                            },
                            {} as Record<string, number>
                          );

                          const mostCommon = Object.entries(counts).sort(
                            ([, a], [, b]) => (b as number) - (a as number)
                          )[0]?.[0];

                          return mostCommon?.replace(/_/g, " ") || "None";
                        })()}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Tips */}
              <div>
                <h2 className="text-xl font-semibold text-gray-900 mb-4">
                  Pro Tips
                </h2>
                <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                  <div className="space-y-4">
                    <div className="flex items-start space-x-3">
                      <div className="flex-shrink-0 w-6 h-6 bg-blue-100 rounded-full flex items-center justify-center">
                        <span className="text-xs font-bold text-blue-600">
                          1
                        </span>
                      </div>
                      <div>
                        <h4 className="text-sm font-medium text-gray-900">
                          Act on Recommendations
                        </h4>
                        <p className="text-xs text-gray-600 mt-1">
                          The more you follow AI suggestions, the better they
                          become.
                        </p>
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <div className="flex-shrink-0 w-6 h-6 bg-green-100 rounded-full flex items-center justify-center">
                        <span className="text-xs font-bold text-green-600">
                          2
                        </span>
                      </div>
                      <div>
                        <h4 className="text-sm font-medium text-gray-900">
                          Provide Feedback
                        </h4>
                        <p className="text-xs text-gray-600 mt-1">
                          Rate recommendations to help improve future
                          suggestions.
                        </p>
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <div className="flex-shrink-0 w-6 h-6 bg-purple-100 rounded-full flex items-center justify-center">
                        <span className="text-xs font-bold text-purple-600">
                          3
                        </span>
                      </div>
                      <div>
                        <h4 className="text-sm font-medium text-gray-900">
                          Check Regularly
                        </h4>
                        <p className="text-xs text-gray-600 mt-1">
                          Visit daily for fresh insights based on your latest
                          activity.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CoachingPage;
