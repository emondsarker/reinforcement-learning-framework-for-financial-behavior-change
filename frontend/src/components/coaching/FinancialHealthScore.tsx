import React from "react";
import {
  HeartIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  MinusIcon,
} from "@heroicons/react/24/outline";
import { useFinancialHealthScore } from "../../hooks/useCoaching";

interface FinancialHealthScoreProps {
  className?: string;
  showDetails?: boolean;
  compact?: boolean;
}

export const FinancialHealthScore: React.FC<FinancialHealthScoreProps> = ({
  className = "",
  showDetails = true,
  compact = false,
}) => {
  const { data: healthScore, isLoading, error } = useFinancialHealthScore();

  if (isLoading) {
    return (
      <div
        className={`bg-white rounded-lg shadow-sm border border-gray-200 p-6 ${className}`}
      >
        <div className="animate-pulse">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-8 h-8 bg-gray-200 rounded-full"></div>
            <div className="h-4 bg-gray-200 rounded w-40"></div>
          </div>
          <div className="space-y-3">
            <div className="h-8 bg-gray-200 rounded w-20"></div>
            <div className="h-2 bg-gray-200 rounded w-full"></div>
            <div className="grid grid-cols-2 gap-4">
              <div className="h-16 bg-gray-200 rounded"></div>
              <div className="h-16 bg-gray-200 rounded"></div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (error || !healthScore) {
    return (
      <div
        className={`bg-white rounded-lg shadow-sm border border-gray-200 p-6 ${className}`}
      >
        <div className="text-center">
          <HeartIcon className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900">
            Health Score Unavailable
          </h3>
          <p className="mt-1 text-sm text-gray-500">
            Unable to calculate your financial health score
          </p>
        </div>
      </div>
    );
  }

  const getScoreColor = (score: number) => {
    if (score >= 80) return "text-green-600";
    if (score >= 60) return "text-yellow-600";
    if (score >= 40) return "text-orange-600";
    return "text-red-600";
  };

  const getScoreBackgroundColor = (score: number) => {
    if (score >= 80) return "bg-green-100";
    if (score >= 60) return "bg-yellow-100";
    if (score >= 40) return "bg-orange-100";
    return "bg-red-100";
  };

  const getProgressColor = (score: number) => {
    if (score >= 80) return "bg-green-500";
    if (score >= 60) return "bg-yellow-500";
    if (score >= 40) return "bg-orange-500";
    return "bg-red-500";
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case "improving":
        return <ArrowTrendingUpIcon className="h-4 w-4 text-green-600" />;
      case "declining":
        return <ArrowTrendingDownIcon className="h-4 w-4 text-red-600" />;
      default:
        return <MinusIcon className="h-4 w-4 text-gray-600" />;
    }
  };

  const getTrendText = (trend: string) => {
    switch (trend) {
      case "improving":
        return "Improving";
      case "declining":
        return "Declining";
      default:
        return "Stable";
    }
  };

  const getTrendColor = (trend: string) => {
    switch (trend) {
      case "improving":
        return "text-green-600";
      case "declining":
        return "text-red-600";
      default:
        return "text-gray-600";
    }
  };

  const getScoreDescription = (score: number) => {
    if (score >= 80) return "Excellent financial health";
    if (score >= 60) return "Good financial health";
    if (score >= 40) return "Fair financial health";
    return "Needs improvement";
  };

  const categoryScores = [
    {
      label: "Spending Control",
      score: healthScore.category_scores.spending_control,
      description: "How well you manage your spending",
    },
    {
      label: "Savings Rate",
      score: healthScore.category_scores.savings_rate,
      description: "Your ability to save money",
    },
    {
      label: "Budget Adherence",
      score: healthScore.category_scores.budget_adherence,
      description: "How well you stick to your budget",
    },
    {
      label: "Financial Stability",
      score: healthScore.category_scores.financial_stability,
      description: "Your overall financial stability",
    },
  ];

  return (
    <div
      className={`bg-white rounded-lg shadow-sm border border-gray-200 ${className}`}
    >
      {/* Header */}
      <div className="p-6 pb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div
              className={`p-2 rounded-full ${getScoreBackgroundColor(healthScore.overall_score)}`}
            >
              <HeartIcon
                className={`h-5 w-5 ${getScoreColor(healthScore.overall_score)}`}
              />
            </div>
            <div>
              <h3 className="text-lg font-medium text-gray-900">
                Financial Health Score
              </h3>
              {!compact && (
                <p className="text-sm text-gray-500">
                  {getScoreDescription(healthScore.overall_score)}
                </p>
              )}
            </div>
          </div>
          <div className="flex items-center space-x-2">
            {getTrendIcon(healthScore.trend)}
            <span
              className={`text-sm font-medium ${getTrendColor(healthScore.trend)}`}
            >
              {getTrendText(healthScore.trend)}
            </span>
          </div>
        </div>
      </div>

      {/* Overall Score */}
      <div className="px-6 pb-4">
        <div className="flex items-end space-x-4">
          <div
            className={`text-4xl font-bold ${getScoreColor(healthScore.overall_score)}`}
          >
            {healthScore.overall_score}
          </div>
          <div className="flex-1 pb-2">
            <div className="flex justify-between text-xs text-gray-500 mb-1">
              <span>0</span>
              <span>100</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className={`h-2 rounded-full transition-all duration-300 ${getProgressColor(healthScore.overall_score)}`}
                style={{ width: `${healthScore.overall_score}%` }}
              ></div>
            </div>
          </div>
        </div>
      </div>

      {/* Category Breakdown */}
      {showDetails && !compact && (
        <div className="px-6 py-4 border-t border-gray-200">
          <h4 className="text-sm font-medium text-gray-900 mb-3">
            Category Breakdown
          </h4>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {categoryScores.map((category) => (
              <div key={category.label} className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium text-gray-700">
                    {category.label}
                  </span>
                  <span
                    className={`text-sm font-bold ${getScoreColor(category.score)}`}
                  >
                    {category.score}
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-1.5">
                  <div
                    className={`h-1.5 rounded-full transition-all duration-300 ${getProgressColor(category.score)}`}
                    style={{ width: `${category.score}%` }}
                  ></div>
                </div>
                <p className="text-xs text-gray-500">{category.description}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Compact Category View */}
      {showDetails && compact && (
        <div className="px-6 py-4 border-t border-gray-200">
          <div className="grid grid-cols-4 gap-2">
            {categoryScores.map((category) => (
              <div key={category.label} className="text-center">
                <div
                  className={`text-sm font-bold ${getScoreColor(category.score)}`}
                >
                  {category.score}
                </div>
                <div className="text-xs text-gray-500 truncate">
                  {category.label.split(" ")[0]}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Last Updated */}
      <div className="px-6 py-3 bg-gray-50 border-t border-gray-200 rounded-b-lg">
        <p className="text-xs text-gray-500">
          Last updated:{" "}
          {new Date(healthScore.last_calculated).toLocaleDateString()}
        </p>
      </div>
    </div>
  );
};

export default FinancialHealthScore;
