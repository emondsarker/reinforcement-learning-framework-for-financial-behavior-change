import React, { useState } from "react";
import {
  SparklesIcon,
  CheckIcon,
  XMarkIcon,
  ArrowPathIcon,
  ClockIcon,
} from "@heroicons/react/24/outline";
import {
  useRecommendation,
  useSubmitRecommendationFeedback,
  useRefreshRecommendation,
  useActionTypeInfo,
} from "../../hooks/useCoaching";
import type { CoachingFeedback } from "../../types";

interface RecommendationCardProps {
  className?: string;
  showActions?: boolean;
  compact?: boolean;
}

export const RecommendationCard: React.FC<RecommendationCardProps> = ({
  className = "",
  showActions = true,
  compact = false,
}) => {
  const {
    data: recommendation,
    isLoading,
    error,
    refetch,
  } = useRecommendation();
  const submitFeedback = useSubmitRecommendationFeedback();
  const refreshRecommendation = useRefreshRecommendation();
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);

  // Always call the hook, but use a fallback action type if no recommendation
  const actionInfo = useActionTypeInfo(
    recommendation?.action_type || "continue_current_behavior"
  );

  if (isLoading) {
    return (
      <div
        className={`bg-white rounded-lg shadow-sm border border-gray-200 p-6 ${className}`}
      >
        <div className="animate-pulse">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-8 h-8 bg-gray-200 rounded-full"></div>
            <div className="h-4 bg-gray-200 rounded w-32"></div>
          </div>
          <div className="space-y-2">
            <div className="h-4 bg-gray-200 rounded w-full"></div>
            <div className="h-4 bg-gray-200 rounded w-3/4"></div>
          </div>
        </div>
      </div>
    );
  }

  if (error || !recommendation) {
    return (
      <div
        className={`bg-white rounded-lg shadow-sm border border-gray-200 p-6 ${className}`}
      >
        <div className="text-center">
          <SparklesIcon className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900">
            No Recommendation Available
          </h3>
          <p className="mt-1 text-sm text-gray-500">
            {error
              ? "Failed to load recommendation"
              : "Check back later for personalized advice"}
          </p>
          <button
            onClick={() => refetch()}
            className="mt-4 inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-black"
          >
            <ArrowPathIcon className="h-4 w-4 mr-2" />
            Try Again
          </button>
        </div>
      </div>
    );
  }

  const getActionColor = (color: string) => {
    const colorMap = {
      red: "bg-red-50 text-red-700 border-red-200",
      green: "bg-green-50 text-green-700 border-green-200",
      blue: "bg-blue-50 text-blue-700 border-blue-200",
      purple: "bg-purple-50 text-purple-700 border-purple-200",
      gray: "bg-gray-50 text-gray-700 border-gray-200",
    };
    return colorMap[color as keyof typeof colorMap] || colorMap.gray;
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return "text-green-600";
    if (confidence >= 0.6) return "text-yellow-600";
    return "text-red-600";
  };

  const handleFeedback = async (
    feedbackType: CoachingFeedback["feedback_type"],
    comment?: string
  ) => {
    try {
      await submitFeedback.mutateAsync({
        recommendationId: `${recommendation.action_id}-${Date.now()}`, // Generate ID
        feedback: {
          recommendation_id: `${recommendation.action_id}-${Date.now()}`,
          feedback_type: feedbackType,
          user_comment: comment,
        },
      });
      setFeedbackSubmitted(true);
    } catch {
      // Error handling is done in the hook
    }
  };

  const handleRefresh = async () => {
    try {
      await refreshRecommendation.mutateAsync();
    } catch {
      // Error handling is done in the hook
    }
  };

  return (
    <div
      className={`bg-white rounded-lg shadow-sm border border-gray-200 ${className}`}
    >
      {/* Header */}
      <div className="p-6 pb-4">
        <div className="flex items-start justify-between">
          <div className="flex items-center space-x-3">
            <div
              className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${getActionColor(actionInfo.color)}`}
            >
              <span className="mr-1">{actionInfo.icon}</span>
              {actionInfo.label}
            </div>
            <div className="flex items-center text-xs text-gray-500">
              <ClockIcon className="h-3 w-3 mr-1" />
              Just now
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <span
              className={`text-xs font-medium ${getConfidenceColor(recommendation.confidence)}`}
            >
              {Math.round(recommendation.confidence * 100)}% confidence
            </span>
            {!compact && (
              <button
                onClick={handleRefresh}
                disabled={refreshRecommendation.isPending}
                className="p-1 text-gray-400 hover:text-gray-600 focus:outline-none focus:ring-2 focus:ring-black focus:ring-offset-2 rounded"
                title="Refresh recommendation"
              >
                <ArrowPathIcon
                  className={`h-4 w-4 ${refreshRecommendation.isPending ? "animate-spin" : ""}`}
                />
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="px-6 pb-4">
        <p
          className={`text-gray-900 ${compact ? "text-sm" : "text-base"} leading-relaxed`}
        >
          {recommendation.recommendation}
        </p>
      </div>

      {/* Actions */}
      {showActions && !feedbackSubmitted && (
        <div className="px-6 py-4 bg-gray-50 border-t border-gray-200 rounded-b-lg">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Was this helpful?</span>
            <div className="flex items-center space-x-2">
              <button
                onClick={() => handleFeedback("helpful")}
                disabled={submitFeedback.isPending}
                className="inline-flex items-center px-3 py-1.5 border border-transparent text-xs font-medium rounded-md text-green-700 bg-green-100 hover:bg-green-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:opacity-50"
              >
                <CheckIcon className="h-3 w-3 mr-1" />
                Helpful
              </button>
              <button
                onClick={() => handleFeedback("not_helpful")}
                disabled={submitFeedback.isPending}
                className="inline-flex items-center px-3 py-1.5 border border-transparent text-xs font-medium rounded-md text-red-700 bg-red-100 hover:bg-red-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 disabled:opacity-50"
              >
                <XMarkIcon className="h-3 w-3 mr-1" />
                Not Helpful
              </button>
              <button
                onClick={() => handleFeedback("implemented")}
                disabled={submitFeedback.isPending}
                className="inline-flex items-center px-3 py-1.5 border border-gray-300 text-xs font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-black disabled:opacity-50"
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Feedback Submitted */}
      {feedbackSubmitted && (
        <div className="px-6 py-4 bg-green-50 border-t border-green-200 rounded-b-lg">
          <div className="flex items-center">
            <CheckIcon className="h-4 w-4 text-green-600 mr-2" />
            <span className="text-sm text-green-800">
              Thank you for your feedback!
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default RecommendationCard;
