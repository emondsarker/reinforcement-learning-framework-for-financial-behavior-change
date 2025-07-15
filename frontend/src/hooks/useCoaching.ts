import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { coachingService } from "../services/coachingService";
import { useToast } from "./useToast";
import { queryKeys, mutationKeys } from "../lib/queryKeys";
import type {
  AIRecommendation,
  FinancialState,
  CoachingFeedback,
  RecommendationHistory,
  FinancialHealthScore,
  CoachingAnalytics,
} from "../types";

/**
 * Hook to get current AI recommendation
 */
export const useRecommendation = () => {
  return useQuery({
    queryKey: queryKeys.coaching.recommendations.list(),
    queryFn: coachingService.getRecommendation,
    staleTime: 5 * 60 * 1000, // 5 minutes
    gcTime: 10 * 60 * 1000, // 10 minutes
    retry: 2,
    refetchOnWindowFocus: false,
  });
};

/**
 * Hook to get financial health summary
 */
export const useFinancialHealthSummary = () => {
  return useQuery({
    queryKey: queryKeys.coaching.financialState(),
    queryFn: coachingService.getFinancialHealthSummary,
    staleTime: 2 * 60 * 1000, // 2 minutes
    gcTime: 5 * 60 * 1000, // 5 minutes
    retry: 2,
  });
};

/**
 * Hook to get financial health score
 */
export const useFinancialHealthScore = () => {
  return useQuery({
    queryKey: [...queryKeys.coaching.financialState(), "score"],
    queryFn: coachingService.getFinancialHealthScore,
    staleTime: 2 * 60 * 1000, // 2 minutes
    gcTime: 5 * 60 * 1000, // 5 minutes
    retry: 2,
  });
};

/**
 * Hook to get recommendation history
 */
export const useRecommendationHistory = (
  limit: number = 20,
  offset: number = 0
) => {
  return useQuery({
    queryKey: [...queryKeys.coaching.recommendations.list(), { limit, offset }],
    queryFn: () => coachingService.getRecommendationHistory(limit, offset),
    staleTime: 5 * 60 * 1000, // 5 minutes
    gcTime: 10 * 60 * 1000, // 10 minutes
    retry: 2,
  });
};

/**
 * Hook to get coaching analytics
 */
export const useCoachingAnalytics = () => {
  return useQuery({
    queryKey: [...queryKeys.coaching.all, "analytics"],
    queryFn: coachingService.getCoachingAnalytics,
    staleTime: 10 * 60 * 1000, // 10 minutes
    gcTime: 15 * 60 * 1000, // 15 minutes
    retry: 2,
  });
};

/**
 * Hook to submit recommendation feedback
 */
export const useSubmitRecommendationFeedback = () => {
  const queryClient = useQueryClient();
  const { addToast } = useToast();

  return useMutation({
    mutationKey: mutationKeys.coaching.provideFeedback,
    mutationFn: ({
      recommendationId,
      feedback,
    }: {
      recommendationId: string;
      feedback: Omit<CoachingFeedback, "submitted_at">;
    }) =>
      coachingService.submitRecommendationFeedback(recommendationId, feedback),
    onSuccess: () => {
      // Invalidate coaching-related queries
      queryClient.invalidateQueries({ queryKey: queryKeys.coaching.all });

      addToast({
        type: "success",
        title: "Feedback Submitted",
        message:
          "Thank you for your feedback! This helps improve our recommendations.",
      });
    },
    onError: (error) => {
      addToast({
        type: "error",
        title: "Feedback Failed",
        message:
          error instanceof Error ? error.message : "Failed to submit feedback",
      });
    },
  });
};

/**
 * Hook to refresh recommendation
 */
export const useRefreshRecommendation = () => {
  const queryClient = useQueryClient();
  const { addToast } = useToast();

  return useMutation({
    mutationKey: ["coaching", "refresh-recommendation"],
    mutationFn: async () => {
      // Invalidate current recommendation to force refetch
      await queryClient.invalidateQueries({
        queryKey: queryKeys.coaching.recommendations.list(),
      });

      // Also invalidate financial state as it affects recommendations
      await queryClient.invalidateQueries({
        queryKey: queryKeys.coaching.financialState(),
      });

      return true;
    },
    onSuccess: () => {
      addToast({
        type: "success",
        title: "Recommendation Updated",
        message: "Your AI coaching recommendation has been refreshed.",
      });
    },
    onError: (error) => {
      addToast({
        type: "error",
        title: "Refresh Failed",
        message:
          error instanceof Error
            ? error.message
            : "Failed to refresh recommendation",
      });
    },
  });
};

/**
 * Hook to get action type information
 */
export const useActionTypeInfo = (actionType: string) => {
  return coachingService.getActionTypeInfo(actionType);
};

/**
 * Hook to prefetch coaching data
 */
export const usePrefetchCoachingData = () => {
  const queryClient = useQueryClient();

  const prefetchRecommendation = () => {
    queryClient.prefetchQuery({
      queryKey: queryKeys.coaching.recommendations.list(),
      queryFn: coachingService.getRecommendation,
      staleTime: 5 * 60 * 1000,
    });
  };

  const prefetchFinancialHealth = () => {
    queryClient.prefetchQuery({
      queryKey: queryKeys.coaching.financialState(),
      queryFn: coachingService.getFinancialHealthSummary,
      staleTime: 2 * 60 * 1000,
    });
  };

  const prefetchHistory = (limit: number = 10) => {
    queryClient.prefetchQuery({
      queryKey: [
        ...queryKeys.coaching.recommendations.list(),
        { limit, offset: 0 },
      ],
      queryFn: () => coachingService.getRecommendationHistory(limit, 0),
      staleTime: 5 * 60 * 1000,
    });
  };

  return {
    prefetchRecommendation,
    prefetchFinancialHealth,
    prefetchHistory,
  };
};

/**
 * Hook to get coaching loading states
 */
export const useCoachingLoadingState = () => {
  const recommendationQuery = useRecommendation();
  const healthQuery = useFinancialHealthSummary();
  const scoreQuery = useFinancialHealthScore();

  return {
    isLoadingRecommendation: recommendationQuery.isLoading,
    isLoadingHealth: healthQuery.isLoading,
    isLoadingScore: scoreQuery.isLoading,
    isLoadingAny:
      recommendationQuery.isLoading ||
      healthQuery.isLoading ||
      scoreQuery.isLoading,
    hasError:
      recommendationQuery.isError || healthQuery.isError || scoreQuery.isError,
    errors: {
      recommendation: recommendationQuery.error,
      health: healthQuery.error,
      score: scoreQuery.error,
    },
  };
};

/**
 * Hook for coaching notifications
 */
export const useCoachingNotifications = () => {
  const { data: recommendation } = useRecommendation();
  const { data: healthScore } = useFinancialHealthScore();
  const { addToast } = useToast();

  const showRecommendationNotification = (rec: AIRecommendation) => {
    const actionInfo = coachingService.getActionTypeInfo(rec.action_type);

    if (actionInfo.priority === "high") {
      addToast({
        type: "warning",
        title: `${actionInfo.icon} ${actionInfo.label}`,
        message: rec.recommendation,
        duration: 8000, // Longer duration for important notifications
      });
    } else if (actionInfo.priority === "medium") {
      addToast({
        type: "info",
        title: `${actionInfo.icon} ${actionInfo.label}`,
        message: rec.recommendation,
        duration: 6000,
      });
    }
  };

  const showHealthScoreNotification = (score: FinancialHealthScore) => {
    if (score.trend === "declining" && score.overall_score < 50) {
      addToast({
        type: "warning",
        title: "⚠️ Financial Health Alert",
        message: `Your financial health score is ${score.overall_score}/100. Consider reviewing your spending habits.`,
        duration: 8000,
      });
    } else if (score.trend === "improving" && score.overall_score > 80) {
      addToast({
        type: "success",
        title: "🎉 Great Progress!",
        message: `Your financial health score is ${score.overall_score}/100. Keep up the excellent work!`,
        duration: 6000,
      });
    }
  };

  return {
    showRecommendationNotification,
    showHealthScoreNotification,
    currentRecommendation: recommendation,
    currentHealthScore: healthScore,
  };
};
