import React from "react";
import { useAuth } from "../hooks/useAuth";
import { RecommendationCard } from "../components/coaching/RecommendationCard";
import { SparklesIcon } from "@heroicons/react/24/outline";

export const CoachingPage: React.FC = () => {
  const { user } = useAuth();

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
            {/* Current Recommendation */}
            <div className="lg:col-span-3">
              <div>
                <h2 className="text-xl font-semibold text-gray-900 mb-4">
                  Current Recommendation
                </h2>
                <RecommendationCard />
              </div>
            </div>

            <div className="space-y-8">
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
