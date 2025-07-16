import React from "react";
import {
  SpendingBreakdown,
  SpendingTrends,
  FinancialMetrics,
} from "../components/analytics";

const AnalyticsPage: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Page Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">
            Financial Analytics
          </h1>
          <p className="mt-2 text-gray-600">
            Comprehensive insights into your financial health and spending
            patterns
          </p>
        </div>

        {/* Analytics Grid */}
        <div className="space-y-8">
          {/* Financial Metrics - Full Width */}
          <FinancialMetrics />

          {/* Charts Row */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Spending Breakdown */}
            <SpendingBreakdown />

            {/* Spending Trends */}
            <SpendingTrends />
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalyticsPage;
