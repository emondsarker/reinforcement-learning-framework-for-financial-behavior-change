import React from "react";
import { WalletCard } from "../components/dashboard/WalletCard";
import { RecentTransactions } from "../components/dashboard/RecentTransactions";
import { QuickActions } from "../components/dashboard/QuickActions";
import { RecommendationCard } from "../components/coaching/RecommendationCard";

export const DashboardPage: React.FC = () => {
  return (
    <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
      <div className="px-4 py-6 sm:px-0">
        {/* AI Financial Coach - full width */}
        <div className="mb-8">
          <RecommendationCard />
        </div>

        {/* Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* Wallet Card - spans 1 column */}
          <div className="lg:col-span-1">
            <WalletCard />
          </div>

          {/* Recent Transactions - spans 2 columns */}
          <div className="lg:col-span-2">
            <RecentTransactions />
          </div>
        </div>

        {/* Quick Actions - full width */}
        <div className="mb-8">
          <QuickActions />
        </div>

        {/* Future sections placeholder */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Spending Analytics - Coming in Task 13 */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="text-center py-8">
              <div className="h-12 w-12 mx-auto bg-orange-100 rounded-full flex items-center justify-center mb-4">
                <svg
                  className="h-6 w-6 text-orange-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                  />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                Spending Analytics
              </h3>
              <p className="text-sm text-gray-500">
                Detailed insights and trends coming soon
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
