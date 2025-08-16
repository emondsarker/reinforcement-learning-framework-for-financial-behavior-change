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

        {/* Future sections can be added here */}
      </div>
    </div>
  );
};
