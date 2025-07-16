import React, { useState } from "react";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts";
import { useSpendingAnalyticsByCategory } from "../../hooks/useFinancial";
import type {
  AnalyticsTimeRange,
  CategorySpendingData,
} from "../../types/financial";

// Category colors for consistent theming
const CATEGORY_COLORS: Record<string, string> = {
  groceries: "#10B981", // green
  dine_out: "#F59E0B", // amber
  entertainment: "#8B5CF6", // violet
  bills: "#EF4444", // red
  transport: "#3B82F6", // blue
  shopping: "#EC4899", // pink
  health: "#06B6D4", // cyan
  fitness: "#84CC16", // lime
  savings: "#22C55E", // green
  income: "#059669", // emerald
  other: "#6B7280", // gray
};

const TIME_RANGES: AnalyticsTimeRange[] = [
  { label: "7 Days", days: 7, value: "7d" },
  { label: "30 Days", days: 30, value: "30d" },
  { label: "90 Days", days: 90, value: "90d" },
];

interface SpendingBreakdownProps {
  className?: string;
}

export const SpendingBreakdown: React.FC<SpendingBreakdownProps> = ({
  className = "",
}) => {
  const [selectedRange, setSelectedRange] = useState<AnalyticsTimeRange>(
    TIME_RANGES[1]
  ); // Default to 30 days

  const {
    data: spendingData,
    isLoading,
    error,
  } = useSpendingAnalyticsByCategory(selectedRange.days);

  // Transform data for chart
  const chartData: CategorySpendingData[] = React.useMemo(() => {
    if (!spendingData) return [];

    return spendingData.map((item) => ({
      category: item.category,
      amount: item.total_amount,
      percentage: item.percentage,
      color: CATEGORY_COLORS[item.category] || CATEGORY_COLORS.other,
      transactions: item.transaction_count,
    }));
  }, [spendingData]);

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
    }).format(value);
  };

  const formatCategoryName = (category: string) => {
    return category.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase());
  };

  const CustomTooltip = ({
    active,
    payload,
  }: {
    active?: boolean;
    payload?: any[];
  }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
          <p className="font-medium text-gray-900">
            {formatCategoryName(data.category)}
          </p>
          <p className="text-sm text-gray-600">
            Amount: {formatCurrency(data.amount)}
          </p>
          <p className="text-sm text-gray-600">
            Percentage: {data.percentage.toFixed(1)}%
          </p>
          <p className="text-sm text-gray-600">
            Transactions: {data.transactions}
          </p>
        </div>
      );
    }
    return null;
  };

  if (error) {
    return (
      <div
        className={`bg-white rounded-lg border border-gray-200 p-6 ${className}`}
      >
        <div className="text-center">
          <div className="text-red-500 mb-2">
            <svg
              className="w-12 h-12 mx-auto"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </div>
          <h3 className="text-lg font-medium text-gray-900 mb-1">
            Failed to Load Spending Data
          </h3>
          <p className="text-gray-500">
            Unable to load spending breakdown. Please try again.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`bg-white rounded-lg border border-gray-200 p-6 ${className}`}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold text-gray-900">
          Spending Breakdown
        </h2>

        {/* Time Range Selector */}
        <div className="flex space-x-1 bg-gray-100 rounded-lg p-1">
          {TIME_RANGES.map((range) => (
            <button
              key={range.value}
              onClick={() => setSelectedRange(range)}
              className={`px-3 py-1 text-sm font-medium rounded-md transition-colors ${
                selectedRange.value === range.value
                  ? "bg-white text-gray-900 shadow-sm"
                  : "text-gray-600 hover:text-gray-900"
              }`}
            >
              {range.label}
            </button>
          ))}
        </div>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
        </div>
      ) : !chartData.length ? (
        <div className="text-center py-12">
          <div className="text-gray-400 mb-2">
            <svg
              className="w-12 h-12 mx-auto"
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
          <h3 className="text-lg font-medium text-gray-900 mb-1">
            No Spending Data
          </h3>
          <p className="text-gray-500">
            No spending transactions found for the selected period.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Pie Chart */}
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={chartData}
                  cx="50%"
                  cy="50%"
                  innerRadius={40}
                  outerRadius={80}
                  paddingAngle={2}
                  dataKey="amount"
                >
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* Category List */}
          <div className="space-y-3">
            {chartData
              .sort((a, b) => b.amount - a.amount)
              .map((item) => (
                <div
                  key={item.category}
                  className="flex items-center justify-between"
                >
                  <div className="flex items-center space-x-3">
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: item.color }}
                    />
                    <div>
                      <p className="text-sm font-medium text-gray-900">
                        {formatCategoryName(item.category)}
                      </p>
                      <p className="text-xs text-gray-500">
                        {item.transactions} transaction
                        {item.transactions !== 1 ? "s" : ""}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-sm font-medium text-gray-900">
                      {formatCurrency(item.amount)}
                    </p>
                    <p className="text-xs text-gray-500">
                      {item.percentage.toFixed(1)}%
                    </p>
                  </div>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default SpendingBreakdown;
