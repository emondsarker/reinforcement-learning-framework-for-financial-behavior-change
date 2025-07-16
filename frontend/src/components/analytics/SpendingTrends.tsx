import React, { useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
} from "recharts";
import { format } from "date-fns";
import { useBalanceHistory, useMonthlySummary } from "../../hooks/useFinancial";
import type { AnalyticsTimeRange } from "../../types/financial";

const TIME_RANGES: AnalyticsTimeRange[] = [
  { label: "1M", days: 30, value: "1m" },
  { label: "3M", days: 90, value: "3m" },
  { label: "6M", days: 180, value: "6m" },
];

interface SpendingTrendsProps {
  className?: string;
}

export const SpendingTrends: React.FC<SpendingTrendsProps> = ({
  className = "",
}) => {
  const [selectedRange, setSelectedRange] = useState<AnalyticsTimeRange>(
    TIME_RANGES[0]
  ); // Default to 1 month

  const {
    data: balanceHistory,
    isLoading: isLoadingBalance,
    error: balanceError,
  } = useBalanceHistory(selectedRange.days);

  // Get current month summary for comparison
  const currentDate = new Date();
  const { data: currentMonthSummary, isLoading: isLoadingCurrent } =
    useMonthlySummary(currentDate.getFullYear(), currentDate.getMonth() + 1);

  // Get previous month summary for comparison
  const previousMonth = new Date(
    currentDate.getFullYear(),
    currentDate.getMonth() - 1
  );
  const { data: previousMonthSummary, isLoading: isLoadingPrevious } =
    useMonthlySummary(
      previousMonth.getFullYear(),
      previousMonth.getMonth() + 1
    );

  const isLoading = isLoadingBalance || isLoadingCurrent || isLoadingPrevious;
  const error = balanceError;

  // Transform balance history for line chart
  const balanceChartData = React.useMemo(() => {
    if (!balanceHistory) return [];

    return balanceHistory.map((point) => ({
      date: format(new Date(point.date), "MMM dd"),
      balance: point.balance,
      fullDate: point.date,
    }));
  }, [balanceHistory]);

  // Monthly comparison data
  const monthlyComparisonData = React.useMemo(() => {
    const data = [];

    if (previousMonthSummary) {
      data.push({
        month: format(previousMonth, "MMM yyyy"),
        spending: previousMonthSummary.total_spending,
        income: previousMonthSummary.total_income,
        net: previousMonthSummary.net_change,
      });
    }

    if (currentMonthSummary) {
      data.push({
        month: format(currentDate, "MMM yyyy"),
        spending: currentMonthSummary.total_spending,
        income: currentMonthSummary.total_income,
        net: currentMonthSummary.net_change,
      });
    }

    return data;
  }, [currentMonthSummary, previousMonthSummary, currentDate, previousMonth]);

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  const CustomTooltip = ({
    active,
    payload,
    label,
  }: {
    active?: boolean;
    payload?: any[];
    label?: string;
  }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
          <p className="font-medium text-gray-900 mb-2">{label}</p>
          {payload.map((entry, index) => (
            <p key={index} className="text-sm" style={{ color: entry.color }}>
              {entry.name}: {formatCurrency(entry.value)}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  const calculateTrend = () => {
    if (!currentMonthSummary || !previousMonthSummary) return null;

    const currentSpending = currentMonthSummary.total_spending;
    const previousSpending = previousMonthSummary.total_spending;

    if (previousSpending === 0) return null;

    const change =
      ((currentSpending - previousSpending) / previousSpending) * 100;
    return {
      change: Math.abs(change),
      type: change > 0 ? "increase" : "decrease",
    };
  };

  const trend = calculateTrend();

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
            Failed to Load Trend Data
          </h3>
          <p className="text-gray-500">
            Unable to load spending trends. Please try again.
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
        <div>
          <h2 className="text-lg font-semibold text-gray-900">
            Spending Trends
          </h2>
          {trend && (
            <p className="text-sm text-gray-600 mt-1">
              <span
                className={`font-medium ${
                  trend.type === "increase" ? "text-red-600" : "text-green-600"
                }`}
              >
                {trend.change.toFixed(1)}% {trend.type}
              </span>{" "}
              from last month
            </p>
          )}
        </div>

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
      ) : (
        <div className="space-y-8">
          {/* Balance Trend Chart */}
          <div>
            <h3 className="text-sm font-medium text-gray-900 mb-4">
              Balance Over Time
            </h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={balanceChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="date" stroke="#6b7280" fontSize={12} />
                  <YAxis
                    stroke="#6b7280"
                    fontSize={12}
                    tickFormatter={formatCurrency}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Line
                    type="monotone"
                    dataKey="balance"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    dot={{ fill: "#3b82f6", strokeWidth: 2, r: 4 }}
                    activeDot={{ r: 6, stroke: "#3b82f6", strokeWidth: 2 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Monthly Comparison */}
          {monthlyComparisonData.length > 0 && (
            <div>
              <h3 className="text-sm font-medium text-gray-900 mb-4">
                Monthly Comparison
              </h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={monthlyComparisonData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="month" stroke="#6b7280" fontSize={12} />
                    <YAxis
                      stroke="#6b7280"
                      fontSize={12}
                      tickFormatter={formatCurrency}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar
                      dataKey="income"
                      fill="#10b981"
                      name="Income"
                      radius={[2, 2, 0, 0]}
                    />
                    <Bar
                      dataKey="spending"
                      fill="#ef4444"
                      name="Spending"
                      radius={[2, 2, 0, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default SpendingTrends;
