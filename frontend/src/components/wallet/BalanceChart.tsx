import React, { useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { format, subDays, eachDayOfInterval } from "date-fns";
import type { Transaction } from "../../types/financial";

interface BalanceChartProps {
  transactions: Transaction[];
  currentBalance: number;
  className?: string;
}

interface ChartDataPoint {
  date: string;
  balance: number;
  formattedDate: string;
}

export const BalanceChart: React.FC<BalanceChartProps> = ({
  transactions,
  currentBalance,
  className = "",
}) => {
  const chartData = useMemo(() => {
    // Generate last 30 days
    const endDate = new Date();
    const startDate = subDays(endDate, 29);
    const dateRange = eachDayOfInterval({ start: startDate, end: endDate });

    // Sort transactions by date (newest first)
    const sortedTransactions = [...transactions].sort(
      (a, b) =>
        new Date(b.transaction_date).getTime() -
        new Date(a.transaction_date).getTime()
    );

    // Calculate balance for each day
    const data: ChartDataPoint[] = [];

    // Work backwards from today
    for (let i = dateRange.length - 1; i >= 0; i--) {
      const date = dateRange[i];
      const dateStr = format(date, "yyyy-MM-dd");

      // If this is today, use current balance
      if (i === dateRange.length - 1) {
        data.unshift({
          date: dateStr,
          balance: currentBalance,
          formattedDate: format(date, "MMM dd"),
        });
      } else {
        // Calculate balance at end of this day
        // Subtract transactions that happened after this day
        const transactionsAfterThisDay = sortedTransactions.filter(
          (t) => new Date(t.transaction_date) > date
        );

        let balanceAtEndOfDay = currentBalance;
        transactionsAfterThisDay.forEach((t) => {
          balanceAtEndOfDay -= t.amount;
        });

        data.unshift({
          date: dateStr,
          balance: balanceAtEndOfDay,
          formattedDate: format(date, "MMM dd"),
        });
      }
    }

    return data;
  }, [transactions, currentBalance]);

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
  }: {
    active?: boolean;
    payload?: Array<{ payload: ChartDataPoint }>;
    label?: string;
  }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
          <p className="text-sm font-medium text-gray-900">
            {data.formattedDate}
          </p>
          <p className="text-sm text-gray-600">
            Balance:{" "}
            <span className="font-semibold text-gray-900">
              {formatCurrency(data.balance)}
            </span>
          </p>
        </div>
      );
    }
    return null;
  };

  // Determine line color based on overall trend
  const firstBalance = chartData[0]?.balance || 0;
  const lastBalance = chartData[chartData.length - 1]?.balance || 0;
  const isPositiveTrend = lastBalance >= firstBalance;

  return (
    <div
      className={`bg-white rounded-lg border border-gray-200 p-6 ${className}`}
    >
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-gray-900">Balance Trend</h3>
        <p className="text-sm text-gray-600">Last 30 days</p>
      </div>

      {chartData.length > 0 ? (
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={chartData}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
              <XAxis
                dataKey="formattedDate"
                tick={{ fontSize: 12, fill: "#6b7280" }}
                tickLine={{ stroke: "#d1d5db" }}
                axisLine={{ stroke: "#d1d5db" }}
                interval="preserveStartEnd"
              />
              <YAxis
                tick={{ fontSize: 12, fill: "#6b7280" }}
                tickLine={{ stroke: "#d1d5db" }}
                axisLine={{ stroke: "#d1d5db" }}
                tickFormatter={formatCurrency}
              />
              <Tooltip content={<CustomTooltip />} />
              <Line
                type="monotone"
                dataKey="balance"
                stroke={isPositiveTrend ? "#10b981" : "#ef4444"}
                strokeWidth={2}
                dot={{
                  fill: isPositiveTrend ? "#10b981" : "#ef4444",
                  strokeWidth: 2,
                  r: 4,
                }}
                activeDot={{
                  r: 6,
                  stroke: isPositiveTrend ? "#10b981" : "#ef4444",
                  strokeWidth: 2,
                }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="h-64 flex items-center justify-center">
          <div className="text-center">
            <div className="w-12 h-12 mx-auto mb-4 bg-gray-100 rounded-full flex items-center justify-center">
              <svg
                className="w-6 h-6 text-gray-400"
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
            <p className="text-sm text-gray-500">
              No transaction data available
            </p>
            <p className="text-xs text-gray-400 mt-1">
              Chart will appear once you have transactions
            </p>
          </div>
        </div>
      )}

      {/* Balance change indicator */}
      {chartData.length > 1 && (
        <div className="mt-4 pt-4 border-t border-gray-100">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600">30-day change:</span>
            <div className="flex items-center">
              {isPositiveTrend ? (
                <svg
                  className="w-4 h-4 text-green-500 mr-1"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M7 11l5-5m0 0l5 5m-5-5v12"
                  />
                </svg>
              ) : (
                <svg
                  className="w-4 h-4 text-red-500 mr-1"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M17 13l-5 5m0 0l-5-5m5 5V6"
                  />
                </svg>
              )}
              <span
                className={`font-medium ${isPositiveTrend ? "text-green-600" : "text-red-600"}`}
              >
                {formatCurrency(Math.abs(lastBalance - firstBalance))}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
