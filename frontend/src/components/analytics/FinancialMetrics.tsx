import React from "react";
import { useFinancialHealthSummary, useWallet } from "../../hooks/useFinancial";
import type { FinancialMetric } from "../../types/financial";

interface FinancialMetricsProps {
  className?: string;
}

export const FinancialMetrics: React.FC<FinancialMetricsProps> = ({
  className = "",
}) => {
  const {
    data: healthSummary,
    isLoading: isLoadingHealth,
    error: healthError,
  } = useFinancialHealthSummary();
  const {
    data: wallet,
    isLoading: isLoadingWallet,
    error: walletError,
  } = useWallet();

  const isLoading = isLoadingHealth || isLoadingWallet;
  const error = healthError || walletError;

  // Calculate additional metrics
  const calculateMetrics = (): FinancialMetric[] => {
    if (!healthSummary || !wallet) return [];

    const metrics: FinancialMetric[] = [];

    // Current Balance
    metrics.push({
      label: "Current Balance",
      value: formatCurrency(wallet.balance),
      format: "currency",
      status:
        wallet.balance > 0
          ? "good"
          : wallet.balance < -100
            ? "danger"
            : "warning",
    });

    // Weekly Spending
    metrics.push({
      label: "Weekly Spending",
      value: formatCurrency(healthSummary.weekly_spending),
      format: "currency",
      status:
        healthSummary.weekly_spending > healthSummary.weekly_income
          ? "danger"
          : "good",
    });

    // Weekly Income
    metrics.push({
      label: "Weekly Income",
      value: formatCurrency(healthSummary.weekly_income),
      format: "currency",
      status: "good",
    });

    // Savings Rate
    const savingsRatePercent = healthSummary.savings_rate * 100;
    metrics.push({
      label: "Savings Rate",
      value: `${savingsRatePercent.toFixed(1)}%`,
      format: "percentage",
      status:
        savingsRatePercent > 20
          ? "good"
          : savingsRatePercent > 10
            ? "warning"
            : "danger",
    });

    // Daily Spending Average
    metrics.push({
      label: "Daily Avg Spending",
      value: formatCurrency(healthSummary.daily_spending_avg),
      format: "currency",
      status:
        healthSummary.daily_spending_avg < 50
          ? "good"
          : healthSummary.daily_spending_avg < 100
            ? "warning"
            : "danger",
    });

    // Emergency Fund Ratio (balance / monthly spending estimate)
    const monthlySpendingEstimate = healthSummary.weekly_spending * 4.33; // Average weeks per month
    const emergencyFundRatio =
      monthlySpendingEstimate > 0
        ? wallet.balance / monthlySpendingEstimate
        : 0;
    metrics.push({
      label: "Emergency Fund",
      value: `${emergencyFundRatio.toFixed(1)}x`,
      format: "number",
      status:
        emergencyFundRatio >= 3
          ? "good"
          : emergencyFundRatio >= 1
            ? "warning"
            : "danger",
    });

    // Transaction Frequency
    metrics.push({
      label: "Weekly Transactions",
      value: healthSummary.transaction_count,
      format: "number",
      status: "good",
    });

    // Net Weekly Change
    const netWeeklyChange =
      healthSummary.weekly_income - healthSummary.weekly_spending;
    metrics.push({
      label: "Weekly Net Change",
      value: formatCurrency(netWeeklyChange),
      format: "currency",
      status:
        netWeeklyChange > 0
          ? "good"
          : netWeeklyChange > -100
            ? "warning"
            : "danger",
    });

    return metrics;
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "good":
        return "text-green-600 bg-green-50 border-green-200";
      case "warning":
        return "text-yellow-600 bg-yellow-50 border-yellow-200";
      case "danger":
        return "text-red-600 bg-red-50 border-red-200";
      default:
        return "text-gray-600 bg-gray-50 border-gray-200";
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "good":
        return (
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
              clipRule="evenodd"
            />
          </svg>
        );
      case "warning":
        return (
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
              clipRule="evenodd"
            />
          </svg>
        );
      case "danger":
        return (
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
              clipRule="evenodd"
            />
          </svg>
        );
      default:
        return (
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
              clipRule="evenodd"
            />
          </svg>
        );
    }
  };

  const metrics = calculateMetrics();

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
            Failed to Load Financial Metrics
          </h3>
          <p className="text-gray-500">
            Unable to load financial health data. Please try again.
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
      <div className="mb-6">
        <h2 className="text-lg font-semibold text-gray-900">
          Financial Health Metrics
        </h2>
        <p className="text-sm text-gray-600 mt-1">
          Key indicators of your financial wellness
        </p>
      </div>

      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {[...Array(8)].map((_, index) => (
            <div key={index} className="animate-pulse">
              <div className="bg-gray-200 rounded-lg h-24"></div>
            </div>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {metrics.map((metric, index) => (
            <div
              key={index}
              className={`p-4 rounded-lg border ${getStatusColor(metric.status || "good")}`}
            >
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm font-medium">{metric.label}</p>
                {getStatusIcon(metric.status || "good")}
              </div>
              <p className="text-2xl font-bold">{metric.value}</p>
              {metric.change && (
                <div className="flex items-center mt-2">
                  <span
                    className={`text-xs font-medium ${
                      metric.changeType === "increase"
                        ? "text-red-600"
                        : metric.changeType === "decrease"
                          ? "text-green-600"
                          : "text-gray-600"
                    }`}
                  >
                    {metric.changeType === "increase"
                      ? "↑"
                      : metric.changeType === "decrease"
                        ? "↓"
                        : "→"}{" "}
                    {Math.abs(metric.change).toFixed(1)}%
                  </span>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Financial Health Tips */}
      {!isLoading && !error && (
        <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <h3 className="text-sm font-medium text-blue-900 mb-2">
            💡 Financial Health Tips
          </h3>
          <ul className="text-sm text-blue-800 space-y-1">
            <li>• Aim for a savings rate of at least 20% of your income</li>
            <li>• Build an emergency fund covering 3-6 months of expenses</li>
            <li>• Track your daily spending to identify improvement areas</li>
            <li>• Review your financial metrics weekly to stay on track</li>
          </ul>
        </div>
      )}
    </div>
  );
};

export default FinancialMetrics;
