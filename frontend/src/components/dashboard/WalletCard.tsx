import React from "react";
import {
  CurrencyDollarIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  ExclamationTriangleIcon,
} from "@heroicons/react/24/outline";
import { useWallet } from "../../hooks/useFinancial";

export const WalletCard: React.FC = () => {
  const { data: wallet, isLoading, error } = useWallet();

  if (isLoading) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="animate-pulse">
          <div className="flex items-center justify-between mb-4">
            <div className="h-6 bg-gray-200 rounded w-24"></div>
            <div className="h-8 w-8 bg-gray-200 rounded"></div>
          </div>
          <div className="h-8 bg-gray-200 rounded w-32 mb-2"></div>
          <div className="h-4 bg-gray-200 rounded w-20"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-red-200 p-6">
        <div className="flex items-center text-red-600">
          <ExclamationTriangleIcon className="h-5 w-5 mr-2" />
          <span className="text-sm">Failed to load wallet data</span>
        </div>
      </div>
    );
  }

  if (!wallet) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="text-center text-gray-500">
          <CurrencyDollarIcon className="h-12 w-12 mx-auto mb-2 text-gray-300" />
          <p className="text-sm">No wallet data available</p>
        </div>
      </div>
    );
  }

  const formatCurrency = (amount: number, currency: string = "USD") => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: currency,
    }).format(amount);
  };

  const formatDate = (date: Date) => {
    return new Intl.DateTimeFormat("en-US", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }).format(new Date(date));
  };

  // Determine trend (this is a placeholder - in a real app you'd compare with previous balance)
  const isPositiveTrend = wallet.balance >= 0;

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">Wallet Balance</h3>
        <div className="flex items-center">
          {isPositiveTrend ? (
            <ArrowTrendingUpIcon className="h-5 w-5 text-green-500" />
          ) : (
            <ArrowTrendingDownIcon className="h-5 w-5 text-red-500" />
          )}
        </div>
      </div>

      <div className="mb-2">
        <span
          className={`text-3xl font-bold ${
            wallet.balance >= 0 ? "text-gray-900" : "text-red-600"
          }`}
        >
          {formatCurrency(wallet.balance, wallet.currency)}
        </span>
      </div>

      <div className="text-sm text-gray-500">
        Last updated: {formatDate(wallet.updated_at)}
      </div>

      {wallet.balance < 0 && (
        <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-md">
          <div className="flex items-center">
            <ExclamationTriangleIcon className="h-4 w-4 text-red-500 mr-2" />
            <span className="text-sm text-red-700">
              Your account balance is negative
            </span>
          </div>
        </div>
      )}

      {wallet.balance > 0 && wallet.balance < 100 && (
        <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-md">
          <div className="flex items-center">
            <ExclamationTriangleIcon className="h-4 w-4 text-yellow-500 mr-2" />
            <span className="text-sm text-yellow-700">
              Low balance - consider adding funds
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
