import React from "react";
import { Link } from "react-router-dom";
import {
  ShoppingBagIcon,
  HomeIcon,
  FilmIcon,
  DocumentTextIcon,
  TruckIcon,
  HeartIcon,
  BanknotesIcon,
  EllipsisHorizontalIcon,
  ArrowRightIcon,
  ExclamationTriangleIcon,
} from "@heroicons/react/24/outline";
import { useRecentTransactions } from "../../hooks/useFinancial";
import type { Transaction } from "../../types/financial";

const getCategoryIcon = (category: string) => {
  const iconClass = "h-5 w-5";

  switch (category.toLowerCase()) {
    case "groceries":
      return <ShoppingBagIcon className={iconClass} />;
    case "dine_out":
      return <HomeIcon className={iconClass} />;
    case "entertainment":
      return <FilmIcon className={iconClass} />;
    case "bills":
      return <DocumentTextIcon className={iconClass} />;
    case "transport":
      return <TruckIcon className={iconClass} />;
    case "health":
    case "fitness":
      return <HeartIcon className={iconClass} />;
    case "income":
    case "savings":
      return <BanknotesIcon className={iconClass} />;
    default:
      return <EllipsisHorizontalIcon className={iconClass} />;
  }
};

const getCategoryColor = (category: string) => {
  switch (category.toLowerCase()) {
    case "groceries":
      return "bg-green-100 text-green-600";
    case "dine_out":
      return "bg-orange-100 text-orange-600";
    case "entertainment":
      return "bg-purple-100 text-purple-600";
    case "bills":
      return "bg-red-100 text-red-600";
    case "transport":
      return "bg-blue-100 text-blue-600";
    case "health":
    case "fitness":
      return "bg-pink-100 text-pink-600";
    case "income":
    case "savings":
      return "bg-emerald-100 text-emerald-600";
    default:
      return "bg-gray-100 text-gray-600";
  }
};

const formatCurrency = (amount: number) => {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
  }).format(Math.abs(amount));
};

const formatDate = (date: Date) => {
  const now = new Date();
  const transactionDate = new Date(date);
  const diffInHours = Math.floor(
    (now.getTime() - transactionDate.getTime()) / (1000 * 60 * 60)
  );

  if (diffInHours < 1) {
    return "Just now";
  } else if (diffInHours < 24) {
    return `${diffInHours}h ago`;
  } else if (diffInHours < 48) {
    return "Yesterday";
  } else {
    return transactionDate.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
    });
  }
};

const TransactionItem: React.FC<{ transaction: Transaction }> = ({
  transaction,
}) => {
  const isCredit = transaction.transaction_type === "credit";

  return (
    <div className="flex items-center justify-between py-3">
      <div className="flex items-center space-x-3">
        <div
          className={`p-2 rounded-full ${getCategoryColor(transaction.category)}`}
        >
          {getCategoryIcon(transaction.category)}
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-gray-900 truncate">
            {transaction.description ||
              transaction.merchant_name ||
              "Transaction"}
          </p>
          <div className="flex items-center space-x-2 text-xs text-gray-500">
            <span className="capitalize">
              {transaction.category.replace("_", " ")}
            </span>
            {transaction.merchant_name &&
              transaction.description !== transaction.merchant_name && (
                <>
                  <span>•</span>
                  <span className="truncate">{transaction.merchant_name}</span>
                </>
              )}
            {transaction.location_city && (
              <>
                <span>•</span>
                <span>{transaction.location_city}</span>
              </>
            )}
          </div>
        </div>
      </div>
      <div className="flex flex-col items-end">
        <span
          className={`text-sm font-semibold ${
            isCredit ? "text-green-600" : "text-gray-900"
          }`}
        >
          {isCredit ? "+" : "-"}
          {formatCurrency(transaction.amount)}
        </span>
        <span className="text-xs text-gray-500">
          {formatDate(transaction.transaction_date)}
        </span>
      </div>
    </div>
  );
};

export const RecentTransactions: React.FC = () => {
  const { data: transactions, isLoading, error } = useRecentTransactions(5);

  if (isLoading) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">
            Recent Transactions
          </h3>
        </div>
        <div className="space-y-3">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="animate-pulse flex items-center space-x-3">
              <div className="h-9 w-9 bg-gray-200 rounded-full"></div>
              <div className="flex-1">
                <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
                <div className="h-3 bg-gray-200 rounded w-1/2"></div>
              </div>
              <div className="text-right">
                <div className="h-4 bg-gray-200 rounded w-16 mb-2"></div>
                <div className="h-3 bg-gray-200 rounded w-12"></div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-red-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">
            Recent Transactions
          </h3>
        </div>
        <div className="flex items-center text-red-600">
          <ExclamationTriangleIcon className="h-5 w-5 mr-2" />
          <span className="text-sm">Failed to load transactions</span>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">
          Recent Transactions
        </h3>
        <Link
          to="/wallet"
          className="flex items-center text-sm text-gray-600 hover:text-gray-900 transition-colors"
        >
          View all
          <ArrowRightIcon className="h-4 w-4 ml-1" />
        </Link>
      </div>

      {!transactions || transactions.length === 0 ? (
        <div className="text-center py-8">
          <BanknotesIcon className="h-12 w-12 mx-auto text-gray-300 mb-3" />
          <p className="text-gray-500 text-sm">No transactions yet</p>
          <p className="text-gray-400 text-xs mt-1">
            Your recent transactions will appear here
          </p>
        </div>
      ) : (
        <div className="divide-y divide-gray-100">
          {transactions.map((transaction) => (
            <TransactionItem key={transaction.id} transaction={transaction} />
          ))}
        </div>
      )}
    </div>
  );
};
