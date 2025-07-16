import React, { useState } from "react";
import {
  PencilIcon,
  TrashIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
} from "@heroicons/react/24/outline";
import { format } from "date-fns";
import type { Transaction } from "../../types/financial";
import type { PaginatedResponse } from "../../types/common";
import { useDeleteTransaction } from "../../hooks/useFinancial";

interface TransactionHistoryProps {
  data: PaginatedResponse<Transaction> | undefined;
  isLoading: boolean;
  error: Error | null;
  currentPage: number;
  onPageChange: (page: number) => void;
  onEditTransaction?: (transaction: Transaction) => void;
  className?: string;
}

const getCategoryIcon = (category: string) => {
  const icons: Record<string, string> = {
    groceries: "🛒",
    dine_out: "🍽️",
    entertainment: "🎬",
    bills: "📄",
    transport: "🚗",
    shopping: "🛍️",
    health: "🏥",
    fitness: "💪",
    savings: "💰",
    income: "💵",
    other: "📝",
  };
  return icons[category] || "📝";
};

const getCategoryColor = (category: string) => {
  const colors: Record<string, string> = {
    groceries: "bg-green-100 text-green-800",
    dine_out: "bg-orange-100 text-orange-800",
    entertainment: "bg-purple-100 text-purple-800",
    bills: "bg-red-100 text-red-800",
    transport: "bg-blue-100 text-blue-800",
    shopping: "bg-pink-100 text-pink-800",
    health: "bg-teal-100 text-teal-800",
    fitness: "bg-indigo-100 text-indigo-800",
    savings: "bg-emerald-100 text-emerald-800",
    income: "bg-green-100 text-green-800",
    other: "bg-gray-100 text-gray-800",
  };
  return colors[category] || "bg-gray-100 text-gray-800";
};

export const TransactionHistory: React.FC<TransactionHistoryProps> = ({
  data,
  isLoading,
  error,
  currentPage,
  onPageChange,
  onEditTransaction,
  className = "",
}) => {
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const deleteTransaction = useDeleteTransaction();

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
    }).format(Math.abs(amount));
  };

  const formatDate = (date: Date) => {
    return format(new Date(date), "MMM dd, yyyy");
  };

  const formatTime = (date: Date) => {
    return format(new Date(date), "h:mm a");
  };

  const handleDelete = async (transaction: Transaction) => {
    if (window.confirm("Are you sure you want to delete this transaction?")) {
      setDeletingId(transaction.id);
      try {
        await deleteTransaction.mutateAsync(parseInt(transaction.id));
      } catch (error) {
        console.error("Failed to delete transaction:", error);
      } finally {
        setDeletingId(null);
      }
    }
  };

  const totalPages = data ? Math.ceil(data.total / (data.limit || 10)) : 0;
  const transactions = data?.data || [];

  if (error) {
    return (
      <div
        className={`bg-white rounded-lg border border-gray-200 p-6 ${className}`}
      >
        <div className="text-center">
          <div className="w-12 h-12 mx-auto mb-4 bg-red-100 rounded-full flex items-center justify-center">
            <svg
              className="w-6 h-6 text-red-600"
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
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            Error Loading Transactions
          </h3>
          <p className="text-gray-600 mb-4">{error.message}</p>
          <button
            onClick={() => window.location.reload()}
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-white rounded-lg border border-gray-200 ${className}`}>
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-medium text-gray-900">
            Transaction History
          </h3>
          {data && (
            <span className="text-sm text-gray-500">
              {data.total} total transactions
            </span>
          )}
        </div>
      </div>

      {/* Loading State */}
      {isLoading && (
        <div className="p-6">
          <div className="space-y-4">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="animate-pulse">
                <div className="flex items-center space-x-4">
                  <div className="w-10 h-10 bg-gray-200 rounded-full"></div>
                  <div className="flex-1 space-y-2">
                    <div className="h-4 bg-gray-200 rounded w-1/4"></div>
                    <div className="h-3 bg-gray-200 rounded w-1/2"></div>
                  </div>
                  <div className="h-4 bg-gray-200 rounded w-20"></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Empty State */}
      {!isLoading && transactions.length === 0 && (
        <div className="p-12 text-center">
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
                d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
              />
            </svg>
          </div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            No transactions found
          </h3>
          <p className="text-gray-600 mb-4">
            No transactions match your current filters. Try adjusting your
            search criteria.
          </p>
        </div>
      )}

      {/* Transaction List */}
      {!isLoading && transactions.length > 0 && (
        <>
          <div className="divide-y divide-gray-200">
            {transactions.map((transaction) => (
              <div
                key={transaction.id}
                className="p-6 hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    {/* Category Icon */}
                    <div className="flex-shrink-0">
                      <div
                        className={`w-10 h-10 rounded-full flex items-center justify-center text-lg ${getCategoryColor(transaction.category)}`}
                      >
                        {getCategoryIcon(transaction.category)}
                      </div>
                    </div>

                    {/* Transaction Details */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center space-x-2">
                        <p className="text-sm font-medium text-gray-900 truncate">
                          {transaction.description}
                        </p>
                        <span
                          className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getCategoryColor(transaction.category)}`}
                        >
                          {transaction.category.replace("_", " ")}
                        </span>
                      </div>
                      <div className="flex items-center space-x-4 mt-1">
                        <p className="text-sm text-gray-500">
                          {formatDate(transaction.transaction_date)} at{" "}
                          {formatTime(transaction.transaction_date)}
                        </p>
                        {transaction.merchant_name && (
                          <p className="text-sm text-gray-500">
                            {transaction.merchant_name}
                          </p>
                        )}
                        {transaction.location_city && (
                          <p className="text-sm text-gray-500">
                            {transaction.location_city}
                          </p>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Amount and Actions */}
                  <div className="flex items-center space-x-4">
                    <div className="text-right">
                      <p
                        className={`text-sm font-medium ${
                          transaction.transaction_type === "credit"
                            ? "text-green-600"
                            : "text-red-600"
                        }`}
                      >
                        {transaction.transaction_type === "credit" ? "+" : "-"}
                        {formatCurrency(transaction.amount)}
                      </p>
                      <p className="text-xs text-gray-500">
                        Balance: {formatCurrency(transaction.balance_after)}
                      </p>
                    </div>

                    {/* Action Buttons */}
                    <div className="flex items-center space-x-2">
                      {onEditTransaction && (
                        <button
                          onClick={() => onEditTransaction(transaction)}
                          className="p-1 text-gray-400 hover:text-gray-600 transition-colors"
                          title="Edit transaction"
                        >
                          <PencilIcon className="h-4 w-4" />
                        </button>
                      )}
                      <button
                        onClick={() => handleDelete(transaction)}
                        disabled={
                          deletingId === transaction.id ||
                          deleteTransaction.isPending
                        }
                        className="p-1 text-gray-400 hover:text-red-600 transition-colors disabled:opacity-50"
                        title="Delete transaction"
                      >
                        {deletingId === transaction.id ? (
                          <div className="h-4 w-4 animate-spin rounded-full border-2 border-gray-300 border-t-red-600"></div>
                        ) : (
                          <TrashIcon className="h-4 w-4" />
                        )}
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="px-6 py-4 border-t border-gray-200">
              <div className="flex items-center justify-between">
                <div className="text-sm text-gray-700">
                  Showing page {currentPage} of {totalPages}
                </div>
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => onPageChange(currentPage - 1)}
                    disabled={currentPage <= 1}
                    className="inline-flex items-center px-3 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <ChevronLeftIcon className="h-4 w-4 mr-1" />
                    Previous
                  </button>

                  {/* Page Numbers */}
                  <div className="flex items-center space-x-1">
                    {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                      let pageNum;
                      if (totalPages <= 5) {
                        pageNum = i + 1;
                      } else if (currentPage <= 3) {
                        pageNum = i + 1;
                      } else if (currentPage >= totalPages - 2) {
                        pageNum = totalPages - 4 + i;
                      } else {
                        pageNum = currentPage - 2 + i;
                      }

                      return (
                        <button
                          key={pageNum}
                          onClick={() => onPageChange(pageNum)}
                          className={`px-3 py-2 text-sm font-medium rounded-md ${
                            currentPage === pageNum
                              ? "bg-blue-600 text-white"
                              : "text-gray-700 hover:bg-gray-50 border border-gray-300"
                          }`}
                        >
                          {pageNum}
                        </button>
                      );
                    })}
                  </div>

                  <button
                    onClick={() => onPageChange(currentPage + 1)}
                    disabled={currentPage >= totalPages}
                    className="inline-flex items-center px-3 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Next
                    <ChevronRightIcon className="h-4 w-4 ml-1" />
                  </button>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};
