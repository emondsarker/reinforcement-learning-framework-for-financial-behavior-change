import React, { useState, useMemo } from "react";
import { PlusIcon } from "@heroicons/react/24/outline";
import {
  BalanceChart,
  TransactionFilters,
  TransactionHistory,
  AddTransactionForm,
  type FilterState,
} from "../components/wallet";
import { WalletCard } from "../components/dashboard/WalletCard";
import { useWallet, useTransactions } from "../hooks/useFinancial";
import type { GetTransactionsParams } from "../services/financialService";
import type { Transaction, TransactionCategory } from "../types/financial";

export const WalletPage: React.FC = () => {
  const [currentPage, setCurrentPage] = useState(1);
  const [showAddForm, setShowAddForm] = useState(false);
  const [filters, setFilters] = useState<FilterState>({
    dateRange: { start: "", end: "" },
    categories: [],
    transactionType: "all",
    amountRange: { min: "", max: "" },
    search: "",
  });

  // Convert filters to API parameters
  const transactionParams = useMemo((): GetTransactionsParams => {
    const params: GetTransactionsParams = {
      page: currentPage,
      limit: 10,
    };

    // Search
    if (filters.search) {
      params.search = filters.search;
    }

    // Date range
    if (filters.dateRange.start) {
      params.start_date = filters.dateRange.start;
    }
    if (filters.dateRange.end) {
      params.end_date = filters.dateRange.end;
    }

    // Categories
    if (filters.categories.length > 0) {
      // API expects single category, so we'll use the first one for now
      // In a real implementation, you might want to modify the API to support multiple categories
      params.category = filters.categories[0];
    }

    return params;
  }, [currentPage, filters]);

  // Fetch data
  const { data: wallet } = useWallet();
  const {
    data: transactionsData,
    isLoading: transactionsLoading,
    error: transactionsError,
  } = useTransactions(transactionParams);

  // Filter transactions on frontend for features not supported by API
  const filteredTransactions = useMemo(() => {
    if (!transactionsData?.data) return transactionsData;

    let filtered = [...transactionsData.data];

    // Transaction type filter
    if (filters.transactionType !== "all") {
      filtered = filtered.filter(
        (t) => t.transaction_type === filters.transactionType
      );
    }

    // Amount range filter
    if (filters.amountRange.min) {
      const minAmount = parseFloat(filters.amountRange.min);
      filtered = filtered.filter((t) => Math.abs(t.amount) >= minAmount);
    }
    if (filters.amountRange.max) {
      const maxAmount = parseFloat(filters.amountRange.max);
      filtered = filtered.filter((t) => Math.abs(t.amount) <= maxAmount);
    }

    // Multiple categories filter (frontend-only)
    if (filters.categories.length > 1) {
      filtered = filtered.filter((t) =>
        filters.categories.includes(t.category as TransactionCategory)
      );
    }

    return {
      ...transactionsData,
      data: filtered,
      total: filtered.length,
    };
  }, [transactionsData, filters]);

  const handlePageChange = (page: number) => {
    setCurrentPage(page);
  };

  const handleFiltersChange = (newFilters: FilterState) => {
    setFilters(newFilters);
    setCurrentPage(1); // Reset to first page when filters change
  };

  const handleEditTransaction = (transaction: Transaction) => {
    // TODO: Implement edit functionality
    console.log("Edit transaction:", transaction);
  };

  return (
    <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
      <div className="px-4 py-6 sm:px-0">
        {/* Page Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Wallet</h1>
              <p className="mt-2 text-gray-600">
                Manage your transactions and track your financial activity
              </p>
            </div>
            <button
              onClick={() => setShowAddForm(true)}
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors"
            >
              <PlusIcon className="h-4 w-4 mr-2" />
              Add Transaction
            </button>
          </div>
        </div>

        {/* Top Section: Wallet Balance + Balance Chart */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Wallet Balance */}
          <WalletCard />

          {/* Balance Chart */}
          <BalanceChart
            transactions={transactionsData?.data || []}
            currentBalance={wallet?.balance || 0}
          />
        </div>

        {/* Filters Section */}
        <div className="mb-6">
          <TransactionFilters
            filters={filters}
            onFiltersChange={handleFiltersChange}
          />
        </div>

        {/* Transaction History */}
        <TransactionHistory
          data={filteredTransactions}
          isLoading={transactionsLoading}
          error={transactionsError}
          currentPage={currentPage}
          onPageChange={handlePageChange}
          onEditTransaction={handleEditTransaction}
        />

        {/* Add Transaction Modal */}
        <AddTransactionForm
          isOpen={showAddForm}
          onClose={() => setShowAddForm(false)}
        />
      </div>
    </div>
  );
};
