import React, { useState, useCallback } from "react";
import { ChevronDownIcon, XMarkIcon } from "@heroicons/react/24/outline";
import type { TransactionCategory } from "../../types/financial";

export interface FilterState {
  dateRange: {
    start: string;
    end: string;
  };
  categories: TransactionCategory[];
  transactionType: "all" | "debit" | "credit";
  amountRange: {
    min: string;
    max: string;
  };
  search: string;
}

interface TransactionFiltersProps {
  filters: FilterState;
  onFiltersChange: (filters: FilterState) => void;
  className?: string;
}

const TRANSACTION_CATEGORIES: { value: TransactionCategory; label: string }[] =
  [
    { value: "groceries", label: "Groceries" },
    { value: "dine_out", label: "Dining Out" },
    { value: "entertainment", label: "Entertainment" },
    { value: "bills", label: "Bills" },
    { value: "transport", label: "Transport" },
    { value: "shopping", label: "Shopping" },
    { value: "health", label: "Health" },
    { value: "fitness", label: "Fitness" },
    { value: "savings", label: "Savings" },
    { value: "income", label: "Income" },
    { value: "other", label: "Other" },
  ];

export const TransactionFilters: React.FC<TransactionFiltersProps> = ({
  filters,
  onFiltersChange,
  className = "",
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [showCategoryDropdown, setShowCategoryDropdown] = useState(false);

  const updateFilters = useCallback(
    (updates: Partial<FilterState>) => {
      onFiltersChange({ ...filters, ...updates });
    },
    [filters, onFiltersChange]
  );

  const handleCategoryToggle = (category: TransactionCategory) => {
    const newCategories = filters.categories.includes(category)
      ? filters.categories.filter((c) => c !== category)
      : [...filters.categories, category];

    updateFilters({ categories: newCategories });
  };

  const clearAllFilters = () => {
    onFiltersChange({
      dateRange: { start: "", end: "" },
      categories: [],
      transactionType: "all",
      amountRange: { min: "", max: "" },
      search: "",
    });
  };

  const getActiveFilterCount = () => {
    let count = 0;
    if (filters.dateRange.start || filters.dateRange.end) count++;
    if (filters.categories.length > 0) count++;
    if (filters.transactionType !== "all") count++;
    if (filters.amountRange.min || filters.amountRange.max) count++;
    if (filters.search) count++;
    return count;
  };

  const activeFilterCount = getActiveFilterCount();

  return (
    <div className={`bg-white rounded-lg border border-gray-200 ${className}`}>
      {/* Filter Header */}
      <div className="p-4 border-b border-gray-100">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <h3 className="text-lg font-medium text-gray-900">Filters</h3>
            {activeFilterCount > 0 && (
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                {activeFilterCount} active
              </span>
            )}
          </div>
          <div className="flex items-center space-x-2">
            {activeFilterCount > 0 && (
              <button
                onClick={clearAllFilters}
                className="text-sm text-gray-500 hover:text-gray-700 transition-colors"
              >
                Clear all
              </button>
            )}
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="flex items-center text-sm text-gray-600 hover:text-gray-900 transition-colors"
            >
              {isExpanded ? "Hide" : "Show"} filters
              <ChevronDownIcon
                className={`ml-1 h-4 w-4 transition-transform ${
                  isExpanded ? "rotate-180" : ""
                }`}
              />
            </button>
          </div>
        </div>
      </div>

      {/* Search Bar - Always Visible */}
      <div className="p-4 border-b border-gray-100">
        <div className="relative">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <svg
              className="h-5 w-5 text-gray-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
              />
            </svg>
          </div>
          <input
            type="text"
            placeholder="Search transactions..."
            value={filters.search}
            onChange={(e) => updateFilters({ search: e.target.value })}
            className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md leading-5 bg-white placeholder-gray-500 focus:outline-none focus:placeholder-gray-400 focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
          />
          {filters.search && (
            <button
              onClick={() => updateFilters({ search: "" })}
              className="absolute inset-y-0 right-0 pr-3 flex items-center"
            >
              <XMarkIcon className="h-4 w-4 text-gray-400 hover:text-gray-600" />
            </button>
          )}
        </div>
      </div>

      {/* Expandable Filters */}
      {isExpanded && (
        <div className="p-4 space-y-6">
          {/* Date Range */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Date Range
            </label>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label className="block text-xs text-gray-500 mb-1">From</label>
                <input
                  type="date"
                  value={filters.dateRange.start}
                  onChange={(e) =>
                    updateFilters({
                      dateRange: {
                        ...filters.dateRange,
                        start: e.target.value,
                      },
                    })
                  }
                  className="block w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-500 mb-1">To</label>
                <input
                  type="date"
                  value={filters.dateRange.end}
                  onChange={(e) =>
                    updateFilters({
                      dateRange: { ...filters.dateRange, end: e.target.value },
                    })
                  }
                  className="block w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
            </div>
          </div>

          {/* Transaction Type */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Transaction Type
            </label>
            <div className="flex space-x-4">
              {[
                { value: "all", label: "All" },
                { value: "credit", label: "Income" },
                { value: "debit", label: "Expenses" },
              ].map((type) => (
                <label key={type.value} className="flex items-center">
                  <input
                    type="radio"
                    name="transactionType"
                    value={type.value}
                    checked={filters.transactionType === type.value}
                    onChange={(e) =>
                      updateFilters({
                        transactionType: e.target
                          .value as FilterState["transactionType"],
                      })
                    }
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                  />
                  <span className="ml-2 text-sm text-gray-700">
                    {type.label}
                  </span>
                </label>
              ))}
            </div>
          </div>

          {/* Categories */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Categories
            </label>
            <div className="relative">
              <button
                onClick={() => setShowCategoryDropdown(!showCategoryDropdown)}
                className="w-full flex items-center justify-between px-3 py-2 border border-gray-300 rounded-md bg-white text-sm focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
              >
                <span className="text-gray-700">
                  {filters.categories.length === 0
                    ? "Select categories..."
                    : `${filters.categories.length} selected`}
                </span>
                <ChevronDownIcon
                  className={`h-4 w-4 text-gray-400 transition-transform ${
                    showCategoryDropdown ? "rotate-180" : ""
                  }`}
                />
              </button>

              {showCategoryDropdown && (
                <div className="absolute z-10 mt-1 w-full bg-white border border-gray-300 rounded-md shadow-lg max-h-60 overflow-auto">
                  {TRANSACTION_CATEGORIES.map((category) => (
                    <label
                      key={category.value}
                      className="flex items-center px-3 py-2 hover:bg-gray-50 cursor-pointer"
                    >
                      <input
                        type="checkbox"
                        checked={filters.categories.includes(category.value)}
                        onChange={() => handleCategoryToggle(category.value)}
                        className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                      />
                      <span className="ml-2 text-sm text-gray-700">
                        {category.label}
                      </span>
                    </label>
                  ))}
                </div>
              )}
            </div>

            {/* Selected Categories Display */}
            {filters.categories.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-2">
                {filters.categories.map((category) => {
                  const categoryInfo = TRANSACTION_CATEGORIES.find(
                    (c) => c.value === category
                  );
                  return (
                    <span
                      key={category}
                      className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800"
                    >
                      {categoryInfo?.label}
                      <button
                        onClick={() => handleCategoryToggle(category)}
                        className="ml-1 inline-flex items-center justify-center w-4 h-4 rounded-full hover:bg-blue-200"
                      >
                        <XMarkIcon className="h-3 w-3" />
                      </button>
                    </span>
                  );
                })}
              </div>
            )}
          </div>

          {/* Amount Range */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Amount Range
            </label>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label className="block text-xs text-gray-500 mb-1">
                  Min ($)
                </label>
                <input
                  type="number"
                  placeholder="0.00"
                  value={filters.amountRange.min}
                  onChange={(e) =>
                    updateFilters({
                      amountRange: {
                        ...filters.amountRange,
                        min: e.target.value,
                      },
                    })
                  }
                  className="block w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-500 mb-1">
                  Max ($)
                </label>
                <input
                  type="number"
                  placeholder="1000.00"
                  value={filters.amountRange.max}
                  onChange={(e) =>
                    updateFilters({
                      amountRange: {
                        ...filters.amountRange,
                        max: e.target.value,
                      },
                    })
                  }
                  className="block w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
