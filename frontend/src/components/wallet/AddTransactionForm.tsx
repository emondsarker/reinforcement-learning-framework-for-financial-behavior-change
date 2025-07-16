import React, { useState } from "react";
import { XMarkIcon } from "@heroicons/react/24/outline";
import { useForm } from "react-hook-form";
import type {
  TransactionCategory,
  TransactionCreate,
} from "../../types/financial";
import { useCreateTransaction } from "../../hooks/useFinancial";

interface AddTransactionFormProps {
  isOpen: boolean;
  onClose: () => void;
}

interface FormData {
  amount: string;
  category: TransactionCategory;
  description: string;
  merchant_name: string;
  location_city: string;
  location_country: string;
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

export const AddTransactionForm: React.FC<AddTransactionFormProps> = ({
  isOpen,
  onClose,
}) => {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const createTransaction = useCreateTransaction();

  const {
    register,
    handleSubmit,
    reset,
    formState: { errors },
    watch,
  } = useForm<FormData>({
    defaultValues: {
      amount: "",
      category: "other",
      description: "",
      merchant_name: "",
      location_city: "",
      location_country: "USA",
    },
  });

  const amount = watch("amount");
  const isExpense = parseFloat(amount) < 0;

  const onSubmit = async (data: FormData) => {
    setIsSubmitting(true);
    try {
      const transactionData: TransactionCreate = {
        amount: parseFloat(data.amount),
        category: data.category,
        description: data.description,
        merchant_name: data.merchant_name || undefined,
        location_city: data.location_city || undefined,
        location_country: data.location_country || undefined,
      };

      await createTransaction.mutateAsync(transactionData);
      reset();
      onClose();
    } catch (error) {
      console.error("Failed to create transaction:", error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleClose = () => {
    reset();
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex items-center justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
        {/* Background overlay */}
        <div
          className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity"
          onClick={handleClose}
        />

        {/* Modal panel */}
        <div className="inline-block align-bottom bg-white rounded-lg px-4 pt-5 pb-4 text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full sm:p-6">
          {/* Header */}
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-medium text-gray-900">
              Add New Transaction
            </h3>
            <button
              onClick={handleClose}
              className="text-gray-400 hover:text-gray-600 transition-colors"
            >
              <XMarkIcon className="h-6 w-6" />
            </button>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
            {/* Amount */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Amount *
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <span className="text-gray-500 sm:text-sm">$</span>
                </div>
                <input
                  type="number"
                  step="0.01"
                  placeholder="0.00"
                  {...register("amount", {
                    required: "Amount is required",
                    validate: (value) => {
                      const num = parseFloat(value);
                      if (isNaN(num)) return "Please enter a valid number";
                      if (num === 0) return "Amount cannot be zero";
                      return true;
                    },
                  })}
                  className={`block w-full pl-7 pr-3 py-2 border rounded-md text-sm focus:outline-none focus:ring-1 ${
                    errors.amount
                      ? "border-red-300 focus:ring-red-500 focus:border-red-500"
                      : "border-gray-300 focus:ring-blue-500 focus:border-blue-500"
                  }`}
                />
              </div>
              {errors.amount && (
                <p className="mt-1 text-sm text-red-600">
                  {errors.amount.message}
                </p>
              )}
              <p className="mt-1 text-xs text-gray-500">
                Use positive numbers for income, negative for expenses
              </p>
            </div>

            {/* Category */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Category *
              </label>
              <select
                {...register("category", { required: "Category is required" })}
                className={`block w-full px-3 py-2 border rounded-md text-sm focus:outline-none focus:ring-1 ${
                  errors.category
                    ? "border-red-300 focus:ring-red-500 focus:border-red-500"
                    : "border-gray-300 focus:ring-blue-500 focus:border-blue-500"
                }`}
              >
                {TRANSACTION_CATEGORIES.map((category) => (
                  <option key={category.value} value={category.value}>
                    {category.label}
                  </option>
                ))}
              </select>
              {errors.category && (
                <p className="mt-1 text-sm text-red-600">
                  {errors.category.message}
                </p>
              )}
            </div>

            {/* Description */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Description *
              </label>
              <input
                type="text"
                placeholder="Enter transaction description"
                {...register("description", {
                  required: "Description is required",
                  minLength: {
                    value: 3,
                    message: "Description must be at least 3 characters",
                  },
                })}
                className={`block w-full px-3 py-2 border rounded-md text-sm focus:outline-none focus:ring-1 ${
                  errors.description
                    ? "border-red-300 focus:ring-red-500 focus:border-red-500"
                    : "border-gray-300 focus:ring-blue-500 focus:border-blue-500"
                }`}
              />
              {errors.description && (
                <p className="mt-1 text-sm text-red-600">
                  {errors.description.message}
                </p>
              )}
            </div>

            {/* Merchant Name */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Merchant Name
              </label>
              <input
                type="text"
                placeholder="Enter merchant or payee name"
                {...register("merchant_name")}
                className="block w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>

            {/* Location */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  City
                </label>
                <input
                  type="text"
                  placeholder="Enter city"
                  {...register("location_city")}
                  className="block w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Country
                </label>
                <input
                  type="text"
                  placeholder="Enter country"
                  {...register("location_country")}
                  className="block w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
            </div>

            {/* Transaction Type Indicator */}
            {amount && (
              <div className="p-3 rounded-md bg-gray-50">
                <div className="flex items-center">
                  <div
                    className={`w-3 h-3 rounded-full mr-2 ${
                      isExpense ? "bg-red-500" : "bg-green-500"
                    }`}
                  />
                  <span className="text-sm text-gray-700">
                    This will be recorded as an{" "}
                    <span
                      className={`font-medium ${isExpense ? "text-red-600" : "text-green-600"}`}
                    >
                      {isExpense ? "expense" : "income"}
                    </span>
                  </span>
                </div>
              </div>
            )}

            {/* Form Actions */}
            <div className="flex items-center justify-end space-x-3 pt-4">
              <button
                type="button"
                onClick={handleClose}
                disabled={isSubmitting}
                className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={isSubmitting}
                className="px-4 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isSubmitting ? (
                  <div className="flex items-center">
                    <div className="w-4 h-4 mr-2 animate-spin rounded-full border-2 border-white border-t-transparent" />
                    Creating...
                  </div>
                ) : (
                  "Create Transaction"
                )}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};
