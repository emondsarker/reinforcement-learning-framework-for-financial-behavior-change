import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  getWallet,
  updateWalletBalance,
  getTransactions,
  createTransaction,
  getTransaction,
  updateTransaction,
  deleteTransaction,
  getSpendingAnalytics,
  getFinancialHealthSummary,
  getRecentTransactions,
} from "../services/financialService";
import type {
  GetTransactionsParams,
  GetSpendingAnalyticsParams,
} from "../services/financialService";
import type { Transaction, TransactionCreate } from "../types/financial";
import type { PaginatedResponse } from "../types/common";
import { queryKeys, invalidateQueries } from "../lib/queryKeys";
import { useToast } from "./useToast";

// Wallet hooks
export const useWallet = () => {
  return useQuery({
    queryKey: queryKeys.financial.wallet(),
    queryFn: getWallet,
    staleTime: 2 * 60 * 1000, // 2 minutes
    gcTime: 5 * 60 * 1000, // 5 minutes
  });
};

export const useUpdateWalletBalance = () => {
  const queryClient = useQueryClient();
  const { addToast } = useToast();

  return useMutation({
    mutationFn: updateWalletBalance,
    onSuccess: (data) => {
      // Update wallet cache
      queryClient.setQueryData(queryKeys.financial.wallet(), data);

      // Invalidate related queries
      queryClient.invalidateQueries({
        queryKey: invalidateQueries.financial(),
      });

      addToast({
        type: "success",
        title: "Wallet Updated",
        message: "Your wallet balance has been updated successfully.",
      });
    },
    onError: () => {
      addToast({
        type: "error",
        title: "Update Failed",
        message: "Failed to update wallet balance. Please try again.",
      });
    },
  });
};

// Transaction hooks
export const useTransactions = (params: GetTransactionsParams = {}) => {
  return useQuery({
    queryKey: queryKeys.financial.transactions.list(params),
    queryFn: () => getTransactions(params),
    staleTime: 2 * 60 * 1000, // 2 minutes
    gcTime: 5 * 60 * 1000, // 5 minutes
  });
};

export const useTransaction = (id: number) => {
  return useQuery({
    queryKey: queryKeys.financial.transactions.detail(id.toString()),
    queryFn: () => getTransaction(id),
    staleTime: 2 * 60 * 1000, // 2 minutes
    gcTime: 5 * 60 * 1000, // 5 minutes
    enabled: !!id,
  });
};

export const useRecentTransactions = (limit: number = 5) => {
  return useQuery({
    queryKey: queryKeys.financial.transactions.list({ limit, page: 1 }),
    queryFn: () => getRecentTransactions(limit),
    staleTime: 2 * 60 * 1000, // 2 minutes
    gcTime: 5 * 60 * 1000, // 5 minutes
  });
};

export const useCreateTransaction = () => {
  const queryClient = useQueryClient();
  const { addToast } = useToast();

  return useMutation({
    mutationFn: createTransaction,
    onMutate: async (newTransaction) => {
      // Cancel outgoing refetches
      await queryClient.cancelQueries({
        queryKey: invalidateQueries.transactions(),
      });

      // Snapshot previous value
      const previousTransactions = queryClient.getQueryData(
        queryKeys.financial.transactions.list()
      );

      // Optimistically update transaction list
      queryClient.setQueryData(
        queryKeys.financial.transactions.list(),
        (old: PaginatedResponse<Transaction> | undefined) => {
          if (!old) return old;

          const optimisticTransaction: Transaction = {
            id: Date.now().toString(), // Temporary ID
            transaction_type: newTransaction.amount > 0 ? "credit" : "debit",
            amount: newTransaction.amount,
            category: newTransaction.category,
            description: newTransaction.description,
            merchant_name: newTransaction.merchant_name,
            location_city: newTransaction.location_city,
            location_country: newTransaction.location_country || "Unknown",
            balance_after: 0, // Will be updated by backend
            transaction_date: new Date(),
          };

          return {
            ...old,
            data: [optimisticTransaction, ...(old.data || [])],
            total: (old.total || 0) + 1,
          };
        }
      );

      return { previousTransactions };
    },
    onError: (err, newTransaction, context) => {
      // Rollback optimistic update
      if (context?.previousTransactions) {
        queryClient.setQueryData(
          queryKeys.financial.transactions.list(),
          context.previousTransactions
        );
      }

      addToast({
        type: "error",
        title: "Transaction Failed",
        message: "Failed to create transaction. Please try again.",
      });
    },
    onSuccess: () => {
      // Invalidate and refetch
      queryClient.invalidateQueries({
        queryKey: invalidateQueries.transactions(),
      });
      queryClient.invalidateQueries({
        queryKey: invalidateQueries.wallet(),
      });

      addToast({
        type: "success",
        title: "Transaction Created",
        message: "Your transaction has been created successfully.",
      });
    },
  });
};

export const useUpdateTransaction = () => {
  const queryClient = useQueryClient();
  const { addToast } = useToast();

  return useMutation({
    mutationFn: ({
      id,
      data,
    }: {
      id: number;
      data: Partial<TransactionCreate>;
    }) => updateTransaction(id, data),
    onSuccess: (data) => {
      // Update specific transaction cache
      queryClient.setQueryData(
        queryKeys.financial.transactions.detail(data.id.toString()),
        data
      );

      // Invalidate transaction lists
      queryClient.invalidateQueries({
        queryKey: invalidateQueries.transactions(),
      });

      addToast({
        type: "success",
        title: "Transaction Updated",
        message: "Your transaction has been updated successfully.",
      });
    },
    onError: () => {
      addToast({
        type: "error",
        title: "Update Failed",
        message: "Failed to update transaction. Please try again.",
      });
    },
  });
};

export const useDeleteTransaction = () => {
  const queryClient = useQueryClient();
  const { addToast } = useToast();

  return useMutation({
    mutationFn: deleteTransaction,
    onSuccess: () => {
      // Invalidate transaction queries
      queryClient.invalidateQueries({
        queryKey: invalidateQueries.transactions(),
      });
      queryClient.invalidateQueries({
        queryKey: invalidateQueries.wallet(),
      });

      addToast({
        type: "success",
        title: "Transaction Deleted",
        message: "Your transaction has been deleted successfully.",
      });
    },
    onError: () => {
      addToast({
        type: "error",
        title: "Delete Failed",
        message: "Failed to delete transaction. Please try again.",
      });
    },
  });
};

// Analytics hooks
export const useSpendingAnalytics = (
  params: GetSpendingAnalyticsParams = {}
) => {
  return useQuery({
    queryKey: queryKeys.financial.analytics.spending(params.period),
    queryFn: () => getSpendingAnalytics(params),
    staleTime: 2 * 60 * 1000, // 2 minutes
    gcTime: 5 * 60 * 1000, // 5 minutes
  });
};

export const useFinancialHealthSummary = () => {
  return useQuery({
    queryKey: queryKeys.financial.analytics.health(),
    queryFn: getFinancialHealthSummary,
    staleTime: 2 * 60 * 1000, // 2 minutes
    gcTime: 5 * 60 * 1000, // 5 minutes
  });
};
