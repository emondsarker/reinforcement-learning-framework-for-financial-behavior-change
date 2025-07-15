import api from "./api";
import type {
  Wallet,
  Transaction,
  TransactionCreate,
  SpendingAnalytics,
  FinancialHealthSummary,
} from "../types/financial";
import type { PaginatedResponse } from "../types/common";

// Wallet operations
export const getWallet = async (): Promise<Wallet> => {
  const response = await api.get("/financial/wallet");
  return response.data;
};

export const updateWalletBalance = async (balance: number): Promise<Wallet> => {
  const response = await api.put("/financial/wallet", { balance });
  return response.data;
};

// Transaction operations
export interface GetTransactionsParams {
  page?: number;
  limit?: number;
  category?: string;
  start_date?: string;
  end_date?: string;
  search?: string;
}

export const getTransactions = async (
  params: GetTransactionsParams = {}
): Promise<PaginatedResponse<Transaction>> => {
  const response = await api.get("/financial/transactions", { params });
  return response.data;
};

export const createTransaction = async (
  data: TransactionCreate
): Promise<Transaction> => {
  const response = await api.post("/financial/transactions", data);
  return response.data;
};

export const getTransaction = async (id: number): Promise<Transaction> => {
  const response = await api.get(`/financial/transactions/${id}`);
  return response.data;
};

export const updateTransaction = async (
  id: number,
  data: Partial<TransactionCreate>
): Promise<Transaction> => {
  const response = await api.put(`/financial/transactions/${id}`, data);
  return response.data;
};

export const deleteTransaction = async (id: number): Promise<void> => {
  await api.delete(`/financial/transactions/${id}`);
};

// Analytics operations
export interface GetSpendingAnalyticsParams {
  period?: "week" | "month" | "quarter" | "year";
  start_date?: string;
  end_date?: string;
}

export const getSpendingAnalytics = async (
  params: GetSpendingAnalyticsParams = {}
): Promise<SpendingAnalytics> => {
  const response = await api.get("/financial/analytics/spending", { params });
  return response.data;
};

export const getFinancialHealthSummary =
  async (): Promise<FinancialHealthSummary> => {
    const response = await api.get("/financial/analytics/health");
    return response.data;
  };

// Recent transactions (for dashboard)
export const getRecentTransactions = async (
  limit: number = 5
): Promise<Transaction[]> => {
  const response = await api.get("/financial/transactions", {
    params: { limit, page: 1 },
  });
  return response.data.items || response.data;
};
