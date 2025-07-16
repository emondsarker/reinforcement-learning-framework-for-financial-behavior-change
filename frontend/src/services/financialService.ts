import api from "./api";
import type {
  Wallet,
  Transaction,
  TransactionCreate,
  SpendingAnalytics,
  FinancialHealthSummary,
  BalanceHistoryPoint,
  MonthlySummary,
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

// Analytics operations - matching backend endpoints
export const getSpendingAnalyticsByCategory = async (
  days: number = 30
): Promise<SpendingAnalytics[]> => {
  const response = await api.get("/financial/analytics/spending-by-category", {
    params: { days },
  });
  return response.data;
};

export const getFinancialHealthSummary =
  async (): Promise<FinancialHealthSummary> => {
    const response = await api.get("/financial/health-summary");
    return response.data;
  };

export const getBalanceHistory = async (
  days: number = 30
): Promise<BalanceHistoryPoint[]> => {
  const response = await api.get("/financial/balance-history", {
    params: { days },
  });
  return response.data.balance_history;
};

export const getMonthlySummary = async (
  year: number,
  month: number
): Promise<MonthlySummary> => {
  const response = await api.get("/financial/monthly-summary", {
    params: { year, month },
  });
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
