// Financial-related types matching backend models

export type TransactionType = "debit" | "credit";

export type TransactionCategory =
  | "groceries"
  | "dine_out"
  | "entertainment"
  | "bills"
  | "transport"
  | "shopping"
  | "health"
  | "fitness"
  | "savings"
  | "income"
  | "other";

export interface TransactionCreate {
  amount: number;
  category: TransactionCategory;
  description: string;
  merchant_name?: string;
  location_city?: string;
  location_country?: string;
}

export interface Transaction {
  id: string;
  transaction_type: TransactionType;
  amount: number;
  category: string;
  description: string;
  merchant_name?: string;
  location_city?: string;
  location_country: string;
  balance_after: number;
  transaction_date: Date;
}

export interface Wallet {
  id: string;
  balance: number;
  currency: string;
  updated_at: Date;
}

export interface SpendingAnalytics {
  category: string;
  total_amount: number;
  transaction_count: number;
  percentage: number;
}

export interface FinancialHealthSummary {
  current_balance: number;
  weekly_spending: number;
  weekly_income: number;
  transaction_count: number;
  savings_rate: number;
  daily_spending_avg: number;
  top_spending_categories: SpendingAnalytics[];
}

export interface TransactionFilter {
  limit?: number;
  days?: number;
  category?: string;
  transaction_type?: TransactionType;
  min_amount?: number;
  max_amount?: number;
}

// Frontend-specific financial types
export interface TransactionFormData {
  amount: string; // String for form input
  category: TransactionCategory;
  description: string;
  merchant_name?: string;
  location_city?: string;
  location_country?: string;
}

export interface TransactionListState {
  transactions: Transaction[];
  isLoading: boolean;
  error: string | null;
  hasMore: boolean;
  page: number;
}

export interface WalletState {
  wallet: Wallet | null;
  isLoading: boolean;
  error: string | null;
  lastUpdated: Date | null;
}

export interface FinancialFilters {
  dateRange: {
    start: Date | null;
    end: Date | null;
  };
  categories: TransactionCategory[];
  transactionTypes: TransactionType[];
  amountRange: {
    min: number | null;
    max: number | null;
  };
  searchTerm: string;
}

export interface SpendingTrend {
  date: Date;
  amount: number;
  category?: string;
}

export interface BudgetGoal {
  id: string;
  category: TransactionCategory;
  monthly_limit: number;
  current_spent: number;
  percentage_used: number;
  is_exceeded: boolean;
}

export interface FinancialInsight {
  type: "warning" | "success" | "info";
  title: string;
  message: string;
  category?: TransactionCategory;
  amount?: number;
  suggestion?: string;
}

export interface MonthlyReport {
  month: string;
  year: number;
  total_income: number;
  total_expenses: number;
  net_savings: number;
  category_breakdown: SpendingAnalytics[];
  insights: FinancialInsight[];
}

// Analytics-specific types
export interface BalanceHistoryPoint {
  date: string;
  balance: number;
  transaction_type: TransactionType;
  amount: number;
}

export interface MonthlySummary {
  month: number;
  year: number;
  total_income: number;
  total_spending: number;
  net_change: number;
  transaction_count: number;
  average_transaction: number;
}

export interface AnalyticsTimeRange {
  label: string;
  days: number;
  value: string;
}

export interface FinancialMetric {
  label: string;
  value: string | number;
  change?: number;
  changeType?: "increase" | "decrease" | "neutral";
  format: "currency" | "percentage" | "number";
  status?: "good" | "warning" | "danger";
}

export interface SpendingTrendData {
  date: string;
  spending: number;
  income: number;
  balance: number;
}

export interface CategorySpendingData {
  category: string;
  amount: number;
  percentage: number;
  color: string;
  transactions: number;
}
