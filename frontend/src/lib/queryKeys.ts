// Query key factory for consistent query key generation
// This follows the TanStack Query best practices for query key organization

export const queryKeys = {
  // Authentication related queries
  auth: {
    all: ["auth"] as const,
    user: () => [...queryKeys.auth.all, "user"] as const,
    profile: () => [...queryKeys.auth.all, "profile"] as const,
  },

  // Financial data queries
  financial: {
    all: ["financial"] as const,
    wallet: () => [...queryKeys.financial.all, "wallet"] as const,
    transactions: {
      all: () => [...queryKeys.financial.all, "transactions"] as const,
      list: (filters?: {
        page?: number;
        limit?: number;
        category?: string;
        dateFrom?: string;
        dateTo?: string;
      }) =>
        [...queryKeys.financial.transactions.all(), "list", filters] as const,
      detail: (id: string) =>
        [...queryKeys.financial.transactions.all(), "detail", id] as const,
    },
    analytics: {
      all: () => [...queryKeys.financial.all, "analytics"] as const,
      spending: (period?: string) =>
        [...queryKeys.financial.analytics.all(), "spending", period] as const,
      health: () => [...queryKeys.financial.analytics.all(), "health"] as const,
      balanceHistory: (days?: string) =>
        [
          ...queryKeys.financial.analytics.all(),
          "balance-history",
          days,
        ] as const,
      monthlySummary: (period?: string) =>
        [
          ...queryKeys.financial.analytics.all(),
          "monthly-summary",
          period,
        ] as const,
    },
  },

  // Product and marketplace queries
  products: {
    all: ["products"] as const,
    categories: () => [...queryKeys.products.all, "categories"] as const,
    list: (filters?: {
      category?: string;
      search?: string;
      page?: number;
      limit?: number;
    }) => [...queryKeys.products.all, "list", filters] as const,
    detail: (id: string) => [...queryKeys.products.all, "detail", id] as const,
    purchases: {
      all: () => [...queryKeys.products.all, "purchases"] as const,
      list: (filters?: { page?: number; limit?: number }) =>
        [...queryKeys.products.purchases.all(), "list", filters] as const,
      detail: (id: string) =>
        [...queryKeys.products.purchases.all(), "detail", id] as const,
    },
  },

  // AI Coaching queries
  coaching: {
    all: ["coaching"] as const,
    recommendations: {
      all: () => [...queryKeys.coaching.all, "recommendations"] as const,
      list: (filters?: { type?: string; status?: string }) =>
        [...queryKeys.coaching.recommendations.all(), "list", filters] as const,
      detail: (id: string) =>
        [...queryKeys.coaching.recommendations.all(), "detail", id] as const,
    },
    financialState: () =>
      [...queryKeys.coaching.all, "financial-state"] as const,
    insights: () => [...queryKeys.coaching.all, "insights"] as const,
  },

  // Admin queries (if needed)
  admin: {
    all: ["admin"] as const,
    users: {
      all: () => [...queryKeys.admin.all, "users"] as const,
      list: (filters?: { page?: number; limit?: number; search?: string }) =>
        [...queryKeys.admin.users.all(), "list", filters] as const,
      detail: (id: string) =>
        [...queryKeys.admin.users.all(), "detail", id] as const,
    },
    system: {
      all: () => [...queryKeys.admin.all, "system"] as const,
      health: () => [...queryKeys.admin.system.all(), "health"] as const,
      metrics: () => [...queryKeys.admin.system.all(), "metrics"] as const,
    },
  },
} as const;

// Helper functions for query invalidation
export const invalidateQueries = {
  // Invalidate all auth-related queries
  auth: () => queryKeys.auth.all,

  // Invalidate all financial queries
  financial: () => queryKeys.financial.all,

  // Invalidate specific financial data
  wallet: () => queryKeys.financial.wallet(),
  transactions: () => queryKeys.financial.transactions.all(),

  // Invalidate all product queries
  products: () => queryKeys.products.all,

  // Invalidate all coaching queries
  coaching: () => queryKeys.coaching.all,

  // Invalidate all admin queries
  admin: () => queryKeys.admin.all,
} as const;

// Mutation keys for consistent mutation identification
export const mutationKeys = {
  auth: {
    login: ["auth", "login"] as const,
    register: ["auth", "register"] as const,
    logout: ["auth", "logout"] as const,
    updateProfile: ["auth", "update-profile"] as const,
  },

  financial: {
    createTransaction: ["financial", "create-transaction"] as const,
    updateTransaction: ["financial", "update-transaction"] as const,
    deleteTransaction: ["financial", "delete-transaction"] as const,
  },

  products: {
    purchase: ["products", "purchase"] as const,
  },

  coaching: {
    provideFeedback: ["coaching", "provide-feedback"] as const,
    updateFinancialState: ["coaching", "update-financial-state"] as const,
  },
} as const;

export default queryKeys;
