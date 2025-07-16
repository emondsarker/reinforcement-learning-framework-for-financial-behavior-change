// Admin-specific types for system management and monitoring
export interface AdminStats {
  totalUsers: number;
  activeUsers: number;
  totalTransactions: number;
  totalRevenue: number;
  totalProducts: number;
  lowStockProducts: number;
  systemHealth: "healthy" | "warning" | "critical";
  lastUpdated: Date;
}

export interface UserOverview {
  id: string;
  email: string;
  firstName: string;
  lastName: string;
  createdAt: Date;
  isActive: boolean;
  lastLogin?: Date;
  transactionCount: number;
  totalSpent: number;
  currentBalance: number;
  walletId?: string;
}

export interface TransactionOverview {
  id: string;
  userId: string;
  userEmail: string;
  userName: string;
  transactionType: "debit" | "credit";
  amount: number;
  category: string;
  description: string;
  merchantName?: string;
  locationCity?: string;
  locationCountry?: string;
  balanceAfter: number;
  transactionDate: Date;
  createdAt: Date;
}

export interface ProductAnalytics {
  id: string;
  name: string;
  price: number;
  categoryName: string;
  merchantName: string;
  stockQuantity: number;
  isAvailable: boolean;
  totalSales: number;
  totalRevenue: number;
  averageRating?: number;
  createdAt: Date;
  updatedAt: Date;
}

export interface SystemHealth {
  status: "healthy" | "warning" | "critical";
  uptime: number;
  apiResponseTime: number;
  databaseConnections: number;
  errorRate: number;
  memoryUsage: number;
  cpuUsage: number;
  lastChecked: Date;
}

export interface AdminDashboardData {
  stats: AdminStats;
  recentUsers: UserOverview[];
  recentTransactions: TransactionOverview[];
  topProducts: ProductAnalytics[];
  systemHealth: SystemHealth;
}

// Admin-specific filter and pagination types
export interface AdminUserFilters {
  search?: string;
  isActive?: boolean;
  dateFrom?: Date;
  dateTo?: Date;
  sortBy?: "createdAt" | "lastLogin" | "totalSpent" | "transactionCount";
  sortOrder?: "asc" | "desc";
}

export interface AdminTransactionFilters {
  search?: string;
  userId?: string;
  transactionType?: "debit" | "credit";
  category?: string;
  amountMin?: number;
  amountMax?: number;
  dateFrom?: Date;
  dateTo?: Date;
  sortBy?: "transactionDate" | "amount" | "balanceAfter";
  sortOrder?: "asc" | "desc";
}

export interface AdminProductFilters {
  search?: string;
  categoryId?: string;
  isAvailable?: boolean;
  lowStock?: boolean;
  sortBy?: "name" | "price" | "stockQuantity" | "totalSales";
  sortOrder?: "asc" | "desc";
}

// Admin action types for audit logging (future enhancement)
export interface AdminAction {
  id: string;
  adminId: string;
  adminEmail: string;
  action: "create" | "update" | "delete" | "view";
  resource: "user" | "product" | "transaction" | "category";
  resourceId: string;
  details?: Record<string, unknown>;
  timestamp: Date;
}

// Admin permissions (for future role-based access)
export interface AdminPermissions {
  canManageUsers: boolean;
  canManageProducts: boolean;
  canViewTransactions: boolean;
  canManageCategories: boolean;
  canViewSystemHealth: boolean;
  canExportData: boolean;
}

// Admin user type extending regular user
export interface AdminUser {
  id: string;
  email: string;
  firstName: string;
  lastName: string;
  isAdmin: boolean;
  permissions: AdminPermissions;
  lastAdminLogin?: Date;
}

// Form types for admin operations
export interface CreateProductForm {
  name: string;
  description: string;
  price: number;
  categoryId: string;
  merchantName: string;
  stockQuantity: number;
  imageUrl?: string;
}

export interface UpdateProductForm extends Partial<CreateProductForm> {
  id: string;
  isAvailable?: boolean;
}

export interface CreateCategoryForm {
  name: string;
  description: string;
  transactionCategory: string;
}

export interface UpdateCategoryForm extends Partial<CreateCategoryForm> {
  id: string;
}

// Admin loading states
export interface AdminLoadingState {
  stats: boolean;
  users: boolean;
  transactions: boolean;
  products: boolean;
  systemHealth: boolean;
}

// Admin error types
export interface AdminError {
  type: "fetch" | "create" | "update" | "delete";
  resource: "users" | "transactions" | "products" | "categories" | "system";
  message: string;
  details?: unknown;
}
