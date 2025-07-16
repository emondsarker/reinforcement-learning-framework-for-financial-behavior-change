import api from "./api";
import type {
  AdminStats,
  UserOverview,
  TransactionOverview,
  ProductAnalytics,
  SystemHealth,
  AdminDashboardData,
  AdminUserFilters,
  AdminTransactionFilters,
  AdminProductFilters,
  CreateProductForm,
  UpdateProductForm,
  CreateCategoryForm,
  UpdateCategoryForm,
} from "../types";
import type {
  User,
  Transaction,
  Product,
  ProductCategory,
  PaginatedResponse,
} from "../types";

export class AdminService {
  // System Statistics
  static async getSystemStats(): Promise<AdminStats> {
    try {
      // Get data from multiple endpoints to calculate stats
      const [usersResponse, transactionsResponse, productsResponse] =
        await Promise.all([
          api.get("/auth/users"), // This endpoint might not exist, will simulate
          api.get("/financial/transactions"),
          api.get("/products"),
        ]);

      // Calculate stats from the responses
      const totalUsers = usersResponse.data?.length || 0;
      const activeUsers =
        usersResponse.data?.filter((user: User) => user.is_active).length || 0;
      const totalTransactions = transactionsResponse.data?.length || 0;
      const totalRevenue =
        transactionsResponse.data
          ?.filter((t: Transaction) => t.transaction_type === "debit")
          .reduce((sum: number, t: Transaction) => sum + Number(t.amount), 0) ||
        0;
      const totalProducts = productsResponse.data?.length || 0;
      const lowStockProducts =
        productsResponse.data?.filter((p: Product) => p.stock_quantity < 10)
          .length || 0;

      return {
        totalUsers,
        activeUsers,
        totalTransactions,
        totalRevenue,
        totalProducts,
        lowStockProducts,
        systemHealth: "healthy" as const,
        lastUpdated: new Date(),
      };
    } catch (error) {
      console.error("Failed to get system stats:", error);
      // Return default stats on error
      return {
        totalUsers: 0,
        activeUsers: 0,
        totalTransactions: 0,
        totalRevenue: 0,
        totalProducts: 0,
        lowStockProducts: 0,
        systemHealth: "critical" as const,
        lastUpdated: new Date(),
      };
    }
  }

  // User Management
  static async getAllUsers(
    filters?: AdminUserFilters,
    page = 1,
    limit = 20
  ): Promise<PaginatedResponse<UserOverview>> {
    try {
      // Since we don't have a dedicated admin users endpoint, we'll simulate this
      // In a real implementation, this would call an admin-specific endpoint
      const response = await api.get("/auth/users", {
        params: { page, limit, ...filters },
      });

      // Transform user data to include admin-specific fields
      const users: UserOverview[] = response.data.map((user: User) => ({
        id: user.id,
        email: user.email,
        firstName: user.first_name,
        lastName: user.last_name,
        createdAt: new Date(user.created_at),
        isActive: user.is_active,
        lastLogin: undefined, // Would come from user activity tracking
        transactionCount: 0, // Would be calculated from transactions
        totalSpent: 0, // Would be calculated from transactions
        currentBalance: 0, // Would come from wallet data
        walletId: undefined, // Would come from wallet relationship
      }));

      return {
        data: users,
        total: users.length,
        page,
        limit,
        has_more: page * limit < users.length,
        total_pages: Math.ceil(users.length / limit),
      };
    } catch (error) {
      console.error("Failed to get users:", error);
      throw new Error("Failed to fetch users");
    }
  }

  // Transaction Oversight
  static async getAllTransactions(
    filters?: AdminTransactionFilters,
    page = 1,
    limit = 50
  ): Promise<PaginatedResponse<TransactionOverview>> {
    try {
      const response = await api.get("/financial/transactions", {
        params: { page, limit, ...filters },
      });

      // Transform transaction data to include user information
      const transactions: TransactionOverview[] = response.data.map(
        (transaction: Transaction) => ({
          id: transaction.id,
          userId: "user-id", // Would come from transaction relationship
          userEmail: "user@example.com", // Would come from user lookup
          userName: "User Name", // Would come from user lookup
          transactionType: transaction.transaction_type,
          amount: Number(transaction.amount),
          category: transaction.category,
          description: transaction.description,
          merchantName: transaction.merchant_name,
          locationCity: transaction.location_city,
          locationCountry: transaction.location_country,
          balanceAfter: Number(transaction.balance_after),
          transactionDate: new Date(transaction.transaction_date),
          createdAt: new Date(transaction.transaction_date), // Use transaction_date as fallback
        })
      );

      return {
        data: transactions,
        total: transactions.length,
        page,
        limit,
        has_more: page * limit < transactions.length,
        total_pages: Math.ceil(transactions.length / limit),
      };
    } catch (error) {
      console.error("Failed to get transactions:", error);
      throw new Error("Failed to fetch transactions");
    }
  }

  // Product Analytics
  static async getProductAnalytics(
    filters?: AdminProductFilters,
    page = 1,
    limit = 20
  ): Promise<PaginatedResponse<ProductAnalytics>> {
    try {
      const [productsResponse, categoriesResponse] = await Promise.all([
        api.get("/products", { params: { page, limit, ...filters } }),
        api.get("/products/categories"),
      ]);

      const categories = categoriesResponse.data;
      const products: ProductAnalytics[] = productsResponse.data.map(
        (product: Product) => {
          const category = categories.find(
            (c: ProductCategory) => c.id === product.category_id
          );
          return {
            id: product.id,
            name: product.name,
            price: Number(product.price),
            categoryName: category?.name || "Unknown",
            merchantName: product.merchant_name,
            stockQuantity: product.stock_quantity,
            isAvailable: product.is_available,
            totalSales: 0, // Would be calculated from purchase data
            totalRevenue: 0, // Would be calculated from purchase data
            averageRating: undefined, // Would come from reviews
            createdAt: new Date(), // Would come from product creation timestamp
            updatedAt: new Date(), // Would come from product update timestamp
          };
        }
      );

      return {
        data: products,
        total: products.length,
        page,
        limit,
        has_more: page * limit < products.length,
        total_pages: Math.ceil(products.length / limit),
      };
    } catch (error) {
      console.error("Failed to get product analytics:", error);
      throw new Error("Failed to fetch product analytics");
    }
  }

  // System Health
  static async getSystemHealth(): Promise<SystemHealth> {
    try {
      const startTime = Date.now();

      // Test API responsiveness
      await api.get("/health");

      const responseTime = Date.now() - startTime;

      return {
        status:
          responseTime < 1000
            ? "healthy"
            : responseTime < 3000
              ? "warning"
              : "critical",
        uptime: Date.now() - (Date.now() - 24 * 60 * 60 * 1000), // Simulate 24h uptime
        apiResponseTime: responseTime,
        databaseConnections: 10, // Simulated
        errorRate: 0.01, // Simulated 1% error rate
        memoryUsage: 65, // Simulated 65% memory usage
        cpuUsage: 45, // Simulated 45% CPU usage
        lastChecked: new Date(),
      };
    } catch (error) {
      console.error("Failed to get system health:", error);
      return {
        status: "critical",
        uptime: 0,
        apiResponseTime: 0,
        databaseConnections: 0,
        errorRate: 1,
        memoryUsage: 0,
        cpuUsage: 0,
        lastChecked: new Date(),
      };
    }
  }

  // Dashboard Data
  static async getDashboardData(): Promise<AdminDashboardData> {
    try {
      const [stats, users, transactions, products, systemHealth] =
        await Promise.all([
          this.getSystemStats(),
          this.getAllUsers(undefined, 1, 5), // Get recent 5 users
          this.getAllTransactions(undefined, 1, 10), // Get recent 10 transactions
          this.getProductAnalytics(undefined, 1, 5), // Get top 5 products
          this.getSystemHealth(),
        ]);

      return {
        stats,
        recentUsers: users.data,
        recentTransactions: transactions.data,
        topProducts: products.data,
        systemHealth,
      };
    } catch (error) {
      console.error("Failed to get dashboard data:", error);
      throw new Error("Failed to fetch dashboard data");
    }
  }

  // Product Management
  static async createProduct(productData: CreateProductForm): Promise<Product> {
    try {
      const response = await api.post("/products", productData);
      return response.data;
    } catch (error) {
      console.error("Failed to create product:", error);
      throw new Error("Failed to create product");
    }
  }

  static async updateProduct(productData: UpdateProductForm): Promise<Product> {
    try {
      const { id, ...updateData } = productData;
      const response = await api.put(`/products/${id}`, updateData);
      return response.data;
    } catch (error) {
      console.error("Failed to update product:", error);
      throw new Error("Failed to update product");
    }
  }

  static async deleteProduct(productId: string): Promise<void> {
    try {
      await api.delete(`/products/${productId}`);
    } catch (error) {
      console.error("Failed to delete product:", error);
      throw new Error("Failed to delete product");
    }
  }

  // Category Management
  static async createCategory(
    categoryData: CreateCategoryForm
  ): Promise<ProductCategory> {
    try {
      const response = await api.post("/products/categories", categoryData);
      return response.data;
    } catch (error) {
      console.error("Failed to create category:", error);
      throw new Error("Failed to create category");
    }
  }

  static async updateCategory(
    categoryData: UpdateCategoryForm
  ): Promise<ProductCategory> {
    try {
      const { id, ...updateData } = categoryData;
      const response = await api.put(`/products/categories/${id}`, updateData);
      return response.data;
    } catch (error) {
      console.error("Failed to update category:", error);
      throw new Error("Failed to update category");
    }
  }

  static async deleteCategory(categoryId: string): Promise<void> {
    try {
      await api.delete(`/products/categories/${categoryId}`);
    } catch (error) {
      console.error("Failed to delete category:", error);
      throw new Error("Failed to delete category");
    }
  }

  // User Management Actions
  static async toggleUserStatus(
    userId: string,
    isActive: boolean
  ): Promise<User> {
    try {
      const response = await api.patch(`/auth/users/${userId}`, {
        is_active: isActive,
      });
      return response.data;
    } catch (error) {
      console.error("Failed to toggle user status:", error);
      throw new Error("Failed to update user status");
    }
  }

  // Export functionality (future enhancement)
  static async exportData(
    type: "users" | "transactions" | "products"
  ): Promise<Blob> {
    try {
      const response = await api.get(`/admin/export/${type}`, {
        responseType: "blob",
      });
      return response.data;
    } catch (error) {
      console.error(`Failed to export ${type}:`, error);
      throw new Error(`Failed to export ${type}`);
    }
  }
}
