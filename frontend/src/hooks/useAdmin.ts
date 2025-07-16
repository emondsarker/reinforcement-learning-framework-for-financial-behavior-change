import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { AdminService } from "../services/adminService";
import { useToast } from "./useToast";
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
  PaginatedResponse,
  User,
  Product,
  ProductCategory,
} from "../types";

// Query Keys
const adminKeys = {
  all: ["admin"] as const,
  stats: () => [...adminKeys.all, "stats"] as const,
  users: () => [...adminKeys.all, "users"] as const,
  usersList: (filters?: AdminUserFilters, page?: number, limit?: number) =>
    [...adminKeys.users(), "list", { filters, page, limit }] as const,
  transactions: () => [...adminKeys.all, "transactions"] as const,
  transactionsList: (
    filters?: AdminTransactionFilters,
    page?: number,
    limit?: number
  ) => [...adminKeys.transactions(), "list", { filters, page, limit }] as const,
  products: () => [...adminKeys.all, "products"] as const,
  productsList: (
    filters?: AdminProductFilters,
    page?: number,
    limit?: number
  ) => [...adminKeys.products(), "list", { filters, page, limit }] as const,
  systemHealth: () => [...adminKeys.all, "systemHealth"] as const,
  dashboard: () => [...adminKeys.all, "dashboard"] as const,
};

// System Statistics Hook
export const useAdminStats = () => {
  return useQuery({
    queryKey: adminKeys.stats(),
    queryFn: AdminService.getSystemStats,
    staleTime: 1 * 60 * 1000, // 1 minute
    gcTime: 5 * 60 * 1000, // 5 minutes
    refetchInterval: 2 * 60 * 1000, // Refetch every 2 minutes
  });
};

// User Management Hooks
export const useAdminUsers = (
  filters?: AdminUserFilters,
  page = 1,
  limit = 20
) => {
  return useQuery({
    queryKey: adminKeys.usersList(filters, page, limit),
    queryFn: () => AdminService.getAllUsers(filters, page, limit),
    staleTime: 1 * 60 * 1000, // 1 minute
    gcTime: 5 * 60 * 1000, // 5 minutes
  });
};

export const useToggleUserStatus = () => {
  const queryClient = useQueryClient();
  const { addToast } = useToast();

  return useMutation({
    mutationFn: ({ userId, isActive }: { userId: string; isActive: boolean }) =>
      AdminService.toggleUserStatus(userId, isActive),
    onSuccess: (updatedUser: User) => {
      // Invalidate users queries
      queryClient.invalidateQueries({ queryKey: adminKeys.users() });
      queryClient.invalidateQueries({ queryKey: adminKeys.stats() });

      addToast({
        type: "success",
        title: "User Status Updated",
        message: `User ${updatedUser.email} has been ${
          updatedUser.is_active ? "activated" : "deactivated"
        }`,
      });
    },
    onError: (error: Error) => {
      addToast({
        type: "error",
        title: "Failed to Update User",
        message: error.message,
      });
    },
  });
};

// Transaction Oversight Hooks
export const useAdminTransactions = (
  filters?: AdminTransactionFilters,
  page = 1,
  limit = 50
) => {
  return useQuery({
    queryKey: adminKeys.transactionsList(filters, page, limit),
    queryFn: () => AdminService.getAllTransactions(filters, page, limit),
    staleTime: 1 * 60 * 1000, // 1 minute
    gcTime: 5 * 60 * 1000, // 5 minutes
  });
};

// Product Analytics Hooks
export const useAdminProducts = (
  filters?: AdminProductFilters,
  page = 1,
  limit = 20
) => {
  return useQuery({
    queryKey: adminKeys.productsList(filters, page, limit),
    queryFn: () => AdminService.getProductAnalytics(filters, page, limit),
    staleTime: 2 * 60 * 1000, // 2 minutes
    gcTime: 10 * 60 * 1000, // 10 minutes
  });
};

export const useCreateProduct = () => {
  const queryClient = useQueryClient();
  const { addToast } = useToast();

  return useMutation({
    mutationFn: (productData: CreateProductForm) =>
      AdminService.createProduct(productData),
    onSuccess: (newProduct: Product) => {
      // Invalidate product queries
      queryClient.invalidateQueries({ queryKey: adminKeys.products() });
      queryClient.invalidateQueries({ queryKey: adminKeys.stats() });

      addToast({
        type: "success",
        title: "Product Created",
        message: `Product "${newProduct.name}" has been created successfully`,
      });
    },
    onError: (error: Error) => {
      addToast({
        type: "error",
        title: "Failed to Create Product",
        message: error.message,
      });
    },
  });
};

export const useUpdateProduct = () => {
  const queryClient = useQueryClient();
  const { addToast } = useToast();

  return useMutation({
    mutationFn: (productData: UpdateProductForm) =>
      AdminService.updateProduct(productData),
    onSuccess: (updatedProduct: Product) => {
      // Invalidate product queries
      queryClient.invalidateQueries({ queryKey: adminKeys.products() });

      addToast({
        type: "success",
        title: "Product Updated",
        message: `Product "${updatedProduct.name}" has been updated successfully`,
      });
    },
    onError: (error: Error) => {
      addToast({
        type: "error",
        title: "Failed to Update Product",
        message: error.message,
      });
    },
  });
};

export const useDeleteProduct = () => {
  const queryClient = useQueryClient();
  const { addToast } = useToast();

  return useMutation({
    mutationFn: (productId: string) => AdminService.deleteProduct(productId),
    onSuccess: () => {
      // Invalidate product queries
      queryClient.invalidateQueries({ queryKey: adminKeys.products() });
      queryClient.invalidateQueries({ queryKey: adminKeys.stats() });

      addToast({
        type: "success",
        title: "Product Deleted",
        message: "Product has been deleted successfully",
      });
    },
    onError: (error: Error) => {
      addToast({
        type: "error",
        title: "Failed to Delete Product",
        message: error.message,
      });
    },
  });
};

// Category Management Hooks
export const useCreateCategory = () => {
  const queryClient = useQueryClient();
  const { addToast } = useToast();

  return useMutation({
    mutationFn: (categoryData: CreateCategoryForm) =>
      AdminService.createCategory(categoryData),
    onSuccess: (newCategory: ProductCategory) => {
      // Invalidate product queries (categories affect products)
      queryClient.invalidateQueries({ queryKey: adminKeys.products() });

      addToast({
        type: "success",
        title: "Category Created",
        message: `Category "${newCategory.name}" has been created successfully`,
      });
    },
    onError: (error: Error) => {
      addToast({
        type: "error",
        title: "Failed to Create Category",
        message: error.message,
      });
    },
  });
};

export const useUpdateCategory = () => {
  const queryClient = useQueryClient();
  const { addToast } = useToast();

  return useMutation({
    mutationFn: (categoryData: UpdateCategoryForm) =>
      AdminService.updateCategory(categoryData),
    onSuccess: (updatedCategory: ProductCategory) => {
      // Invalidate product queries (categories affect products)
      queryClient.invalidateQueries({ queryKey: adminKeys.products() });

      addToast({
        type: "success",
        title: "Category Updated",
        message: `Category "${updatedCategory.name}" has been updated successfully`,
      });
    },
    onError: (error: Error) => {
      addToast({
        type: "error",
        title: "Failed to Update Category",
        message: error.message,
      });
    },
  });
};

export const useDeleteCategory = () => {
  const queryClient = useQueryClient();
  const { addToast } = useToast();

  return useMutation({
    mutationFn: (categoryId: string) => AdminService.deleteCategory(categoryId),
    onSuccess: () => {
      // Invalidate product queries (categories affect products)
      queryClient.invalidateQueries({ queryKey: adminKeys.products() });

      addToast({
        type: "success",
        title: "Category Deleted",
        message: "Category has been deleted successfully",
      });
    },
    onError: (error: Error) => {
      addToast({
        type: "error",
        title: "Failed to Delete Category",
        message: error.message,
      });
    },
  });
};

// System Health Hook
export const useSystemHealth = () => {
  return useQuery({
    queryKey: adminKeys.systemHealth(),
    queryFn: AdminService.getSystemHealth,
    staleTime: 30 * 1000, // 30 seconds
    gcTime: 2 * 60 * 1000, // 2 minutes
    refetchInterval: 60 * 1000, // Refetch every minute
  });
};

// Dashboard Data Hook
export const useAdminDashboard = () => {
  return useQuery({
    queryKey: adminKeys.dashboard(),
    queryFn: AdminService.getDashboardData,
    staleTime: 1 * 60 * 1000, // 1 minute
    gcTime: 5 * 60 * 1000, // 5 minutes
    refetchInterval: 2 * 60 * 1000, // Refetch every 2 minutes
  });
};

// Export Data Hook
export const useExportData = () => {
  const { addToast } = useToast();

  return useMutation({
    mutationFn: (type: "users" | "transactions" | "products") =>
      AdminService.exportData(type),
    onSuccess: (blob: Blob, variables) => {
      // Create download link
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${variables}_export_${new Date().toISOString().split("T")[0]}.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

      addToast({
        type: "success",
        title: "Export Successful",
        message: `${variables} data has been exported successfully`,
      });
    },
    onError: (error: Error) => {
      addToast({
        type: "error",
        title: "Export Failed",
        message: error.message,
      });
    },
  });
};

// Utility Hooks
export const useAdminLoadingState = () => {
  const statsQuery = useAdminStats();
  const healthQuery = useSystemHealth();

  return {
    isLoading: statsQuery.isLoading || healthQuery.isLoading,
    isError: statsQuery.isError || healthQuery.isError,
    error: statsQuery.error || healthQuery.error,
  };
};

export const useRefreshAdminData = () => {
  const queryClient = useQueryClient();

  return () => {
    queryClient.invalidateQueries({ queryKey: adminKeys.all });
  };
};

// Prefetch hooks for performance
export const usePrefetchAdminData = () => {
  const queryClient = useQueryClient();

  return {
    prefetchStats: () => {
      queryClient.prefetchQuery({
        queryKey: adminKeys.stats(),
        queryFn: AdminService.getSystemStats,
        staleTime: 1 * 60 * 1000,
      });
    },
    prefetchUsers: (filters?: AdminUserFilters, page = 1, limit = 20) => {
      queryClient.prefetchQuery({
        queryKey: adminKeys.usersList(filters, page, limit),
        queryFn: () => AdminService.getAllUsers(filters, page, limit),
        staleTime: 1 * 60 * 1000,
      });
    },
    prefetchTransactions: (
      filters?: AdminTransactionFilters,
      page = 1,
      limit = 50
    ) => {
      queryClient.prefetchQuery({
        queryKey: adminKeys.transactionsList(filters, page, limit),
        queryFn: () => AdminService.getAllTransactions(filters, page, limit),
        staleTime: 1 * 60 * 1000,
      });
    },
    prefetchProducts: (filters?: AdminProductFilters, page = 1, limit = 20) => {
      queryClient.prefetchQuery({
        queryKey: adminKeys.productsList(filters, page, limit),
        queryFn: () => AdminService.getProductAnalytics(filters, page, limit),
        staleTime: 2 * 60 * 1000,
      });
    },
  };
};
