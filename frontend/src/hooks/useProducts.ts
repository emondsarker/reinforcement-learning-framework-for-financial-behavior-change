import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  productService,
  type ProductFilters,
  type PurchaseFilters,
} from "../services/productService";
import { queryKeys, mutationKeys, invalidateQueries } from "../lib/queryKeys";
import { useToast } from "./useToast";
import type { PurchaseRequest, Purchase } from "../types";

// ===== PRODUCT CATEGORIES =====

/**
 * Hook to fetch all product categories
 * Uses aggressive caching (10 minutes) since categories are relatively static
 */
export const useProductCategories = () => {
  return useQuery({
    queryKey: queryKeys.products.categories(),
    queryFn: productService.getProductCategories,
    staleTime: 10 * 60 * 1000, // 10 minutes
    gcTime: 15 * 60 * 1000, // 15 minutes
  });
};

/**
 * Mutation hook to create a new product category (admin functionality)
 */
export const useCreateProductCategory = () => {
  const queryClient = useQueryClient();
  const { addToast } = useToast();

  return useMutation({
    mutationKey: ["products", "create-category"],
    mutationFn: productService.createProductCategory,
    onSuccess: (newCategory) => {
      // Invalidate categories query to refetch
      queryClient.invalidateQueries({
        queryKey: queryKeys.products.categories(),
      });

      addToast({
        type: "success",
        title: "Category Created",
        message: `Category "${newCategory.name}" created successfully!`,
      });
    },
    onError: (error: Error) => {
      addToast({
        type: "error",
        title: "Creation Failed",
        message: `Failed to create category: ${error.message}`,
      });
    },
  });
};

// ===== PRODUCTS =====

/**
 * Hook to fetch products with optional filtering
 * Uses moderate caching (5 minutes) for product lists
 */
export const useProducts = (filters: ProductFilters = {}) => {
  return useQuery({
    queryKey: queryKeys.products.list(filters),
    queryFn: () => productService.getProducts(filters),
    staleTime: 5 * 60 * 1000, // 5 minutes
    gcTime: 10 * 60 * 1000, // 10 minutes
  });
};

/**
 * Hook to fetch a specific product by ID
 */
export const useProduct = (productId: string) => {
  return useQuery({
    queryKey: queryKeys.products.detail(productId),
    queryFn: () => productService.getProductById(productId),
    staleTime: 5 * 60 * 1000, // 5 minutes
    gcTime: 10 * 60 * 1000, // 10 minutes
    enabled: !!productId, // Only run query if productId is provided
  });
};

/**
 * Hook to fetch popular products
 */
export const usePopularProducts = (limit: number = 10, days: number = 30) => {
  return useQuery({
    queryKey: ["products", "popular", { limit, days }],
    queryFn: () => productService.getPopularProducts(limit, days),
    staleTime: 5 * 60 * 1000, // 5 minutes
    gcTime: 10 * 60 * 1000, // 10 minutes
  });
};

/**
 * Hook to fetch products by category
 */
export const useProductsByCategory = (
  categoryId: string,
  limit: number = 20
) => {
  return useQuery({
    queryKey: ["products", "by-category", categoryId, { limit }],
    queryFn: () => productService.getProductsByCategory(categoryId, limit),
    staleTime: 5 * 60 * 1000, // 5 minutes
    gcTime: 10 * 60 * 1000, // 10 minutes
    enabled: !!categoryId, // Only run query if categoryId is provided
  });
};

/**
 * Hook to search products
 */
export const useSearchProducts = (searchTerm: string, limit: number = 20) => {
  return useQuery({
    queryKey: ["products", "search", searchTerm, { limit }],
    queryFn: () => productService.searchProducts(searchTerm, limit),
    staleTime: 2 * 60 * 1000, // 2 minutes (search results change more frequently)
    gcTime: 5 * 60 * 1000, // 5 minutes
    enabled: !!searchTerm && searchTerm.length > 0, // Only search if term is provided
  });
};

/**
 * Mutation hook to create a new product (admin functionality)
 */
export const useCreateProduct = () => {
  const queryClient = useQueryClient();
  const { addToast } = useToast();

  return useMutation({
    mutationKey: ["products", "create-product"],
    mutationFn: productService.createProduct,
    onSuccess: (newProduct) => {
      // Invalidate all product queries to refetch
      queryClient.invalidateQueries({
        queryKey: invalidateQueries.products(),
      });

      addToast({
        type: "success",
        title: "Product Created",
        message: `Product "${newProduct.name}" created successfully!`,
      });
    },
    onError: (error: Error) => {
      addToast({
        type: "error",
        title: "Creation Failed",
        message: `Failed to create product: ${error.message}`,
      });
    },
  });
};

// ===== PURCHASES =====

/**
 * Hook to fetch user's purchase history
 */
export const useUserPurchases = (filters: PurchaseFilters = {}) => {
  return useQuery({
    queryKey: queryKeys.products.purchases.list(filters),
    queryFn: () => productService.getUserPurchases(filters),
    staleTime: 2 * 60 * 1000, // 2 minutes
    gcTime: 5 * 60 * 1000, // 5 minutes
  });
};

/**
 * Mutation hook to purchase a product
 * Includes optimistic updates and proper cache invalidation
 */
export const usePurchaseProduct = () => {
  const queryClient = useQueryClient();
  const { addToast } = useToast();

  return useMutation({
    mutationKey: mutationKeys.products.purchase,
    mutationFn: productService.purchaseProduct,
    onMutate: async (purchaseData: PurchaseRequest) => {
      // Show immediate feedback
      addToast({
        type: "info",
        title: "Processing Purchase",
        message: "Processing purchase...",
      });

      // Cancel any outgoing refetches for wallet and purchases
      await queryClient.cancelQueries({
        queryKey: queryKeys.financial.wallet(),
      });
      await queryClient.cancelQueries({
        queryKey: queryKeys.products.purchases.all(),
      });

      // Return context for rollback if needed
      return { purchaseData };
    },
    onSuccess: (purchase: Purchase) => {
      // Invalidate related queries to refetch fresh data
      queryClient.invalidateQueries({
        queryKey: queryKeys.financial.wallet(),
      });
      queryClient.invalidateQueries({
        queryKey: queryKeys.financial.transactions.all(),
      });
      queryClient.invalidateQueries({
        queryKey: queryKeys.products.purchases.all(),
      });

      addToast({
        type: "success",
        title: "Purchase Successful",
        message: `Successfully purchased ${purchase.product.name}!`,
      });
    },
    onError: (error: Error) => {
      // Rollback optimistic updates if any were made
      queryClient.invalidateQueries({
        queryKey: queryKeys.financial.wallet(),
      });
      queryClient.invalidateQueries({
        queryKey: queryKeys.products.purchases.all(),
      });

      addToast({
        type: "error",
        title: "Purchase Failed",
        message: `Purchase failed: ${error.message}`,
      });
    },
  });
};

// ===== UTILITY HOOKS =====

/**
 * Hook to get all product-related loading states
 * Useful for showing global loading indicators
 */
export const useProductsLoadingState = () => {
  const categoriesQuery = useProductCategories();
  const productsQuery = useProducts();

  return {
    isLoading: categoriesQuery.isLoading || productsQuery.isLoading,
    isError: categoriesQuery.isError || productsQuery.isError,
    error: categoriesQuery.error || productsQuery.error,
  };
};

/**
 * Hook to prefetch product data
 * Useful for preloading data on hover or navigation
 */
export const usePrefetchProduct = () => {
  const queryClient = useQueryClient();

  const prefetchProduct = (productId: string) => {
    queryClient.prefetchQuery({
      queryKey: queryKeys.products.detail(productId),
      queryFn: () => productService.getProductById(productId),
      staleTime: 5 * 60 * 1000,
    });
  };

  const prefetchProducts = (filters: ProductFilters = {}) => {
    queryClient.prefetchQuery({
      queryKey: queryKeys.products.list(filters),
      queryFn: () => productService.getProducts(filters),
      staleTime: 5 * 60 * 1000,
    });
  };

  return {
    prefetchProduct,
    prefetchProducts,
  };
};
