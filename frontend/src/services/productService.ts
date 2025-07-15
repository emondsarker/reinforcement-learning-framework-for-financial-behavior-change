import api from "./api";
import type {
  ProductCategory,
  Product,
  PurchaseRequest,
  Purchase,
} from "../types";

// Product filters interface for API calls
export interface ProductFilters {
  category_id?: string;
  search?: string;
  min_price?: number;
  max_price?: number;
  limit?: number;
  available_only?: boolean;
}

// Purchase history filters
export interface PurchaseFilters {
  limit?: number;
}

/**
 * Product Service - Handles all product-related API calls
 * Follows the same pattern as financialService for consistency
 */
export const productService = {
  // ===== PRODUCT CATEGORIES =====

  /**
   * Get all product categories
   * Used for category filtering in marketplace
   */
  async getProductCategories(): Promise<ProductCategory[]> {
    const response = await api.get("/products/categories");
    return response.data;
  },

  /**
   * Create a new product category (admin functionality)
   */
  async createProductCategory(categoryData: {
    name: string;
    description: string;
    transaction_category: string;
  }): Promise<ProductCategory> {
    const response = await api.post("/products/categories", categoryData);
    return response.data;
  },

  // ===== PRODUCTS =====

  /**
   * Get products with optional filtering
   * Supports category, search, price range, and pagination
   */
  async getProducts(filters: ProductFilters = {}): Promise<Product[]> {
    const params = new URLSearchParams();

    if (filters.category_id) params.append("category_id", filters.category_id);
    if (filters.search) params.append("search", filters.search);
    if (filters.min_price !== undefined)
      params.append("min_price", filters.min_price.toString());
    if (filters.max_price !== undefined)
      params.append("max_price", filters.max_price.toString());
    if (filters.limit !== undefined)
      params.append("limit", filters.limit.toString());
    if (filters.available_only !== undefined)
      params.append("available_only", filters.available_only.toString());

    const response = await api.get(`/products/?${params.toString()}`);
    return response.data;
  },

  /**
   * Get a specific product by ID
   */
  async getProductById(productId: string): Promise<Product> {
    const response = await api.get(`/products/${productId}`);
    return response.data;
  },

  /**
   * Create a new product (admin functionality)
   */
  async createProduct(productData: {
    name: string;
    description: string;
    price: number;
    category_id: string;
    merchant_name: string;
    stock_quantity?: number;
    image_url?: string;
  }): Promise<Product> {
    const response = await api.post("/products/", productData);
    return response.data;
  },

  /**
   * Get popular products based on purchase history
   */
  async getPopularProducts(
    limit: number = 10,
    days: number = 30
  ): Promise<Product[]> {
    const params = new URLSearchParams();
    params.append("limit", limit.toString());
    params.append("days", days.toString());

    const response = await api.get(`/products/popular?${params.toString()}`);
    return response.data;
  },

  /**
   * Get products by category ID
   */
  async getProductsByCategory(
    categoryId: string,
    limit: number = 20
  ): Promise<Product[]> {
    const params = new URLSearchParams();
    params.append("limit", limit.toString());

    const response = await api.get(
      `/products/category/${categoryId}/products?${params.toString()}`
    );
    return response.data;
  },

  /**
   * Search products by name, description, or merchant
   */
  async searchProducts(
    searchTerm: string,
    limit: number = 20
  ): Promise<Product[]> {
    const params = new URLSearchParams();
    params.append("limit", limit.toString());

    const response = await api.get(
      `/products/search/${encodeURIComponent(searchTerm)}?${params.toString()}`
    );
    return response.data;
  },

  // ===== PURCHASES =====

  /**
   * Purchase a product
   * Creates a transaction and updates wallet balance
   */
  async purchaseProduct(purchaseData: PurchaseRequest): Promise<Purchase> {
    const response = await api.post("/products/purchase", purchaseData);
    return response.data;
  },

  /**
   * Get user's purchase history
   */
  async getUserPurchases(filters: PurchaseFilters = {}): Promise<Purchase[]> {
    const params = new URLSearchParams();
    if (filters.limit !== undefined)
      params.append("limit", filters.limit.toString());

    const response = await api.get(
      `/products/purchases/history?${params.toString()}`
    );
    return response.data;
  },
};

export default productService;
