// Product and marketplace-related types matching backend models

export interface ProductCategory {
  id: string;
  name: string;
  description?: string;
  transaction_category: string;
}

export interface Product {
  id: string;
  name: string;
  description?: string;
  price: number;
  category_id: string;
  merchant_name: string;
  is_available: boolean;
  stock_quantity: number;
  image_url?: string;
}

export interface ProductCreate {
  name: string;
  description?: string;
  price: number;
  category_id: string;
  merchant_name: string;
  stock_quantity?: number;
  image_url?: string;
}

export interface PurchaseRequest {
  product_id: string;
  quantity?: number;
}

export interface Purchase {
  id: string;
  product: Product;
  quantity: number;
  total_amount: number;
  transaction_id: string;
  purchase_date: Date;
}

export interface ProductFilter {
  category_id?: string;
  search?: string;
  min_price?: number;
  max_price?: number;
  limit?: number;
  available_only?: boolean;
}

export interface CategoryCreate {
  name: string;
  description?: string;
  transaction_category: string;
}

// Frontend-specific product types
export interface ProductWithCategory extends Product {
  category: ProductCategory;
}

export interface CartItem {
  product: Product;
  quantity: number;
  subtotal: number;
}

export interface ShoppingCart {
  items: CartItem[];
  total_items: number;
  total_amount: number;
  updated_at: Date;
}

export interface ProductListState {
  products: Product[];
  categories: ProductCategory[];
  isLoading: boolean;
  error: string | null;
  hasMore: boolean;
  page: number;
}

export interface ProductFilters {
  categories: string[];
  priceRange: {
    min: number | null;
    max: number | null;
  };
  searchTerm: string;
  sortBy: "name" | "price" | "popularity" | "newest";
  sortOrder: "asc" | "desc";
  availableOnly: boolean;
}

export interface PurchaseFormData {
  product_id: string;
  quantity: number;
  notes?: string;
}

export interface PurchaseHistory {
  purchases: Purchase[];
  total_count: number;
  total_spent: number;
  date_range: {
    start: Date;
    end: Date;
  };
}

export interface ProductReview {
  id: string;
  product_id: string;
  user_id: string;
  rating: number;
  comment?: string;
  created_at: Date;
}

export interface ProductWithReviews extends Product {
  reviews: ProductReview[];
  average_rating: number;
  review_count: number;
}

export interface WishlistItem {
  id: string;
  product: Product;
  added_at: Date;
  priority: "low" | "medium" | "high";
}

export interface MarketplaceStats {
  total_products: number;
  total_categories: number;
  popular_products: Product[];
  recent_purchases: Purchase[];
  trending_categories: ProductCategory[];
}

export interface ProductSearchResult {
  products: Product[];
  total_count: number;
  search_term: string;
  filters_applied: ProductFilters;
  suggestions: string[];
}
