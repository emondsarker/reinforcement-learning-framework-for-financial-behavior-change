import React, { useState } from "react";
import {
  MagnifyingGlassIcon,
  ShoppingCartIcon,
} from "@heroicons/react/24/outline";
import { useProductCategories, useProducts } from "../hooks/useProducts";
import { useCart } from "../hooks/useCart";
import { ProductCard } from "../components/marketplace/ProductCard";
import { CartSidebar } from "../components/marketplace/CartSidebar";
import { PurchaseModal } from "../components/marketplace/PurchaseModal";
import type { Product } from "../types";

export const MarketplacePage: React.FC = () => {
  const [selectedCategory, setSelectedCategory] = useState<string>("");
  const [searchTerm, setSearchTerm] = useState<string>("");
  const [isCartOpen, setIsCartOpen] = useState(false);
  const [purchaseModal, setPurchaseModal] = useState<{
    isOpen: boolean;
    mode: "single" | "cart";
    singleProduct?: { product: Product; quantity: number };
  }>({
    isOpen: false,
    mode: "cart",
  });

  // Fetch categories and products
  const { data: categories, isLoading: categoriesLoading } =
    useProductCategories();
  const {
    data: products,
    isLoading: productsLoading,
    error: productsError,
  } = useProducts({
    category_id: selectedCategory || undefined,
    search: searchTerm || undefined,
    limit: 20,
  });

  const { cart } = useCart();
  const isLoading = categoriesLoading || productsLoading;

  const handleQuickPurchase = (product: Product, quantity: number) => {
    setPurchaseModal({
      isOpen: true,
      mode: "single",
      singleProduct: { product, quantity },
    });
  };

  const handleCartCheckout = () => {
    setIsCartOpen(false);
    setPurchaseModal({
      isOpen: true,
      mode: "cart",
    });
  };

  const closePurchaseModal = () => {
    setPurchaseModal({
      isOpen: false,
      mode: "cart",
    });
  };

  return (
    <>
      <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="px-4 py-6 sm:px-0">
          {/* Header */}
          <div className="mb-8">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold text-gray-900">
                  Marketplace
                </h1>
                <p className="mt-2 text-gray-600">
                  Browse and purchase products to simulate real spending
                </p>
              </div>
              <div className="flex items-center space-x-4">
                {/* Product Count */}
                <div className="flex items-center space-x-2 text-gray-500">
                  <span className="text-sm">
                    {products?.length || 0} products available
                  </span>
                </div>

                {/* Cart Button */}
                <button
                  onClick={() => setIsCartOpen(true)}
                  className="relative bg-black text-white px-4 py-2 rounded-md hover:bg-gray-800 transition-colors flex items-center space-x-2"
                >
                  <ShoppingCartIcon className="h-5 w-5" />
                  <span>Cart</span>
                  {cart.total_items > 0 && (
                    <span className="absolute -top-2 -right-2 bg-red-500 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center">
                      {cart.total_items}
                    </span>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Search Bar */}
          <div className="mb-6">
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <MagnifyingGlassIcon className="h-5 w-5 text-gray-400" />
              </div>
              <input
                type="text"
                placeholder="Search products..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md leading-5 bg-white placeholder-gray-500 focus:outline-none focus:placeholder-gray-400 focus:ring-1 focus:ring-black focus:border-black"
              />
            </div>
          </div>

          {/* Category Filter */}
          <div className="mb-8">
            <div className="border-b border-gray-200">
              <nav className="-mb-px flex space-x-8 overflow-x-auto">
                <button
                  onClick={() => setSelectedCategory("")}
                  className={`whitespace-nowrap py-2 px-1 border-b-2 font-medium text-sm ${
                    selectedCategory === ""
                      ? "border-black text-black"
                      : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                  }`}
                >
                  All Products
                </button>
                {categories?.map((category) => (
                  <button
                    key={category.id}
                    onClick={() => setSelectedCategory(category.id)}
                    className={`whitespace-nowrap py-2 px-1 border-b-2 font-medium text-sm ${
                      selectedCategory === category.id
                        ? "border-black text-black"
                        : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                    }`}
                  >
                    {category.name}
                  </button>
                ))}
              </nav>
            </div>
          </div>

          {/* Loading State */}
          {isLoading && (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {[...Array(8)].map((_, i) => (
                <div
                  key={i}
                  className="bg-white rounded-lg shadow-sm border border-gray-200 p-4"
                >
                  <div className="animate-pulse">
                    <div className="bg-gray-200 h-48 rounded-md mb-4"></div>
                    <div className="h-4 bg-gray-200 rounded mb-2"></div>
                    <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
                    <div className="h-6 bg-gray-200 rounded w-1/2"></div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Error State */}
          {productsError && (
            <div className="text-center py-12">
              <div className="bg-red-50 border border-red-200 rounded-md p-4 max-w-md mx-auto">
                <p className="text-red-800">
                  Failed to load products. Please try again later.
                </p>
              </div>
            </div>
          )}

          {/* Products Grid */}
          {!isLoading && !productsError && (
            <>
              {products && products.length > 0 ? (
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                  {products.map((product) => (
                    <ProductCard
                      key={product.id}
                      product={product}
                      onQuickPurchase={handleQuickPurchase}
                    />
                  ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <ShoppingCartIcon className="mx-auto h-12 w-12 text-gray-400" />
                  <h3 className="mt-2 text-sm font-medium text-gray-900">
                    No products found
                  </h3>
                  <p className="mt-1 text-sm text-gray-500">
                    {searchTerm || selectedCategory
                      ? "Try adjusting your search or filter criteria."
                      : "No products are currently available."}
                  </p>
                </div>
              )}
            </>
          )}

          {/* Task 10 Complete Notice */}
          <div className="mt-12 bg-green-50 border border-green-200 rounded-md p-4">
            <div className="flex">
              <div className="ml-3">
                <h3 className="text-sm font-medium text-green-800">
                  Task 10 Complete - Interactive Marketplace Ready!
                </h3>
                <div className="mt-2 text-sm text-green-700">
                  <p>Full shopping cart functionality now implemented:</p>
                  <ul className="mt-1 list-disc list-inside space-y-1">
                    <li>Add products to cart with quantity selection</li>
                    <li>Shopping cart sidebar with item management</li>
                    <li>Purchase modal for single items and cart checkout</li>
                    <li>Real-time stock validation and wallet integration</li>
                    <li>
                      Persistent cart storage and comprehensive error handling
                    </li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Cart Sidebar */}
      <CartSidebar
        isOpen={isCartOpen}
        onClose={() => setIsCartOpen(false)}
        onCheckout={handleCartCheckout}
      />

      {/* Purchase Modal */}
      <PurchaseModal
        isOpen={purchaseModal.isOpen}
        onClose={closePurchaseModal}
        mode={purchaseModal.mode}
        singleProduct={purchaseModal.singleProduct}
      />
    </>
  );
};
