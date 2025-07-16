import React, { useState } from "react";
import type { ProductAnalytics } from "../../types";
import {
  useAdminProducts,
  useCreateProduct,
  useUpdateProduct,
  useDeleteProduct,
} from "../../hooks/useAdmin";
import {
  CubeIcon,
  MagnifyingGlassIcon,
  PlusIcon,
  PencilIcon,
  TrashIcon,
  ExclamationTriangleIcon,
} from "@heroicons/react/24/outline";

interface ProductManagementProps {
  topProducts?: ProductAnalytics[];
  compact?: boolean;
  className?: string;
}

export const ProductManagement: React.FC<ProductManagementProps> = ({
  topProducts,
  compact = false,
  className = "",
}) => {
  const [searchTerm, setSearchTerm] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = compact ? 5 : 10;

  const {
    data: productsData,
    isLoading,
    error,
  } = useAdminProducts({ search: searchTerm }, currentPage, pageSize);
  const createProduct = useCreateProduct();
  const updateProduct = useUpdateProduct();
  const deleteProduct = useDeleteProduct();

  const products = compact ? topProducts || [] : productsData?.data || [];
  const totalPages = compact ? 1 : productsData?.total_pages || 1;

  const getStockStatus = (quantity: number) => {
    if (quantity === 0)
      return { label: "Out of Stock", color: "text-red-600 bg-red-50" };
    if (quantity < 10)
      return { label: "Low Stock", color: "text-yellow-600 bg-yellow-50" };
    return { label: "In Stock", color: "text-green-600 bg-green-50" };
  };

  const handleDeleteProduct = async (
    productId: string,
    productName: string
  ) => {
    if (window.confirm(`Are you sure you want to delete "${productName}"?`)) {
      try {
        await deleteProduct.mutateAsync(productId);
      } catch (error) {
        console.error("Failed to delete product:", error);
      }
    }
  };

  if (compact) {
    return (
      <div className={className}>
        {topProducts && topProducts.length > 0 ? (
          <div className="space-y-3">
            {topProducts.slice(0, 5).map((product) => {
              const stockStatus = getStockStatus(product.stockQuantity);
              return (
                <div
                  key={product.id}
                  className="flex items-center justify-between py-2 border-b border-gray-100 last:border-b-0"
                >
                  <div className="flex items-center">
                    <CubeIcon className="h-8 w-8 text-gray-400 mr-3" />
                    <div>
                      <p className="text-sm font-medium text-gray-900">
                        {product.name}
                      </p>
                      <p className="text-xs text-gray-500">
                        {product.categoryName} • ${product.price.toFixed(2)}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <div
                      className={`inline-flex px-2 py-1 text-xs font-medium rounded-full ${stockStatus.color}`}
                    >
                      {stockStatus.label}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      {product.stockQuantity} units
                    </div>
                  </div>
                </div>
              );
            })}
            <div className="pt-3 border-t border-gray-200">
              <button className="text-sm text-blue-600 hover:text-blue-800 font-medium">
                View all products →
              </button>
            </div>
          </div>
        ) : (
          <div className="text-center py-8">
            <CubeIcon className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-2 text-sm font-medium text-gray-900">
              No products
            </h3>
            <p className="mt-1 text-sm text-gray-500">
              Top products will appear here.
            </p>
          </div>
        )}
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className={`bg-white rounded-lg shadow ${className}`}>
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <CubeIcon className="h-6 w-6 text-gray-400 mr-3" />
              <h2 className="text-lg font-semibold text-gray-900">
                Product Management
              </h2>
            </div>
          </div>
        </div>
        <div className="p-6">
          <div className="space-y-4">
            {[...Array(5)].map((_, i) => (
              <div
                key={i}
                className="flex items-center justify-between p-4 border border-gray-200 rounded-lg animate-pulse"
              >
                <div className="flex items-center">
                  <div className="h-12 w-12 bg-gray-200 rounded mr-4"></div>
                  <div>
                    <div className="h-4 bg-gray-200 rounded w-32 mb-2"></div>
                    <div className="h-3 bg-gray-200 rounded w-24"></div>
                  </div>
                </div>
                <div className="h-6 bg-gray-200 rounded w-20"></div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`bg-white rounded-lg shadow ${className}`}>
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center">
            <CubeIcon className="h-6 w-6 text-gray-400 mr-3" />
            <h2 className="text-lg font-semibold text-gray-900">
              Product Management
            </h2>
          </div>
        </div>
        <div className="p-6">
          <div className="text-center py-8">
            <ExclamationTriangleIcon className="mx-auto h-12 w-12 text-red-400" />
            <h3 className="mt-2 text-sm font-medium text-gray-900">
              Error loading products
            </h3>
            <p className="mt-1 text-sm text-gray-500">
              There was an error loading product data. Please try again.
            </p>
            <button
              onClick={() => window.location.reload()}
              className="mt-4 px-4 py-2 bg-black text-white rounded-md hover:bg-gray-800 transition-colors"
            >
              Retry
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-white rounded-lg shadow ${className}`}>
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <CubeIcon className="h-6 w-6 text-gray-400 mr-3" />
            <h2 className="text-lg font-semibold text-gray-900">
              Product Management
            </h2>
          </div>
          <div className="flex items-center space-x-3">
            <div className="text-sm text-gray-500">
              {productsData?.total || 0} total products
            </div>
            <button className="flex items-center px-3 py-2 bg-black text-white text-sm rounded-md hover:bg-gray-800 transition-colors">
              <PlusIcon className="h-4 w-4 mr-2" />
              Add Product
            </button>
          </div>
        </div>
      </div>

      <div className="p-6">
        {/* Search */}
        <div className="mb-6">
          <div className="relative">
            <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search products..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-black focus:border-transparent"
            />
          </div>
        </div>

        {/* Products List */}
        {products.length > 0 ? (
          <div className="space-y-4">
            {products.map((product) => {
              const stockStatus = getStockStatus(product.stockQuantity);
              return (
                <div
                  key={product.id}
                  className="flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  <div className="flex items-center">
                    <div className="h-12 w-12 bg-gray-100 rounded flex items-center justify-center mr-4">
                      <CubeIcon className="h-6 w-6 text-gray-400" />
                    </div>
                    <div>
                      <h3 className="text-sm font-medium text-gray-900">
                        {product.name}
                      </h3>
                      <p className="text-sm text-gray-500">
                        {product.categoryName} • {product.merchantName}
                      </p>
                      <div className="flex items-center mt-1 text-xs text-gray-500">
                        <span>
                          Created{" "}
                          {new Date(product.createdAt).toLocaleDateString()}
                        </span>
                        <span className="mx-2">•</span>
                        <span>
                          Updated{" "}
                          {new Date(product.updatedAt).toLocaleDateString()}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center space-x-4">
                    <div className="text-right">
                      <div className="text-sm font-medium text-gray-900">
                        ${product.price.toFixed(2)}
                      </div>
                      <div
                        className={`inline-flex px-2 py-1 text-xs font-medium rounded-full ${stockStatus.color}`}
                      >
                        {product.stockQuantity} units
                      </div>
                    </div>

                    <div className="flex items-center space-x-2">
                      <button className="p-2 text-gray-400 hover:text-blue-600 transition-colors">
                        <PencilIcon className="h-4 w-4" />
                      </button>
                      <button
                        onClick={() =>
                          handleDeleteProduct(product.id, product.name)
                        }
                        disabled={deleteProduct.isPending}
                        className="p-2 text-gray-400 hover:text-red-600 transition-colors disabled:opacity-50"
                      >
                        <TrashIcon className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="text-center py-8">
            <CubeIcon className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-2 text-sm font-medium text-gray-900">
              No products found
            </h3>
            <p className="mt-1 text-sm text-gray-500">
              {searchTerm
                ? "Try adjusting your search terms."
                : "No products have been added yet."}
            </p>
            <button className="mt-4 flex items-center mx-auto px-4 py-2 bg-black text-white text-sm rounded-md hover:bg-gray-800 transition-colors">
              <PlusIcon className="h-4 w-4 mr-2" />
              Add First Product
            </button>
          </div>
        )}

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="mt-6 flex items-center justify-between border-t border-gray-200 pt-6">
            <div className="text-sm text-gray-500">
              Page {currentPage} of {totalPages}
            </div>
            <div className="flex space-x-2">
              <button
                onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                disabled={currentPage === 1}
                className="px-3 py-1 text-sm border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Previous
              </button>
              <button
                onClick={() =>
                  setCurrentPage(Math.min(totalPages, currentPage + 1))
                }
                disabled={currentPage === totalPages}
                className="px-3 py-1 text-sm border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
