import React, { useState } from "react";
import {
  ShoppingBagIcon,
  PlusIcon,
  MinusIcon,
  ShoppingCartIcon,
} from "@heroicons/react/24/outline";
import { useCart } from "../../hooks/useCart";
import { useToast } from "../../hooks/useToast";
import type { Product } from "../../types";

interface ProductCardProps {
  product: Product;
  onQuickPurchase?: (product: Product, quantity: number) => void;
}

export const ProductCard: React.FC<ProductCardProps> = ({
  product,
  onQuickPurchase,
}) => {
  const [quantity, setQuantity] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const { addItem, getItemQuantity, isInCart } = useCart();
  const { addToast } = useToast();

  const currentCartQuantity = getItemQuantity(product.id);
  const maxQuantity = product.stock_quantity - currentCartQuantity;
  const canAddToCart = product.is_available && maxQuantity > 0;

  const handleAddToCart = async () => {
    if (!canAddToCart) return;

    setIsLoading(true);
    try {
      addItem(product, quantity);
      addToast({
        type: "success",
        title: "Added to Cart",
        message: `${product.name} added to cart!`,
      });
      setQuantity(1); // Reset quantity after adding
    } catch (error) {
      addToast({
        type: "error",
        title: "Failed to Add",
        message:
          error instanceof Error ? error.message : "Failed to add to cart",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleQuickPurchase = () => {
    if (onQuickPurchase && product.is_available) {
      onQuickPurchase(product, quantity);
    }
  };

  const incrementQuantity = () => {
    if (quantity < maxQuantity) {
      setQuantity(quantity + 1);
    }
  };

  const decrementQuantity = () => {
    if (quantity > 1) {
      setQuantity(quantity - 1);
    }
  };

  const getStockStatusColor = () => {
    if (!product.is_available) return "text-red-600";
    if (product.stock_quantity <= 5) return "text-orange-600";
    return "text-green-600";
  };

  const getStockStatusText = () => {
    if (!product.is_available) return "Out of Stock";
    if (product.stock_quantity <= 5)
      return `Only ${product.stock_quantity} left`;
    return "In Stock";
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 hover:shadow-md transition-shadow duration-200 group">
      <div className="p-4">
        {/* Product Image Placeholder */}
        <div className="bg-gray-100 h-48 rounded-md mb-4 flex items-center justify-center relative overflow-hidden">
          {product.image_url ? (
            <img
              src={product.image_url}
              alt={product.name}
              className="w-full h-full object-cover"
            />
          ) : (
            <ShoppingBagIcon className="h-12 w-12 text-gray-400" />
          )}

          {/* Stock status badge */}
          <div className="absolute top-2 right-2">
            <span
              className={`px-2 py-1 text-xs font-medium rounded-full bg-white shadow-sm ${getStockStatusColor()}`}
            >
              {getStockStatusText()}
            </span>
          </div>

          {/* Cart indicator */}
          {isInCart(product.id) && (
            <div className="absolute top-2 left-2">
              <div className="bg-black text-white rounded-full p-1">
                <ShoppingCartIcon className="h-4 w-4" />
              </div>
            </div>
          )}
        </div>

        {/* Product Info */}
        <div className="space-y-2">
          <h3 className="font-medium text-gray-900 line-clamp-2 group-hover:text-black transition-colors">
            {product.name}
          </h3>

          {product.description && (
            <p className="text-sm text-gray-600 line-clamp-2">
              {product.description}
            </p>
          )}

          <p className="text-xs text-gray-500">by {product.merchant_name}</p>

          <div className="flex items-center justify-between pt-2">
            <span className="text-lg font-bold text-gray-900">
              ${product.price.toFixed(2)}
            </span>

            {currentCartQuantity > 0 && (
              <span className="text-sm text-gray-600">
                {currentCartQuantity} in cart
              </span>
            )}
          </div>
        </div>

        {/* Quantity Selector */}
        {canAddToCart && (
          <div className="mt-4 flex items-center justify-center space-x-3">
            <button
              onClick={decrementQuantity}
              disabled={quantity <= 1}
              className="p-1 rounded-full border border-gray-300 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <MinusIcon className="h-4 w-4" />
            </button>

            <span className="font-medium text-gray-900 min-w-[2rem] text-center">
              {quantity}
            </span>

            <button
              onClick={incrementQuantity}
              disabled={quantity >= maxQuantity}
              className="p-1 rounded-full border border-gray-300 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <PlusIcon className="h-4 w-4" />
            </button>
          </div>
        )}

        {/* Action Buttons */}
        <div className="mt-4 space-y-2">
          {canAddToCart ? (
            <>
              <button
                onClick={handleAddToCart}
                disabled={isLoading}
                className="w-full bg-gray-100 text-gray-900 px-4 py-2 rounded-md text-sm font-medium hover:bg-gray-200 transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
              >
                <ShoppingCartIcon className="h-4 w-4" />
                <span>{isLoading ? "Adding..." : "Add to Cart"}</span>
              </button>

              {onQuickPurchase && (
                <button
                  onClick={handleQuickPurchase}
                  className="w-full bg-black text-white px-4 py-2 rounded-md text-sm font-medium hover:bg-gray-800 transition-colors duration-200"
                >
                  Buy Now
                </button>
              )}
            </>
          ) : (
            <button
              disabled
              className="w-full bg-gray-100 text-gray-500 px-4 py-2 rounded-md text-sm font-medium cursor-not-allowed"
            >
              {!product.is_available ? "Out of Stock" : "Cannot Add More"}
            </button>
          )}
        </div>

        {/* Additional Info */}
        {maxQuantity < product.stock_quantity && maxQuantity > 0 && (
          <p className="mt-2 text-xs text-gray-500 text-center">
            You can add {maxQuantity} more to cart
          </p>
        )}
      </div>
    </div>
  );
};
