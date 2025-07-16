import React from "react";
import {
  PlusIcon,
  MinusIcon,
  TrashIcon,
  ShoppingBagIcon,
} from "@heroicons/react/24/outline";
import { useCart } from "../../hooks/useCart";
import { useToast } from "../../hooks/useToast";
import type { CartItem as CartItemType } from "../../types";

interface CartItemProps {
  item: CartItemType;
}

export const CartItem: React.FC<CartItemProps> = ({ item }) => {
  const { updateQuantity, removeItem } = useCart();
  const { addToast } = useToast();

  const handleQuantityChange = (newQuantity: number) => {
    try {
      updateQuantity(item.product.id, newQuantity);
    } catch (error) {
      addToast({
        type: "error",
        title: "Update Failed",
        message:
          error instanceof Error ? error.message : "Failed to update quantity",
      });
    }
  };

  const handleRemove = () => {
    removeItem(item.product.id);
    addToast({
      type: "success",
      title: "Item Removed",
      message: `${item.product.name} removed from cart`,
    });
  };

  const incrementQuantity = () => {
    if (item.quantity < item.product.stock_quantity) {
      handleQuantityChange(item.quantity + 1);
    }
  };

  const decrementQuantity = () => {
    if (item.quantity > 1) {
      handleQuantityChange(item.quantity - 1);
    }
  };

  const canIncrement = item.quantity < item.product.stock_quantity;
  const canDecrement = item.quantity > 1;

  return (
    <div className="flex items-start space-x-4 py-4 border-b border-gray-200 last:border-b-0">
      {/* Product Image */}
      <div className="flex-shrink-0 w-16 h-16 bg-gray-100 rounded-md flex items-center justify-center overflow-hidden">
        {item.product.image_url ? (
          <img
            src={item.product.image_url}
            alt={item.product.name}
            className="w-full h-full object-cover"
          />
        ) : (
          <ShoppingBagIcon className="h-8 w-8 text-gray-400" />
        )}
      </div>

      {/* Product Details */}
      <div className="flex-1 min-w-0">
        <div className="flex items-start justify-between">
          <div className="flex-1 min-w-0">
            <h4 className="text-sm font-medium text-gray-900 line-clamp-2">
              {item.product.name}
            </h4>
            <p className="text-xs text-gray-500 mt-1">
              by {item.product.merchant_name}
            </p>
            <p className="text-sm font-medium text-gray-900 mt-1">
              ${Number(item.product.price).toFixed(2)} each
            </p>
          </div>

          {/* Remove Button */}
          <button
            onClick={handleRemove}
            className="ml-2 p-1 text-gray-400 hover:text-red-500 transition-colors"
            title="Remove item"
          >
            <TrashIcon className="h-4 w-4" />
          </button>
        </div>

        {/* Quantity Controls */}
        <div className="flex items-center justify-between mt-3">
          <div className="flex items-center space-x-2">
            <button
              onClick={decrementQuantity}
              disabled={!canDecrement}
              className="p-1 rounded-full border border-gray-300 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <MinusIcon className="h-3 w-3" />
            </button>

            <span className="font-medium text-gray-900 min-w-[2rem] text-center text-sm">
              {item.quantity}
            </span>

            <button
              onClick={incrementQuantity}
              disabled={!canIncrement}
              className="p-1 rounded-full border border-gray-300 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <PlusIcon className="h-3 w-3" />
            </button>
          </div>

          {/* Subtotal */}
          <div className="text-right">
            <p className="text-sm font-medium text-gray-900">
              ${Number(item.subtotal).toFixed(2)}
            </p>
            {item.quantity > 1 && (
              <p className="text-xs text-gray-500">
                ${Number(item.product.price).toFixed(2)} × {item.quantity}
              </p>
            )}
          </div>
        </div>

        {/* Stock Warning */}
        {item.quantity >= item.product.stock_quantity && (
          <p className="text-xs text-orange-600 mt-1">
            Maximum quantity reached
          </p>
        )}

        {/* Low Stock Warning */}
        {item.product.stock_quantity <= 5 &&
          item.product.stock_quantity > 0 && (
            <p className="text-xs text-orange-600 mt-1">
              Only {item.product.stock_quantity} left in stock
            </p>
          )}
      </div>
    </div>
  );
};
