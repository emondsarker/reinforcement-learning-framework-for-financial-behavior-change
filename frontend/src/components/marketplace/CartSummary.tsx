import React from "react";
import type { ShoppingCart } from "../../types";

interface CartSummaryProps {
  cart: ShoppingCart;
}

export const CartSummary: React.FC<CartSummaryProps> = ({ cart }) => {
  // Calculate potential savings or fees (for future features)
  const subtotal = cart.total_amount;
  const tax = 0; // No tax for now
  const shipping = 0; // Free shipping for now
  const discount = 0; // No discounts for now
  const total = subtotal + tax + shipping - discount;

  return (
    <div className="space-y-3">
      {/* Items Summary */}
      <div className="flex justify-between text-sm">
        <span className="text-gray-600">
          Items ({cart.total_items} {cart.total_items === 1 ? "item" : "items"})
        </span>
        <span className="font-medium text-gray-900">
          ${subtotal.toFixed(2)}
        </span>
      </div>

      {/* Tax (if applicable) */}
      {tax > 0 && (
        <div className="flex justify-between text-sm">
          <span className="text-gray-600">Tax</span>
          <span className="font-medium text-gray-900">${tax.toFixed(2)}</span>
        </div>
      )}

      {/* Shipping (if applicable) */}
      {shipping > 0 ? (
        <div className="flex justify-between text-sm">
          <span className="text-gray-600">Shipping</span>
          <span className="font-medium text-gray-900">
            ${shipping.toFixed(2)}
          </span>
        </div>
      ) : (
        <div className="flex justify-between text-sm">
          <span className="text-gray-600">Shipping</span>
          <span className="font-medium text-green-600">Free</span>
        </div>
      )}

      {/* Discount (if applicable) */}
      {discount > 0 && (
        <div className="flex justify-between text-sm">
          <span className="text-gray-600">Discount</span>
          <span className="font-medium text-green-600">
            -${discount.toFixed(2)}
          </span>
        </div>
      )}

      {/* Divider */}
      <div className="border-t border-gray-200 pt-3">
        <div className="flex justify-between">
          <span className="text-base font-medium text-gray-900">Total</span>
          <span className="text-lg font-bold text-gray-900">
            ${total.toFixed(2)}
          </span>
        </div>
      </div>

      {/* Additional Info */}
      <div className="text-xs text-gray-500 space-y-1">
        <p>• All purchases are simulated for demonstration</p>
        <p>• Transactions will be recorded in your financial history</p>
        {cart.total_amount > 100 && (
          <p className="text-orange-600">
            • Large purchase detected - consider your budget
          </p>
        )}
      </div>

      {/* Cart Updated Time */}
      <div className="text-xs text-gray-400 text-center">
        Cart updated: {cart.updated_at.toLocaleTimeString()}
      </div>
    </div>
  );
};
