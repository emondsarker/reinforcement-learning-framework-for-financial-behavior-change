import React, { Fragment, useState } from "react";
import { Dialog, Transition } from "@headlessui/react";
import {
  XMarkIcon,
  ShoppingCartIcon,
  CreditCardIcon,
  ExclamationTriangleIcon,
} from "@heroicons/react/24/outline";
import { usePurchaseProduct } from "../../hooks/useProducts";
import { useCart } from "../../hooks/useCart";
import { useToast } from "../../hooks/useToast";
import { useWallet } from "../../hooks/useFinancial";
import type { Product, CartItem } from "../../types";

interface PurchaseModalProps {
  isOpen: boolean;
  onClose: () => void;
  mode: "single" | "cart";
  singleProduct?: { product: Product; quantity: number };
}

export const PurchaseModal: React.FC<PurchaseModalProps> = ({
  isOpen,
  onClose,
  mode,
  singleProduct,
}) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const { cart, clearCart } = useCart();
  const { data: wallet } = useWallet();
  const purchaseProduct = usePurchaseProduct();
  const { addToast } = useToast();

  // Determine items to purchase
  const itemsToPurchase: CartItem[] =
    mode === "single" && singleProduct
      ? [
          {
            product: singleProduct.product,
            quantity: singleProduct.quantity,
            subtotal: singleProduct.product.price * singleProduct.quantity,
          },
        ]
      : cart.items;

  const totalAmount = itemsToPurchase.reduce(
    (sum, item) => sum + item.subtotal,
    0
  );
  const totalItems = itemsToPurchase.reduce(
    (sum, item) => sum + item.quantity,
    0
  );

  const hasInsufficientFunds =
    wallet && Number(wallet.balance || 0) < totalAmount;
  const canPurchase = !hasInsufficientFunds && itemsToPurchase.length > 0;

  const handlePurchase = async () => {
    if (!canPurchase) return;

    setIsProcessing(true);
    try {
      // Process each item as a separate purchase
      for (const item of itemsToPurchase) {
        await purchaseProduct.mutateAsync({
          product_id: item.product.id,
          quantity: item.quantity,
        });
      }

      // Clear cart if purchasing from cart
      if (mode === "cart") {
        clearCart();
      }

      addToast({
        type: "success",
        title: "Purchase Successful!",
        message: `Successfully purchased ${totalItems} ${
          totalItems === 1 ? "item" : "items"
        } for $${totalAmount.toFixed(2)}`,
      });

      onClose();
    } catch (error) {
      addToast({
        type: "error",
        title: "Purchase Failed",
        message:
          error instanceof Error
            ? error.message
            : "Failed to complete purchase",
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const getModalTitle = () => {
    if (mode === "single") return "Confirm Purchase";
    return "Checkout";
  };

  const getModalDescription = () => {
    if (mode === "single") return "Review your purchase details below.";
    return "Review your cart and complete your purchase.";
  };

  return (
    <Transition.Root show={isOpen} as={Fragment}>
      <Dialog as="div" className="relative z-50" onClose={onClose}>
        <Transition.Child
          as={Fragment}
          enter="ease-out duration-300"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="ease-in duration-200"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <div className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity" />
        </Transition.Child>

        <div className="fixed inset-0 z-10 overflow-y-auto">
          <div className="flex min-h-full items-end justify-center p-4 text-center sm:items-center sm:p-0">
            <Transition.Child
              as={Fragment}
              enter="ease-out duration-300"
              enterFrom="opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95"
              enterTo="opacity-100 translate-y-0 sm:scale-100"
              leave="ease-in duration-200"
              leaveFrom="opacity-100 translate-y-0 sm:scale-100"
              leaveTo="opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95"
            >
              <Dialog.Panel className="relative transform overflow-hidden rounded-lg bg-white px-4 pb-4 pt-5 text-left shadow-xl transition-all sm:my-8 sm:w-full sm:max-w-lg sm:p-6">
                {/* Header */}
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-2">
                    <CreditCardIcon className="h-6 w-6 text-gray-900" />
                    <Dialog.Title className="text-lg font-medium text-gray-900">
                      {getModalTitle()}
                    </Dialog.Title>
                  </div>
                  <button
                    type="button"
                    className="rounded-md text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-black"
                    onClick={onClose}
                  >
                    <XMarkIcon className="h-6 w-6" />
                  </button>
                </div>

                <p className="text-sm text-gray-600 mb-6">
                  {getModalDescription()}
                </p>

                {/* Items List */}
                <div className="mb-6">
                  <h3 className="text-sm font-medium text-gray-900 mb-3">
                    Order Summary
                  </h3>
                  <div className="bg-gray-50 rounded-lg p-4 space-y-3 max-h-60 overflow-y-auto">
                    {itemsToPurchase.map((item) => (
                      <div
                        key={item.product.id}
                        className="flex items-center justify-between"
                      >
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-gray-900 truncate">
                            {item.product.name}
                          </p>
                          <p className="text-xs text-gray-500">
                            ${item.product.price.toFixed(2)} × {item.quantity}
                          </p>
                        </div>
                        <p className="text-sm font-medium text-gray-900 ml-4">
                          ${item.subtotal.toFixed(2)}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Total */}
                <div className="border-t border-gray-200 pt-4 mb-6">
                  <div className="flex justify-between items-center">
                    <span className="text-base font-medium text-gray-900">
                      Total ({totalItems} {totalItems === 1 ? "item" : "items"})
                    </span>
                    <span className="text-lg font-bold text-gray-900">
                      ${totalAmount.toFixed(2)}
                    </span>
                  </div>
                </div>

                {/* Wallet Balance */}
                <div className="mb-6 p-3 bg-gray-50 rounded-lg">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">
                      Wallet Balance
                    </span>
                    <span
                      className={`text-sm font-medium ${
                        hasInsufficientFunds ? "text-red-600" : "text-green-600"
                      }`}
                    >
                      $
                      {wallet?.balance
                        ? Number(wallet.balance).toFixed(2)
                        : "0.00"}
                    </span>
                  </div>
                  {hasInsufficientFunds && (
                    <div className="mt-2 flex items-center space-x-2 text-red-600">
                      <ExclamationTriangleIcon className="h-4 w-4" />
                      <span className="text-xs">
                        Insufficient funds. Need $
                        {(totalAmount - Number(wallet?.balance || 0)).toFixed(
                          2
                        )}{" "}
                        more.
                      </span>
                    </div>
                  )}
                </div>

                {/* Action Buttons */}
                <div className="flex space-x-3">
                  <button
                    type="button"
                    className="flex-1 bg-white px-4 py-2 text-sm font-medium text-gray-700 border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-black focus:ring-offset-2"
                    onClick={onClose}
                    disabled={isProcessing}
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    onClick={handlePurchase}
                    disabled={!canPurchase || isProcessing}
                    className="flex-1 bg-black text-white px-4 py-2 text-sm font-medium rounded-md hover:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-black focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                  >
                    {isProcessing ? (
                      <>
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                        <span>Processing...</span>
                      </>
                    ) : (
                      <>
                        <ShoppingCartIcon className="h-4 w-4" />
                        <span>
                          {hasInsufficientFunds
                            ? "Insufficient Funds"
                            : `Purchase $${totalAmount.toFixed(2)}`}
                        </span>
                      </>
                    )}
                  </button>
                </div>

                {/* Additional Info */}
                <div className="mt-4 text-xs text-gray-500 text-center space-y-1">
                  <p>
                    This is a simulated purchase for demonstration purposes.
                  </p>
                  <p>Your wallet balance will be updated accordingly.</p>
                </div>
              </Dialog.Panel>
            </Transition.Child>
          </div>
        </div>
      </Dialog>
    </Transition.Root>
  );
};
