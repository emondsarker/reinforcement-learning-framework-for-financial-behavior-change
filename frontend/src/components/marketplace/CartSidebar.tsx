import React, { Fragment } from "react";
import { Dialog, Transition } from "@headlessui/react";
import {
  XMarkIcon,
  ShoppingCartIcon,
  TrashIcon,
} from "@heroicons/react/24/outline";
import { useCart } from "../../hooks/useCart";
import { CartItem } from "./CartItem";
import { CartSummary } from "./CartSummary";

interface CartSidebarProps {
  isOpen: boolean;
  onClose: () => void;
  onCheckout: () => void;
}

export const CartSidebar: React.FC<CartSidebarProps> = ({
  isOpen,
  onClose,
  onCheckout,
}) => {
  const { cart, clearCart } = useCart();

  const handleClearCart = () => {
    if (window.confirm("Are you sure you want to clear your cart?")) {
      clearCart();
    }
  };

  return (
    <Transition.Root show={isOpen} as={Fragment}>
      <Dialog as="div" className="relative z-50" onClose={onClose}>
        <Transition.Child
          as={Fragment}
          enter="ease-in-out duration-300"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="ease-in-out duration-300"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <div className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity" />
        </Transition.Child>

        <div className="fixed inset-0 overflow-hidden">
          <div className="absolute inset-0 overflow-hidden">
            <div className="pointer-events-none fixed inset-y-0 right-0 flex max-w-full pl-10">
              <Transition.Child
                as={Fragment}
                enter="transform transition ease-in-out duration-300"
                enterFrom="translate-x-full"
                enterTo="translate-x-0"
                leave="transform transition ease-in-out duration-300"
                leaveFrom="translate-x-0"
                leaveTo="translate-x-full"
              >
                <Dialog.Panel className="pointer-events-auto w-screen max-w-md">
                  <div className="flex h-full flex-col bg-white shadow-xl">
                    {/* Header */}
                    <div className="flex items-center justify-between px-4 py-6 border-b border-gray-200">
                      <div className="flex items-center space-x-2">
                        <ShoppingCartIcon className="h-6 w-6 text-gray-900" />
                        <Dialog.Title className="text-lg font-medium text-gray-900">
                          Shopping Cart
                        </Dialog.Title>
                        {cart.total_items > 0 && (
                          <span className="bg-black text-white text-xs rounded-full px-2 py-1">
                            {cart.total_items}
                          </span>
                        )}
                      </div>
                      <button
                        type="button"
                        className="rounded-md text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-black"
                        onClick={onClose}
                      >
                        <span className="sr-only">Close panel</span>
                        <XMarkIcon className="h-6 w-6" aria-hidden="true" />
                      </button>
                    </div>

                    {/* Cart Content */}
                    <div className="flex-1 overflow-y-auto">
                      {cart.items.length === 0 ? (
                        /* Empty Cart */
                        <div className="flex flex-col items-center justify-center h-full px-4 text-center">
                          <ShoppingCartIcon className="h-12 w-12 text-gray-400 mb-4" />
                          <h3 className="text-lg font-medium text-gray-900 mb-2">
                            Your cart is empty
                          </h3>
                          <p className="text-gray-500 mb-6">
                            Add some products to get started!
                          </p>
                          <button
                            onClick={onClose}
                            className="bg-black text-white px-6 py-2 rounded-md hover:bg-gray-800 transition-colors"
                          >
                            Continue Shopping
                          </button>
                        </div>
                      ) : (
                        /* Cart Items */
                        <div className="px-4 py-6">
                          <div className="space-y-4">
                            {cart.items.map((item) => (
                              <CartItem key={item.product.id} item={item} />
                            ))}
                          </div>

                          {/* Clear Cart Button */}
                          {cart.items.length > 0 && (
                            <div className="mt-6 pt-6 border-t border-gray-200">
                              <button
                                onClick={handleClearCart}
                                className="flex items-center space-x-2 text-red-600 hover:text-red-700 text-sm font-medium"
                              >
                                <TrashIcon className="h-4 w-4" />
                                <span>Clear Cart</span>
                              </button>
                            </div>
                          )}
                        </div>
                      )}
                    </div>

                    {/* Cart Summary & Checkout */}
                    {cart.items.length > 0 && (
                      <div className="border-t border-gray-200 px-4 py-6">
                        <CartSummary cart={cart} />

                        <div className="mt-6">
                          <button
                            onClick={onCheckout}
                            className="w-full bg-black text-white px-6 py-3 rounded-md font-medium hover:bg-gray-800 transition-colors focus:outline-none focus:ring-2 focus:ring-black focus:ring-offset-2"
                          >
                            Proceed to Checkout
                          </button>
                        </div>

                        <div className="mt-4 text-center">
                          <button
                            onClick={onClose}
                            className="text-sm text-gray-600 hover:text-gray-700 underline"
                          >
                            Continue Shopping
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                </Dialog.Panel>
              </Transition.Child>
            </div>
          </div>
        </div>
      </Dialog>
    </Transition.Root>
  );
};
