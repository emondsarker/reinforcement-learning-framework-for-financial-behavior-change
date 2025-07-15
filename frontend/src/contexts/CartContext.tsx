import React, { createContext, useReducer, useEffect } from "react";
import type { Product, CartItem, ShoppingCart } from "../types";

// Cart action types
type CartAction =
  | { type: "ADD_ITEM"; payload: { product: Product; quantity: number } }
  | { type: "REMOVE_ITEM"; payload: { productId: string } }
  | {
      type: "UPDATE_QUANTITY";
      payload: { productId: string; quantity: number };
    }
  | { type: "CLEAR_CART" }
  | { type: "LOAD_CART"; payload: ShoppingCart };

// Cart context type
interface CartContextType {
  cart: ShoppingCart;
  addItem: (product: Product, quantity?: number) => void;
  removeItem: (productId: string) => void;
  updateQuantity: (productId: string, quantity: number) => void;
  clearCart: () => void;
  getItemQuantity: (productId: string) => number;
  isInCart: (productId: string) => boolean;
}

// Initial cart state
const initialCart: ShoppingCart = {
  items: [],
  total_items: 0,
  total_amount: 0,
  updated_at: new Date(),
};

// Cart reducer
const cartReducer = (state: ShoppingCart, action: CartAction): ShoppingCart => {
  switch (action.type) {
    case "ADD_ITEM": {
      const { product, quantity } = action.payload;
      const existingItemIndex = state.items.findIndex(
        (item) => item.product.id === product.id
      );

      let newItems: CartItem[];

      if (existingItemIndex >= 0) {
        // Update existing item quantity
        newItems = state.items.map((item, index) =>
          index === existingItemIndex
            ? {
                ...item,
                quantity: item.quantity + quantity,
                subtotal: (item.quantity + quantity) * product.price,
              }
            : item
        );
      } else {
        // Add new item
        const newItem: CartItem = {
          product,
          quantity,
          subtotal: quantity * product.price,
        };
        newItems = [...state.items, newItem];
      }

      const total_items = newItems.reduce(
        (sum, item) => sum + item.quantity,
        0
      );
      const total_amount = newItems.reduce(
        (sum, item) => sum + item.subtotal,
        0
      );

      return {
        items: newItems,
        total_items,
        total_amount,
        updated_at: new Date(),
      };
    }

    case "REMOVE_ITEM": {
      const newItems = state.items.filter(
        (item) => item.product.id !== action.payload.productId
      );

      const total_items = newItems.reduce(
        (sum, item) => sum + item.quantity,
        0
      );
      const total_amount = newItems.reduce(
        (sum, item) => sum + item.subtotal,
        0
      );

      return {
        items: newItems,
        total_items,
        total_amount,
        updated_at: new Date(),
      };
    }

    case "UPDATE_QUANTITY": {
      const { productId, quantity } = action.payload;

      if (quantity <= 0) {
        // Remove item if quantity is 0 or negative
        return cartReducer(state, {
          type: "REMOVE_ITEM",
          payload: { productId },
        });
      }

      const newItems = state.items.map((item) =>
        item.product.id === productId
          ? {
              ...item,
              quantity,
              subtotal: quantity * item.product.price,
            }
          : item
      );

      const total_items = newItems.reduce(
        (sum, item) => sum + item.quantity,
        0
      );
      const total_amount = newItems.reduce(
        (sum, item) => sum + item.subtotal,
        0
      );

      return {
        items: newItems,
        total_items,
        total_amount,
        updated_at: new Date(),
      };
    }

    case "CLEAR_CART":
      return {
        ...initialCart,
        updated_at: new Date(),
      };

    case "LOAD_CART":
      return action.payload;

    default:
      return state;
  }
};

// Create context
export const CartContext = createContext<CartContextType | undefined>(
  undefined
);

// Local storage key
const CART_STORAGE_KEY = "fincoach_cart";

// Cart provider component
export const CartProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [cart, dispatch] = useReducer(cartReducer, initialCart);

  // Load cart from localStorage on mount
  useEffect(() => {
    try {
      const savedCart = localStorage.getItem(CART_STORAGE_KEY);
      if (savedCart) {
        const parsedCart = JSON.parse(savedCart);
        // Convert date strings back to Date objects
        parsedCart.updated_at = new Date(parsedCart.updated_at);
        dispatch({ type: "LOAD_CART", payload: parsedCart });
      }
    } catch (error) {
      console.error("Failed to load cart from localStorage:", error);
    }
  }, []);

  // Save cart to localStorage whenever it changes
  useEffect(() => {
    try {
      localStorage.setItem(CART_STORAGE_KEY, JSON.stringify(cart));
    } catch (error) {
      console.error("Failed to save cart to localStorage:", error);
    }
  }, [cart]);

  // Cart operations
  const addItem = (product: Product, quantity: number = 1) => {
    // Validate stock availability
    if (!product.is_available || product.stock_quantity < quantity) {
      throw new Error("Product is not available or insufficient stock");
    }

    dispatch({ type: "ADD_ITEM", payload: { product, quantity } });
  };

  const removeItem = (productId: string) => {
    dispatch({ type: "REMOVE_ITEM", payload: { productId } });
  };

  const updateQuantity = (productId: string, quantity: number) => {
    // Find the product to validate stock
    const cartItem = cart.items.find((item) => item.product.id === productId);
    if (cartItem && quantity > cartItem.product.stock_quantity) {
      throw new Error("Insufficient stock for requested quantity");
    }

    dispatch({ type: "UPDATE_QUANTITY", payload: { productId, quantity } });
  };

  const clearCart = () => {
    dispatch({ type: "CLEAR_CART" });
  };

  const getItemQuantity = (productId: string): number => {
    const item = cart.items.find((item) => item.product.id === productId);
    return item ? item.quantity : 0;
  };

  const isInCart = (productId: string): boolean => {
    return cart.items.some((item) => item.product.id === productId);
  };

  const contextValue: CartContextType = {
    cart,
    addItem,
    removeItem,
    updateQuantity,
    clearCart,
    getItemQuantity,
    isInCart,
  };

  return (
    <CartContext.Provider value={contextValue}>{children}</CartContext.Provider>
  );
};
