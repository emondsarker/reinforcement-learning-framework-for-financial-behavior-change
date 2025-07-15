import { QueryClient } from "@tanstack/react-query";
import type { AxiosError } from "axios";

// Create and configure the QueryClient
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Cache times based on data type
      staleTime: 2 * 60 * 1000, // 2 minutes default for user data
      gcTime: 10 * 60 * 1000, // 10 minutes garbage collection

      // Retry configuration
      retry: (failureCount, error: Error) => {
        // Don't retry on 401, 403, or 404 errors
        const axiosError = error as AxiosError;
        if (
          axiosError?.response?.status === 401 ||
          axiosError?.response?.status === 403 ||
          axiosError?.response?.status === 404
        ) {
          return false;
        }
        // Retry up to 3 times for other errors
        return failureCount < 3;
      },

      // Retry delay with exponential backoff
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),

      // Refetch on window focus for real-time data
      refetchOnWindowFocus: true,

      // Refetch on reconnect
      refetchOnReconnect: true,

      // Error handling
      throwOnError: false,
    },
    mutations: {
      // Retry mutations once on network errors
      retry: (failureCount, error: Error) => {
        const axiosError = error as AxiosError;
        if (
          axiosError?.code === "ECONNABORTED" ||
          axiosError?.message === "Network Error"
        ) {
          return failureCount < 1;
        }
        return false;
      },

      // Error handling for mutations
      throwOnError: false,
    },
  },
});

// Cache time configurations for different data types
export const cacheConfig = {
  // Static data (products, categories) - cache for 10 minutes
  static: {
    staleTime: 10 * 60 * 1000, // 10 minutes
    gcTime: 30 * 60 * 1000, // 30 minutes
  },

  // User financial data (wallet, transactions) - cache for 2 minutes
  financial: {
    staleTime: 2 * 60 * 1000, // 2 minutes
    gcTime: 10 * 60 * 1000, // 10 minutes
  },

  // Real-time data (notifications, live updates) - cache for 30 seconds
  realtime: {
    staleTime: 30 * 1000, // 30 seconds
    gcTime: 2 * 60 * 1000, // 2 minutes
  },

  // User profile data - cache for 5 minutes
  profile: {
    staleTime: 5 * 60 * 1000, // 5 minutes
    gcTime: 15 * 60 * 1000, // 15 minutes
  },
};

export default queryClient;
