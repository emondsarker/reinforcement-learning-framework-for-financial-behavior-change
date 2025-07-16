// Error handling utilities
export {
  useErrorHandler,
  parseApiError,
  getUserFriendlyErrorMessage,
  createRetryHandler,
  logError,
  formatValidationErrors,
  isNetworkError,
  isTimeoutError,
  isAuthError,
} from "./errorHandling";

// Re-export common components for convenience
export { default as ErrorBoundary } from "../components/common/ErrorBoundary";
export {
  default as LoadingSpinner,
  InlineSpinner,
  PageLoader,
  SkeletonLoader,
  ButtonSpinner,
} from "../components/common/LoadingSpinner";
