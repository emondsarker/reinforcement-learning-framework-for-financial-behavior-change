// Hook version for functional components
export const useErrorHandler = () => {
  const handleError = (error: Error, errorInfo?: string) => {
    console.error("Error caught by useErrorHandler:", error);

    if (process.env.NODE_ENV === "production") {
      // TODO: Send to error monitoring service
      console.error("Production error:", {
        error: error.message,
        stack: error.stack,
        info: errorInfo,
      });
    }
  };

  return { handleError };
};

// API error parsing utilities
export const parseApiError = (error: unknown): string => {
  if (typeof error === "string") {
    return error;
  }

  if (error && typeof error === "object") {
    // Axios error format
    if (
      "response" in error &&
      error.response &&
      typeof error.response === "object"
    ) {
      const response = error.response as {
        data?: { detail?: string; message?: string };
        statusText?: string;
      };
      if (response.data?.detail) {
        return response.data.detail;
      }
      if (response.data?.message) {
        return response.data.message;
      }
      if (response.statusText) {
        return response.statusText;
      }
    }

    // Standard Error object
    if ("message" in error && typeof error.message === "string") {
      return error.message;
    }
  }

  return "An unexpected error occurred";
};

// User-friendly error messages
export const getUserFriendlyErrorMessage = (error: unknown): string => {
  const message = parseApiError(error);

  // Map common error messages to user-friendly versions
  const errorMap: Record<string, string> = {
    "Network Error":
      "Unable to connect to the server. Please check your internet connection.",
    "Request timeout":
      "The request took too long to complete. Please try again.",
    Unauthorized: "Your session has expired. Please log in again.",
    Forbidden: "You don't have permission to perform this action.",
    "Not Found": "The requested resource was not found.",
    "Internal Server Error":
      "Something went wrong on our end. Please try again later.",
    "Bad Request": "Invalid request. Please check your input and try again.",
  };

  return errorMap[message] || message;
};

// Retry mechanism utility
export const createRetryHandler = (
  fn: () => Promise<unknown>,
  maxRetries: number = 3,
  delay: number = 1000
) => {
  return async (): Promise<unknown> => {
    let lastError: Error;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        return await fn();
      } catch (error) {
        lastError = error as Error;

        if (attempt === maxRetries) {
          throw lastError;
        }

        // Exponential backoff
        await new Promise((resolve) =>
          setTimeout(resolve, delay * Math.pow(2, attempt - 1))
        );
      }
    }

    throw lastError!;
  };
};

// Error logging utility
export const logError = (error: Error, context?: string) => {
  const errorInfo = {
    message: error.message,
    stack: error.stack,
    context,
    timestamp: new Date().toISOString(),
    userAgent: navigator.userAgent,
    url: window.location.href,
  };

  console.error("Error logged:", errorInfo);

  if (process.env.NODE_ENV === "production") {
    // TODO: Send to error monitoring service (e.g., Sentry, LogRocket)
    // Example: Sentry.captureException(error, { extra: errorInfo });
  }
};

// Validation error helpers
export const formatValidationErrors = (
  errors: Record<string, string[]>
): string => {
  return Object.entries(errors)
    .map(([field, messages]) => `${field}: ${messages.join(", ")}`)
    .join("; ");
};

// Network error detection
export const isNetworkError = (error: unknown): boolean => {
  if (typeof error === "object" && error !== null) {
    const err = error as {
      code?: string;
      message?: string;
      name?: string;
    };
    return (
      err.code === "NETWORK_ERROR" ||
      err.message === "Network Error" ||
      err.name === "NetworkError" ||
      !navigator.onLine
    );
  }
  return false;
};

// Timeout error detection
export const isTimeoutError = (error: unknown): boolean => {
  if (typeof error === "object" && error !== null) {
    const err = error as {
      code?: string;
      message?: string;
      name?: string;
    };
    return (
      err.code === "ECONNABORTED" ||
      (err.message?.includes("timeout") ?? false) ||
      err.name === "TimeoutError"
    );
  }
  return false;
};

// Auth error detection
export const isAuthError = (error: unknown): boolean => {
  if (typeof error === "object" && error !== null) {
    const err = error as {
      response?: { status?: number };
      message?: string;
    };
    return (
      err.response?.status === 401 ||
      err.response?.status === 403 ||
      (err.message?.includes("Unauthorized") ?? false) ||
      (err.message?.includes("Forbidden") ?? false)
    );
  }
  return false;
};
