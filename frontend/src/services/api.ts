import axios from "axios";
import type {
  AxiosInstance,
  InternalAxiosRequestConfig,
  AxiosResponse,
  AxiosError,
} from "axios";

// Create axios instance with base configuration
const api: AxiosInstance = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:8000",
  timeout: 10000,
  headers: {
    "Content-Type": "application/json",
  },
});

// Request interceptor to add authentication token
api.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const token = localStorage.getItem("access_token");
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor to handle authentication errors
api.interceptors.response.use(
  (response: AxiosResponse) => {
    return response;
  },
  (error) => {
    // Handle 401 Unauthorized errors
    if (error.response?.status === 401) {
      // Clear stored authentication data
      localStorage.removeItem("access_token");
      localStorage.removeItem("user");

      // Redirect to login page if not already there
      if (window.location.pathname !== "/login") {
        window.location.href = "/login";
      }
    }

    return Promise.reject(error);
  }
);

// Helper functions for token management
export const tokenManager = {
  getToken: (): string | null => {
    return localStorage.getItem("access_token");
  },

  setToken: (token: string): void => {
    localStorage.setItem("access_token", token);
  },

  removeToken: (): void => {
    localStorage.removeItem("access_token");
    localStorage.removeItem("user");
  },

  isAuthenticated: (): boolean => {
    const token = localStorage.getItem("access_token");
    return !!token;
  },
};

// Helper function to handle API errors
export const handleApiError = (error: AxiosError): string => {
  const responseData = error.response?.data as { detail?: string } | undefined;

  if (responseData?.detail) {
    return responseData.detail;
  }

  if (error.response?.status === 401) {
    return "Authentication failed. Please log in again.";
  }

  if (error.response?.status === 403) {
    return "You do not have permission to perform this action.";
  }

  if (error.response?.status === 404) {
    return "The requested resource was not found.";
  }

  if (error.response && error.response.status >= 500) {
    return "Server error. Please try again later.";
  }

  if (error.code === "ECONNABORTED") {
    return "Request timeout. Please try again.";
  }

  if (error.message === "Network Error") {
    return "Network error. Please check your connection.";
  }

  return error.message || "An unexpected error occurred.";
};

export default api;
