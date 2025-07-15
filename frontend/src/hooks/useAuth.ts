import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AuthService } from "../services/authService";
import { useAuthContext } from "../contexts/AuthContext";
import type { LoginRequest, RegisterRequest } from "../types";
import { queryKeys } from "../lib/queryKeys";

// Hook for current user query
export const useCurrentUser = () => {
  const { isAuthenticated } = useAuthContext();

  return useQuery({
    queryKey: queryKeys.auth.user(),
    queryFn: AuthService.getCurrentUser,
    enabled: isAuthenticated,
    staleTime: 5 * 60 * 1000, // 5 minutes
    retry: (failureCount, error: unknown) => {
      // Don't retry on 401 errors (token expired)
      if (
        (error as { response?: { status?: number } })?.response?.status === 401
      ) {
        return false;
      }
      return failureCount < 3;
    },
  });
};

// Hook for login mutation
export const useLogin = () => {
  const { login } = useAuthContext();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (credentials: LoginRequest) => login(credentials),
    onSuccess: (_, variables) => {
      // Invalidate and refetch user data
      queryClient.invalidateQueries({ queryKey: queryKeys.auth.user() });

      // Show success message could be added here
      console.log("Login successful for:", variables.email);
    },
    onError: (error: unknown) => {
      console.error("Login failed:", error);
      // Error handling will be done by the calling component
    },
  });
};

// Hook for register mutation
export const useRegister = () => {
  const { register } = useAuthContext();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (userData: RegisterRequest) => register(userData),
    onSuccess: (_, variables) => {
      // Invalidate and refetch user data
      queryClient.invalidateQueries({ queryKey: queryKeys.auth.user() });

      // Show success message could be added here
      console.log("Registration successful for:", variables.email);
    },
    onError: (error: unknown) => {
      console.error("Registration failed:", error);
      // Error handling will be done by the calling component
    },
  });
};

// Hook for logout mutation
export const useLogout = () => {
  const { logout } = useAuthContext();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: () => Promise.resolve(logout()),
    onSuccess: () => {
      // Clear all cached data on logout
      queryClient.clear();

      console.log("Logout successful");
    },
    onError: (error: unknown) => {
      console.error("Logout failed:", error);
    },
  });
};

// Hook for token refresh
export const useRefreshToken = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: AuthService.refreshToken,
    onSuccess: (data) => {
      // Update user data in cache
      queryClient.setQueryData(queryKeys.auth.user(), data.user);

      console.log("Token refreshed successfully");
    },
    onError: (error: unknown) => {
      console.error("Token refresh failed:", error);
      // Clear cache and redirect to login
      queryClient.clear();
    },
  });
};

// Convenience hook that combines auth context and queries
export const useAuth = () => {
  const authContext = useAuthContext();
  const currentUserQuery = useCurrentUser();
  const loginMutation = useLogin();
  const registerMutation = useRegister();
  const logoutMutation = useLogout();
  const refreshTokenMutation = useRefreshToken();

  return {
    // Auth state
    ...authContext,

    // Queries
    currentUserQuery,

    // Mutations
    loginMutation,
    registerMutation,
    logoutMutation,
    refreshTokenMutation,

    // Convenience flags
    isLoginLoading: loginMutation.isPending,
    isRegisterLoading: registerMutation.isPending,
    isLogoutLoading: logoutMutation.isPending,
  };
};
