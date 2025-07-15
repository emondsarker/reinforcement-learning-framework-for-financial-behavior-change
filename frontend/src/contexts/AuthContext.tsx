import React, { createContext, useContext, useEffect, useState } from "react";
import type {
  User,
  AuthContextType,
  LoginRequest,
  RegisterRequest,
  UserProfileUpdate,
} from "../types";
import { AuthService } from "../services/authService";

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuthContext = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuthContext must be used within an AuthProvider");
  }
  return context;
};

interface AuthProviderProps {
  children: React.ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  // Initialize auth state on mount
  useEffect(() => {
    const initializeAuth = async () => {
      try {
        const currentToken = AuthService.getToken();
        if (currentToken && AuthService.isAuthenticated()) {
          const userData = await AuthService.getCurrentUser();
          setUser(userData);
          setToken(currentToken);
          setIsAuthenticated(true);
        }
      } catch (error) {
        // Token might be expired or invalid
        console.warn("Failed to initialize auth:", error);
        AuthService.removeToken();
        setUser(null);
        setToken(null);
        setIsAuthenticated(false);
      } finally {
        setIsLoading(false);
      }
    };

    initializeAuth();
  }, []);

  const login = async (credentials: LoginRequest): Promise<void> => {
    setIsLoading(true);
    try {
      const response = await AuthService.login(credentials);
      setUser(response.user);
      setToken(response.access_token);
      setIsAuthenticated(true);
    } finally {
      setIsLoading(false);
    }
  };

  const register = async (userData: RegisterRequest): Promise<void> => {
    setIsLoading(true);
    try {
      const response = await AuthService.register(userData);
      setUser(response.user);
      setToken(response.access_token);
      setIsAuthenticated(true);
    } finally {
      setIsLoading(false);
    }
  };

  const logout = (): void => {
    setIsLoading(true);
    try {
      AuthService.logout();
    } catch (error) {
      console.warn("Logout error:", error);
    } finally {
      setUser(null);
      setToken(null);
      setIsAuthenticated(false);
      setIsLoading(false);
    }
  };

  const updateProfile = async (updates: UserProfileUpdate): Promise<void> => {
    // This would be implemented when we add profile update functionality
    // For now, just a placeholder
    console.log("Profile update not implemented yet:", updates);
  };

  const value: AuthContextType = {
    user,
    token,
    isAuthenticated,
    isLoading,
    login,
    register,
    logout,
    updateProfile,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};
