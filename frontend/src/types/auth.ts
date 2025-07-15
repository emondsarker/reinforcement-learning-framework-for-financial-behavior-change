// Authentication-related types matching backend models

export interface User {
  id: string;
  email: string;
  first_name: string;
  last_name: string;
  created_at: Date;
  is_active: boolean;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  password: string;
  first_name: string;
  last_name: string;
  date_of_birth?: string; // ISO date string (YYYY-MM-DD)
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface UserProfileUpdate {
  monthly_income?: number;
  savings_goal?: number;
  risk_tolerance?: "low" | "medium" | "high";
  financial_goals?: string[];
}

// Frontend-specific auth types
export interface AuthContextType {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (credentials: LoginRequest) => Promise<void>;
  register: (userData: RegisterRequest) => Promise<void>;
  logout: () => void;
  updateProfile: (updates: UserProfileUpdate) => Promise<void>;
}

export interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
}

export interface LoginFormData {
  email: string;
  password: string;
  rememberMe?: boolean;
}

export interface RegisterFormData {
  email: string;
  password: string;
  confirmPassword: string;
  first_name: string;
  last_name: string;
  date_of_birth?: string;
  agreeToTerms: boolean;
}

export interface PasswordResetRequest {
  email: string;
}

export interface PasswordResetConfirm {
  token: string;
  new_password: string;
  confirm_password: string;
}
