import api from "./api";
import type {
  LoginRequest,
  RegisterRequest,
  TokenResponse,
  User,
} from "../types";

const TOKEN_KEY = "access_token";

export class AuthService {
  // Token management
  static getToken(): string | null {
    return localStorage.getItem(TOKEN_KEY);
  }

  static setToken(token: string): void {
    localStorage.setItem(TOKEN_KEY, token);
  }

  static removeToken(): void {
    localStorage.removeItem(TOKEN_KEY);
  }

  static isTokenExpired(token: string): boolean {
    try {
      const payload = JSON.parse(atob(token.split(".")[1]));
      const currentTime = Date.now() / 1000;
      return payload.exp < currentTime;
    } catch {
      return true;
    }
  }

  static isAuthenticated(): boolean {
    const token = this.getToken();
    if (!token) return false;
    return !this.isTokenExpired(token);
  }

  // API calls
  static async login(credentials: LoginRequest): Promise<TokenResponse> {
    const response = await api.post<TokenResponse>("/auth/login", credentials);

    if (response.data.access_token) {
      this.setToken(response.data.access_token);
    }

    return response.data;
  }

  static async register(userData: RegisterRequest): Promise<TokenResponse> {
    const response = await api.post<TokenResponse>("/auth/register", userData);

    if (response.data.access_token) {
      this.setToken(response.data.access_token);
    }

    return response.data;
  }

  static async getCurrentUser(): Promise<User> {
    const response = await api.get<User>("/auth/me");
    return response.data;
  }

  static async logout(): Promise<void> {
    try {
      await api.post("/auth/logout");
    } catch (error) {
      // Even if logout fails on server, we should clear local token
      console.warn("Logout request failed:", error);
    } finally {
      this.removeToken();
    }
  }

  static async refreshToken(): Promise<TokenResponse> {
    const response = await api.post<TokenResponse>("/auth/refresh");

    if (response.data.access_token) {
      this.setToken(response.data.access_token);
    }

    return response.data;
  }
}
