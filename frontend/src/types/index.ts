// Central export file for all TypeScript types

// Authentication types
export type {
  User,
  LoginRequest,
  RegisterRequest,
  TokenResponse,
  UserProfileUpdate,
  AuthContextType,
  AuthState,
  LoginFormData,
  RegisterFormData,
  PasswordResetRequest,
  PasswordResetConfirm,
} from "./auth";

// Financial types
export type {
  TransactionType,
  TransactionCategory,
  TransactionCreate,
  Transaction,
  Wallet,
  SpendingAnalytics,
  FinancialHealthSummary,
  TransactionFilter,
  TransactionFormData,
  TransactionListState,
  WalletState,
  FinancialFilters,
  SpendingTrend,
  BudgetGoal,
  FinancialInsight,
  MonthlyReport,
} from "./financial";

// Product types
export type {
  ProductCategory,
  Product,
  ProductCreate,
  PurchaseRequest,
  Purchase,
  ProductFilter,
  CategoryCreate,
  ProductWithCategory,
  CartItem,
  ShoppingCart,
  ProductListState,
  ProductFilters,
  PurchaseFormData,
  PurchaseHistory,
  ProductReview,
  ProductWithReviews,
  WishlistItem,
  MarketplaceStats,
  ProductSearchResult,
} from "./products";

// Coaching types
export type {
  AIRecommendation,
  FinancialState,
  CoachingAction,
  ModelMetrics,
  RecommendationHistory,
  CoachingSession,
  CoachingFeedback,
  CoachingGoal,
  CoachingInsight,
  CoachingDashboard,
  RecommendationCard,
  CoachingPreferences,
  FinancialHealthScore,
  CoachingAnalytics,
} from "./coaching";

// Common types
export type {
  ApiError,
  PaginatedResponse,
  ApiResponse,
  LoadingState,
  SelectOption,
  TableColumn,
  SortConfig,
  FilterConfig,
  SearchConfig,
  DateRange,
  NumberRange,
  ValidationError,
  FormState,
  ModalState,
  ToastNotification,
  BreadcrumbItem,
  TabItem,
  MenuItem,
  ChartDataPoint,
  ChartConfig,
  FileUpload,
  InfiniteScrollState,
  KeyboardShortcut,
  Theme,
  Currency,
  Language,
  UserPreferences,
} from "./common";
