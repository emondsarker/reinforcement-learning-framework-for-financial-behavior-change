import React from "react";
import { useAuth } from "../hooks/useAuth";
import { useAdminDashboard, useAdminLoadingState } from "../hooks/useAdmin";
import { UserManagement } from "../components/admin/UserManagement";
import { ProductManagement } from "../components/admin/ProductManagement";
import { SystemHealth } from "../components/admin/SystemHealth";
import { AdminStats } from "../components/admin/AdminStats";
import {
  UsersIcon,
  CubeIcon,
  ChartBarIcon,
  Cog6ToothIcon,
} from "@heroicons/react/24/outline";

export const AdminPage: React.FC = () => {
  const { user, isAdmin } = useAuth();
  const { data: dashboardData, isLoading, error } = useAdminDashboard();
  const { isLoading: globalLoading, isError: globalError } =
    useAdminLoadingState();

  // Redirect if not admin
  if (!isAdmin) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="mx-auto h-12 w-12 text-red-400">
            <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.728-.833-2.498 0L4.316 18.5c-.77.833.192 2.5 1.732 2.5z"
              />
            </svg>
          </div>
          <h1 className="mt-4 text-xl font-semibold text-gray-900">
            Access Denied
          </h1>
          <p className="mt-2 text-gray-600">
            You don't have permission to access the admin panel.
          </p>
        </div>
      </div>
    );
  }

  if (globalLoading || isLoading) {
    return (
      <div className="min-h-screen bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Header Skeleton */}
          <div className="mb-8">
            <div className="h-8 bg-gray-200 rounded w-64 mb-2 animate-pulse"></div>
            <div className="h-4 bg-gray-200 rounded w-96 animate-pulse"></div>
          </div>

          {/* Stats Grid Skeleton */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            {[...Array(4)].map((_, i) => (
              <div
                key={i}
                className="bg-white p-6 rounded-lg shadow animate-pulse"
              >
                <div className="h-4 bg-gray-200 rounded w-24 mb-2"></div>
                <div className="h-8 bg-gray-200 rounded w-16"></div>
              </div>
            ))}
          </div>

          {/* Content Skeleton */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="bg-white p-6 rounded-lg shadow animate-pulse">
              <div className="h-6 bg-gray-200 rounded w-32 mb-4"></div>
              <div className="space-y-3">
                {[...Array(5)].map((_, i) => (
                  <div key={i} className="h-4 bg-gray-200 rounded"></div>
                ))}
              </div>
            </div>
            <div className="bg-white p-6 rounded-lg shadow animate-pulse">
              <div className="h-6 bg-gray-200 rounded w-32 mb-4"></div>
              <div className="space-y-3">
                {[...Array(5)].map((_, i) => (
                  <div key={i} className="h-4 bg-gray-200 rounded"></div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (globalError || error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="mx-auto h-12 w-12 text-red-400">
            <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </div>
          <h1 className="mt-4 text-xl font-semibold text-gray-900">
            Error Loading Admin Panel
          </h1>
          <p className="mt-2 text-gray-600">
            There was an error loading the admin dashboard. Please try again.
          </p>
          <button
            onClick={() => window.location.reload()}
            className="mt-4 px-4 py-2 bg-black text-white rounded-md hover:bg-gray-800 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Admin Dashboard</h1>
          <p className="mt-2 text-gray-600">
            Welcome back, {user?.first_name}. Manage your FinCoach system.
          </p>
        </div>

        {/* System Stats */}
        {dashboardData?.stats && (
          <AdminStats stats={dashboardData.stats} className="mb-8" />
        )}

        {/* System Health */}
        {dashboardData?.systemHealth && (
          <div className="mb-8">
            <SystemHealth health={dashboardData.systemHealth} />
          </div>
        )}

        {/* Management Sections */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* User Management */}
          <div className="bg-white rounded-lg shadow">
            <div className="p-6 border-b border-gray-200">
              <div className="flex items-center">
                <UsersIcon className="h-6 w-6 text-gray-400 mr-3" />
                <h2 className="text-lg font-semibold text-gray-900">
                  User Management
                </h2>
              </div>
            </div>
            <div className="p-6">
              <UserManagement
                recentUsers={dashboardData?.recentUsers || []}
                compact={true}
              />
            </div>
          </div>

          {/* Product Management */}
          <div className="bg-white rounded-lg shadow">
            <div className="p-6 border-b border-gray-200">
              <div className="flex items-center">
                <CubeIcon className="h-6 w-6 text-gray-400 mr-3" />
                <h2 className="text-lg font-semibold text-gray-900">
                  Product Management
                </h2>
              </div>
            </div>
            <div className="p-6">
              <ProductManagement
                topProducts={dashboardData?.topProducts || []}
                compact={true}
              />
            </div>
          </div>
        </div>

        {/* Recent Activity */}
        <div className="bg-white rounded-lg shadow">
          <div className="p-6 border-b border-gray-200">
            <div className="flex items-center">
              <ChartBarIcon className="h-6 w-6 text-gray-400 mr-3" />
              <h2 className="text-lg font-semibold text-gray-900">
                Recent Activity
              </h2>
            </div>
          </div>
          <div className="p-6">
            {dashboardData?.recentTransactions &&
            dashboardData.recentTransactions.length > 0 ? (
              <div className="space-y-4">
                {dashboardData.recentTransactions
                  .slice(0, 5)
                  .map((transaction) => (
                    <div
                      key={transaction.id}
                      className="flex items-center justify-between py-3 border-b border-gray-100 last:border-b-0"
                    >
                      <div className="flex items-center">
                        <div className="flex-shrink-0">
                          <div
                            className={`w-2 h-2 rounded-full ${
                              transaction.transactionType === "credit"
                                ? "bg-green-400"
                                : "bg-red-400"
                            }`}
                          ></div>
                        </div>
                        <div className="ml-3">
                          <p className="text-sm font-medium text-gray-900">
                            {transaction.userName}
                          </p>
                          <p className="text-sm text-gray-500">
                            {transaction.description} • {transaction.category}
                          </p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p
                          className={`text-sm font-medium ${
                            transaction.transactionType === "credit"
                              ? "text-green-600"
                              : "text-red-600"
                          }`}
                        >
                          {transaction.transactionType === "credit" ? "+" : "-"}
                          ${Math.abs(transaction.amount).toFixed(2)}
                        </p>
                        <p className="text-xs text-gray-500">
                          {new Date(
                            transaction.transactionDate
                          ).toLocaleDateString()}
                        </p>
                      </div>
                    </div>
                  ))}
              </div>
            ) : (
              <div className="text-center py-8">
                <ChartBarIcon className="mx-auto h-12 w-12 text-gray-400" />
                <h3 className="mt-2 text-sm font-medium text-gray-900">
                  No recent activity
                </h3>
                <p className="mt-1 text-sm text-gray-500">
                  Recent transactions will appear here.
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Quick Actions */}
        <div className="mt-8 bg-white rounded-lg shadow">
          <div className="p-6 border-b border-gray-200">
            <div className="flex items-center">
              <Cog6ToothIcon className="h-6 w-6 text-gray-400 mr-3" />
              <h2 className="text-lg font-semibold text-gray-900">
                Quick Actions
              </h2>
            </div>
          </div>
          <div className="p-6">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <button className="p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors text-left">
                <UsersIcon className="h-6 w-6 text-gray-400 mb-2" />
                <h3 className="text-sm font-medium text-gray-900">
                  Manage Users
                </h3>
                <p className="text-xs text-gray-500">
                  View and manage user accounts
                </p>
              </button>

              <button className="p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors text-left">
                <CubeIcon className="h-6 w-6 text-gray-400 mb-2" />
                <h3 className="text-sm font-medium text-gray-900">
                  Add Product
                </h3>
                <p className="text-xs text-gray-500">
                  Add new products to marketplace
                </p>
              </button>

              <button className="p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors text-left">
                <ChartBarIcon className="h-6 w-6 text-gray-400 mb-2" />
                <h3 className="text-sm font-medium text-gray-900">
                  View Reports
                </h3>
                <p className="text-xs text-gray-500">
                  System analytics and reports
                </p>
              </button>

              <button className="p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors text-left">
                <Cog6ToothIcon className="h-6 w-6 text-gray-400 mb-2" />
                <h3 className="text-sm font-medium text-gray-900">
                  System Settings
                </h3>
                <p className="text-xs text-gray-500">
                  Configure system settings
                </p>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
