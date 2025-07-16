import React, { useState } from "react";
import type { UserOverview } from "../../types";
import { useAdminUsers, useToggleUserStatus } from "../../hooks/useAdmin";
import {
  UsersIcon,
  MagnifyingGlassIcon,
  EyeIcon,
  UserCircleIcon,
  CheckCircleIcon,
  XCircleIcon,
} from "@heroicons/react/24/outline";

interface UserManagementProps {
  recentUsers?: UserOverview[];
  compact?: boolean;
  className?: string;
}

export const UserManagement: React.FC<UserManagementProps> = ({
  recentUsers,
  compact = false,
  className = "",
}) => {
  const [searchTerm, setSearchTerm] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = compact ? 5 : 10;

  const {
    data: usersData,
    isLoading,
    error,
  } = useAdminUsers({ search: searchTerm }, currentPage, pageSize);
  const toggleUserStatus = useToggleUserStatus();

  const users = compact ? recentUsers || [] : usersData?.data || [];
  const totalPages = compact ? 1 : usersData?.total_pages || 1;

  const handleToggleUserStatus = async (
    userId: string,
    currentStatus: boolean
  ) => {
    try {
      await toggleUserStatus.mutateAsync({
        userId,
        isActive: !currentStatus,
      });
    } catch (error) {
      console.error("Failed to toggle user status:", error);
    }
  };

  if (compact) {
    return (
      <div className={className}>
        {recentUsers && recentUsers.length > 0 ? (
          <div className="space-y-3">
            {recentUsers.slice(0, 5).map((user) => (
              <div
                key={user.id}
                className="flex items-center justify-between py-2 border-b border-gray-100 last:border-b-0"
              >
                <div className="flex items-center">
                  <UserCircleIcon className="h-8 w-8 text-gray-400 mr-3" />
                  <div>
                    <p className="text-sm font-medium text-gray-900">
                      {user.firstName} {user.lastName}
                    </p>
                    <p className="text-xs text-gray-500">{user.email}</p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <div
                    className={`w-2 h-2 rounded-full ${
                      user.isActive ? "bg-green-400" : "bg-red-400"
                    }`}
                  ></div>
                  <span className="text-xs text-gray-500">
                    {user.isActive ? "Active" : "Inactive"}
                  </span>
                </div>
              </div>
            ))}
            <div className="pt-3 border-t border-gray-200">
              <button className="text-sm text-blue-600 hover:text-blue-800 font-medium">
                View all users →
              </button>
            </div>
          </div>
        ) : (
          <div className="text-center py-8">
            <UsersIcon className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-2 text-sm font-medium text-gray-900">No users</h3>
            <p className="mt-1 text-sm text-gray-500">
              Recent users will appear here.
            </p>
          </div>
        )}
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className={`bg-white rounded-lg shadow ${className}`}>
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <UsersIcon className="h-6 w-6 text-gray-400 mr-3" />
              <h2 className="text-lg font-semibold text-gray-900">
                User Management
              </h2>
            </div>
          </div>
        </div>
        <div className="p-6">
          <div className="space-y-4">
            {[...Array(5)].map((_, i) => (
              <div
                key={i}
                className="flex items-center justify-between py-3 animate-pulse"
              >
                <div className="flex items-center">
                  <div className="h-10 w-10 bg-gray-200 rounded-full mr-3"></div>
                  <div>
                    <div className="h-4 bg-gray-200 rounded w-32 mb-1"></div>
                    <div className="h-3 bg-gray-200 rounded w-48"></div>
                  </div>
                </div>
                <div className="h-6 bg-gray-200 rounded w-16"></div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`bg-white rounded-lg shadow ${className}`}>
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center">
            <UsersIcon className="h-6 w-6 text-gray-400 mr-3" />
            <h2 className="text-lg font-semibold text-gray-900">
              User Management
            </h2>
          </div>
        </div>
        <div className="p-6">
          <div className="text-center py-8">
            <XCircleIcon className="mx-auto h-12 w-12 text-red-400" />
            <h3 className="mt-2 text-sm font-medium text-gray-900">
              Error loading users
            </h3>
            <p className="mt-1 text-sm text-gray-500">
              There was an error loading user data. Please try again.
            </p>
            <button
              onClick={() => window.location.reload()}
              className="mt-4 px-4 py-2 bg-black text-white rounded-md hover:bg-gray-800 transition-colors"
            >
              Retry
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-white rounded-lg shadow ${className}`}>
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <UsersIcon className="h-6 w-6 text-gray-400 mr-3" />
            <h2 className="text-lg font-semibold text-gray-900">
              User Management
            </h2>
          </div>
          <div className="text-sm text-gray-500">
            {usersData?.total || 0} total users
          </div>
        </div>
      </div>

      <div className="p-6">
        {/* Search */}
        <div className="mb-6">
          <div className="relative">
            <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search users..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-black focus:border-transparent"
            />
          </div>
        </div>

        {/* Users List */}
        {users.length > 0 ? (
          <div className="space-y-4">
            {users.map((user) => (
              <div
                key={user.id}
                className="flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-center">
                  <UserCircleIcon className="h-12 w-12 text-gray-400 mr-4" />
                  <div>
                    <h3 className="text-sm font-medium text-gray-900">
                      {user.firstName} {user.lastName}
                    </h3>
                    <p className="text-sm text-gray-500">{user.email}</p>
                    <div className="flex items-center mt-1 text-xs text-gray-500">
                      <span>
                        Joined {new Date(user.createdAt).toLocaleDateString()}
                      </span>
                      {user.lastLogin && (
                        <>
                          <span className="mx-2">•</span>
                          <span>
                            Last login{" "}
                            {new Date(user.lastLogin).toLocaleDateString()}
                          </span>
                        </>
                      )}
                    </div>
                  </div>
                </div>

                <div className="flex items-center space-x-4">
                  <div className="text-right">
                    <div className="text-sm font-medium text-gray-900">
                      ${user.totalSpent.toFixed(2)}
                    </div>
                    <div className="text-xs text-gray-500">
                      {user.transactionCount} transactions
                    </div>
                  </div>

                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() =>
                        handleToggleUserStatus(user.id, user.isActive)
                      }
                      disabled={toggleUserStatus.isPending}
                      className={`flex items-center px-3 py-1 rounded-full text-xs font-medium transition-colors ${
                        user.isActive
                          ? "bg-green-100 text-green-800 hover:bg-green-200"
                          : "bg-red-100 text-red-800 hover:bg-red-200"
                      } ${toggleUserStatus.isPending ? "opacity-50 cursor-not-allowed" : ""}`}
                    >
                      {user.isActive ? (
                        <CheckCircleIcon className="h-3 w-3 mr-1" />
                      ) : (
                        <XCircleIcon className="h-3 w-3 mr-1" />
                      )}
                      {user.isActive ? "Active" : "Inactive"}
                    </button>

                    <button className="p-2 text-gray-400 hover:text-gray-600 transition-colors">
                      <EyeIcon className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8">
            <UsersIcon className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-2 text-sm font-medium text-gray-900">
              No users found
            </h3>
            <p className="mt-1 text-sm text-gray-500">
              {searchTerm
                ? "Try adjusting your search terms."
                : "No users have been registered yet."}
            </p>
          </div>
        )}

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="mt-6 flex items-center justify-between border-t border-gray-200 pt-6">
            <div className="text-sm text-gray-500">
              Page {currentPage} of {totalPages}
            </div>
            <div className="flex space-x-2">
              <button
                onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                disabled={currentPage === 1}
                className="px-3 py-1 text-sm border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Previous
              </button>
              <button
                onClick={() =>
                  setCurrentPage(Math.min(totalPages, currentPage + 1))
                }
                disabled={currentPage === totalPages}
                className="px-3 py-1 text-sm border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
