import React from "react";
import { useAuthContext } from "../../contexts/AuthContext";
import { useAuth } from "../../hooks/useAuth";
import { useToast } from "../../hooks/useToast";

export const Header: React.FC = () => {
  const { user } = useAuthContext();
  const { logoutMutation } = useAuth();
  const { addToast } = useToast();

  const handleLogout = async () => {
    try {
      await logoutMutation.mutateAsync();
      addToast({
        type: "success",
        title: "Logged Out",
        message: "You have been successfully logged out.",
      });
    } catch {
      addToast({
        type: "error",
        title: "Logout Error",
        message: "There was an error logging out.",
      });
    }
  };

  return (
    <header className="bg-white border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <div className="flex-shrink-0">
            <h1 className="text-2xl font-bold text-black">FinCoach</h1>
          </div>

          {/* User Menu */}
          <div className="flex items-center space-x-4">
            {user && (
              <>
                <span className="text-sm text-gray-700">
                  Welcome, {user.first_name}
                </span>
                <button
                  onClick={handleLogout}
                  disabled={logoutMutation.isPending}
                  className="text-sm text-gray-600 hover:text-black transition-colors duration-200 disabled:opacity-50"
                >
                  {logoutMutation.isPending ? "Logging out..." : "Logout"}
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};
