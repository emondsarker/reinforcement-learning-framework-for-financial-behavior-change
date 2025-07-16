import React from "react";
import type { AdminStats as AdminStatsType } from "../../types";
import {
  UsersIcon,
  CreditCardIcon,
  CubeIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  XCircleIcon,
} from "@heroicons/react/24/outline";

interface AdminStatsProps {
  stats: AdminStatsType;
  className?: string;
}

export const AdminStats: React.FC<AdminStatsProps> = ({
  stats,
  className = "",
}) => {
  const getHealthIcon = (health: AdminStatsType["systemHealth"]) => {
    switch (health) {
      case "healthy":
        return <CheckCircleIcon className="h-5 w-5 text-green-500" />;
      case "warning":
        return <ExclamationTriangleIcon className="h-5 w-5 text-yellow-500" />;
      case "critical":
        return <XCircleIcon className="h-5 w-5 text-red-500" />;
      default:
        return <ExclamationTriangleIcon className="h-5 w-5 text-gray-500" />;
    }
  };

  const getHealthColor = (health: AdminStatsType["systemHealth"]) => {
    switch (health) {
      case "healthy":
        return "text-green-600 bg-green-50";
      case "warning":
        return "text-yellow-600 bg-yellow-50";
      case "critical":
        return "text-red-600 bg-red-50";
      default:
        return "text-gray-600 bg-gray-50";
    }
  };

  const statsData = [
    {
      name: "Total Users",
      value: stats.totalUsers.toLocaleString(),
      subValue: `${stats.activeUsers} active`,
      icon: <UsersIcon className="h-6 w-6" />,
      color: "text-blue-600 bg-blue-50",
    },
    {
      name: "Total Revenue",
      value: `$${stats.totalRevenue.toLocaleString()}`,
      subValue: `${stats.totalTransactions} transactions`,
      icon: <CreditCardIcon className="h-6 w-6" />,
      color: "text-green-600 bg-green-50",
    },
    {
      name: "Products",
      value: stats.totalProducts.toLocaleString(),
      subValue: `${stats.lowStockProducts} low stock`,
      icon: <CubeIcon className="h-6 w-6" />,
      color: "text-purple-600 bg-purple-50",
    },
    {
      name: "System Health",
      value:
        stats.systemHealth.charAt(0).toUpperCase() +
        stats.systemHealth.slice(1),
      subValue: `Updated ${new Date(stats.lastUpdated).toLocaleTimeString()}`,
      icon: getHealthIcon(stats.systemHealth),
      color: getHealthColor(stats.systemHealth),
    },
  ];

  return (
    <div
      className={`grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 ${className}`}
    >
      {statsData.map((stat) => (
        <div key={stat.name} className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <div className={`p-2 rounded-lg ${stat.color}`}>{stat.icon}</div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">{stat.name}</p>
              <p className="text-2xl font-semibold text-gray-900">
                {stat.value}
              </p>
            </div>
          </div>
          <div className="mt-4">
            <p className="text-sm text-gray-500">{stat.subValue}</p>
          </div>
        </div>
      ))}
    </div>
  );
};
