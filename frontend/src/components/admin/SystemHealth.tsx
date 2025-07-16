import React from "react";
import type { SystemHealth as SystemHealthType } from "../../types";
import {
  CheckCircleIcon,
  ExclamationTriangleIcon,
  XCircleIcon,
  ClockIcon,
  ServerIcon,
  CpuChipIcon,
  CircleStackIcon,
} from "@heroicons/react/24/outline";

interface SystemHealthProps {
  health: SystemHealthType;
  className?: string;
}

export const SystemHealth: React.FC<SystemHealthProps> = ({
  health,
  className = "",
}) => {
  const getStatusIcon = (status: SystemHealthType["status"]) => {
    switch (status) {
      case "healthy":
        return <CheckCircleIcon className="h-6 w-6 text-green-500" />;
      case "warning":
        return <ExclamationTriangleIcon className="h-6 w-6 text-yellow-500" />;
      case "critical":
        return <XCircleIcon className="h-6 w-6 text-red-500" />;
      default:
        return <ExclamationTriangleIcon className="h-6 w-6 text-gray-500" />;
    }
  };

  const getStatusColor = (status: SystemHealthType["status"]) => {
    switch (status) {
      case "healthy":
        return "text-green-600 bg-green-50 border-green-200";
      case "warning":
        return "text-yellow-600 bg-yellow-50 border-yellow-200";
      case "critical":
        return "text-red-600 bg-red-50 border-red-200";
      default:
        return "text-gray-600 bg-gray-50 border-gray-200";
    }
  };

  const formatUptime = (uptime: number) => {
    const hours = Math.floor(uptime / (1000 * 60 * 60));
    const days = Math.floor(hours / 24);
    const remainingHours = hours % 24;

    if (days > 0) {
      return `${days}d ${remainingHours}h`;
    }
    return `${hours}h`;
  };

  const getMetricColor = (
    value: number,
    thresholds: { warning: number; critical: number }
  ) => {
    if (value >= thresholds.critical) return "text-red-600";
    if (value >= thresholds.warning) return "text-yellow-600";
    return "text-green-600";
  };

  const metrics = [
    {
      name: "API Response Time",
      value: `${health.apiResponseTime}ms`,
      icon: <ClockIcon className="h-5 w-5" />,
      color: getMetricColor(health.apiResponseTime, {
        warning: 1000,
        critical: 3000,
      }),
    },
    {
      name: "Database Connections",
      value: health.databaseConnections.toString(),
      icon: <CircleStackIcon className="h-5 w-5" />,
      color: getMetricColor(health.databaseConnections, {
        warning: 80,
        critical: 95,
      }),
    },
    {
      name: "Memory Usage",
      value: `${health.memoryUsage}%`,
      icon: <ServerIcon className="h-5 w-5" />,
      color: getMetricColor(health.memoryUsage, { warning: 80, critical: 90 }),
    },
    {
      name: "CPU Usage",
      value: `${health.cpuUsage}%`,
      icon: <CpuChipIcon className="h-5 w-5" />,
      color: getMetricColor(health.cpuUsage, { warning: 70, critical: 85 }),
    },
  ];

  return (
    <div className={`bg-white rounded-lg shadow ${className}`}>
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <ServerIcon className="h-6 w-6 text-gray-400 mr-3" />
            <h2 className="text-lg font-semibold text-gray-900">
              System Health
            </h2>
          </div>
          <div
            className={`flex items-center px-3 py-1 rounded-full border ${getStatusColor(health.status)}`}
          >
            {getStatusIcon(health.status)}
            <span className="ml-2 text-sm font-medium capitalize">
              {health.status}
            </span>
          </div>
        </div>
      </div>

      <div className="p-6">
        {/* Overall Status */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-gray-600">
              System Status
            </span>
            <span className="text-sm text-gray-500">
              Last checked: {new Date(health.lastChecked).toLocaleTimeString()}
            </span>
          </div>
          <div className="flex items-center">
            <div className="flex-1">
              <div className="text-2xl font-semibold text-gray-900 capitalize">
                {health.status}
              </div>
              <div className="text-sm text-gray-500">
                Uptime: {formatUptime(health.uptime)}
              </div>
            </div>
            <div className="text-right">
              <div className="text-sm text-gray-600">Error Rate</div>
              <div
                className={`text-lg font-semibold ${getMetricColor(health.errorRate * 100, { warning: 5, critical: 10 })}`}
              >
                {(health.errorRate * 100).toFixed(2)}%
              </div>
            </div>
          </div>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {metrics.map((metric) => (
            <div
              key={metric.name}
              className="p-4 border border-gray-200 rounded-lg"
            >
              <div className="flex items-center mb-2">
                <div className="text-gray-400 mr-2">{metric.icon}</div>
                <span className="text-xs font-medium text-gray-600 truncate">
                  {metric.name}
                </span>
              </div>
              <div className={`text-lg font-semibold ${metric.color}`}>
                {metric.value}
              </div>
            </div>
          ))}
        </div>

        {/* Status Indicators */}
        <div className="mt-6 pt-6 border-t border-gray-200">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center space-x-4">
              <div className="flex items-center">
                <div className="w-2 h-2 bg-green-400 rounded-full mr-2"></div>
                <span className="text-gray-600">Healthy</span>
              </div>
              <div className="flex items-center">
                <div className="w-2 h-2 bg-yellow-400 rounded-full mr-2"></div>
                <span className="text-gray-600">Warning</span>
              </div>
              <div className="flex items-center">
                <div className="w-2 h-2 bg-red-400 rounded-full mr-2"></div>
                <span className="text-gray-600">Critical</span>
              </div>
            </div>
            <button
              onClick={() => window.location.reload()}
              className="px-3 py-1 text-xs font-medium text-gray-600 hover:text-gray-900 border border-gray-300 rounded hover:bg-gray-50 transition-colors"
            >
              Refresh
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
