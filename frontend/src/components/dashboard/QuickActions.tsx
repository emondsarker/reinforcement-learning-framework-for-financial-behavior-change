import React from "react";
import { Link } from "react-router-dom";
import {
  PlusIcon,
  WalletIcon,
  ChartBarIcon,
  ArrowTrendingUpIcon,
  CreditCardIcon,
} from "@heroicons/react/24/outline";

interface QuickActionProps {
  to: string;
  icon: React.ReactNode;
  title: string;
  description: string;
  color: string;
}

const QuickAction: React.FC<QuickActionProps> = ({
  to,
  icon,
  title,
  description,
  color,
}) => {
  return (
    <Link
      to={to}
      className="group relative bg-white rounded-lg shadow-sm border border-gray-200 p-6 hover:shadow-md transition-all duration-200 hover:border-gray-300"
    >
      <div className="flex items-center">
        <div
          className={`p-3 rounded-lg ${color} group-hover:scale-110 transition-transform duration-200`}
        >
          {icon}
        </div>
        <div className="ml-4 flex-1">
          <h4 className="text-sm font-semibold text-gray-900 group-hover:text-gray-700">
            {title}
          </h4>
          <p className="text-xs text-gray-500 mt-1">{description}</p>
        </div>
      </div>
      <div className="absolute inset-0 rounded-lg ring-2 ring-transparent group-hover:ring-gray-200 transition-all duration-200" />
    </Link>
  );
};

export const QuickActions: React.FC = () => {
  const actions = [
    {
      to: "/wallet?action=add",
      icon: <PlusIcon className="h-6 w-6 text-white" />,
      title: "Add Transaction",
      description: "Record a new transaction",
      color: "bg-blue-500",
    },
    {
      to: "/wallet",
      icon: <WalletIcon className="h-6 w-6 text-white" />,
      title: "View Wallet",
      description: "Manage your finances",
      color: "bg-green-500",
    },
    {
      to: "/analytics",
      icon: <ChartBarIcon className="h-6 w-6 text-white" />,
      title: "Analytics",
      description: "View spending insights",
      color: "bg-orange-500",
    },
    {
      to: "/coaching",
      icon: <ArrowTrendingUpIcon className="h-6 w-6 text-white" />,
      title: "AI Coaching",
      description: "Get financial advice",
      color: "bg-indigo-500",
    },
    {
      to: "/wallet?action=balance",
      icon: <CreditCardIcon className="h-6 w-6 text-white" />,
      title: "Update Balance",
      description: "Adjust wallet balance",
      color: "bg-gray-500",
    },
  ];

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-gray-900">Quick Actions</h3>
        <p className="text-sm text-gray-500 mt-1">Common tasks and shortcuts</p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {actions.map((action, index) => (
          <QuickAction
            key={index}
            to={action.to}
            icon={action.icon}
            title={action.title}
            description={action.description}
            color={action.color}
          />
        ))}
      </div>
    </div>
  );
};
