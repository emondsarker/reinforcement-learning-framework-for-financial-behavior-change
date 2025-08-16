import React from "react";
import { NavLink, useLocation } from "react-router-dom";
import {
  HomeIcon,
  WalletIcon,
  ShoppingBagIcon,
} from "@heroicons/react/24/outline";

const navigationItems = [
  {
    name: "Dashboard",
    href: "/dashboard",
    icon: HomeIcon,
    subItems: [
      {
        name: "Overview",
        href: "/dashboard",
      },
      {
        name: "Analytics",
        href: "/analytics",
      },
    ],
  },
  {
    name: "Wallet",
    href: "/wallet",
    icon: WalletIcon,
  },
  {
    name: "Marketplace",
    href: "/marketplace",
    icon: ShoppingBagIcon,
  },
];

export const Navigation: React.FC = () => {
  const location = useLocation();

  const isItemActive = (item: (typeof navigationItems)[0]) => {
    if (item.subItems) {
      return item.subItems.some(
        (subItem) => location.pathname === subItem.href
      );
    }
    return location.pathname === item.href;
  };

  return (
    <nav className="bg-white border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Main Navigation */}
        <div className="flex space-x-8">
          {navigationItems.map((item) => (
            <NavLink
              key={item.name}
              to={item.href}
              className={() =>
                `flex items-center px-1 py-4 text-sm font-medium border-b-2 transition-colors duration-200 ${
                  isItemActive(item)
                    ? "border-black text-black"
                    : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                }`
              }
            >
              <item.icon className="w-5 h-5 mr-2" />
              {item.name}
            </NavLink>
          ))}
        </div>

        {/* Sub Navigation for Dashboard */}
        {(location.pathname === "/dashboard" ||
          location.pathname === "/analytics") && (
          <div className="border-t border-gray-100">
            <div className="flex space-x-6 py-2">
              <NavLink
                to="/dashboard"
                className={({ isActive }) =>
                  `px-3 py-2 text-sm font-medium rounded-md transition-colors duration-200 ${
                    isActive
                      ? "bg-gray-100 text-gray-900"
                      : "text-gray-600 hover:text-gray-900 hover:bg-gray-50"
                  }`
                }
              >
                Overview
              </NavLink>
              <NavLink
                to="/analytics"
                className={({ isActive }) =>
                  `px-3 py-2 text-sm font-medium rounded-md transition-colors duration-200 ${
                    isActive
                      ? "bg-gray-100 text-gray-900"
                      : "text-gray-600 hover:text-gray-900 hover:bg-gray-50"
                  }`
                }
              >
                Analytics
              </NavLink>
            </div>
          </div>
        )}
      </div>
    </nav>
  );
};
