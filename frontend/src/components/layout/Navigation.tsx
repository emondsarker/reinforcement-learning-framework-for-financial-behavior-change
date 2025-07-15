import React from "react";
import { NavLink } from "react-router-dom";
import {
  HomeIcon,
  WalletIcon,
  ShoppingBagIcon,
  SparklesIcon,
} from "@heroicons/react/24/outline";

const navigationItems = [
  {
    name: "Dashboard",
    href: "/dashboard",
    icon: HomeIcon,
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
  {
    name: "AI Coach",
    href: "/coaching",
    icon: SparklesIcon,
  },
];

export const Navigation: React.FC = () => {
  return (
    <nav className="bg-white border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex space-x-8">
          {navigationItems.map((item) => (
            <NavLink
              key={item.name}
              to={item.href}
              className={({ isActive }) =>
                `flex items-center px-1 py-4 text-sm font-medium border-b-2 transition-colors duration-200 ${
                  isActive
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
      </div>
    </nav>
  );
};
