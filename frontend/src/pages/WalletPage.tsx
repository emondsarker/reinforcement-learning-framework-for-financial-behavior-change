import React from "react";

export const WalletPage: React.FC = () => {
  return (
    <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
      <div className="px-4 py-6 sm:px-0">
        <div className="border-4 border-dashed border-gray-200 rounded-lg h-96 flex items-center justify-center">
          <div className="text-center">
            <h1 className="text-3xl font-bold text-gray-900 mb-4">Wallet</h1>
            <p className="text-gray-600 mb-4">
              Your wallet management center! This is where you'll find:
            </p>
            <ul className="text-left text-gray-600 space-y-2">
              <li>• Complete transaction history with filtering</li>
              <li>• Manual transaction creation</li>
              <li>• Balance trends and analytics</li>
              <li>• Transaction search and categorization</li>
            </ul>
            <p className="text-sm text-gray-500 mt-6">
              This page will be implemented in Task 12
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
