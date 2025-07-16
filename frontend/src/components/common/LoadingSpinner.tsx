import React from "react";

interface LoadingSpinnerProps {
  size?: "small" | "medium" | "large";
  text?: string;
  className?: string;
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = "medium",
  text,
  className = "",
}) => {
  const sizeClasses = {
    small: "h-4 w-4",
    medium: "h-8 w-8",
    large: "h-12 w-12",
  };

  const textSizeClasses = {
    small: "text-sm",
    medium: "text-base",
    large: "text-lg",
  };

  return (
    <div className={`flex flex-col items-center justify-center ${className}`}>
      <div
        className={`animate-spin rounded-full border-2 border-gray-300 border-t-black ${sizeClasses[size]}`}
        role="status"
        aria-label="Loading"
      />
      {text && (
        <p className={`mt-2 text-gray-600 ${textSizeClasses[size]}`}>{text}</p>
      )}
    </div>
  );
};

export default LoadingSpinner;

// Inline spinner for buttons and small spaces
export const InlineSpinner: React.FC<{ className?: string }> = ({
  className = "",
}) => (
  <div
    className={`inline-block h-4 w-4 animate-spin rounded-full border-2 border-gray-300 border-t-current ${className}`}
    role="status"
    aria-label="Loading"
  />
);

// Full page loading overlay
export const PageLoader: React.FC<{ text?: string }> = ({
  text = "Loading...",
}) => (
  <div className="fixed inset-0 z-50 flex items-center justify-center bg-white bg-opacity-75">
    <LoadingSpinner size="large" text={text} />
  </div>
);

// Card/section loading skeleton
export const SkeletonLoader: React.FC<{
  lines?: number;
  className?: string;
}> = ({ lines = 3, className = "" }) => (
  <div className={`animate-pulse ${className}`}>
    {Array.from({ length: lines }).map((_, index) => (
      <div
        key={index}
        className={`mb-2 h-4 rounded bg-gray-200 ${
          index === lines - 1 ? "w-3/4" : "w-full"
        }`}
      />
    ))}
  </div>
);

// Button loading state
export const ButtonSpinner: React.FC = () => <InlineSpinner className="mr-2" />;
