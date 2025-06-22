/**
 * LoadingSkeleton component for chat messages
 */

import React, { memo } from 'react';
import { motion } from 'framer-motion';
import Image from 'next/image';

interface LoadingSkeletonProps {
  variant?: 'typing' | 'thinking' | 'processing';
  className?: string;
}

export const LoadingSkeleton = memo<LoadingSkeletonProps>(({ 
  variant = 'typing',
  className = '' 
}) => {
  const renderLoadingContent = () => {
    switch (variant) {
      case 'typing':
        return (
          <div className="flex items-center space-x-1">
            <div className="h-2 w-2 animate-pulse rounded-full bg-gray-400 dark:bg-gray-500" />
            <div 
              className="h-2 w-2 animate-pulse rounded-full bg-gray-400 dark:bg-gray-500" 
              style={{ animationDelay: '0.2s' }} 
            />
            <div 
              className="h-2 w-2 animate-pulse rounded-full bg-gray-400 dark:bg-gray-500" 
              style={{ animationDelay: '0.4s' }} 
            />
          </div>
        );
      
      case 'thinking':
        return (
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <div className="h-3 w-3 animate-spin rounded-full border-2 border-gray-300 border-t-gray-600 dark:border-gray-600 dark:border-t-gray-300" />
              <span className="text-sm text-gray-500 dark:text-gray-400">
                Analysiere Ihre Anfrage...
              </span>
            </div>
            <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded animate-pulse w-3/4" />
            <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded animate-pulse w-1/2" />
          </div>
        );
      
      case 'processing':
        return (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <div className="h-4 w-4 animate-pulse">
                <svg className="animate-spin" viewBox="0 0 24 24" fill="none">
                  <circle 
                    className="opacity-25" 
                    cx="12" 
                    cy="12" 
                    r="10" 
                    stroke="currentColor" 
                    strokeWidth="4"
                  />
                  <path 
                    className="opacity-75" 
                    fill="currentColor" 
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  />
                </svg>
              </div>
              <span className="text-sm text-gray-500 dark:text-gray-400">
                Suche relevante Dokumente...
              </span>
            </div>
            <div className="space-y-2">
              <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded animate-pulse w-full" />
              <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded animate-pulse w-5/6" />
              <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded animate-pulse w-4/6" />
            </div>
          </div>
        );
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.3 }}
      className={`flex ${className}`}
    >
      <div className="mr-3 flex-shrink-0">
        <Image
          src="/luota-current.jpeg"
          alt="Luota Logo"
          width={32}
          height={32}
          className="rounded-sm"
          priority
        />
      </div>
      <div className="rounded-2xl rounded-tl-sm bg-white dark:bg-gray-800 px-5 py-4 shadow-sm min-w-[200px]">
        {renderLoadingContent()}
      </div>
    </motion.div>
  );
});

LoadingSkeleton.displayName = 'LoadingSkeleton';

/**
 * MessageSkeleton component for placeholder messages
 */
interface MessageSkeletonProps {
  role?: 'user' | 'assistant';
  className?: string;
}

export const MessageSkeleton = memo<MessageSkeletonProps>(({ 
  role = 'assistant',
  className = '' 
}) => {
  const isUser = role === 'user';

  return (
    <div className={`flex mb-8 ${isUser ? 'justify-end' : 'justify-start'} ${className}`}>
      {isUser ? (
        <div className="flex items-start gap-3 max-w-[92%]">
          <div className="bg-gray-100 dark:bg-gray-700 rounded-2xl rounded-tr-sm px-5 py-4 shadow-sm">
            <div className="space-y-2 animate-pulse">
              <div className="h-3 bg-gray-300 dark:bg-gray-600 rounded w-48" />
              <div className="h-3 bg-gray-300 dark:bg-gray-600 rounded w-36" />
            </div>
          </div>
          <div className="flex-shrink-0 mt-1">
            <div className="w-8 h-8 rounded-full bg-gray-200 dark:bg-gray-700 animate-pulse" />
          </div>
        </div>
      ) : (
        <div className="flex items-start gap-3 max-w-[92%] w-full">
          <div className="flex-shrink-0 mt-1">
            <div className="w-8 h-8 rounded-sm bg-gray-200 dark:bg-gray-700 animate-pulse" />
          </div>
          <div className="w-full bg-white dark:bg-gray-800 rounded-2xl rounded-tl-sm px-5 py-4 shadow-sm">
            <div className="space-y-3 animate-pulse">
              <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-full" />
              <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-5/6" />
              <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-4/6" />
              <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded w-20 mb-2" />
                <div className="space-y-1">
                  <div className="h-8 bg-gray-100 dark:bg-gray-800 rounded" />
                  <div className="h-8 bg-gray-100 dark:bg-gray-800 rounded" />
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
});

MessageSkeleton.displayName = 'MessageSkeleton';