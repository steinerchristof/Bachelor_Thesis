/**
 * MessageBubble component for displaying chat messages
 */

import React, { useState, memo, useCallback } from 'react';
import Image from 'next/image';
import { motion } from 'framer-motion';
import { Copy, CheckCheck, RotateCw, Trash2, User } from 'lucide-react';

import type { Message, PDFTab, SourceObject } from '@/types/chat';
import { formatTime, formatRelativeTime, copyToClipboard } from '@/utils/chatUtils';
import { BlackWhiteMessageContent } from './BlackWhiteMessageContent';

interface MessageBubbleProps {
  message: Message;
  onSourceClick?: (source: SourceObject) => void;
  onRetry?: (messageId: string) => void;
  onDelete?: (messageId: string) => void;
  pdfTabs?: PDFTab[];
  showActions?: boolean;
  showTimestamp?: boolean;
  className?: string;
}

export const MessageBubble = memo<MessageBubbleProps>(({
  message,
  onSourceClick,
  onRetry,
  onDelete,
  pdfTabs = [],
  showActions = true,
  showTimestamp = true,
  className = ''
}) => {
  const [copied, setCopied] = useState(false);
  const [showRelativeTime, setShowRelativeTime] = useState(false);

  const handleCopy = useCallback(async () => {
    const success = await copyToClipboard(message.text);
    if (success) {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }, [message.text]);

  const toggleTimeFormat = useCallback(() => {
    setShowRelativeTime(prev => !prev);
  }, []);

  const isUser = message.role === 'user';
  const hasError = message.error;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.3 }}
      className={`flex mb-4 ${isUser ? 'justify-end' : 'justify-start'} ${className}`}
    >
      {isUser ? (
        <div className="flex items-start gap-3 max-w-[92%]">
          <div className={`
            w-fit rounded-2xl px-4 py-3 max-w-[70vw]
            ${hasError 
              ? 'bg-red-50 dark:bg-red-900/20 text-red-800 dark:text-red-200' 
              : 'bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-gray-100'
            }
          `}>
            <BlackWhiteMessageContent 
              content={message.text} 
              sources={message.sources}
              isUserMessage={true}
            />
            
            {showTimestamp && (
              <button
                onClick={toggleTimeFormat}
                className="mt-2 text-right text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
                aria-label="Toggle time format"
              >
                {showRelativeTime 
                  ? formatRelativeTime(message.timestamp)
                  : formatTime(message.timestamp)
                }
              </button>
            )}
            
            {showActions && (
              <div className="flex items-center gap-1 mt-2 justify-end">
                <button
                  onClick={handleCopy}
                  className="p-1.5 rounded hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                  aria-label="Copy message"
                  title="Nachricht kopieren"
                >
                  {copied ? (
                    <CheckCheck className="h-3.5 w-3.5 text-green-600" />
                  ) : (
                    <Copy className="h-3.5 w-3.5 text-gray-500" />
                  )}
                </button>
                
                {onDelete && (
                  <button
                    onClick={() => onDelete(message.id)}
                    className="p-1.5 rounded hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                    aria-label="Delete message"
                    title="Nachricht löschen"
                  >
                    <Trash2 className="h-3.5 w-3.5 text-gray-500" />
                  </button>
                )}
              </div>
            )}
          </div>
          
          <div className="flex-shrink-0">
            <div className="w-8 h-8 rounded-full bg-gray-300 dark:bg-gray-600 flex items-center justify-center">
              <User className="h-4 w-4 text-gray-700 dark:text-gray-200" />
            </div>
          </div>
        </div>
      ) : (
        <div className="flex items-start gap-3 max-w-[92%] w-full">
          <div className="flex-shrink-0 mt-1">
            <Image
              src="/luota-current.jpeg"
              alt="Luota Logo"
              width={32}
              height={32}
              className="rounded-sm"
              priority
            />
          </div>
          
          <div className={`
            w-full rounded-2xl px-4 py-3
            ${hasError 
              ? 'bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800' 
              : 'bg-white dark:bg-gray-800 border border-gray-100 dark:border-gray-700 shadow-sm'
            }
          `}>
            <BlackWhiteMessageContent 
              content={message.text} 
              sources={message.sources}
              onSourceClick={onSourceClick}
              pdfTabs={pdfTabs}
              isUserMessage={false}
            />
            
            <div className="flex items-center justify-between mt-2">
              {showTimestamp && (
                <button
                  onClick={toggleTimeFormat}
                  className="text-xs text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
                  aria-label="Toggle time format"
                >
                  {showRelativeTime 
                    ? formatRelativeTime(message.timestamp)
                    : formatTime(message.timestamp)
                  }
                </button>
              )}
              
              {showActions && (
                <div className="flex items-center gap-1">
                  <button
                    onClick={handleCopy}
                    className="p-1.5 rounded hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                    aria-label="Copy message"
                    title="Nachricht kopieren"
                  >
                    {copied ? (
                      <CheckCheck className="h-3.5 w-3.5 text-green-600" />
                    ) : (
                      <Copy className="h-3.5 w-3.5 text-gray-500" />
                    )}
                  </button>
                  
                  {hasError && onRetry && (
                    <button
                      onClick={() => onRetry(message.id)}
                      className="p-1.5 rounded hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                      aria-label="Retry message"
                      title="Erneut versuchen"
                    >
                      <RotateCw className="h-3.5 w-3.5 text-gray-500" />
                    </button>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </motion.div>
  );
});

MessageBubble.displayName = 'MessageBubble';