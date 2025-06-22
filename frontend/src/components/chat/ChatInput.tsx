/**
 * ChatInput component for message input
 */

import React, { useRef, useEffect, useCallback, memo, useState } from 'react';
import { Send, Loader2, Paperclip, Mic, StopCircle } from 'lucide-react';
import { debounce } from '@/utils/chatUtils';

interface ChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  onKeyDown?: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  loading?: boolean;
  disabled?: boolean;
  placeholder?: string;
  maxLength?: number;
  autoFocus?: boolean;
  className?: string;
  onAttachment?: () => void;
  onVoiceInput?: () => void;
}

const ChatInputComponent = React.forwardRef<HTMLTextAreaElement, ChatInputProps>(({
  value,
  onChange,
  onSend,
  onKeyDown,
  loading = false,
  disabled = false,
  placeholder = "Nachricht senden…",
  maxLength = 4000,
  autoFocus = false,
  className = '',
  onAttachment,
  onVoiceInput
}, forwardedRef) => {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const ref = forwardedRef || textareaRef;
  const [isRecording, setIsRecording] = useState(false);
  const [charCount, setCharCount] = useState(0);

  // Auto-resize textarea
  const resizeTextarea = useCallback(() => {
    const textarea = typeof ref === 'object' && ref?.current;
    if (!textarea) return;
    
    requestAnimationFrame(() => {
      if (!textarea) return;
      
      const scrollPos = window.scrollY;
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
      window.scrollTo(0, scrollPos);
    });
  }, [ref]);

  // Debounced resize for performance
  const debouncedResize = useCallback(
    debounce(resizeTextarea, 100),
    [resizeTextarea]
  );

  useEffect(() => {
    resizeTextarea();
  }, [value, resizeTextarea]);

  useEffect(() => {
    const handleResize = () => debouncedResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [debouncedResize]);

  useEffect(() => {
    setCharCount(value.length);
  }, [value]);

  useEffect(() => {
    const textarea = typeof ref === 'object' && ref?.current;
    if (autoFocus && textarea) {
      textarea.focus();
    }
  }, [autoFocus, ref]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && !loading) {
      e.preventDefault();
      if (value.trim()) {
        onSend();
      }
    }
    
    if (e.key === 'Escape') {
      e.preventDefault();
      onChange('');
      const textarea = typeof ref === 'object' && ref?.current;
      textarea?.blur();
    }
    
    onKeyDown?.(e);
  }, [loading, value, onSend, onChange, onKeyDown]);

  const handleVoiceToggle = useCallback(() => {
    if (isRecording) {
      setIsRecording(false);
      // Stop recording logic here
    } else {
      setIsRecording(true);
      onVoiceInput?.();
    }
  }, [isRecording, onVoiceInput]);

  const isNearLimit = charCount > maxLength * 0.9;
  const isOverLimit = charCount > maxLength;

  return (
    <div className={`relative ${className}`}>
      <div className={`
        relative rounded-full border transition-all duration-200
        ${disabled 
          ? 'border-gray-200 bg-gray-50 dark:border-gray-700 dark:bg-gray-800/50' 
          : 'border-gray-200/70 bg-gray-50/80 dark:border-gray-700/50 dark:bg-gray-800/50'
        }
        ${isOverLimit && 'border-red-300 dark:border-red-600/70'}
      `}>
        <div className="flex items-end">
          {/* Attachment button */}
          {onAttachment && !loading && (
            <button
              onClick={onAttachment}
              disabled={disabled}
              className="p-2.5 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 disabled:opacity-50 transition-colors"
              aria-label="Datei anhängen"
              title="Datei anhängen"
            >
              <Paperclip className="h-4.5 w-4.5" />
            </button>
          )}

          {/* Textarea */}
          <textarea
            ref={ref}
            rows={1}
            value={value}
            onChange={(e) => {
              if (!disabled && e.target.value.length <= maxLength) {
                onChange(e.target.value);
              }
            }}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={disabled || loading}
            className={`
              flex-1 resize-none overflow-y-auto bg-transparent py-2.5 px-4 text-sm
              focus:outline-none focus:ring-0 disabled:opacity-50 disabled:cursor-not-allowed
              ${onAttachment ? 'pl-0' : ''}
              ${onVoiceInput ? 'pr-0' : ''}
            `}
            aria-label="Nachricht eingeben"
            aria-invalid={isOverLimit}
            aria-describedby={isNearLimit ? 'char-count' : undefined}
          />

          {/* Voice input button */}
          {onVoiceInput && !loading && !value && (
            <button
              onClick={handleVoiceToggle}
              disabled={disabled}
              className={`
                p-2.5 transition-colors mr-1
                ${isRecording 
                  ? 'text-red-500 hover:text-red-600 animate-pulse' 
                  : 'text-gray-400 hover:text-gray-600 dark:hover:text-gray-200'
                }
                disabled:opacity-50
              `}
              aria-label={isRecording ? "Aufnahme stoppen" : "Sprachaufnahme starten"}
              title={isRecording ? "Aufnahme stoppen" : "Sprachaufnahme starten"}
            >
              {isRecording ? (
                <StopCircle className="h-4.5 w-4.5" />
              ) : (
                <Mic className="h-4.5 w-4.5" />
              )}
            </button>
          )}

          {/* Send button */}
          <button
            onClick={onSend}
            disabled={!value.trim() || loading || disabled || isOverLimit}
            className={`
              p-2.5 mr-2.5 rounded-full text-gray-500 transition-all
              ${!value.trim() || loading || disabled || isOverLimit
                ? 'opacity-50 cursor-not-allowed' 
                : 'hover:bg-gray-100 hover:text-gray-700 dark:hover:bg-gray-700 dark:hover:text-gray-300'
              }
            `}
            aria-label="Nachricht senden"
            title={loading ? "Wird gesendet..." : "Nachricht senden (Enter)"}
          >
            {loading ? (
              <Loader2 className="h-4.5 w-4.5 animate-spin" />
            ) : (
              <Send className="h-4.5 w-4.5" />
            )}
          </button>
        </div>

        {/* Character count */}
        {isNearLimit && (
          <div 
            id="char-count"
            className={`
              absolute -top-6 right-2 text-xs
              ${isOverLimit ? 'text-red-500' : 'text-gray-500 dark:text-gray-400'}
            `}
            role="status"
            aria-live="polite"
          >
            {charCount} / {maxLength}
          </div>
        )}
      </div>

      {/* Recording indicator */}
      {isRecording && (
        <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 flex items-center gap-2 bg-red-50 dark:bg-red-900/20 px-3 py-1 rounded-full">
          <div className="h-2 w-2 bg-red-500 rounded-full animate-pulse" />
          <span className="text-xs text-red-600 dark:text-red-400">Aufnahme läuft...</span>
        </div>
      )}

      {/* Helper text - only show when over limit */}
      {isOverLimit && (
        <div className="mt-1 px-4 text-xs text-red-500">
          Nachricht zu lang ({charCount - maxLength} Zeichen zu viel)
        </div>
      )}
    </div>
  );
});

ChatInputComponent.displayName = 'ChatInput';

export const ChatInput = memo(ChatInputComponent);