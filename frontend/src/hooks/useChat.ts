/**
 * Custom Hook: useChat
 * 
 * Manages the entire chat state and communication with the backend API.
 * This hook encapsulates all chat-related logic, providing a clean interface
 * for the UI components.
 * 
 * Features:
 * - Message state management
 * - API communication with retry logic
 * - Error handling
 * - Loading states
 * - Message history
 * 
 * @param {UseChatOptions} options - Configuration options
 * @returns {Object} Chat state and control functions
 * 
 * @example
 * const { messages, loading, sendMessage } = useChat({
 *   onError: (error) => console.error(error),
 *   onSuccess: (message) => console.log('Message sent:', message)
 * });
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import type { Message, ChatState } from '@/types/chat';
import { generateMessageId } from '@/utils/chatUtils';
import { apiService } from '@/services/api.service';
import { CHAT_CONFIG } from '@/config/chat.config';

interface UseChatOptions {
  apiUrl?: string;
  maxRetries?: number;
  retryDelay?: number;
  onError?: (error: Error) => void;
  onSuccess?: (message: Message) => void;
}

export const useChat = ({
  apiUrl,
  maxRetries = CHAT_CONFIG.api.retryAttempts,
  retryDelay = CHAT_CONFIG.api.retryDelay,
  onError,
  onSuccess
}: UseChatOptions = {}) => {
  const [state, setState] = useState<ChatState>({
    messages: [],
    loading: false,
    error: null,
    retryCount: 0
  });

  const abortControllerRef = useRef<AbortController | null>(null);

  // Persist messages to localStorage
  useEffect(() => {
    const savedMessages = localStorage.getItem(CHAT_CONFIG.storage.messageKey);
    if (savedMessages) {
      try {
        const parsed = JSON.parse(savedMessages);
        setState(prev => ({ ...prev, messages: parsed }));
      } catch (error) {
        console.error('Failed to load saved messages:', error);
      }
    }
  }, []);

  useEffect(() => {
    if (state.messages.length > 0) {
      localStorage.setItem(CHAT_CONFIG.storage.messageKey, JSON.stringify(state.messages));
    }
  }, [state.messages]);


  const sendMessage = useCallback(async (text: string): Promise<void> => {
    const trimmedText = text.trim();
    if (!trimmedText || state.loading) return;

    // Cancel any pending request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    // Create new abort controller
    abortControllerRef.current = new AbortController();

    const userMessage: Message = {
      id: generateMessageId(),
      role: 'user',
      text: trimmedText,
      timestamp: Date.now()
    };

    setState(prev => ({
      ...prev,
      messages: [...prev.messages, userMessage],
      loading: true,
      error: null
    }));

    try {
      const data = await apiService.sendMessage(
        trimmedText,
        abortControllerRef.current.signal
      );

      const assistantMessage: Message = {
        id: generateMessageId(),
        role: 'assistant',
        text: data.answer,
        sources: data.sources,
        timestamp: Date.now()
      };

      setState(prev => ({
        ...prev,
        messages: [...prev.messages, assistantMessage],
        loading: false,
        error: null,
        retryCount: 0
      }));

      onSuccess?.(assistantMessage);
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unbekannter Fehler';
      
      setState(prev => ({
        ...prev,
        loading: false,
        error: `Fehler: ${errorMessage}`,
        retryCount: prev.retryCount + 1
      }));

      onError?.(error as Error);

      // Auto-retry logic
      if (state.retryCount < maxRetries) {
        setTimeout(() => {
          sendMessage(trimmedText);
        }, retryDelay);
      } else {
        const errorAssistantMessage: Message = {
          id: generateMessageId(),
          role: 'assistant',
          text: `⚠️ Fehler: ${errorMessage}\n\nBitte überprüfen Sie Ihre Verbindung und versuchen Sie es später erneut.`,
          timestamp: Date.now(),
          error: true
        };

        setState(prev => ({
          ...prev,
          messages: [...prev.messages, errorAssistantMessage]
        }));
      }
    }
  }, [apiUrl, maxRetries, retryDelay, onError, onSuccess, state.loading, state.retryCount]);

  const clearMessages = useCallback(() => {
    setState(prev => ({
      ...prev,
      messages: [],
      error: null,
      retryCount: 0
    }));
    localStorage.removeItem(CHAT_CONFIG.storage.messageKey);
  }, []);

  const deleteMessage = useCallback((messageId: string) => {
    setState(prev => ({
      ...prev,
      messages: prev.messages.filter(msg => msg.id !== messageId)
    }));
  }, []);

  const retryLastMessage = useCallback(() => {
    const lastUserMessage = [...state.messages].reverse().find(msg => msg.role === 'user');
    if (lastUserMessage) {
      // Remove any error messages
      setState(prev => ({
        ...prev,
        messages: prev.messages.filter(msg => !msg.error)
      }));
      sendMessage(lastUserMessage.text);
    }
  }, [state.messages, sendMessage]);

  const cancelRequest = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setState(prev => ({
        ...prev,
        loading: false
      }));
    }
  }, []);

  return {
    messages: state.messages,
    loading: state.loading,
    error: state.error,
    sendMessage,
    clearMessages,
    deleteMessage,
    retryLastMessage,
    cancelRequest
  };
};