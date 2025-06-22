/**
 * TypeScript Type Definitions
 * 
 * Central type definitions for the entire application.
 * These types ensure type safety across all components and functions.
 * 
 * Design Principles:
 * - Single source of truth for types
 * - Clear, descriptive interface names
 * - Optional properties marked explicitly
 * - No use of 'any' type
 * - Proper union types for enums
 * 
 * Type Categories:
 * - Message types: Communication between user and AI
 * - API types: Backend communication contracts
 * - UI types: Component-specific types
 * - State types: Application state management
 * 
 * @module types/chat
 */

export interface SourceObject {
  title: string;
  metadata?: Record<string, any>;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  sources?: SourceObject[];
  timestamp: number;
  error?: boolean;
  retryCount?: number;
}

export interface ApiResponse {
  answer: string;
  sources?: SourceObject[];
  error?: string;
  status?: number;
}

export interface PDFTab {
  id: string;
  title: string;
  url: string;
  loading: boolean;
  error: string | null;
  lastAccessed?: number;
  pageNumber?: number;
}

export interface ChatState {
  messages: Message[];
  loading: boolean;
  error: string | null;
  retryCount: number;
}

export interface ChatContextType extends ChatState {
  sendMessage: (text: string) => Promise<void>;
  clearMessages: () => void;
  deleteMessage: (id: string) => void;
  retryMessage: (id: string) => void;
}

export interface SourceDocument {
  title: string;
  id: string;
  relevanceScore?: number;
  excerpt?: string;
}

export interface ChatPreferences {
  autoScroll: boolean;
  enableMarkdown: boolean;
  enableMath: boolean;
  theme: 'light' | 'dark' | 'system';
}

export interface ApiError extends Error {
  status?: number;
  code?: string;
  details?: unknown;
}

export type MessageStatus = 'sending' | 'sent' | 'failed' | 'retrying';

export interface ExtendedMessage extends Message {
  status?: MessageStatus;
  editedAt?: number;
  reactions?: string[];
}

export interface ChatMetrics {
  totalMessages: number;
  sessionDuration: number;
  averageResponseTime: number;
  errorRate: number;
}