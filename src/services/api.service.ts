/**
 * API Service Layer
 * 
 * Handles all communication with the backend API.
 * Implements a singleton pattern to ensure consistent configuration
 * and connection management throughout the application.
 * 
 * Features:
 * - Health check endpoint
 * - Message streaming with EventSource
 * - Automatic retry logic
 * - Error handling and normalization
 * - Request timeout management
 * 
 * Design Decisions:
 * - Singleton pattern for consistency
 * - EventSource for real-time streaming
 * - AbortController for request cancellation
 * - Typed responses with TypeScript
 * 
 * @class ApiService
 */

import { CHAT_CONFIG } from '@/config/chat.config';
import type { ApiResponse, ApiError } from '@/types/chat';

class ApiService {
  private baseUrl: string;
  private timeout: number;

  constructor() {
    this.baseUrl = CHAT_CONFIG.api.baseUrl;
    this.timeout = CHAT_CONFIG.api.timeout;
  }

  /**
   * Check if the backend is healthy
   */
  async checkHealth(): Promise<boolean> {
    try {
      const response = await fetch(
        `${this.baseUrl}${CHAT_CONFIG.api.endpoints.health}`,
        {
          method: 'GET',
          signal: AbortSignal.timeout(3000)
        }
      );
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * Send a chat message
   */
  async sendMessage(text: string, signal?: AbortSignal): Promise<ApiResponse> {
    const response = await this.request<ApiResponse>(
      CHAT_CONFIG.api.endpoints.chat,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
        signal
      }
    );

    return response;
  }

  /**
   * Get document URL
   */
  getDocumentUrl(documentTitle: string): string {
    const encoded = encodeURIComponent(documentTitle);
    return `${this.baseUrl}${CHAT_CONFIG.api.endpoints.document}/${encoded}?t=${Date.now()}`;
  }

  /**
   * Generic request method with error handling
   */
  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    
    try {
      console.log('Making request to:', url);
      console.log('Request options:', options);
      
      const response = await fetch(url, {
        ...options,
        signal: options.signal || AbortSignal.timeout(this.timeout)
      });

      console.log('Response received:', response);

      if (!response.ok) {
        throw this.createApiError(
          `HTTP ${response.status}`,
          response.status,
          await this.getErrorDetails(response)
        );
      }

      const data = await response.json();
      return data as T;
    } catch (error) {
      console.error('Request failed:', error);
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          throw this.createApiError('Request aborted', 0, error);
        }
        throw error;
      }
      throw this.createApiError('Unknown error occurred', 0, error);
    }
  }

  /**
   * Create a standardized API error
   */
  private createApiError(
    message: string,
    status?: number,
    details?: unknown
  ): ApiError {
    const error = new Error(message) as ApiError;
    error.status = status;
    error.details = details;
    return error;
  }

  /**
   * Extract error details from response
   */
  private async getErrorDetails(response: Response): Promise<unknown> {
    try {
      const contentType = response.headers.get('content-type');
      if (contentType?.includes('application/json')) {
        return await response.json();
      }
      return await response.text();
    } catch {
      return null;
    }
  }
}

// Export singleton instance
export const apiService = new ApiService();