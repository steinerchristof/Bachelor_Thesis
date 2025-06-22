/**
 * Main Chat Interface Component
 * 
 * This is the core component of the Luota financial document analysis system.
 * It provides a conversational interface for users to interact with their financial documents.
 * 
 * Key Features:
 * - Real-time chat with AI backend
 * - PDF document viewer integration
 * - Swiss number formatting
 * - Math formula rendering (KaTeX)
 * - Code syntax highlighting
 * - Responsive design
 * 
 * Architecture Decisions:
 * - Uses custom hooks for separation of concerns
 * - Implements error boundaries for robustness
 * - Memoization for performance optimization
 * - Suspense for lazy loading
 * - Framer Motion for smooth animations
 * 
 * @module pages/chat
 */

import React, { Suspense, useCallback, useRef, useState, useEffect } from 'react';
import Image from 'next/image';
import { AnimatePresence, motion } from 'framer-motion';
import { AlertCircle } from 'lucide-react';
import type { SourceObject } from '@/types/chat';

// Custom hooks
import { useChatWithPersistence } from '@/hooks/useChatWithPersistence';
import { usePDFTabs } from '@/hooks/usePDFTabs';
import { useAutoScroll } from '@/hooks/useAutoScroll';

// Components
import {
  MessageBubble,
  TabbedPDFViewer,
  LoadingSkeleton,
  MessageSkeleton,
  ChatInput,
  ErrorBoundary,
  AsyncErrorBoundary,
  ConversationSidebar
} from '@/components/chat';

// Styles
import 'katex/dist/katex.min.css';
import 'highlight.js/styles/github.css';

// Constants
const EXAMPLE_MESSAGES = [
  'Ermittle den effektiven Steueraufwand der Jahresrechnungen 2022 und 2023 und gib sie mir in % an',
  'Jahresrechnung 2023 vs. 2022',
  'Jahresrechnung 2023: Berechne alle üblichen Kennzahlen und bewerte sie',
  'Wie hoch ist die Liquidität der Firma im Jahr 2023?',
  'Welche Bilanzkennzahlen (z. B. Eigenkapitalquote, Liquiditätsgrad) sind kritisch und wie kann ich diese verbessern? 2023 Jahresrechnung',
  'Wie haben sich flüssige Mittel von 2022 zu 2023 verändert?'
] as const;

// Luota Logo Component
interface LogoProps {
  size?: number;
  className?: string;
}

const LuotaLogo: React.FC<LogoProps> = ({ size = 24, className = "" }) => (
  <Image
    src="/luota-current.jpeg"
    alt="Luota Logo"
    width={size}
    height={size}
    className={`rounded-sm ${className}`}
    priority
    unoptimized={false}
  />
);

// Welcome Screen Component
const WelcomeScreen: React.FC<{ onExampleClick: (text: string) => void }> = ({ onExampleClick }) => {
  // Split messages into rows for better organization
  const messageRows = [
    EXAMPLE_MESSAGES.slice(0, 2),
    EXAMPLE_MESSAGES.slice(2, 4),
    EXAMPLE_MESSAGES.slice(4, 6)
  ];

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="flex flex-col items-center justify-center min-h-[calc(100vh-280px)] max-w-5xl mx-auto px-4"
    >
      {/* Logo and Title */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: "easeOut" }}
        className="mb-16 text-center"
      >
        <div className="relative inline-block mb-6">
          <div className="absolute inset-0 blur-xl bg-gradient-to-r from-blue-400/20 to-purple-400/20 rounded-full" />
          <LuotaLogo size={56} className="relative" />
        </div>
        <h1 className="text-4xl font-light text-gray-900 dark:text-gray-100 mb-3 tracking-tight">
          Wie kann ich Ihnen helfen?
        </h1>
        <p className="text-base text-gray-500 dark:text-gray-400 max-w-md mx-auto">
          Stellen Sie Fragen zu Ihren Finanzdokumenten
        </p>
      </motion.div>
      
      {/* Example Questions - Bento Grid Style */}
      <div className="w-full space-y-3">
        {messageRows.map((row, rowIndex) => (
          <div key={rowIndex} className="flex flex-col md:flex-row gap-3">
            {row.map((message, index) => (
              <motion.button
                key={message}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ 
                  duration: 0.4, 
                  delay: 0.1 + (rowIndex * 2 + index) * 0.05,
                  ease: "easeOut"
                }}
                onClick={() => onExampleClick(message)}
                className="group relative flex-1 p-5 rounded-2xl text-left overflow-hidden
                         bg-gradient-to-br from-gray-50 to-gray-100/50 dark:from-gray-800/50 dark:to-gray-900/30
                         border border-gray-200/60 dark:border-gray-700/40
                         hover:border-gray-300 dark:hover:border-gray-600
                         transition-all duration-300 hover:shadow-xl hover:-translate-y-0.5"
              >
                {/* Gradient Overlay on Hover */}
                <div className="absolute inset-0 bg-gradient-to-br from-blue-500/0 via-purple-500/0 to-pink-500/0 
                              group-hover:from-blue-500/10 group-hover:via-purple-500/10 group-hover:to-pink-500/10 
                              transition-all duration-500" />
                
                {/* Arrow Icon */}
                <div className="absolute top-5 right-5 w-6 h-6 rounded-full bg-gray-200/50 dark:bg-gray-700/50
                              group-hover:bg-white dark:group-hover:bg-gray-600
                              flex items-center justify-center transition-all duration-300
                              group-hover:scale-110 opacity-0 group-hover:opacity-100">
                  <svg className="w-3 h-3 text-gray-600 dark:text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </div>
                
                {/* Text Content */}
                <span className="relative block text-sm font-normal text-gray-700 dark:text-gray-200 
                               leading-relaxed pr-8 group-hover:text-gray-900 dark:group-hover:text-gray-50
                               transition-colors duration-300">
                  {message}
                </span>
                
                {/* Subtle Animation Line */}
                <div className="absolute bottom-0 left-0 h-0.5 bg-gradient-to-r from-blue-500 to-purple-500 
                              w-0 group-hover:w-full transition-all duration-500" />
              </motion.button>
            ))}
          </div>
        ))}
      </div>
      
      {/* CTA Section */}
      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6, delay: 0.6 }}
        className="mt-12 text-center"
      >
        <p className="text-sm text-gray-400 dark:text-gray-500">
          Wählen Sie ein Beispiel oder tippen Sie Ihre eigene Frage ein
        </p>
      </motion.div>
    </motion.div>
  );
};

// Main Chat Component
function ChatPage() {
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  
  // Custom hooks for state management
  const {
    messages,
    loading,
    error,
    sendMessage,
    clearMessages,
    deleteMessage,
    retryLastMessage,
    conversations,
    currentConversationId,
    switchConversation,
    newConversation,
    deleteConversation,
    renameConversation
  } = useChatWithPersistence({
    onError: (error) => {
      console.error('Chat error:', error);
    }
  });

  const {
    tabs: pdfTabs,
    activeTabId,
    openTab,
    closeTab,
    closeAllTabs,
    changeTab,
    updateTabError,
    updateTabLoading
  } = usePDFTabs({
    onTabOpen: (tab) => {
      console.log('PDF tab opened:', tab.title);
    }
  });

  const {
    endRef,
    shouldAutoScroll,
    scrollToBottom,
    forceScrollToBottom
  } = useAutoScroll();

  // Input state
  const [input, setInput] = useState('');

  // Handlers
  const handleSend = useCallback(async () => {
    if (!input.trim() || loading) return;
    
    await sendMessage(input);
    setInput('');
    scrollToBottom();
  }, [input, loading, sendMessage, scrollToBottom]);

  const handleSourceClick = useCallback((source: SourceObject) => {
    openTab(source.title);
  }, [openTab]);

  const handleExampleClick = useCallback((text: string) => {
    setInput(text);
    setTimeout(() => {
      inputRef.current?.focus();
    }, 50);
  }, []);

  // Wrap newConversation to also close PDFs
  const handleNewConversation = useCallback(() => {
    closeAllTabs(); // Close all PDFs
    newConversation(); // Create new conversation
  }, [newConversation, closeAllTabs]);

  // Wrap switchConversation to also close PDFs
  const handleSwitchConversation = useCallback((conversationId: string) => {
    closeAllTabs(); // Close all PDFs
    switchConversation(conversationId); // Switch conversation
  }, [switchConversation, closeAllTabs]);

  const handleTabError = useCallback((tabId: string, error: string) => {
    updateTabError(tabId, error);
  }, [updateTabError]);

  const handleTabLoad = useCallback((tabId: string) => {
    updateTabLoading(tabId, false);
  }, [updateTabLoading]);

  // Cleanup: Close all PDFs when leaving the page
  useEffect(() => {
    return () => {
      closeAllTabs();
    };
  }, [closeAllTabs]);

  // Render
  return (
    <ErrorBoundary>
      <AsyncErrorBoundary>
        <div className="flex h-screen">
          {/* Conversation Sidebar */}
          <ConversationSidebar
            conversations={conversations}
            currentConversationId={currentConversationId}
            onSelectConversation={handleSwitchConversation}
            onNewConversation={handleNewConversation}
            onDeleteConversation={deleteConversation}
            onRenameConversation={renameConversation}
            isOpen={sidebarOpen}
            onToggle={() => setSidebarOpen(!sidebarOpen)}
          />

          {/* Main Chat Area */}
          <div className={`flex-1 flex flex-col ${
            pdfTabs.length > 0 ? 'pr-0 md:pr-[55%] lg:pr-[50%] xl:pr-[45%]' : ''
          } transition-all duration-300`}>

          {/* Header */}
          <header className="sticky top-0 z-10 border-b py-4 backdrop-blur-sm bg-white/80 dark:bg-gray-900/80" style={{ borderColor: 'var(--color-border)' }}>
            <div className="mx-auto flex max-w-5xl items-center justify-between px-4">
              <h1 className="flex items-center gap-3">
                <LuotaLogo size={28} />
                <span className="text-base font-medium">Luota Assistent</span>
              </h1>
              
              <div className="flex items-center gap-3">
                {messages.length > 0 && (
                  <button
                    onClick={handleNewConversation}
                    className="text-sm opacity-60 hover:opacity-100 transition-opacity"
                    aria-label="Neue Unterhaltung"
                  >
                    Neu
                  </button>
                )}
              </div>
            </div>
          </header>

          {/* Main Content */}
          <main 
            className="flex-1 overflow-y-auto"
            role="main"
            aria-label="Chat messages"
          >
            <div className="mx-auto max-w-5xl px-4 py-6">
              {/* Welcome screen or messages */}
              {messages.length === 0 && !loading ? (
                <WelcomeScreen onExampleClick={handleExampleClick} />
              ) : (
                <Suspense fallback={<MessageSkeleton />}>
                  <AnimatePresence mode="popLayout">
                    {messages.map((message) => (
                      <MessageBubble
                        key={message.id}
                        message={message}
                        onSourceClick={handleSourceClick}
                        onRetry={retryLastMessage}
                        onDelete={deleteMessage}
                        pdfTabs={pdfTabs}
                        showActions
                        showTimestamp
                      />
                    ))}
                  </AnimatePresence>
                </Suspense>
              )}

              {/* Loading state */}
              {loading && (
                <LoadingSkeleton variant="thinking" />
              )}

              {/* Scroll anchor */}
              <div ref={endRef} aria-hidden="true" />
            </div>
          </main>

          {/* PDF Viewer */}
          <AnimatePresence mode="wait">
            {pdfTabs.length > 0 && (
              <TabbedPDFViewer
                pdfTabs={pdfTabs}
                activeTabId={activeTabId}
                onTabChange={changeTab}
                onTabClose={closeTab}
                onCloseAll={closeAllTabs}
                onTabError={handleTabError}
                onTabLoad={handleTabLoad}
              />
            )}
          </AnimatePresence>

          {/* Input Footer */}
          <footer 
            className="sticky bottom-0 border-t py-3 backdrop-blur-sm bg-white/95 dark:bg-gray-900/95"
            role="contentinfo"
            style={{ borderColor: 'var(--color-border)' }}
          >
            <div className="mx-auto max-w-2xl px-4">
              <ChatInput
                ref={inputRef}
                value={input}
                onChange={setInput}
                onSend={handleSend}
                loading={loading}
                disabled={false}
                placeholder="Stellen Sie eine Frage…"
                maxLength={4000}
                autoFocus
              />
              
              {error && (
                <p className="mt-1.5 text-center text-xs" role="alert" style={{ color: 'var(--color-accent)' }}>
                  <AlertCircle className="mr-1 inline h-3 w-3" aria-hidden="true" />
                  {error}
                </p>
              )}
            </div>
          </footer>
          </div>
        </div>
      </AsyncErrorBoundary>
    </ErrorBoundary>
  );
}

export default ChatPage;