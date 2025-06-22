/**
 * TabbedPDFViewer component for displaying PDF documents in tabs
 */

import React, { useRef, useEffect, memo, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Loader2, AlertCircle, ChevronLeft, ChevronRight, Maximize2, Minimize2 } from 'lucide-react';

import type { PDFTab } from '@/types/chat';

interface TabbedPDFViewerProps {
  pdfTabs: PDFTab[];
  activeTabId: string;
  onTabChange: (tabId: string) => void;
  onTabClose: (tabId: string) => void;
  onCloseAll: () => void;
  onTabError?: (tabId: string, error: string) => void;
  onTabLoad?: (tabId: string) => void;
}

export const TabbedPDFViewer = memo<TabbedPDFViewerProps>(({
  pdfTabs,
  activeTabId,
  onTabChange,
  onTabClose,
  onCloseAll,
  onTabError,
  onTabLoad
}) => {
  const objectRefs = useRef<{ [key: string]: HTMLObjectElement | null }>({});
  const [isFullscreen, setIsFullscreen] = React.useState(false);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Escape to close all tabs
      if (e.key === 'Escape' && !isFullscreen) {
        e.preventDefault();
        onCloseAll();
      }
      
      // F11 or Cmd/Ctrl+Shift+F for fullscreen toggle
      if (e.key === 'F11' || ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'f')) {
        e.preventDefault();
        setIsFullscreen(prev => !prev);
      }
      
      // Tab switching with Ctrl/Cmd + number
      if ((e.ctrlKey || e.metaKey) && e.key >= '1' && e.key <= '9') {
        e.preventDefault();
        const tabIndex = parseInt(e.key) - 1;
        if (pdfTabs[tabIndex]) {
          onTabChange(pdfTabs[tabIndex].id);
        }
      }
      
      // Next/Previous tab with Ctrl/Cmd + Tab
      if ((e.ctrlKey || e.metaKey) && e.key === 'Tab') {
        e.preventDefault();
        const currentIndex = pdfTabs.findIndex(tab => tab.id === activeTabId);
        if (currentIndex !== -1) {
          const nextIndex = e.shiftKey 
            ? (currentIndex - 1 + pdfTabs.length) % pdfTabs.length
            : (currentIndex + 1) % pdfTabs.length;
          onTabChange(pdfTabs[nextIndex].id);
        }
      }
      
      // Close current tab with Ctrl/Cmd + W
      if ((e.ctrlKey || e.metaKey) && e.key === 'w') {
        e.preventDefault();
        if (activeTabId) {
          onTabClose(activeTabId);
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [pdfTabs, activeTabId, isFullscreen, onTabChange, onCloseAll, onTabClose]);

  const handleIframeLoad = useCallback((tabId: string) => {
    onTabLoad?.(tabId);
  }, [onTabLoad]);

  const handleIframeError = useCallback((tabId: string) => {
    onTabError?.(tabId, 'Dokument konnte nicht geladen werden');
  }, [onTabError]);

  const activeTab = pdfTabs.find(tab => tab.id === activeTabId);

  if (pdfTabs.length === 0) return null;

  return (
    <motion.div
      initial={{ opacity: 0, x: '100%' }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: '100%' }}
      transition={{ duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] }}
      className={`
        fixed top-0 right-0 h-screen bg-white dark:bg-gray-800 shadow-2xl flex flex-col 
        border-l border-gray-200 dark:border-gray-700 
        ${isFullscreen ? 'w-full' : 'w-full md:w-[55%] lg:w-[50%] xl:w-[45%]'}
      `}
      style={{ zIndex: 9999 }}
      role="complementary"
      aria-label="PDF-Dokumentenansicht"
    >
      {/* Tab Bar */}
      <div className="flex flex-col border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
        {/* Tab Navigation */}
        <div className="flex items-center justify-between px-2 py-1">
          <div className="flex items-center flex-1 min-w-0 overflow-x-auto scrollbar-hide">
            {pdfTabs.map((tab, index) => (
              <motion.button
                key={tab.id}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                transition={{ duration: 0.2, delay: index * 0.05 }}
                onClick={() => onTabChange(tab.id)}
                className={`
                  group relative flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium 
                  transition-all duration-200 min-w-0 max-w-[200px] whitespace-nowrap
                  ${tab.id === activeTabId 
                    ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 shadow-sm ring-1 ring-blue-500 dark:ring-blue-400' 
                    : 'text-gray-600 dark:text-gray-400 hover:bg-white/60 dark:hover:bg-gray-700/60 hover:text-gray-900 dark:hover:text-gray-200'
                  }
                `}
                title={tab.title}
                aria-label={`Tab ${index + 1}: ${tab.title}`}
                aria-selected={tab.id === activeTabId}
                role="tab"
              >
                {/* Tab title */}
                <span className="truncate flex-1 text-left">
                  {tab.title}
                </span>
                
                {/* Tab number badge */}
                {pdfTabs.length > 1 && (
                  <span 
                    className="flex-shrink-0 w-4 h-4 rounded bg-gray-200 dark:bg-gray-600 text-gray-600 dark:text-gray-300 text-[10px] font-medium flex items-center justify-center"
                    aria-hidden="true"
                  >
                    {index + 1}
                  </span>
                )}
                
                {/* Close button */}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onTabClose(tab.id);
                  }}
                  className="flex-shrink-0 p-0.5 rounded hover:bg-gray-200 dark:hover:bg-gray-600 opacity-0 group-hover:opacity-100 transition-opacity"
                  title={`Tab schließen (${tab.title})`}
                  aria-label={`Tab ${tab.title} schließen`}
                >
                  <X className="h-3 w-3" />
                </button>
              </motion.button>
            ))}
          </div>
          
          {/* Global controls */}
          <div className="flex items-center gap-1 ml-2">
            {/* Fullscreen toggle */}
            <button
              onClick={() => setIsFullscreen(prev => !prev)}
              className="p-1.5 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
              title={isFullscreen ? "Vollbild beenden (F11)" : "Vollbild (F11)"}
              aria-label={isFullscreen ? "Vollbild beenden" : "Vollbild"}
            >
              {isFullscreen ? (
                <Minimize2 className="h-4 w-4" />
              ) : (
                <Maximize2 className="h-4 w-4" />
              )}
            </button>
            
            {/* Close all tabs */}
            <button
              onClick={onCloseAll}
              className="p-1.5 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
              title="Alle Tabs schließen (ESC)"
              aria-label="Alle Tabs schließen"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>
        
        {/* Active tab info bar */}
        {activeTab && (
          <div className="px-3 py-1.5 text-xs text-gray-500 dark:text-gray-400 border-t border-gray-200/50 dark:border-gray-700/50">
            <div className="flex items-center justify-between">
              <span className="truncate">
                {activeTab.loading ? 'Wird geladen...' : 
                 activeTab.error ? 'Fehler beim Laden' : 
                 `Dokument ${pdfTabs.findIndex(t => t.id === activeTab.id) + 1} von ${pdfTabs.length}`}
              </span>
              <div className="flex items-center gap-3 text-[10px] opacity-60">
                <span>⌘{pdfTabs.findIndex(t => t.id === activeTab.id) + 1}</span>
                <span>⌘W schließen</span>
                <span>⌘Tab wechseln</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* PDF Content Area */}
      <div className="flex-1 relative bg-gray-50 dark:bg-gray-950">
        <AnimatePresence mode="wait">
          {pdfTabs.map((tab) => (
            <motion.div
              key={tab.id}
              initial={{ opacity: 0 }}
              animate={{ opacity: tab.id === activeTabId ? 1 : 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className={`absolute inset-0 ${
                tab.id === activeTabId ? 'z-10' : 'z-0 pointer-events-none'
              }`}
            >
              {/* Loading state */}
              {tab.loading && (
                <div className="absolute inset-0 flex items-center justify-center bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm z-20">
                  <div className="text-center">
                    <Loader2 className="h-8 w-8 animate-spin text-blue-500 mx-auto mb-3" />
                    <p className="text-sm text-gray-600 dark:text-gray-400">Dokument wird geladen...</p>
                    <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">{tab.title}</p>
                  </div>
                </div>
              )}

              {/* Error state */}
              {tab.error && !tab.loading && (
                <div className="absolute inset-0 flex items-center justify-center p-6 z-20">
                  <div className="text-center max-w-md">
                    <AlertCircle className="h-12 w-12 text-red-400 mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
                      Dokument nicht verfügbar
                    </h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                      {tab.error}
                    </p>
                    <div className="flex gap-2 justify-center">
                      <button
                        onClick={() => {
                          // Retry loading
                          if (iframeRefs.current[tab.id]) {
                            iframeRefs.current[tab.id]!.src = tab.url;
                          }
                        }}
                        className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm"
                      >
                        Erneut versuchen
                      </button>
                      <button
                        onClick={() => onTabClose(tab.id)}
                        className="px-4 py-2 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors text-sm"
                      >
                        Tab schließen
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {/* PDF viewer - Using object tag for better browser compatibility */}
              <object
                ref={(el) => { objectRefs.current[tab.id] = el; }}
                data={tab.url}
                type="application/pdf"
                className="w-full h-full"
                aria-label={`PDF: ${tab.title}`}
                onLoad={() => handleIframeLoad(tab.id)}
                onError={() => handleIframeError(tab.id)}
              >
                <embed
                  src={tab.url}
                  type="application/pdf"
                  className="w-full h-full"
                />
                <p className="p-4 text-center text-gray-600 dark:text-gray-400">
                  PDF kann nicht angezeigt werden. 
                  <a 
                    href={tab.url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="text-blue-500 hover:underline ml-2"
                  >
                    In neuem Tab öffnen
                  </a>
                </p>
              </object>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {/* Navigation hints for mobile */}
      {pdfTabs.length > 1 && (
        <div className="md:hidden absolute bottom-4 left-1/2 transform -translate-x-1/2 flex gap-2">
          <button
            onClick={() => {
              const currentIndex = pdfTabs.findIndex(tab => tab.id === activeTabId);
              const prevIndex = (currentIndex - 1 + pdfTabs.length) % pdfTabs.length;
              onTabChange(pdfTabs[prevIndex].id);
            }}
            className="p-2 bg-white dark:bg-gray-800 rounded-full shadow-lg"
            aria-label="Vorheriger Tab"
          >
            <ChevronLeft className="h-5 w-5" />
          </button>
          <button
            onClick={() => {
              const currentIndex = pdfTabs.findIndex(tab => tab.id === activeTabId);
              const nextIndex = (currentIndex + 1) % pdfTabs.length;
              onTabChange(pdfTabs[nextIndex].id);
            }}
            className="p-2 bg-white dark:bg-gray-800 rounded-full shadow-lg"
            aria-label="Nächster Tab"
          >
            <ChevronRight className="h-5 w-5" />
          </button>
        </div>
      )}
    </motion.div>
  );
});

TabbedPDFViewer.displayName = 'TabbedPDFViewer';