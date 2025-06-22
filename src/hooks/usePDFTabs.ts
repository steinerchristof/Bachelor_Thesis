/**
 * Custom hook for managing PDF tabs
 */

import { useState, useCallback, useEffect } from 'react';
import type { PDFTab } from '@/types/chat';
import { generateTabId, extractDocumentTitle } from '@/utils/chatUtils';

interface UsePDFTabsOptions {
  maxTabs?: number;
  apiUrl?: string;
  onTabOpen?: (tab: PDFTab) => void;
  onTabClose?: (tabId: string) => void;
  onTabChange?: (tabId: string) => void;
}

export const usePDFTabs = ({
  maxTabs = 5,
  apiUrl = '/api',
  onTabOpen,
  onTabClose,
  onTabChange
}: UsePDFTabsOptions = {}) => {
  const [tabs, setTabs] = useState<PDFTab[]>([]);
  const [activeTabId, setActiveTabId] = useState<string>('');

  // Persist tabs to sessionStorage
  useEffect(() => {
    // Clear PDF tabs on page load to ensure fresh start
    sessionStorage.removeItem('pdf_tabs');
    sessionStorage.removeItem('pdf_active_tab');
  }, []);

  useEffect(() => {
    if (tabs.length > 0) {
      sessionStorage.setItem('pdf_tabs', JSON.stringify(tabs));
      sessionStorage.setItem('pdf_active_tab', activeTabId);
    } else {
      sessionStorage.removeItem('pdf_tabs');
      sessionStorage.removeItem('pdf_active_tab');
    }
  }, [tabs, activeTabId]);

  const openTab = useCallback((source: string) => {
    const documentTitle = extractDocumentTitle(source);
    
    // Check if tab already exists
    const existingTab = tabs.find(tab => tab.title === documentTitle);
    
    if (existingTab) {
      // If already open, close it (toggle functionality)
      setTabs(prevTabs => {
        const newTabs = prevTabs.filter(tab => tab.id !== existingTab.id);
        
        // Handle active tab closure
        if (existingTab.id === activeTabId && newTabs.length > 0) {
          const closedTabIndex = prevTabs.findIndex(tab => tab.id === existingTab.id);
          const nextTabIndex = Math.min(closedTabIndex, newTabs.length - 1);
          setActiveTabId(newTabs[nextTabIndex].id);
        } else if (newTabs.length === 0) {
          setActiveTabId('');
        }
        
        return newTabs;
      });
      
      onTabClose?.(existingTab.id);
      return null;
    }
    
    // Create new tab
    const newTabId = generateTabId();
    const searchUrl = `${apiUrl}/document/${encodeURIComponent(documentTitle)}?t=${Date.now()}`;
    
    const newTab: PDFTab = {
      id: newTabId,
      title: documentTitle,
      url: searchUrl,
      loading: true,
      error: null,
      lastAccessed: Date.now()
    };
    
    setTabs(prevTabs => {
      // Enforce max tabs limit
      if (prevTabs.length >= maxTabs) {
        // Remove the oldest tab (by lastAccessed)
        const sortedTabs = [...prevTabs].sort((a, b) => 
          (a.lastAccessed || 0) - (b.lastAccessed || 0)
        );
        return [...sortedTabs.slice(1), newTab];
      }
      return [...prevTabs, newTab];
    });
    
    setActiveTabId(newTabId);
    onTabOpen?.(newTab);
    
    // Fetch the document info to get the actual filename
    fetch(`${apiUrl}/document/${encodeURIComponent(documentTitle)}/info`)
      .then(response => response.json())
      .then(data => {
        if (data.found && data.title) {
          setTabs(prev => prev.map(tab => 
            tab.id === newTabId ? { 
              ...tab, 
              loading: false,
              // Update title with actual filename from the backend
              title: data.title
            } : tab
          ));
        } else {
          setTabs(prev => prev.map(tab => 
            tab.id === newTabId ? { ...tab, loading: false } : tab
          ));
        }
      })
      .catch(() => {
        // Fallback to original behavior if fetch fails
        setTabs(prev => prev.map(tab => 
          tab.id === newTabId ? { ...tab, loading: false } : tab
        ));
      });
    
    return newTabId;
  }, [tabs, maxTabs, apiUrl, onTabOpen, onTabChange, onTabClose, activeTabId]);

  const closeTab = useCallback((tabId: string) => {
    setTabs(prevTabs => {
      const newTabs = prevTabs.filter(tab => tab.id !== tabId);
      
      // Handle active tab closure
      if (tabId === activeTabId && newTabs.length > 0) {
        const closedTabIndex = prevTabs.findIndex(tab => tab.id === tabId);
        const nextTabIndex = Math.min(closedTabIndex, newTabs.length - 1);
        setActiveTabId(newTabs[nextTabIndex].id);
      } else if (newTabs.length === 0) {
        setActiveTabId('');
      }
      
      return newTabs;
    });
    
    onTabClose?.(tabId);
  }, [activeTabId, onTabClose]);

  const closeAllTabs = useCallback(() => {
    setTabs([]);
    setActiveTabId('');
  }, []);

  const changeTab = useCallback((tabId: string) => {
    if (tabs.some(tab => tab.id === tabId)) {
      setActiveTabId(tabId);
      setTabs(prev => prev.map(tab => 
        tab.id === tabId 
          ? { ...tab, lastAccessed: Date.now() }
          : tab
      ));
      onTabChange?.(tabId);
    }
  }, [tabs, onTabChange]);

  const updateTabError = useCallback((tabId: string, error: string | null) => {
    setTabs(prev => prev.map(tab => 
      tab.id === tabId ? { ...tab, error, loading: false } : tab
    ));
  }, []);

  const updateTabLoading = useCallback((tabId: string, loading: boolean) => {
    setTabs(prev => prev.map(tab => 
      tab.id === tabId ? { ...tab, loading } : tab
    ));
  }, []);

  const getNextTab = useCallback(() => {
    const currentIndex = tabs.findIndex(tab => tab.id === activeTabId);
    if (currentIndex !== -1 && tabs.length > 1) {
      const nextIndex = (currentIndex + 1) % tabs.length;
      return tabs[nextIndex];
    }
    return null;
  }, [tabs, activeTabId]);

  const getPreviousTab = useCallback(() => {
    const currentIndex = tabs.findIndex(tab => tab.id === activeTabId);
    if (currentIndex !== -1 && tabs.length > 1) {
      const prevIndex = (currentIndex - 1 + tabs.length) % tabs.length;
      return tabs[prevIndex];
    }
    return null;
  }, [tabs, activeTabId]);

  return {
    tabs,
    activeTabId,
    activeTab: tabs.find(tab => tab.id === activeTabId),
    openTab,
    closeTab,
    closeAllTabs,
    changeTab,
    updateTabError,
    updateTabLoading,
    getNextTab,
    getPreviousTab
  };
};