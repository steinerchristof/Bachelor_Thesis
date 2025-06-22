/**
 * Custom hook for managing auto-scroll behavior
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { throttle, isAtBottom, scrollToElement } from '@/utils/chatUtils';

interface UseAutoScrollOptions {
  threshold?: number;
  delay?: number;
  smooth?: boolean;
}

export const useAutoScroll = ({
  threshold = 100,
  delay = 100,
  smooth = true
}: UseAutoScrollOptions = {}) => {
  const [shouldAutoScroll, setShouldAutoScroll] = useState(true);
  const [isUserScrolling, setIsUserScrolling] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const scrollTimeoutRef = useRef<NodeJS.Timeout>();

  // Throttled scroll handler
  const handleScroll = useCallback(
    throttle(() => {
      const atBottom = isAtBottom(threshold);
      setShouldAutoScroll(atBottom);
      
      // Detect user scrolling
      setIsUserScrolling(true);
      clearTimeout(scrollTimeoutRef.current);
      scrollTimeoutRef.current = setTimeout(() => {
        setIsUserScrolling(false);
      }, 150);
    }, 100),
    [threshold]
  );

  useEffect(() => {
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => {
      window.removeEventListener('scroll', handleScroll);
      clearTimeout(scrollTimeoutRef.current);
    };
  }, [handleScroll]);

  const scrollToBottom = useCallback(() => {
    if (endRef.current && shouldAutoScroll && !isUserScrolling) {
      setTimeout(() => {
        if (endRef.current) {
          scrollToElement(endRef.current, {
            behavior: smooth ? 'smooth' : 'auto',
            block: 'end'
          });
        }
      }, delay);
    }
  }, [shouldAutoScroll, isUserScrolling, delay, smooth]);

  const forceScrollToBottom = useCallback(() => {
    if (endRef.current) {
      scrollToElement(endRef.current, {
        behavior: smooth ? 'smooth' : 'auto',
        block: 'end'
      });
      setShouldAutoScroll(true);
    }
  }, [smooth]);

  const enableAutoScroll = useCallback(() => {
    setShouldAutoScroll(true);
    forceScrollToBottom();
  }, [forceScrollToBottom]);

  const disableAutoScroll = useCallback(() => {
    setShouldAutoScroll(false);
  }, []);

  return {
    endRef,
    shouldAutoScroll,
    isUserScrolling,
    scrollToBottom,
    forceScrollToBottom,
    enableAutoScroll,
    disableAutoScroll
  };
};