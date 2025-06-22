# Swiss Financial Document RAG System - Frontend Architecture

## Executive Summary

This frontend implements a sophisticated chat interface for a financial document analysis system, featuring Swiss-specific formatting, advanced PDF viewing capabilities, and a responsive design optimized for professional use. Built with Next.js and TypeScript, it provides a seamless user experience for querying and analyzing financial documents.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Technologies](#core-technologies)
3. [Key Features and Components](#key-features-and-components)
4. [State Management Architecture](#state-management-architecture)
5. [Component Deep Dive](#component-deep-dive)
6. [Swiss-Specific Implementations](#swiss-specific-implementations)
7. [Performance Optimizations](#performance-optimizations)
8. [Testing Strategy](#testing-strategy)
9. [Deployment and Configuration](#deployment-and-configuration)
10. [Future Enhancements](#future-enhancements)

## Architecture Overview

The frontend follows a modern React architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interface                         │
├─────────────────────────────────────────────────────────────┤
│                    Page Components                          │
│                   (pages/chat.tsx)                          │
├─────────────────────────────────────────────────────────────┤
│                  Feature Components                         │
│              (MessageBubble, ChatInput,                     │
│               TabbedPDFViewer, etc.)                        │
├─────────────────────────────────────────────────────────────┤
│                    Custom Hooks                             │
│           (useChat, usePDFTabs, etc.)                       │
├─────────────────────────────────────────────────────────────┤
│                 Services & Utils                            │
│            (API Service, Formatters)                        │
├─────────────────────────────────────────────────────────────┤
│                    Next.js API                              │
│              (Proxy endpoints)                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    Backend Services
```

## Core Technologies

- **Next.js 13.4**: React framework with SSR/SSG capabilities
- **TypeScript 5.1**: Type-safe development with 100% coverage
- **Tailwind CSS 3.3**: Utility-first CSS framework
- **Framer Motion**: Animation library for smooth transitions
- **React Markdown**: Markdown rendering with custom components
- **KaTeX**: LaTeX math rendering
- **Highlight.js**: Syntax highlighting for code blocks
- **Lucide Icons**: Modern icon library

## Key Features and Components

### 1. Chat Interface

The chat system provides a sophisticated conversational interface:

```typescript
// Main chat component structure
const ChatPage = () => {
  const {
    messages,
    loading,
    sendMessage,
    deleteMessage,
    retryLastMessage
  } = useChat();
  
  const {
    pdfTabs,
    addPDFTab,
    closePDFTab,
    setActiveTab
  } = usePDFTabs();
  
  return (
    <div className="flex h-screen">
      <div className="flex-1 flex flex-col">
        <MessageList messages={messages} />
        <ChatInput onSend={sendMessage} loading={loading} />
      </div>
      {pdfTabs.length > 0 && (
        <TabbedPDFViewer tabs={pdfTabs} />
      )}
    </div>
  );
};
```

### 2. Message Bubble Component

Advanced message rendering with rich features:

```typescript
export const MessageBubble = memo<MessageBubbleProps>(({
  message,
  onSourceClick,
  onRetry,
  onDelete,
  pdfTabs = [],
  showActions = true,
  showTimestamp = true
}) => {
  const [copied, setCopied] = useState(false);
  const [showRelativeTime, setShowRelativeTime] = useState(false);

  // Features:
  // - Copy to clipboard functionality
  // - Timestamp toggle (absolute/relative)
  // - Retry failed messages
  // - Delete messages
  // - Source document references
  // - Error state handling
  // - Smooth animations with Framer Motion
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`flex mb-4 ${isUser ? 'justify-end' : 'justify-start'}`}
    >
      {/* Message content with formatting */}
      <BlackWhiteMessageContent 
        content={message.text} 
        sources={message.sources}
        onSourceClick={onSourceClick}
      />
    </motion.div>
  );
});
```

### 3. Swiss Number Formatting

Specialized formatting for Swiss financial standards:

```typescript
export const formatSwissNumbers = (text: string): string => {
  // Format numbers with apostrophe separators (1'234'567)
  return text.replace(
    /\b(\d{1,3})(\d{3})(\d{3})?(\d{3})?\b/g,
    (match, p1, p2, p3, p4) => {
      let result = p1 + "'" + p2;
      if (p3) result += "'" + p3;
      if (p4) result += "'" + p4;
      return result;
    }
  );
};

export const enhanceFinancialMarkdown = (text: string): string => {
  let enhanced = text;

  // Bold CHF amounts
  enhanced = enhanced.replace(
    /CHF\s*([+-]?\d+(?:[''']?\d{3})*(?:[.,]\d{2})?)/g,
    '**CHF $1**'
  );

  // Format percentages with directional indicators
  enhanced = enhanced.replace(
    /([+-]?\d+(?:[.,]\d+)?)\s*%/g,
    (match, num) => {
      const value = parseFloat(num.replace(',', '.'));
      if (value > 0) return `**+${num}%** ↑`;
      if (value < 0) return `**${num}%** ↓`;
      return `**${num}%**`;
    }
  );

  return enhanced;
};
```

### 4. Advanced PDF Viewer

Tabbed PDF viewer with keyboard navigation:

```typescript
export const TabbedPDFViewer = memo<TabbedPDFViewerProps>(({
  pdfTabs,
  activeTabId,
  onTabChange,
  onTabClose,
  onCloseAll
}) => {
  // Keyboard shortcuts implementation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // F11 - Toggle fullscreen
      if (e.key === 'F11') {
        e.preventDefault();
        setIsFullscreen(prev => !prev);
      }
      
      // Ctrl/Cmd + Tab - Switch tabs
      if ((e.ctrlKey || e.metaKey) && e.key === 'Tab') {
        e.preventDefault();
        const nextIndex = e.shiftKey 
          ? (currentIndex - 1 + pdfTabs.length) % pdfTabs.length
          : (currentIndex + 1) % pdfTabs.length;
        onTabChange(pdfTabs[nextIndex].id);
      }
      
      // Ctrl/Cmd + W - Close current tab
      if ((e.ctrlKey || e.metaKey) && e.key === 'w') {
        e.preventDefault();
        onTabClose(activeTabId);
      }
      
      // Ctrl/Cmd + 1-9 - Direct tab selection
      if ((e.ctrlKey || e.metaKey) && e.key >= '1' && e.key <= '9') {
        const tabIndex = parseInt(e.key) - 1;
        if (pdfTabs[tabIndex]) {
          onTabChange(pdfTabs[tabIndex].id);
        }
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [pdfTabs, activeTabId]);
  
  // Features:
  // - Multiple PDF tabs
  // - Keyboard navigation
  // - Fullscreen mode
  // - Loading/error states
  // - Mobile-friendly swipe navigation
  // - Smooth animations
});
```

## State Management Architecture

### 1. Custom Hook Pattern

The application uses custom hooks for state management:

```typescript
// Core chat logic hook
export const useChat = (options: UseChatOptions = {}) => {
  const [state, setState] = useState<ChatState>({
    messages: [],
    loading: false,
    error: null,
    retryCount: 0
  });

  // Message persistence
  useEffect(() => {
    const savedMessages = localStorage.getItem(CHAT_CONFIG.storage.messageKey);
    if (savedMessages) {
      setState(prev => ({ ...prev, messages: JSON.parse(savedMessages) }));
    }
  }, []);

  // API communication with retry logic
  const sendMessage = useCallback(async (text: string) => {
    // Cancel any pending request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    try {
      const data = await apiService.sendMessage(text, signal);
      
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
        loading: false
      }));
    } catch (error) {
      // Automatic retry with exponential backoff
      if (state.retryCount < maxRetries) {
        setTimeout(() => sendMessage(text), retryDelay * Math.pow(2, state.retryCount));
      }
    }
  }, [state.retryCount, maxRetries, retryDelay]);

  return { messages: state.messages, loading: state.loading, sendMessage };
};
```

### 2. PDF Tab Management

Sophisticated PDF tab state management:

```typescript
export const usePDFTabs = () => {
  const [tabs, setTabs] = useState<PDFTab[]>([]);
  const [activeTabId, setActiveTabId] = useState<string>('');

  const addPDFTab = useCallback((source: SourceObject) => {
    const existingTab = tabs.find(tab => tab.url === source.url);
    
    if (existingTab) {
      setActiveTabId(existingTab.id);
      return;
    }

    const newTab: PDFTab = {
      id: generateTabId(),
      title: source.title,
      url: source.url,
      loading: true,
      error: null
    };

    setTabs(prev => [...prev, newTab]);
    setActiveTabId(newTab.id);
  }, [tabs]);

  const closePDFTab = useCallback((tabId: string) => {
    setTabs(prev => {
      const filtered = prev.filter(tab => tab.id !== tabId);
      
      // Update active tab if closing current
      if (tabId === activeTabId && filtered.length > 0) {
        const currentIndex = prev.findIndex(tab => tab.id === tabId);
        const newIndex = Math.min(currentIndex, filtered.length - 1);
        setActiveTabId(filtered[newIndex].id);
      }
      
      return filtered;
    });
  }, [activeTabId]);

  return { tabs, activeTabId, addPDFTab, closePDFTab, setActiveTabId };
};
```

## Component Deep Dive

### 1. Message Content Renderer

Rich content rendering with multiple formats:

```typescript
export const BlackWhiteMessageContent: React.FC<Props> = ({
  content,
  sources = [],
  onSourceClick,
  isUserMessage = false
}) => {
  const processedContent = useMemo(() => {
    if (isUserMessage) return content;
    
    // Apply formatting pipeline
    let formatted = content;
    formatted = formatSwissNumbers(formatted);
    formatted = structureMarkdownText(formatted);
    formatted = formatMarkdownTables(formatted);
    formatted = enhanceFinancialMarkdown(formatted);
    
    return formatted;
  }, [content, isUserMessage]);

  return (
    <div className="message-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeHighlight]}
        components={{
          // Custom component renderers
          code: ({ inline, className, children }) => {
            const match = /language-(\w+)/.exec(className || '');
            return !inline && match ? (
              <SyntaxHighlighter language={match[1]}>
                {String(children).replace(/\n$/, '')}
              </SyntaxHighlighter>
            ) : (
              <code className="bg-gray-100 px-1 py-0.5 rounded">
                {children}
              </code>
            );
          },
          table: ({ children }) => (
            <div className="overflow-x-auto my-4">
              <table className="min-w-full divide-y divide-gray-200">
                {children}
              </table>
            </div>
          )
        }}
      >
        {processedContent}
      </ReactMarkdown>
      
      {/* Source references */}
      {sources.length > 0 && (
        <SourceReferences 
          sources={sources} 
          onSourceClick={onSourceClick}
        />
      )}
    </div>
  );
};
```

### 2. Chat Input Component

Advanced input handling with auto-resize:

```typescript
export const ChatInput: React.FC<ChatInputProps> = ({
  onSend,
  disabled = false,
  placeholder = "Stellen Sie Ihre Frage...",
  maxLength = 4000
}) => {
  const [value, setValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [value]);

  // Handle keyboard shortcuts
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSubmit = () => {
    if (value.trim() && !disabled) {
      onSend(value.trim());
      setValue('');
    }
  };

  return (
    <div className="border-t border-gray-200 p-4">
      <div className="relative max-w-4xl mx-auto">
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          maxLength={maxLength}
          className="w-full resize-none rounded-lg border p-3 pr-12"
          rows={1}
        />
        
        <div className="absolute bottom-3 right-3 flex items-center gap-2">
          <span className="text-xs text-gray-400">
            {value.length}/{maxLength}
          </span>
          
          <button
            onClick={handleSubmit}
            disabled={!value.trim() || disabled}
            className="p-2 rounded-lg bg-blue-600 text-white"
          >
            <Send className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
};
```

## Swiss-Specific Implementations

### 1. Number Formatting

The system implements Swiss number formatting standards throughout:

- **Thousands separator**: Apostrophe (') instead of comma
- **Decimal separator**: Period (.) for decimals
- **Currency display**: CHF with proper spacing
- **Percentage formatting**: With directional indicators

### 2. Language Support

- **German UI**: All interface elements in German
- **Swiss German considerations**: Proper character encoding for umlauts
- **Professional terminology**: Financial terms in Swiss context

### 3. Document Formatting

Special handling for Swiss financial documents:
- Jahresrechnung (Annual Reports)
- Bilanz (Balance Sheets)
- Erfolgsrechnung (Income Statements)
- Revisionsbericht (Audit Reports)

## Performance Optimizations

### 1. Component Memoization

```typescript
// Memoized components prevent unnecessary re-renders
export const MessageBubble = memo<MessageBubbleProps>(({ ... }) => {
  // Component implementation
});

// Memoized expensive computations
const processedContent = useMemo(() => {
  return formatForDisplay(content);
}, [content]);

// Stable callback references
const handleCopy = useCallback(async () => {
  await copyToClipboard(message.text);
}, [message.text]);
```

### 2. Lazy Loading

```typescript
// Dynamic imports for heavy components
const PDFViewer = dynamic(
  () => import('./TabbedPDFViewer'),
  { 
    loading: () => <LoadingSkeleton />,
    ssr: false 
  }
);
```

### 3. Virtual Scrolling Consideration

For large message lists:
```typescript
// Future enhancement for performance
import { VariableSizeList } from 'react-window';

const VirtualMessageList = ({ messages }) => (
  <VariableSizeList
    height={window.innerHeight}
    itemCount={messages.length}
    itemSize={getItemSize}
    width="100%"
  >
    {({ index, style }) => (
      <div style={style}>
        <MessageBubble message={messages[index]} />
      </div>
    )}
  </VariableSizeList>
);
```

### 4. Bundle Optimization

- **Code splitting**: Automatic with Next.js routing
- **Tree shaking**: Removes unused code
- **Minification**: Production builds are optimized
- **Image optimization**: Next.js Image component with lazy loading

## Testing Strategy

### 1. Unit Tests

```typescript
// Example test for chat hook
describe('useChat', () => {
  it('should send message and update state', async () => {
    const { result } = renderHook(() => useChat());
    
    await act(async () => {
      await result.current.sendMessage('Test question');
    });
    
    expect(result.current.messages).toHaveLength(2);
    expect(result.current.messages[0].role).toBe('user');
    expect(result.current.messages[1].role).toBe('assistant');
  });
  
  it('should retry on failure', async () => {
    apiService.sendMessage = jest.fn()
      .mockRejectedValueOnce(new Error('Network error'))
      .mockResolvedValueOnce({ answer: 'Success', sources: [] });
    
    const { result } = renderHook(() => useChat({ maxRetries: 1 }));
    
    await act(async () => {
      await result.current.sendMessage('Test');
    });
    
    expect(apiService.sendMessage).toHaveBeenCalledTimes(2);
  });
});
```

### 2. Component Tests

```typescript
describe('MessageBubble', () => {
  it('should render user message correctly', () => {
    const message: Message = {
      id: '1',
      role: 'user',
      text: 'Test message',
      timestamp: Date.now()
    };
    
    render(<MessageBubble message={message} />);
    
    expect(screen.getByText('Test message')).toBeInTheDocument();
  });
  
  it('should copy message to clipboard', async () => {
    const user = userEvent.setup();
    
    render(<MessageBubble message={message} />);
    
    await user.click(screen.getByLabelText('Copy message'));
    
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith('Test message');
  });
});
```

## Deployment and Configuration

### 1. Environment Variables

```bash
# .env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_ENVIRONMENT=production
NEXT_PUBLIC_VERSION=2.0.0
```

### 2. Build Configuration

```javascript
// next.config.js
module.exports = {
  reactStrictMode: true,
  swcMinify: true,
  images: {
    domains: ['localhost'],
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${process.env.BACKEND_URL}/api/:path*`,
      },
    ];
  },
};
```

### 3. Production Optimization

```bash
# Build commands
npm run build        # Creates optimized production build
npm run analyze      # Bundle size analysis
npm run lighthouse   # Performance audit
```

## Future Enhancements

### 1. Planned Features

- **Voice Input**: Speech-to-text for queries
- **File Upload**: Direct document analysis
- **Export Functionality**: Save conversations as PDF/Word
- **Multi-language Support**: French, Italian, English
- **Collaborative Features**: Share conversations and documents
- **Advanced Analytics**: Usage statistics and insights

### 2. Performance Improvements

- **Virtual Scrolling**: For large message lists
- **Service Worker**: Offline capability
- **WebSocket**: Real-time streaming responses
- **IndexedDB**: Local message storage

### 3. UI/UX Enhancements

- **Dark Mode**: Full theme support
- **Mobile App**: React Native implementation
- **Accessibility**: WCAG AAA compliance
- **Customization**: User preferences and layouts

## Conclusion

This frontend provides a professional, performant, and user-friendly interface for financial document analysis. The architecture supports scalability, maintainability, and future enhancements while delivering an excellent user experience tailored for Swiss financial professionals.

The combination of modern React patterns, TypeScript safety, and thoughtful Swiss-specific implementations creates a robust foundation for enterprise-grade financial analysis tools.