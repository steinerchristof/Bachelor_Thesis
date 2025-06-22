/**
 * Black & White Message Content - Pure minimalistic design
 */

import React, { useMemo, memo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { FileText } from 'lucide-react';
import 'katex/dist/katex.min.css';

import type { PDFTab, SourceDocument, SourceObject } from '@/types/chat';
import { parseSourceToDocument } from '@/utils/chatUtils';
import { normalizeYears } from '@/utils/professionalFormatting';

interface BlackWhiteMessageContentProps {
  content: string;
  sources?: SourceObject[];
  onSourceClick?: (source: SourceObject) => void;
  pdfTabs?: PDFTab[];
  isUserMessage?: boolean;
}

export const BlackWhiteMessageContent = memo<BlackWhiteMessageContentProps>(({
  content,
  sources,
  onSourceClick,
  pdfTabs = [],
  isUserMessage = false
}) => {
  // Process content
  const processedContent = useMemo(() => {
    return normalizeYears(content);
  }, [content]);

  // Parse sources
  const sourceDocuments = useMemo(() => {
    return sources?.map(source => ({
      title: source.title,
      id: source.metadata?.id || source.title,
      relevanceScore: source.metadata?.score
    })) || [];
  }, [sources]);

  const isDocumentOpen = (doc: SourceDocument) => {
    return pdfTabs.some(tab => tab.title === doc.title);
  };

  return (
    <div className="bw-message">
      {/* Main Content */}
      <div className="content">
        <ReactMarkdown
          remarkPlugins={[remarkGfm, remarkMath]}
          rehypePlugins={[rehypeKatex]}
          components={{
            h1: ({ children }) => <h1>{children}</h1>,
            h2: ({ children }) => <h2>{children}</h2>,
            h3: ({ children }) => <h3>{children}</h3>,
            p: ({ children }) => <p>{children}</p>,
            strong: ({ children }) => <strong>{children}</strong>,
            ul: ({ children }) => <ul>{children}</ul>,
            ol: ({ children }) => <ol>{children}</ol>,
            li: ({ children }) => <li>{children}</li>,
            table: ({ children }) => <table>{children}</table>,
            th: ({ children }) => <th>{children}</th>,
            td: ({ children }) => <td>{children}</td>,
            code: ({ inline, children }) => 
              inline ? 
                <code className="inline">{children}</code> : 
                <pre><code>{children}</code></pre>,
          }}
        >
          {processedContent}
        </ReactMarkdown>
      </div>

      {/* Sources Section */}
      {!isUserMessage && sourceDocuments.length > 0 && onSourceClick && (
        <div className="sources-section">
          <h4>Quellen</h4>
          <div className="sources-list">
            {sourceDocuments.map((doc, index) => {
              const isOpen = isDocumentOpen(doc);
              
              return (
                <button
                  key={`${doc.id}-${index}`}
                  onClick={() => onSourceClick(sources[index])}
                  className={`source ${isOpen ? 'open' : ''}`}
                >
                  <FileText size={16} />
                  <span>{normalizeYears(doc.title)}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Pure Black & White CSS */}
      <style jsx>{`
        .bw-message {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          color: #000;
          font-size: 16px;
          line-height: 1.6;
        }

        .content {
          color: #000;
        }

        /* Typography */
        h1 {
          font-size: 24px;
          font-weight: bold;
          margin: 24px 0 16px;
          color: #000;
        }

        h2 {
          font-size: 20px;
          font-weight: bold;
          margin: 20px 0 12px;
          color: #000;
        }

        h3 {
          font-size: 16px;
          font-weight: bold;
          margin: 16px 0 8px;
          color: #000;
        }

        p {
          margin: 0 0 16px;
          color: #000;
        }

        strong {
          font-weight: bold;
          color: #000;
        }

        /* Lists */
        ul, ol {
          margin: 0 0 16px;
          padding-left: 24px;
          color: #000;
        }

        li {
          margin: 4px 0;
          color: #000;
        }

        /* Tables */
        table {
          width: 100%;
          border-collapse: collapse;
          margin: 16px 0;
        }

        th {
          text-align: left;
          padding: 8px;
          border-bottom: 2px solid #000;
          font-weight: bold;
          color: #000;
        }

        td {
          padding: 8px;
          border-bottom: 1px solid #000;
          color: #000;
        }

        /* Code */
        code {
          font-family: monospace;
          font-size: 14px;
          color: #000;
        }

        code.inline {
          background: #f5f5f5;
          padding: 2px 4px;
        }

        pre {
          background: #f5f5f5;
          padding: 12px;
          overflow-x: auto;
          margin: 12px 0;
        }

        pre code {
          background: none;
          padding: 0;
        }

        /* Sources */
        .sources-section {
          margin-top: 32px;
          padding-top: 24px;
          border-top: 1px solid #000;
        }

        .sources-section h4 {
          font-size: 14px;
          font-weight: bold;
          margin-bottom: 12px;
          color: #000;
        }

        .sources-list {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }

        .source {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 8px 12px;
          background: #fff;
          border: 1px solid #000;
          font-size: 14px;
          color: #000;
          cursor: pointer;
          text-align: left;
          width: 100%;
          font-family: inherit;
        }

        .source:hover {
          background: #f5f5f5;
        }

        .source.open {
          background: #f0f0f0;
          border: 2px solid #000;
        }

        /* Math/KaTeX styling */
        .katex {
          font-size: 1em !important;
          color: #000 !important;
        }
        
        .katex-display {
          margin: 16px 0 !important;
          text-align: center;
        }
        
        .katex .base {
          color: #000 !important;
        }

        /* Force all text black */
        .bw-message * {
          color: #000 !important;
        }


        /* Dark mode - still black on white */
        @media (prefers-color-scheme: dark) {
          .bw-message,
          .bw-message * {
            color: #000 !important;
          }
          
          .bw-message {
            background: #fff;
          }
          
          .source.open {
            background: #f0f0f0;
            border: 2px solid #000;
          }
          
          .katex,
          .katex * {
            color: #000 !important;
          }
        }
      `}</style>
    </div>
  );
});

BlackWhiteMessageContent.displayName = 'BlackWhiteMessageContent';