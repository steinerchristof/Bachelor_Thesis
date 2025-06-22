import React, { useState, useMemo, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Plus, 
  Trash2, 
  Edit2, 
  Check,
  X,
  Clock,
  ChevronRight,
  ChevronLeft,
  Search,
  MessageCircle
} from 'lucide-react';
import { Conversation } from '@/hooks/useConversationPersistence';

interface ConversationSidebarProps {
  conversations: Conversation[];
  currentConversationId: string | null;
  onSelectConversation: (id: string) => void;
  onNewConversation: () => void;
  onDeleteConversation: (id: string) => void;
  onRenameConversation: (id: string, title: string) => void;
  isOpen: boolean;
  onToggle: () => void;
}

export const ConversationSidebar: React.FC<ConversationSidebarProps> = ({
  conversations,
  currentConversationId,
  onSelectConversation,
  onNewConversation,
  onDeleteConversation,
  onRenameConversation,
  isOpen,
  onToggle
}) => {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const sidebarRef = useRef<HTMLDivElement>(null);

  // Handle click outside to close sidebar
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (isOpen && sidebarRef.current && !sidebarRef.current.contains(event.target as Node)) {
        onToggle();
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen, onToggle]);

  // Handle escape key to close sidebar
  useEffect(() => {
    const handleEscapeKey = (event: KeyboardEvent) => {
      if (isOpen && event.key === 'Escape' && !editingId) {
        onToggle();
      }
    };

    document.addEventListener('keydown', handleEscapeKey);
    return () => document.removeEventListener('keydown', handleEscapeKey);
  }, [isOpen, onToggle, editingId]);

  // Filter conversations based on search
  const filteredConversations = useMemo(() => {
    if (!searchQuery.trim()) return conversations;
    
    const query = searchQuery.toLowerCase();
    return conversations.filter(conv => 
      conv.title.toLowerCase().includes(query) ||
      conv.messages.some(msg => msg.text.toLowerCase().includes(query))
    );
  }, [conversations, searchQuery]);

  const formatDate = (timestamp: number) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffDays === 0) {
      return date.toLocaleTimeString('de-CH', { hour: '2-digit', minute: '2-digit' });
    } else if (diffDays === 1) {
      return 'Gestern';
    } else if (diffDays < 7) {
      return `vor ${diffDays} Tagen`;
    } else {
      return date.toLocaleDateString('de-CH');
    }
  };

  const startEditing = (id: string, currentTitle: string) => {
    setEditingId(id);
    setEditTitle(currentTitle);
  };

  const saveEdit = () => {
    if (editingId && editTitle.trim()) {
      onRenameConversation(editingId, editTitle.trim());
    }
    setEditingId(null);
    setEditTitle('');
  };

  const cancelEdit = () => {
    setEditingId(null);
    setEditTitle('');
  };

  return (
    <>
      {/* Mobile Overlay */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 bg-black/30 backdrop-blur-sm z-10 md:hidden"
            onClick={onToggle}
            aria-hidden="true"
          />
        )}
      </AnimatePresence>

      {/* Collapsible Sidebar */}
      <div 
        ref={sidebarRef}
        className={`fixed left-0 top-1/2 -translate-y-1/2 z-20 transition-all duration-300 ${
          isOpen ? 'w-80' : 'w-12'
        }`}
      >
        <div className="flex h-[500px] bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-r-lg shadow-lg overflow-hidden">
          {/* Sidebar Content */}
          <AnimatePresence mode="wait">
            {isOpen && (
              <motion.div
                initial={{ opacity: 0, width: 0 }}
                animate={{ opacity: 1, width: 320 }}
                exit={{ opacity: 0, width: 0 }}
                transition={{ duration: 0.2 }}
                className="flex flex-col w-full"
              >
                {/* Header */}
                <div className="p-4 border-b border-gray-200 dark:border-gray-800">
                  <div className="flex items-center justify-between mb-3">
                    <h2 className="text-sm font-medium text-gray-900 dark:text-gray-100">
                      Ihre Unterhaltungen
                    </h2>
                    <button
                      onClick={onToggle}
                      className="p-1 text-gray-500 hover:text-gray-700 dark:hover:text-gray-300
                               hover:bg-gray-100 dark:hover:bg-gray-800 rounded-md transition-colors
                               md:hidden"
                      aria-label="Seitenleiste schließen"
                      title="Schließen (ESC)"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                  
                  <button
                    onClick={() => {
                      onNewConversation();
                      onToggle();
                    }}
                    className="w-full flex items-center justify-center gap-2 px-3 py-2 
                             text-sm font-medium text-gray-700 dark:text-gray-300
                             bg-gray-50 dark:bg-gray-800 rounded-md
                             hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                  >
                    <Plus className="w-4 h-4" />
                    <span>Neue Unterhaltung</span>
                  </button>

                  {/* Search */}
                  {conversations.length > 3 && (
                    <div className="relative mt-3">
                      <Search className="absolute left-3 top-1/2 -translate-y-1/2 
                                       w-4 h-4 text-gray-400" />
                      <input
                        type="text"
                        placeholder="Suchen..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="w-full pl-9 pr-3 py-2 text-sm bg-gray-50 dark:bg-gray-800 
                                 border border-gray-200 dark:border-gray-700 rounded-md
                                 focus:outline-none focus:border-gray-300 dark:focus:border-gray-600
                                 placeholder-gray-400"
                      />
                      {searchQuery && (
                        <button
                          onClick={() => setSearchQuery('')}
                          className="absolute right-2 top-1/2 -translate-y-1/2 p-1 
                                   hover:bg-gray-200 dark:hover:bg-gray-700 rounded"
                        >
                          <X className="w-3 h-3 text-gray-400" />
                        </button>
                      )}
                    </div>
                  )}
                </div>

                {/* Conversations List */}
                <div className="flex-1 overflow-y-auto">
                  {conversations.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-64 text-center px-4">
                      <MessageCircle className="w-12 h-12 text-gray-300 dark:text-gray-700 mb-3" />
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        Keine Unterhaltungen vorhanden
                      </p>
                    </div>
                  ) : filteredConversations.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-64 text-center px-4">
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        Keine Ergebnisse gefunden
                      </p>
                    </div>
                  ) : (
                    <div className="py-1">
                      {filteredConversations.map((conversation) => (
                        <div
                          key={conversation.id}
                          className={`
                            group relative px-4 py-3 cursor-pointer
                            ${conversation.id === currentConversationId 
                              ? 'bg-gray-50 dark:bg-gray-800' 
                              : 'hover:bg-gray-50 dark:hover:bg-gray-800'
                            }
                            transition-colors
                          `}
                          onClick={() => {
                            if (editingId !== conversation.id) {
                              onSelectConversation(conversation.id);
                              onToggle();
                            }
                          }}
                        >
                          {editingId === conversation.id ? (
                            // Edit Mode
                            <div className="flex items-center gap-2">
                              <input
                                type="text"
                                value={editTitle}
                                onChange={(e) => setEditTitle(e.target.value)}
                                onKeyDown={(e) => {
                                  if (e.key === 'Enter') saveEdit();
                                  if (e.key === 'Escape') cancelEdit();
                                }}
                                className="flex-1 px-2 py-1 text-sm border border-gray-300 
                                         dark:border-gray-600 rounded focus:outline-none 
                                         focus:border-blue-500 bg-white dark:bg-gray-800"
                                autoFocus
                                onClick={(e) => e.stopPropagation()}
                              />
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  saveEdit();
                                }}
                                className="p-1 text-green-600 hover:bg-green-50 
                                         dark:hover:bg-green-900/20 rounded"
                              >
                                <Check className="w-4 h-4" />
                              </button>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  cancelEdit();
                                }}
                                className="p-1 text-gray-500 hover:bg-gray-100 
                                         dark:hover:bg-gray-700 rounded"
                              >
                                <X className="w-4 h-4" />
                              </button>
                            </div>
                          ) : (
                            // Normal Mode
                            <>
                              {/* Active Indicator */}
                              {conversation.id === currentConversationId && (
                                <div className="absolute left-0 top-0 bottom-0 w-0.5 bg-blue-500" />
                              )}
                              
                              <div className="pr-16">
                                <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100 
                                             truncate">
                                  {conversation.title}
                                </h3>
                                <div className="flex items-center gap-2 mt-1 text-xs text-gray-500 
                                              dark:text-gray-400">
                                  <span>{formatDate(conversation.updatedAt)}</span>
                                  <span>·</span>
                                  <span>{conversation.messages.length} Nachrichten</span>
                                </div>
                              </div>

                              {/* Action Buttons */}
                              <div className="absolute right-3 top-1/2 -translate-y-1/2 
                                            opacity-0 group-hover:opacity-100 transition-opacity
                                            flex items-center gap-1">
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    startEditing(conversation.id, conversation.title);
                                  }}
                                  className="p-1 text-gray-500 hover:bg-gray-100 
                                           dark:hover:bg-gray-700 rounded"
                                  aria-label="Rename conversation"
                                >
                                  <Edit2 className="w-3.5 h-3.5" />
                                </button>
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    if (confirm('Diese Unterhaltung löschen?')) {
                                      onDeleteConversation(conversation.id);
                                    }
                                  }}
                                  className="p-1 text-gray-500 hover:text-red-600 
                                           hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
                                  aria-label="Delete conversation"
                                >
                                  <Trash2 className="w-3.5 h-3.5" />
                                </button>
                              </div>
                            </>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Footer */}
                {conversations.length > 0 && (
                  <div className="p-3 border-t border-gray-200 dark:border-gray-800">
                    <p className="text-xs text-gray-500 dark:text-gray-400 text-center">
                      {conversations.length} von 50 Unterhaltungen
                    </p>
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Toggle Tab */}
          <button
            onClick={onToggle}
            className="flex items-center justify-center w-12 h-full bg-white dark:bg-gray-900 
                     hover:bg-gray-50 dark:hover:bg-gray-800 transition-all duration-200
                     border-l border-gray-200 dark:border-gray-800 group
                     focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-inset"
            aria-label={isOpen ? "Verlauf schließen" : "Verlauf öffnen"}
            title={isOpen ? "Verlauf schließen (ESC)" : "Verlauf öffnen"}
          >
            {isOpen ? (
              <ChevronLeft className="w-5 h-5 text-gray-600 dark:text-gray-400 
                                    group-hover:text-gray-900 dark:group-hover:text-gray-100 
                                    transition-colors" />
            ) : (
              <div className="flex flex-col items-center gap-2">
                <MessageCircle className="w-4 h-4 text-gray-600 dark:text-gray-400 
                                        group-hover:text-gray-900 dark:group-hover:text-gray-100 
                                        transition-colors" />
                {conversations.length > 0 && (
                  <span className="text-xs font-medium text-gray-600 dark:text-gray-400 
                                 bg-gray-100 dark:bg-gray-800 rounded-full px-1.5 py-0.5 
                                 min-w-[20px] text-center group-hover:bg-gray-200 
                                 dark:group-hover:bg-gray-700 transition-colors">
                    {conversations.length}
                  </span>
                )}
              </div>
            )}
          </button>
        </div>
      </div>

    </>
  );
};