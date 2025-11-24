import { useState, useEffect, useRef } from 'react'
import { ChevronDown, ChevronRight, BookOpen } from 'lucide-react'
import { previewCorpus, DatasetInfo } from '../api/client'
import Pagination from './Pagination'

export interface CorpusDocument {
  doc_id: string
  content: string
  title?: string
  metadata?: Record<string, any>
}

export interface CorpusBrowserProps {
  datasetName: string
  subset?: string
  highlightedDocIds?: string[]
  onDocumentClick?: (docId: string) => void
  onNavigateToSample?: (sampleId: string) => void
  reverseReferences?: { [docId: string]: string[] } // Map of doc_id to sample_ids
}

export default function CorpusBrowser({
  datasetName,
  subset,
  highlightedDocIds = [],
  onDocumentClick,
  onNavigateToSample,
  reverseReferences = {},
}: CorpusBrowserProps) {
  const [documents, setDocuments] = useState<CorpusDocument[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [expandedDocs, setExpandedDocs] = useState<Set<string>>(new Set())
  
  // Pagination state
  const [currentPage, setCurrentPage] = useState(1)
  const [pageSize, setPageSize] = useState(20)
  const [totalCount, setTotalCount] = useState(0)
  const [totalPages, setTotalPages] = useState(0)
  
  // Refs for scrolling to highlighted documents
  const documentRefs = useRef<{ [key: string]: HTMLDivElement | null }>({})

  useEffect(() => {
    loadDocuments()
  }, [datasetName, subset, currentPage, pageSize])

  // Scroll to and expand highlighted documents
  useEffect(() => {
    if (highlightedDocIds.length > 0 && documents.length > 0) {
      const firstHighlightedId = highlightedDocIds[0]
      
      // Auto-expand highlighted documents
      setExpandedDocs(prev => {
        const newExpanded = new Set(prev)
        highlightedDocIds.forEach(id => newExpanded.add(id))
        return newExpanded
      })
      
      // Scroll to the first highlighted document
      setTimeout(() => {
        const element = documentRefs.current[firstHighlightedId]
        if (element) {
          element.scrollIntoView({ behavior: 'smooth', block: 'center' })
        }
      }, 100)
    }
  }, [highlightedDocIds, documents])

  const loadDocuments = async () => {
    try {
      setLoading(true)
      setError(null)
      
      const datasetInfo: DatasetInfo = {
        name: datasetName,
        ...(subset && { subset }),
      }
      
      const response = await previewCorpus(datasetInfo, currentPage, pageSize)
      
      setDocuments(response.data.documents || [])
      setTotalCount(response.data.total_count || 0)
      setTotalPages(response.data.total_pages || 0)
    } catch (err: any) {
      console.error('加载语料库失败:', err)
      
      // Handle page out of range error
      if (err.response?.status === 400 && err.response?.data?.detail?.total_pages) {
        const validTotalPages = err.response.data.detail.total_pages
        console.warn(`页码 ${currentPage} 超出范围，自动跳转到最后一页 ${validTotalPages}`)
        // Retry with the last valid page
        if (validTotalPages > 0 && currentPage !== validTotalPages) {
          setCurrentPage(validTotalPages)
          return // The useEffect will trigger loadDocuments again
        }
      }
      
      setError('加载语料库文档失败')
      setDocuments([])
      setTotalCount(0)
      setTotalPages(0)
    } finally {
      setLoading(false)
    }
  }

  const handlePageChange = (page: number) => {
    setCurrentPage(page)
  }

  const handlePageSizeChange = (size: number) => {
    setPageSize(size)
    setCurrentPage(1) // Reset to first page when changing page size
  }

  const toggleDocument = (docId: string) => {
    const newExpanded = new Set(expandedDocs)
    if (newExpanded.has(docId)) {
      newExpanded.delete(docId)
    } else {
      newExpanded.add(docId)
    }
    setExpandedDocs(newExpanded)
  }

  const handleDocumentClick = (docId: string) => {
    toggleDocument(docId)
    if (onDocumentClick) {
      onDocumentClick(docId)
    }
  }

  const isHighlighted = (docId: string) => {
    return highlightedDocIds.includes(docId)
  }

  if (loading) {
    return (
      <div className="flex justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="text-center py-8">
        <p className="text-red-500">{error}</p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold flex items-center">
          <BookOpen className="w-5 h-5 mr-2" />
          语料库文档
        </h3>
        <span className="text-sm text-gray-500">
          共 {totalCount} 个文档
        </span>
      </div>

      {/* Documents List */}
      <div className="space-y-3">
        {documents.length > 0 ? (
          documents.map((doc) => {
            const isExpanded = expandedDocs.has(doc.doc_id)
            const highlighted = isHighlighted(doc.doc_id)
            
            return (
              <div
                key={doc.doc_id}
                ref={(el) => { documentRefs.current[doc.doc_id] = el }}
                className={`
                  border rounded-lg transition-all
                  ${highlighted 
                    ? 'border-blue-500 bg-blue-50 shadow-md' 
                    : 'border-gray-200 hover:border-gray-300'
                  }
                `}
              >
                <button
                  onClick={() => handleDocumentClick(doc.doc_id)}
                  className="w-full flex items-start space-x-3 p-4 text-left hover:bg-gray-50 rounded-lg transition-colors"
                >
                  {isExpanded ? (
                    <ChevronDown className="w-5 h-5 mt-0.5 flex-shrink-0 text-gray-500" />
                  ) : (
                    <ChevronRight className="w-5 h-5 mt-0.5 flex-shrink-0 text-gray-500" />
                  )}
                  
                  <div className="flex-1 min-w-0">
                    {/* Document Header */}
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <span className="text-xs font-semibold text-gray-500 uppercase">
                          文档 ID:
                        </span>
                        <span className={`
                          text-sm font-medium
                          ${highlighted ? 'text-blue-700' : 'text-gray-900'}
                        `}>
                          {doc.doc_id}
                        </span>
                        {highlighted && (
                          <span className="px-2 py-0.5 text-xs font-medium bg-blue-500 text-white rounded">
                            已引用
                          </span>
                        )}
                      </div>
                      <span className="text-xs text-gray-400">
                        长度: {doc.content?.length || 0} 字符
                      </span>
                    </div>

                    {/* Document Title (if available) */}
                    {doc.title && (
                      <div className="mb-2">
                        <span className="text-sm font-medium text-gray-900">
                          {doc.title}
                        </span>
                      </div>
                    )}

                    {/* Content Preview (when collapsed) */}
                    {!isExpanded && (
                      <p className="text-sm text-gray-700 line-clamp-2">
                        {doc.content?.substring(0, 150)}
                        {doc.content && doc.content.length > 150 ? '...' : ''}
                      </p>
                    )}
                  </div>
                </button>

                {/* Expanded Content */}
                {isExpanded && (
                  <div className="px-4 pb-4 border-t border-gray-200 mt-2 pt-4">
                    {/* Reverse References - Show which samples reference this document */}
                    {reverseReferences[doc.doc_id] && reverseReferences[doc.doc_id].length > 0 && (
                      <div className="mb-3 p-3 bg-green-50 border border-green-200 rounded">
                        <span className="text-xs font-semibold text-green-700 uppercase mb-2 block">
                          被引用 ({reverseReferences[doc.doc_id].length} 个样本)
                        </span>
                        <div className="flex flex-wrap gap-2">
                          {reverseReferences[doc.doc_id].map((sampleId: string) => (
                            <button
                              key={sampleId}
                              onClick={() => onNavigateToSample && onNavigateToSample(sampleId)}
                              className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-green-100 text-green-700 hover:bg-green-200 transition-colors"
                              title={`点击查看样本: ${sampleId}`}
                            >
                              样本 {sampleId.length > 15 ? `${sampleId.substring(0, 15)}...` : sampleId}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    {/* Metadata (if available) */}
                    {doc.metadata && Object.keys(doc.metadata).length > 0 && (
                      <div className="mb-3 p-3 bg-gray-50 rounded">
                        <span className="text-xs font-semibold text-gray-500 uppercase mb-2 block">
                          元数据
                        </span>
                        <div className="space-y-1">
                          {Object.entries(doc.metadata).map(([key, value]) => (
                            <div key={key} className="flex items-start space-x-2 text-xs">
                              <span className="font-medium text-gray-600">{key}:</span>
                              <span className="text-gray-700">{String(value)}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Full Content */}
                    <div>
                      <span className="text-xs font-semibold text-gray-500 uppercase mb-2 block">
                        完整内容
                      </span>
                      <p className="text-sm text-gray-700 whitespace-pre-wrap leading-relaxed">
                        {doc.content}
                      </p>
                    </div>
                  </div>
                )}
              </div>
            )
          })
        ) : (
          <div className="text-center py-12 text-gray-500">
            <BookOpen className="w-12 h-12 mx-auto mb-3 text-gray-300" />
            <p>暂无语料库文档</p>
          </div>
        )}
      </div>

      {/* Pagination */}
      {totalCount > 0 && (
        <Pagination
          currentPage={currentPage}
          totalPages={totalPages}
          pageSize={pageSize}
          totalCount={totalCount}
          onPageChange={handlePageChange}
          onPageSizeChange={handlePageSizeChange}
        />
      )}
    </div>
  )
}
