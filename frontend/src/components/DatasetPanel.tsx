import { useState, useEffect } from 'react'
import { listDatasets, getDatasetStats, sampleDataset, DatasetStats } from '../api/client'
import { Loader2, Database, FileText, ChevronDown, ChevronRight, BookOpen, ExternalLink, ArrowLeft } from 'lucide-react'
import Pagination from './Pagination'
import SearchBar from './SearchBar'
import CorpusBrowser from './CorpusBrowser'

export default function DatasetPanel() {
  const [datasets, setDatasets] = useState<string[]>([])
  const [selectedDataset, setSelectedDataset] = useState<string>('')
  const [stats, setStats] = useState<DatasetStats | null>(null)
  const [samples, setSamples] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [expandedContexts, setExpandedContexts] = useState<Set<string>>(new Set())
  
  // Pagination state for samples
  const [samplesCurrentPage, setSamplesCurrentPage] = useState(1)
  const [samplesPageSize, setSamplesPageSize] = useState(20)
  const [samplesTotalCount, setSamplesTotalCount] = useState(0)
  const [samplesTotalPages, setSamplesTotalPages] = useState(0)
  
  // Pagination state for corpus (managed by CorpusBrowser internally)
  // We don't need separate corpus pagination state here since CorpusBrowser manages its own
  
  // Search state
  const [searchQuery, setSearchQuery] = useState('')
  
  // View toggle state
  const [currentView, setCurrentView] = useState<'samples' | 'corpus'>('samples')
  
  // Highlighted corpus documents (for navigation from samples)
  const [highlightedDocIds, setHighlightedDocIds] = useState<string[]>([])
  
  // Reverse references mapping (doc_id -> sample_ids)
  const [reverseReferences, setReverseReferences] = useState<{ [docId: string]: string[] }>({})
  
  // Highlighted sample (for navigation from corpus)
  const [highlightedSampleId, setHighlightedSampleId] = useState<string | null>(null)
  
  // Navigation history
  const [navigationHistory, setNavigationHistory] = useState<Array<{
    view: 'samples' | 'corpus'
    highlightedDocIds?: string[]
    highlightedSampleId?: string | null
  }>>([])
  const [canGoBack, setCanGoBack] = useState(false)

  useEffect(() => {
    loadDatasets()
  }, [])

  const loadDatasets = async () => {
    try {
      const response = await listDatasets()
      setDatasets(response.data)
    } catch (error) {
      console.error('加载数据集失败:', error)
    }
  }

  const handleSelectDataset = async (name: string) => {
    setSelectedDataset(name)
    setLoading(true)
    setExpandedContexts(new Set())
    // Reset pagination and search when selecting a new dataset
    setSamplesCurrentPage(1)
    setSearchQuery('')
    // Clear navigation history and highlights
    setNavigationHistory([])
    setCanGoBack(false)
    setHighlightedDocIds([])
    setHighlightedSampleId(null)
    try {
      const statsRes = await getDatasetStats({ name })
      setStats(statsRes.data)
      // Load samples (corpus view handles its own loading)
      if (currentView === 'samples') {
        await loadSamples(name, 1, samplesPageSize, '')
      }
    } catch (error) {
      console.error('加载数据集信息失败:', error)
    } finally {
      setLoading(false)
    }
  }

  const loadSamples = async (datasetName: string, page: number, size: number, search: string) => {
    try {
      setLoading(true)
      const response = await sampleDataset(
        { name: datasetName },
        size,
        page,
        search || undefined
      )
      setSamples(response.data)
      setSamplesTotalCount(response.data.total_count || 0)
      setSamplesTotalPages(response.data.total_pages || 0)
      
      // Build reverse references mapping
      buildReverseReferences(response.data.samples || [])
    } catch (error: any) {
      console.error('加载样本失败:', error)
      
      // Handle page out of range error
      if (error.response?.status === 400 && error.response?.data?.detail?.total_pages) {
        const validTotalPages = error.response.data.detail.total_pages
        console.warn(`页码 ${page} 超出范围，自动跳转到最后一页 ${validTotalPages}`)
        // Retry with the last valid page
        if (validTotalPages > 0 && page !== validTotalPages) {
          setSamplesCurrentPage(validTotalPages)
          return loadSamples(datasetName, validTotalPages, size, search)
        }
      }
      
      setSamples(null)
      setSamplesTotalCount(0)
      setSamplesTotalPages(0)
    } finally {
      setLoading(false)
    }
  }

  // Build reverse references mapping from samples
  const buildReverseReferences = (samplesData: any[]) => {
    const mapping: { [docId: string]: string[] } = {}
    
    samplesData.forEach((sample: any) => {
      if (sample.reference_context_ids && Array.isArray(sample.reference_context_ids)) {
        sample.reference_context_ids.forEach((docId: string) => {
          if (!mapping[docId]) {
            mapping[docId] = []
          }
          // Use sample.id if available, otherwise use a placeholder
          const sampleId = sample.id || `sample-${samplesData.indexOf(sample)}`
          if (!mapping[docId].includes(sampleId)) {
            mapping[docId].push(sampleId)
          }
        })
      }
    })
    
    setReverseReferences(mapping)
  }



  // Handle pagination changes (only for samples view)
  const handlePageChange = (page: number) => {
    setSamplesCurrentPage(page)
    if (selectedDataset && currentView === 'samples') {
      loadSamples(selectedDataset, page, samplesPageSize, searchQuery)
    }
  }

  const handlePageSizeChange = (size: number) => {
    setSamplesPageSize(size)
    setSamplesCurrentPage(1) // Reset to first page when changing page size
    if (selectedDataset && currentView === 'samples') {
      loadSamples(selectedDataset, 1, size, searchQuery)
    }
  }

  // Handle search
  const handleSearch = (query: string) => {
    setSearchQuery(query)
    setSamplesCurrentPage(1) // Reset to first page when searching
    if (selectedDataset) {
      loadSamples(selectedDataset, 1, samplesPageSize, query)
    }
  }

  // Handle view toggle
  const handleViewChange = (view: 'samples' | 'corpus') => {
    if (view === currentView) return
    
    setCurrentView(view)
    setExpandedContexts(new Set())
    
    // Clear navigation history and highlights when manually switching views
    setNavigationHistory([])
    setCanGoBack(false)
    setHighlightedDocIds([])
    setHighlightedSampleId(null)
    
    // Reset samples page when switching to samples view
    // Corpus view manages its own pagination state internally
    if (view === 'samples') {
      setSamplesCurrentPage(1)
      if (selectedDataset) {
        loadSamples(selectedDataset, 1, samplesPageSize, searchQuery)
      }
    }
  }

  const toggleContext = (sampleIdx: number, contextIdx: number) => {
    const key = `${sampleIdx}-${contextIdx}`
    const newExpanded = new Set(expandedContexts)
    if (newExpanded.has(key)) {
      newExpanded.delete(key)
    } else {
      newExpanded.add(key)
    }
    setExpandedContexts(newExpanded)
  }

  // Handle clicking on a reference_context_id to navigate to corpus
  const handleReferenceClick = (docId: string) => {
    // Save current state to history
    setNavigationHistory(prev => [...prev, {
      view: currentView,
      highlightedDocIds,
      highlightedSampleId
    }])
    setCanGoBack(true)
    
    setHighlightedDocIds([docId])
    setHighlightedSampleId(null) // Clear sample highlight
    setCurrentView('corpus')
    // Scroll to top when switching views
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  // Handle clicking on a sample reference from corpus to navigate back to samples
  const handleNavigateToSample = (sampleId: string) => {
    // Save current state to history
    setNavigationHistory(prev => [...prev, {
      view: currentView,
      highlightedDocIds,
      highlightedSampleId
    }])
    setCanGoBack(true)
    
    setHighlightedSampleId(sampleId)
    setHighlightedDocIds([]) // Clear corpus highlight
    setCurrentView('samples')
    // Scroll to top when switching views
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  // Handle back navigation
  const handleGoBack = () => {
    if (navigationHistory.length === 0) return
    
    const previousState = navigationHistory[navigationHistory.length - 1]
    setNavigationHistory(prev => prev.slice(0, -1))
    setCanGoBack(navigationHistory.length > 1)
    
    setCurrentView(previousState.view)
    setHighlightedDocIds(previousState.highlightedDocIds || [])
    setHighlightedSampleId(previousState.highlightedSampleId || null)
    
    // Scroll to top when going back
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-lg font-semibold mb-4 flex items-center">
          <Database className="w-5 h-5 mr-2" />
          可用数据集
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {datasets.map((dataset) => (
            <button
              key={dataset}
              onClick={() => handleSelectDataset(dataset)}
              className={`
                p-3 rounded-lg border-2 text-sm font-medium transition-colors
                ${selectedDataset === dataset
                  ? 'border-blue-500 bg-blue-50 text-blue-700'
                  : 'border-gray-200 hover:border-gray-300 text-gray-700'
                }
              `}
            >
              {dataset}
            </button>
          ))}
        </div>
      </div>

      {loading && (
        <div className="flex justify-center py-12">
          <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
        </div>
      )}

      {stats && !loading && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">数据集统计</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="text-sm text-gray-500">记录数</div>
              <div className="text-2xl font-bold text-gray-900">{stats.record_count}</div>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="text-sm text-gray-500">语料库大小</div>
              <div className="text-2xl font-bold text-gray-900">{stats.corpus_count}</div>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="text-sm text-gray-500">平均问题长度</div>
              <div className="text-2xl font-bold text-gray-900">{Math.round(stats.avg_input_length)}</div>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="text-sm text-gray-500">平均上下文数</div>
              <div className="text-2xl font-bold text-gray-900">{stats.avg_contexts_per_record.toFixed(1)}</div>
            </div>
          </div>
        </div>
      )}

      {selectedDataset && !loading && (
        <div className="bg-white rounded-lg shadow p-6">
          {/* View Toggle */}
          <div className="flex items-center justify-between mb-4">
            <div className="flex space-x-2">
              <button
                onClick={() => handleViewChange('samples')}
                className={`
                  flex items-center px-4 py-2 rounded-lg font-medium text-sm transition-colors
                  ${currentView === 'samples'
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }
                `}
              >
                <FileText className="w-4 h-4 mr-2" />
                数据样本
              </button>
              <button
                onClick={() => handleViewChange('corpus')}
                className={`
                  flex items-center px-4 py-2 rounded-lg font-medium text-sm transition-colors
                  ${currentView === 'corpus'
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }
                `}
              >
                <BookOpen className="w-4 h-4 mr-2" />
                语料库
              </button>
            </div>
            
            {/* Back button */}
            {canGoBack && (
              <button
                onClick={handleGoBack}
                className="flex items-center px-3 py-2 rounded-lg font-medium text-sm bg-gray-100 text-gray-700 hover:bg-gray-200 transition-colors"
                title="返回上一个视图"
              >
                <ArrowLeft className="w-4 h-4 mr-1" />
                返回
              </button>
            )}
          </div>

          {/* Search Bar (only for samples view) */}
          {currentView === 'samples' && (
            <div className="mb-4">
              <SearchBar
                value={searchQuery}
                onChange={setSearchQuery}
                onSearch={handleSearch}
                placeholder="搜索问题..."
              />
            </div>
          )}

          {/* Samples View */}
          {currentView === 'samples' && samples && (
            <>
              <div className="space-y-4 mb-4">
                {samples.samples && samples.samples.length > 0 ? (
                  samples.samples.map((sample: any, idx: number) => {
                    const sampleId = sample.id || `sample-${idx}`
                    const isHighlighted = highlightedSampleId === sampleId
                    
                    return (
                    <div 
                      key={idx} 
                      className={`
                        border rounded-lg p-4 transition-all
                        ${isHighlighted 
                          ? 'border-green-500 bg-green-50 shadow-md' 
                          : 'border-gray-200'
                        }
                      `}
                    >
                      {isHighlighted && (
                        <div className="mb-2 px-2 py-1 bg-green-500 text-white text-xs font-medium rounded inline-block">
                          从语料库导航
                        </div>
                      )}
                      <div className="mb-2">
                        <span className="text-xs font-semibold text-gray-500 uppercase">问题</span>
                        <p className="text-sm text-gray-900 mt-1">{sample.user_input}</p>
                      </div>
                      <div className="mb-2">
                        <span className="text-xs font-semibold text-gray-500 uppercase">参考答案</span>
                        <p className="text-sm text-gray-700 mt-1">{sample.reference}</p>
                      </div>
                      
                      {/* Display reference_context_ids as clickable badges */}
                      {sample.reference_context_ids && sample.reference_context_ids.length > 0 && (
                        <div className="mb-2">
                          <span className="text-xs font-semibold text-gray-500 uppercase">
                            引用文档 ({sample.reference_context_ids.length})
                          </span>
                          <div className="mt-1 flex flex-wrap gap-2">
                            {sample.reference_context_ids.map((docId: string, i: number) => (
                              <button
                                key={i}
                                onClick={() => handleReferenceClick(docId)}
                                className="inline-flex items-center px-2.5 py-1 rounded-md text-xs font-medium bg-blue-100 text-blue-700 hover:bg-blue-200 transition-colors"
                                title={`点击查看文档: ${docId}`}
                              >
                                <ExternalLink className="w-3 h-3 mr-1" />
                                {docId.length > 20 ? `${docId.substring(0, 20)}...` : docId}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      <div>
                        <span className="text-xs font-semibold text-gray-500 uppercase">
                          参考上下文 ({sample.reference_contexts?.length || 0})
                        </span>
                        <div className="mt-1 space-y-2">
                          {sample.reference_contexts?.map((ctx: string, i: number) => {
                            const key = `${idx}-${i}`
                            const isExpanded = expandedContexts.has(key)
                            return (
                              <div key={i} className="bg-gray-50 rounded">
                                <button
                                  onClick={() => toggleContext(idx, i)}
                                  className="w-full flex items-start space-x-2 p-2 text-left hover:bg-gray-100 rounded"
                                >
                                  {isExpanded ? (
                                    <ChevronDown className="w-4 h-4 mt-0.5 flex-shrink-0 text-gray-500" />
                                  ) : (
                                    <ChevronRight className="w-4 h-4 mt-0.5 flex-shrink-0 text-gray-500" />
                                  )}
                                  <div className="flex-1 min-w-0">
                                    <span className="text-xs font-medium text-gray-700">
                                      上下文 {i + 1}
                                    </span>
                                    {!isExpanded && (
                                      <p className="text-xs text-gray-600 truncate mt-0.5">
                                        {ctx.substring(0, 80)}...
                                      </p>
                                    )}
                                  </div>
                                </button>
                                {isExpanded && (
                                  <div className="px-2 pb-2">
                                    <p className="text-xs text-gray-700 whitespace-pre-wrap">
                                      {ctx}
                                    </p>
                                  </div>
                                )}
                              </div>
                            )
                          })}
                        </div>
                      </div>
                    </div>
                  )
                  })
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    {searchQuery ? '未找到匹配的样本' : '暂无数据样本'}
                  </div>
                )}
              </div>

              {/* Pagination for samples */}
              {samplesTotalCount > 0 && (
                <Pagination
                  currentPage={samplesCurrentPage}
                  totalPages={samplesTotalPages}
                  pageSize={samplesPageSize}
                  totalCount={samplesTotalCount}
                  onPageChange={handlePageChange}
                  onPageSizeChange={handlePageSizeChange}
                />
              )}
            </>
          )}

          {/* Corpus View */}
          {currentView === 'corpus' && (
            <CorpusBrowser
              datasetName={selectedDataset}
              highlightedDocIds={highlightedDocIds}
              reverseReferences={reverseReferences}
              onNavigateToSample={handleNavigateToSample}
            />
          )}
        </div>
      )}
    </div>
  )
}
