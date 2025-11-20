import { useState, useEffect } from 'react'
import { createRAG, listRAGs, deleteRAG, queryRAG, indexDocuments, listModels, ModelInfo, IndexStatus } from '../api/client'
import { Cpu, Plus, Search, Loader2, Trash2, Upload, Database, Clock, HardDrive } from 'lucide-react'

interface RAGInstance {
  name: string
  rag_type: 'baseline' | 'api'
  model_info: any
  rag_config: any
  api_config?: any
  index_status?: IndexStatus
}

// Helper functions
const formatBytes = (bytes?: number): string => {
  if (!bytes) return 'N/A'
  const sizes = ['B', 'KB', 'MB', 'GB']
  if (bytes === 0) return '0 B'
  const i = Math.floor(Math.log(bytes) / Math.log(1024))
  return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i]
}

const formatDate = (dateStr?: string): string => {
  if (!dateStr) return 'N/A'
  try {
    const date = new Date(dateStr)
    return date.toLocaleString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit'
    })
  } catch {
    return 'N/A'
  }
}

export default function RAGPanel() {
  const [rags, setRags] = useState<RAGInstance[]>([])
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [selectedRAG, setSelectedRAG] = useState<RAGInstance | null>(null)
  const [query, setQuery] = useState('')
  const [queryResult, setQueryResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [llmModels, setLlmModels] = useState<ModelInfo[]>([])
  const [embeddingModels, setEmbeddingModels] = useState<ModelInfo[]>([])
  const [datasets, setDatasets] = useState<string[]>([])
  const [showIndexDialog, setShowIndexDialog] = useState(false)
  const [indexData, setIndexData] = useState({
    dataset_name: '',
    subset: ''
  })
  const [corpusDocuments, setCorpusDocuments] = useState<any[]>([])
  const [selectedDocIds, setSelectedDocIds] = useState<Set<string>>(new Set())
  const [loadingCorpus, setLoadingCorpus] = useState(false)
  const [showDocumentPreview, setShowDocumentPreview] = useState(false)

  const [formData, setFormData] = useState({
    name: '',
    rag_type: 'baseline' as 'baseline' | 'api',
    llm_model_id: '',
    embedding_model_id: '',
    top_k: 5,
    temperature: 0.7,
    retrieval_endpoint: '',
    generation_endpoint: '',
    api_key: ''
  })

  useEffect(() => {
    loadRAGs()
    loadModels()
    loadDatasets()
  }, [])

  const loadRAGs = async () => {
    try {
      const response = await listRAGs()
      console.log('RAG列表响应:', response.data)
      console.log('RAG数组:', response.data.rags)
      setRags(response.data.rags)
    } catch (error) {
      console.error('加载RAG列表失败:', error)
    }
  }

  const loadModels = async () => {
    try {
      const response = await listModels()
      setLlmModels(response.data.llm_models)
      setEmbeddingModels(response.data.embedding_models)
      
      // 设置默认选择
      if (response.data.llm_models.length > 0 && !formData.llm_model_id) {
        setFormData(prev => ({ ...prev, llm_model_id: response.data.llm_models[0].model_id }))
      }
      if (response.data.embedding_models.length > 0 && !formData.embedding_model_id) {
        setFormData(prev => ({ ...prev, embedding_model_id: response.data.embedding_models[0].model_id }))
      }
    } catch (error) {
      console.error('加载模型列表失败:', error)
    }
  }

  const loadDatasets = async () => {
    try {
      const { listDatasets } = await import('../api/client')
      const response = await listDatasets()
      setDatasets(response.data)
    } catch (error) {
      console.error('加载数据集列表失败:', error)
    }
  }

  const handleCreateRAG = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    try {
      // Baseline类型需要模型配置
      if (formData.rag_type === 'baseline') {
        if (!formData.llm_model_id || !formData.embedding_model_id) {
          alert('请先在模型仓库中注册模型')
          return
        }
      }
      
      // API类型需要端点配置
      if (formData.rag_type === 'api') {
        if (!formData.retrieval_endpoint || !formData.generation_endpoint) {
          alert('请配置API端点')
          return
        }
      }
      
      const result = await createRAG({
        name: formData.name,
        rag_type: formData.rag_type,
        model_info: {
          llm_model_id: formData.llm_model_id,
          embedding_model_id: formData.embedding_model_id,
        },
        rag_config: {
          top_k: formData.top_k,
          temperature: formData.temperature,
        },
        api_config: formData.rag_type === 'api' ? {
          retrieval_endpoint: formData.retrieval_endpoint,
          generation_endpoint: formData.generation_endpoint,
          api_key: formData.api_key
        } : undefined
      })
      console.log('创建RAG成功:', result.data)
      await loadRAGs()
      setShowCreateForm(false)
      setFormData({
        name: '',
        rag_type: 'baseline',
        llm_model_id: llmModels[0]?.model_id || '',
        embedding_model_id: embeddingModels[0]?.model_id || '',
        top_k: 5,
        temperature: 0.7,
        retrieval_endpoint: '',
        generation_endpoint: '',
        api_key: ''
      })
    } catch (error: any) {
      alert('创建失败: ' + (error.response?.data?.detail || error.message))
    } finally {
      setLoading(false)
    }
  }

  const handleQuery = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!selectedRAG || !query) return
    
    setLoading(true)
    try {
      const response = await queryRAG({
        rag_name: selectedRAG.name,
        query: query,
      })
      setQueryResult(response.data)
    } catch (error: any) {
      alert('查询失败: ' + (error.response?.data?.detail || error.message))
    } finally {
      setLoading(false)
    }
  }

  const handleDeleteRAG = async (ragName: string) => {
    if (!confirm(`确定要删除RAG实例 "${ragName}" 吗？`)) {
      return
    }
    
    try {
      await deleteRAG(ragName)
      await loadRAGs()
      if (selectedRAG?.name === ragName) {
        setSelectedRAG(null)
        setQueryResult(null)
      }
    } catch (error: any) {
      alert('删除失败: ' + (error.response?.data?.detail || error.message))
    }
  }

  const handleIndexDocuments = async () => {
    if (!selectedRAG) return
    
    // 只有baseline类型才能索引
    if (selectedRAG.rag_type !== 'baseline') {
      alert('只有Baseline类型的RAG支持索引文档')
      return
    }
    
    setShowIndexDialog(true)
  }

  const handleLoadCorpus = async () => {
    if (!indexData.dataset_name) {
      alert('请选择数据集')
      return
    }
    
    setLoadingCorpus(true)
    try {
      const { previewCorpus } = await import('../api/client')
      const result = await previewCorpus({
        name: indexData.dataset_name,
        subset: indexData.subset || undefined
      }, 100)
      
      setCorpusDocuments(result.data.documents)
      setShowDocumentPreview(true)
      
      // 默认全选
      const allIds = new Set<string>(result.data.documents.map((doc: any) => doc.id as string))
      setSelectedDocIds(allIds)
    } catch (error: any) {
      alert('加载文档失败: ' + (error.response?.data?.detail || error.message))
    } finally {
      setLoadingCorpus(false)
    }
  }

  const handleToggleDocument = (docId: string) => {
    const newSelected = new Set(selectedDocIds)
    if (newSelected.has(docId)) {
      newSelected.delete(docId)
    } else {
      newSelected.add(docId)
    }
    setSelectedDocIds(newSelected)
  }

  const handleToggleAll = () => {
    if (selectedDocIds.size === corpusDocuments.length) {
      setSelectedDocIds(new Set<string>())
    } else {
      const allIds = new Set<string>(corpusDocuments.map(doc => doc.id as string))
      setSelectedDocIds(allIds)
    }
  }

  const handleIndexFromDataset = async () => {
    if (!indexData.dataset_name) {
      alert('请选择数据集')
      return
    }
    
    if (selectedDocIds.size === 0) {
      alert('请至少选择一个文档')
      return
    }
    
    setLoading(true)
    try {
      const result = await indexDocuments({
        rag_name: selectedRAG!.name,
        dataset_name: indexData.dataset_name,
        subset: indexData.subset || undefined,
        document_ids: Array.from(selectedDocIds)
      })
      alert(result.data.message)
      setShowIndexDialog(false)
      setShowDocumentPreview(false)
      setIndexData({ dataset_name: '', subset: '' })
      setCorpusDocuments([])
      setSelectedDocIds(new Set())
    } catch (error: any) {
      alert('索引失败: ' + (error.response?.data?.detail || error.message))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-semibold flex items-center">
            <Cpu className="w-5 h-5 mr-2" />
            RAG实例
          </h2>
          <button
            onClick={() => setShowCreateForm(!showCreateForm)}
            className="flex items-center space-x-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
          >
            <Plus className="w-4 h-4" />
            <span>创建RAG</span>
          </button>
        </div>

        {showCreateForm && (
          <form onSubmit={handleCreateRAG} className="mb-6 p-4 bg-gray-50 rounded-lg space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">名称</label>
              <input
                type="text"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                required
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">RAG类型</label>
              <select
                value={formData.rag_type}
                onChange={(e) => setFormData({ ...formData, rag_type: e.target.value as 'baseline' | 'api' })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              >
                <option value="baseline">📦 Baseline (本地向量存储)</option>
                <option value="api">🌐 API (外部服务)</option>
              </select>
            </div>

            {formData.rag_type === 'baseline' && (
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">LLM模型</label>
                <select
                  value={formData.llm_model_id}
                  onChange={(e) => setFormData({ ...formData, llm_model_id: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                  required
                >
                  {llmModels.length === 0 ? (
                    <option value="">请先在模型仓库中注册LLM模型</option>
                  ) : (
                    llmModels.map(model => (
                      <option key={model.model_id} value={model.model_id}>
                        {model.model_id} ({model.model_name})
                      </option>
                    ))
                  )}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Embedding模型</label>
                <select
                  value={formData.embedding_model_id}
                  onChange={(e) => setFormData({ ...formData, embedding_model_id: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                  required
                >
                  {embeddingModels.length === 0 ? (
                    <option value="">请先在模型仓库中注册Embedding模型</option>
                  ) : (
                    embeddingModels.map(model => (
                      <option key={model.model_id} value={model.model_id}>
                        {model.model_id} ({model.model_name})
                      </option>
                    ))
                  )}
                </select>
              </div>
              </div>
            )}

            {formData.rag_type === 'api' && (
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">检索API端点</label>
                  <input
                    type="url"
                    value={formData.retrieval_endpoint}
                    onChange={(e) => setFormData({ ...formData, retrieval_endpoint: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                    placeholder="https://api.example.com/retrieve"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">生成API端点</label>
                  <input
                    type="url"
                    value={formData.generation_endpoint}
                    onChange={(e) => setFormData({ ...formData, generation_endpoint: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                    placeholder="https://api.example.com/generate"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">API Key (可选)</label>
                  <input
                    type="password"
                    value={formData.api_key}
                    onChange={(e) => setFormData({ ...formData, api_key: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                    placeholder="your-api-key"
                  />
                </div>
              </div>
            )}

            {formData.rag_type === 'baseline' && (
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Top K</label>
                  <input
                    type="number"
                    value={formData.top_k}
                    onChange={(e) => setFormData({ ...formData, top_k: parseInt(e.target.value) })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                    min="1"
                    max="20"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Temperature</label>
                  <input
                    type="number"
                    step="0.1"
                    value={formData.temperature}
                    onChange={(e) => setFormData({ ...formData, temperature: parseFloat(e.target.value) })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                    min="0"
                    max="2"
                  />
                </div>
              </div>
            )}
            <div className="flex space-x-3">
              <button
                type="submit"
                disabled={loading}
                className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50"
              >
                {loading ? '创建中...' : '创建'}
              </button>
              <button
                type="button"
                onClick={() => setShowCreateForm(false)}
                className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300"
              >
                取消
              </button>
            </div>
          </form>
        )}

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {rags.length === 0 ? (
            <p className="text-sm text-gray-500 col-span-full text-center py-4">
              暂无RAG实例
            </p>
          ) : (
            rags.map((rag) => (
              <div key={rag.name} className="relative">
                <button
                  onClick={() => setSelectedRAG(rag)}
                  className={`
                    w-full p-3 rounded-lg border-2 text-sm font-medium transition-colors
                    ${selectedRAG?.name === rag.name
                      ? 'border-blue-500 bg-blue-50 text-blue-700'
                      : 'border-gray-200 hover:border-gray-300 text-gray-700'
                    }
                  `}
                >
                  <div className="text-left">
                    <div className="flex items-center space-x-2 mb-1">
                      <span className="font-semibold">{rag.name}</span>
                      <span className="text-xs px-2 py-0.5 rounded bg-gray-200">
                        {rag.rag_type === 'baseline' ? '📦 Baseline' : '🌐 API'}
                      </span>
                    </div>
                    {rag.rag_type === 'baseline' ? (
                      <>
                        <div className="text-xs text-gray-500 mt-1">
                          LLM: {rag.model_info?.llm_model_id || 'N/A'}
                        </div>
                        <div className="text-xs text-gray-500">
                          Embedding: {rag.model_info?.embedding_model_id || 'N/A'}
                        </div>
                        {/* Index Status */}
                        {rag.index_status && (
                          <div className="mt-2 pt-2 border-t border-gray-200">
                            {rag.index_status.has_index ? (
                              <div className="space-y-1">
                                <div className="flex items-center space-x-1 text-xs text-green-600">
                                  <Database className="w-3 h-3" />
                                  <span>{rag.index_status.document_count || 0} 文档</span>
                                </div>
                                {rag.index_status.total_size_bytes && (
                                  <div className="flex items-center space-x-1 text-xs text-gray-500">
                                    <HardDrive className="w-3 h-3" />
                                    <span>{formatBytes(rag.index_status.total_size_bytes)}</span>
                                  </div>
                                )}
                              </div>
                            ) : (
                              <div className="text-xs text-gray-400 italic">未索引</div>
                            )}
                          </div>
                        )}
                      </>
                    ) : (
                      <div className="text-xs text-gray-500 mt-1">
                        API端点已配置
                      </div>
                    )}
                  </div>
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    handleDeleteRAG(rag.name)
                  }}
                  className="absolute top-2 right-2 p-1 text-red-500 hover:bg-red-50 rounded"
                  title="删除"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      {selectedRAG && (
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold flex items-center">
              <Search className="w-5 h-5 mr-2" />
              查询测试 - {selectedRAG.name}
              <span className="ml-2 text-xs px-2 py-0.5 rounded bg-gray-200">
                {selectedRAG.rag_type === 'baseline' ? '📦 Baseline' : '🌐 API'}
              </span>
            </h3>
            {selectedRAG.rag_type === 'baseline' && (
              <button
                onClick={handleIndexDocuments}
                className="flex items-center space-x-2 px-3 py-1.5 bg-green-500 text-white text-sm rounded-lg hover:bg-green-600"
              >
                <Upload className="w-4 h-4" />
                <span>索引文档</span>
              </button>
            )}
          </div>
          
          {/* Index Status Display for Baseline RAG */}
          {selectedRAG.rag_type === 'baseline' && selectedRAG.index_status && (
            <div className="mb-4 p-4 bg-gray-50 rounded-lg">
              <h4 className="text-sm font-semibold text-gray-700 mb-3 flex items-center">
                <Database className="w-4 h-4 mr-2" />
                索引状态
              </h4>
              {selectedRAG.index_status.has_index ? (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <div className="text-gray-500 text-xs mb-1">文档数量</div>
                    <div className="font-semibold text-gray-900">
                      {selectedRAG.index_status.document_count?.toLocaleString() || 0}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-500 text-xs mb-1">向量维度</div>
                    <div className="font-semibold text-gray-900">
                      {selectedRAG.index_status.embedding_dimension || 'N/A'}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-500 text-xs mb-1">索引大小</div>
                    <div className="font-semibold text-gray-900">
                      {formatBytes(selectedRAG.index_status.total_size_bytes)}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-500 text-xs mb-1 flex items-center">
                      <Clock className="w-3 h-3 mr-1" />
                      更新时间
                    </div>
                    <div className="font-semibold text-gray-900 text-xs">
                      {formatDate(selectedRAG.index_status.updated_at)}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-sm text-gray-500 italic">
                  该 RAG 实例尚未索引任何文档。点击"索引文档"按钮开始索引。
                </div>
              )}
            </div>
          )}
          <form onSubmit={handleQuery} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">查询内容</label>
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                rows={3}
                placeholder="输入你的问题..."
                required
              />
            </div>
            <button
              type="submit"
              disabled={loading}
              className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 flex items-center space-x-2"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>查询中...</span>
                </>
              ) : (
                <>
                  <Search className="w-4 h-4" />
                  <span>查询</span>
                </>
              )}
            </button>
          </form>

          {queryResult && (
            <div className="mt-6 space-y-4">
              <div className="border border-gray-200 rounded-lg p-4">
                <h4 className="text-sm font-semibold text-gray-700 mb-2">答案</h4>
                <p className="text-gray-900">{queryResult.answer}</p>
              </div>
              <div className="border border-gray-200 rounded-lg p-4">
                <h4 className="text-sm font-semibold text-gray-700 mb-2">检索到的上下文</h4>
                <div className="space-y-2">
                  {queryResult.contexts.map((ctx: string, idx: number) => (
                    <div key={idx} className="bg-gray-50 p-3 rounded text-sm">
                      <div className="flex justify-between items-start mb-1">
                        <span className="text-xs font-medium text-gray-500">上下文 {idx + 1}</span>
                        {queryResult.scores && (
                          <span className="text-xs text-blue-600">
                            相似度: {queryResult.scores[idx]?.toFixed(3)}
                          </span>
                        )}
                      </div>
                      <p className="text-gray-700">{ctx}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* 索引文档对话框 */}
      {showIndexDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-4xl w-full mx-4 max-h-[90vh] overflow-y-auto">
            <h3 className="text-lg font-semibold mb-4">索引文档 - {selectedRAG?.name}</h3>
            
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    选择数据集
                  </label>
                  <select
                    value={indexData.dataset_name}
                    onChange={(e) => {
                      setIndexData({ ...indexData, dataset_name: e.target.value })
                      setShowDocumentPreview(false)
                      setCorpusDocuments([])
                    }}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="">选择数据集</option>
                    {datasets.map((ds) => (
                      <option key={ds} value={ds}>{ds}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    子集 (可选)
                  </label>
                  <input
                    type="text"
                    value={indexData.subset}
                    onChange={(e) => {
                      setIndexData({ ...indexData, subset: e.target.value })
                      setShowDocumentPreview(false)
                      setCorpusDocuments([])
                    }}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                    placeholder="例如: dev, test"
                  />
                </div>
              </div>

              {!showDocumentPreview && (
                <button
                  onClick={handleLoadCorpus}
                  disabled={loadingCorpus || !indexData.dataset_name}
                  className="w-full px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 disabled:opacity-50"
                >
                  {loadingCorpus ? '加载中...' : '加载并预览文档'}
                </button>
              )}

              {showDocumentPreview && corpusDocuments.length > 0 && (
                <div className="border rounded-lg p-4 bg-gray-50">
                  <div className="flex justify-between items-center mb-3">
                    <div className="flex items-center space-x-4">
                      <h4 className="font-semibold">文档列表 ({corpusDocuments.length})</h4>
                      <span className="text-sm text-gray-600">
                        已选择: {selectedDocIds.size} / {corpusDocuments.length}
                      </span>
                    </div>
                    <button
                      onClick={handleToggleAll}
                      className="text-sm px-3 py-1 bg-blue-100 text-blue-700 rounded hover:bg-blue-200"
                    >
                      {selectedDocIds.size === corpusDocuments.length ? '取消全选' : '全选'}
                    </button>
                  </div>

                  <div className="max-h-96 overflow-y-auto space-y-2">
                    {corpusDocuments.map((doc) => (
                      <div
                        key={doc.id}
                        className={`p-3 rounded border-2 cursor-pointer transition-colors ${
                          selectedDocIds.has(doc.id)
                            ? 'border-blue-500 bg-blue-50'
                            : 'border-gray-200 bg-white hover:border-gray-300'
                        }`}
                        onClick={() => handleToggleDocument(doc.id)}
                      >
                        <div className="flex items-start space-x-3">
                          <input
                            type="checkbox"
                            checked={selectedDocIds.has(doc.id)}
                            onChange={() => handleToggleDocument(doc.id)}
                            className="mt-1"
                            onClick={(e) => e.stopPropagation()}
                          />
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center space-x-2 mb-1">
                              <span className="text-xs font-mono text-gray-500">{doc.id}</span>
                              <span className="text-xs text-gray-400">({doc.length} 字符)</span>
                            </div>
                            <p className="text-sm text-gray-700 line-clamp-3">
                              {doc.content}
                            </p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="flex space-x-3">
                {showDocumentPreview && (
                  <button
                    onClick={handleIndexFromDataset}
                    disabled={loading || selectedDocIds.size === 0}
                    className="flex-1 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50"
                  >
                    {loading ? '索引中...' : `索引选中的 ${selectedDocIds.size} 个文档`}
                  </button>
                )}
                <button
                  onClick={() => {
                    setShowIndexDialog(false)
                    setShowDocumentPreview(false)
                    setIndexData({ dataset_name: '', subset: '' })
                    setCorpusDocuments([])
                    setSelectedDocIds(new Set())
                  }}
                  className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300"
                >
                  取消
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
