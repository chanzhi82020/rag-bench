import { useState, useEffect } from 'react'
import { listDatasets, listRAGs, startEvaluation, getEvaluationStatus, listEvaluationTasks, listModels, ModelInfo } from '../api/client'
import { Play, Loader2, CheckCircle, XCircle, Clock } from 'lucide-react'

interface RAGInstance {
  name: string
  model_info: any
  rag_config: any
}

export default function EvaluationPanel() {
  const [datasets, setDatasets] = useState<string[]>([])
  const [rags, setRags] = useState<RAGInstance[]>([])
  const [llmModels, setLlmModels] = useState<ModelInfo[]>([])
  const [embeddingModels, setEmbeddingModels] = useState<ModelInfo[]>([])
  const [formData, setFormData] = useState({
    dataset_name: '',
    rag_name: '',
    eval_type: 'e2e' as 'e2e' | 'retrieval' | 'generation',
    sample_size: 10,
    llm_model_id: '',
    embedding_model_id: '',
  })
  const [tasks, setTasks] = useState<any[]>([])
  const [selectedTask, setSelectedTask] = useState<any>(null)
  const [polling, setPolling] = useState(false)
  const [showCreateForm, setShowCreateForm] = useState(false)

  useEffect(() => {
    loadOptions()
    loadTasks()
  }, [])

  useEffect(() => {
    if (polling && selectedTask) {
      const interval = setInterval(async () => {
        try {
          const response = await getEvaluationStatus(selectedTask.task_id)
          setSelectedTask(response.data)
          setTasks(prev => prev.map(t => 
            t.task_id === response.data.task_id ? response.data : t
          ))
          
          if (response.data.status === 'completed' || response.data.status === 'failed') {
            setPolling(false)
          }
        } catch (error) {
          console.error('获取任务状态失败:', error)
          setPolling(false)
        }
      }, 2000)

      return () => clearInterval(interval)
    }
  }, [polling, selectedTask])

  const loadOptions = async () => {
    try {
      const [datasetsRes, ragsRes, modelsRes] = await Promise.all([
        listDatasets(),
        listRAGs(),
        listModels()
      ])
      setDatasets(datasetsRes.data)
      setRags(ragsRes.data.rags)
      setLlmModels(modelsRes.data.llm_models)
      setEmbeddingModels(modelsRes.data.embedding_models)
      
      if (modelsRes.data.llm_models.length > 0) {
        setFormData(prev => ({ ...prev, llm_model_id: modelsRes.data.llm_models[0].model_id }))
      }
      if (modelsRes.data.embedding_models.length > 0) {
        setFormData(prev => ({ ...prev, embedding_model_id: modelsRes.data.embedding_models[0].model_id }))
      }
    } catch (error) {
      console.error('加载选项失败:', error)
    }
  }

  const loadTasks = async () => {
    try {
      const response = await listEvaluationTasks()
      setTasks(response.data.tasks || [])
    } catch (error) {
      console.error('加载任务列表失败:', error)
    }
  }

  const handleStartEvaluation = async (e: React.FormEvent) => {
    e.preventDefault()
    
    try {
      const response = await startEvaluation({
        dataset_name: formData.dataset_name,
        rag_name: formData.rag_name,
        eval_type: formData.eval_type,
        sample_size: formData.sample_size,
        model_info: {
          llm_model_id: formData.llm_model_id,
          embedding_model_id: formData.embedding_model_id,
        }
      })
      
      const newTask = {
        task_id: response.data.task_id,
        status: 'pending',
        progress: 0,
        created_at: new Date().toISOString(),
        request: {
          dataset_name: formData.dataset_name,
          rag_name: formData.rag_name,
          eval_type: formData.eval_type,
        }
      }
      
      setTasks(prev => [newTask, ...prev])
      setSelectedTask(newTask)
      setPolling(true)
      setShowCreateForm(false)
    } catch (error: any) {
      alert('启动评测失败: ' + (error.response?.data?.detail || error.message))
    }
  }

  const handleSelectTask = async (task: any) => {
    try {
      const response = await getEvaluationStatus(task.task_id)
      setSelectedTask(response.data)
      
      if (response.data.status === 'running' || response.data.status === 'pending') {
        setPolling(true)
      } else {
        setPolling(false)
      }
    } catch (error: any) {
      alert('加载任务详情失败: ' + (error.response?.data?.detail || error.message))
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-500" />
      case 'failed':
        return <XCircle className="w-4 h-4 text-red-500" />
      case 'running':
        return <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />
      default:
        return <Clock className="w-4 h-4 text-gray-400" />
    }
  }

  const getStatusText = (status: string) => {
    switch (status) {
      case 'completed': return '已完成'
      case 'failed': return '失败'
      case 'running': return '运行中'
      default: return '等待中'
    }
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* 左侧：任务列表 */}
      <div className="lg:col-span-1">
        <div className="bg-white rounded-lg shadow p-4">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-lg font-semibold">评测任务</h2>
            <button
              onClick={() => setShowCreateForm(!showCreateForm)}
              className="px-3 py-1.5 bg-blue-500 text-white text-sm rounded-lg hover:bg-blue-600"
            >
              {showCreateForm ? '取消' : '新建'}
            </button>
          </div>

          {showCreateForm && (
            <form onSubmit={handleStartEvaluation} className="mb-4 p-3 bg-gray-50 rounded-lg space-y-2">
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">数据集</label>
                <select
                  value={formData.dataset_name}
                  onChange={(e) => setFormData({ ...formData, dataset_name: e.target.value })}
                  className="w-full px-2 py-1.5 text-sm border rounded"
                  required
                >
                  <option value="">选择数据集</option>
                  {datasets.map((ds) => (
                    <option key={ds} value={ds}>{ds}</option>
                  ))}
                </select>
              </div>
              
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">RAG系统</label>
                <select
                  value={formData.rag_name}
                  onChange={(e) => setFormData({ ...formData, rag_name: e.target.value })}
                  className="w-full px-2 py-1.5 text-sm border rounded"
                  required
                >
                  <option value="">选择RAG</option>
                  {rags.map((rag) => (
                    <option key={rag.name} value={rag.name}>{rag.name}</option>
                  ))}
                </select>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">类型</label>
                  <select
                    value={formData.eval_type}
                    onChange={(e) => setFormData({ ...formData, eval_type: e.target.value as any })}
                    className="w-full px-2 py-1.5 text-sm border rounded"
                  >
                    <option value="e2e">端到端</option>
                    <option value="retrieval">检索</option>
                    <option value="generation">生成</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">样本数</label>
                  <input
                    type="number"
                    value={formData.sample_size}
                    onChange={(e) => setFormData({ ...formData, sample_size: parseInt(e.target.value) })}
                    className="w-full px-2 py-1.5 text-sm border rounded"
                    min="1"
                    max="100"
                  />
                </div>
              </div>

              <button
                type="submit"
                className="w-full flex items-center justify-center space-x-2 px-3 py-2 bg-blue-500 text-white text-sm rounded hover:bg-blue-600"
              >
                <Play className="w-4 h-4" />
                <span>启动评测</span>
              </button>
            </form>
          )}

          {/* 任务列表 */}
          <div className="space-y-2 max-h-[600px] overflow-y-auto">
            {tasks.length === 0 ? (
              <p className="text-sm text-gray-500 text-center py-8">暂无评测任务</p>
            ) : (
              tasks.map((task) => (
                <button
                  key={task.task_id}
                  onClick={() => handleSelectTask(task)}
                  className={`w-full text-left p-3 rounded-lg border-2 transition-colors ${
                    selectedTask?.task_id === task.task_id
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <div className="flex items-start justify-between mb-1">
                    <div className="flex items-center space-x-2">
                      {getStatusIcon(task.status)}
                      <span className="text-sm font-medium">{getStatusText(task.status)}</span>
                    </div>
                    <span className="text-xs text-gray-500">
                      {Math.round((task.progress || 0) * 100)}%
                    </span>
                  </div>
                  <div className="text-xs text-gray-600">
                    <div>{task.request?.dataset_name} / {task.request?.rag_name}</div>
                    <div className="text-gray-400 mt-1">
                      {new Date(task.created_at).toLocaleString('zh-CN', {
                        month: '2-digit',
                        day: '2-digit',
                        hour: '2-digit',
                        minute: '2-digit'
                      })}
                    </div>
                  </div>
                </button>
              ))
            )}
          </div>
        </div>
      </div>

      {/* 右侧：任务详情 */}
      <div className="lg:col-span-2">
        {selectedTask ? (
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold mb-4">任务详情</h3>
            
            <div className="space-y-4">
              {/* 状态信息 */}
              <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  {getStatusIcon(selectedTask.status)}
                  <div>
                    <div className="font-medium">{getStatusText(selectedTask.status)}</div>
                    {selectedTask.current_stage && (
                      <div className="text-sm text-gray-600">{selectedTask.current_stage}</div>
                    )}
                  </div>
                </div>
                <div className="text-2xl font-bold text-blue-600">
                  {Math.round((selectedTask.progress || 0) * 100)}%
                </div>
              </div>

              {/* 进度条 */}
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-500 h-2 rounded-full transition-all"
                  style={{ width: `${(selectedTask.progress || 0) * 100}%` }}
                />
              </div>

              {/* 错误信息 */}
              {selectedTask.error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                  <h4 className="font-semibold text-red-900 mb-2">错误信息</h4>
                  <p className="text-sm text-red-800 whitespace-pre-wrap font-mono">
                    {selectedTask.error}
                  </p>
                </div>
              )}

              {/* 评测结果 */}
              {selectedTask.result && (
                <div className="space-y-4">
                  <div className="border rounded-lg p-4">
                    <h4 className="font-semibold mb-3">评测指标（平均值）</h4>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                      {Object.entries(selectedTask.result.metrics).map(([key, value]: [string, any]) => (
                        <div key={key} className="bg-gray-50 p-3 rounded">
                          <div className="text-xs text-gray-500 uppercase">{key}</div>
                          <div className="text-lg font-bold">
                            {typeof value === 'number' ? value.toFixed(4) : value}
                          </div>
                        </div>
                      ))}
                    </div>
                    <div className="mt-3 text-sm text-gray-600">
                      样本数: {selectedTask.result.sample_count} | 
                      评测类型: {selectedTask.result.eval_type}
                    </div>
                  </div>

                  {/* 详细结果 */}
                  {selectedTask.result.detailed_results && selectedTask.result.detailed_results.length > 0 && (
                    <div className="border rounded-lg p-4">
                      <h4 className="font-semibold mb-3">详细评测结果 ({selectedTask.result.detailed_results.length} 个样本)</h4>
                      <div className="space-y-3">
                        {selectedTask.result.detailed_results.map((row: any, idx: number) => (
                          <details key={idx} className="group border rounded-lg">
                            <summary className="cursor-pointer list-none p-3 hover:bg-gray-50 flex items-center justify-between">
                              <div className="flex items-center space-x-4">
                                <span className="font-medium text-gray-700">样本 #{idx + 1}</span>
                                <div className="flex space-x-2 text-xs">
                                  {Object.keys(selectedTask.result.metrics).map(metric => (
                                    <span key={metric} className="px-2 py-1 bg-gray-100 rounded">
                                      {metric}: {typeof row[metric] === 'number' ? row[metric].toFixed(3) : '-'}
                                    </span>
                                  ))}
                                </div>
                              </div>
                              <span className="text-gray-400 group-open:rotate-180 transition-transform">▼</span>
                            </summary>
                            
                            <div className="p-4 bg-gray-50 space-y-4 text-sm">
                              {/* 用户输入 */}
                              {row.user_input && (
                                <div>
                                  <div className="font-semibold text-gray-700 mb-1">❓ 用户问题</div>
                                  <div className="bg-white p-3 rounded border">
                                    {row.user_input}
                                  </div>
                                </div>
                              )}
                              
                              {/* 检索上下文 */}
                              {row.retrieved_contexts && row.retrieved_contexts.length > 0 && (
                                <div>
                                  <div className="font-semibold text-gray-700 mb-1">
                                    🔍 检索到的上下文 ({row.retrieved_contexts.length})
                                  </div>
                                  <div className="space-y-2">
                                    {row.retrieved_contexts.map((ctx: string, ctxIdx: number) => (
                                      <div key={ctxIdx} className="bg-white p-3 rounded border">
                                        <div className="text-xs text-gray-500 mb-1">上下文 {ctxIdx + 1}</div>
                                        <div className="text-gray-800">{ctx}</div>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}
                              
                              {/* 参考上下文 */}
                              {row.reference_contexts && row.reference_contexts.length > 0 && (
                                <div>
                                  <div className="font-semibold text-gray-700 mb-1">
                                    📚 参考上下文 ({row.reference_contexts.length})
                                  </div>
                                  <div className="space-y-2">
                                    {row.reference_contexts.map((ctx: string, ctxIdx: number) => (
                                      <div key={ctxIdx} className="bg-blue-50 p-3 rounded border border-blue-200">
                                        <div className="text-xs text-blue-600 mb-1">
                                          参考 {ctxIdx + 1}
                                          {row.reference_context_ids && row.reference_context_ids[ctxIdx] && (
                                            <span className="ml-2 text-gray-500">ID: {row.reference_context_ids[ctxIdx]}</span>
                                          )}
                                        </div>
                                        <div className="text-gray-800">{ctx}</div>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}
                              
                              {/* 系统回答 */}
                              {row.response && (
                                <div>
                                  <div className="font-semibold text-gray-700 mb-1">💬 系统回答</div>
                                  <div className="bg-green-50 p-3 rounded border border-green-200">
                                    {row.response}
                                  </div>
                                </div>
                              )}
                              
                              {/* 参考答案 */}
                              {row.reference && (
                                <div>
                                  <div className="font-semibold text-gray-700 mb-1">✅ 参考答案</div>
                                  <div className="bg-yellow-50 p-3 rounded border border-yellow-200">
                                    {row.reference}
                                  </div>
                                </div>
                              )}
                              
                              {/* 评测指标 */}
                              <div>
                                <div className="font-semibold text-gray-700 mb-2">📊 评测指标</div>
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                                  {Object.keys(selectedTask.result.metrics).map(metric => (
                                    <div key={metric} className="bg-white p-2 rounded border">
                                      <div className="text-xs text-gray-500">{metric}</div>
                                      <div className="font-bold text-lg">
                                        {typeof row[metric] === 'number' ? row[metric].toFixed(4) : '-'}
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            </div>
                          </details>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="bg-white rounded-lg shadow p-6">
            <p className="text-gray-500 text-center py-12">
              请从左侧选择一个任务查看详情
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
