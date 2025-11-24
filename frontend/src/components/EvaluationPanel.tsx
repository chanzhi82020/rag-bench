import { useState, useEffect } from 'react'
import { listDatasets, listRAGs, startEvaluation, getEvaluationStatus, listEvaluationTasks, listModels, deleteEvaluationTask, sampleDataset, ModelInfo, SampleSelection } from '../api/client'
import { Play, Loader2, CheckCircle, XCircle, Clock, Users, Trash2, Eye } from 'lucide-react'

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
  const [sampleSelection, setSampleSelection] = useState<SampleSelection>({
    strategy: 'random',
    sample_size: 10,
  })
  const [tasks, setTasks] = useState<any[]>([])
  const [selectedTask, setSelectedTask] = useState<any>(null)
  const [polling, setPolling] = useState(false)
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [deleteConfirmTask, setDeleteConfirmTask] = useState<any>(null)
  const [deleting, setDeleting] = useState(false)
  
  // Sample preview states
  const [datasetSamples, setDatasetSamples] = useState<any[]>([])
  const [selectedSampleIds, setSelectedSampleIds] = useState<Set<string>>(new Set())
  const [loadingSamples, setLoadingSamples] = useState(false)
  const [showSamplePreview, setShowSamplePreview] = useState(false)

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
      }, 5000)

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

  const handleLoadSamples = async () => {
    if (!formData.dataset_name) {
      alert('请先选择数据集')
      return
    }
    
    setLoadingSamples(true)
    try {
      const result = await sampleDataset({
        name: formData.dataset_name,
        subset: undefined
      }, 100) // Load up to 100 samples for preview
      
      setDatasetSamples(result.data.samples || [])
      setShowSamplePreview(true)
      
      // Default: select all - use record IDs from API response
      const allIds = new Set<string>(
        (result.data.samples || []).map((sample: any) => sample.id)
      )
      setSelectedSampleIds(allIds)
    } catch (error: any) {
      alert('加载样本失败: ' + (error.response?.data?.detail || error.message))
    } finally {
      setLoadingSamples(false)
    }
  }

  const handleToggleSample = (sampleId: string) => {
    const newSelected = new Set(selectedSampleIds)
    if (newSelected.has(sampleId)) {
      newSelected.delete(sampleId)
    } else {
      newSelected.add(sampleId)
    }
    setSelectedSampleIds(newSelected)
  }

  const handleToggleAllSamples = () => {
    if (selectedSampleIds.size === datasetSamples.length) {
      setSelectedSampleIds(new Set<string>())
    } else {
      const allIds = new Set<string>(
        datasetSamples.map((sample: any) => sample.id)
      )
      setSelectedSampleIds(allIds)
    }
  }

  const handleStartEvaluation = async (e: React.FormEvent) => {
    e.preventDefault()
    
    try {
      // Prepare sample selection based on strategy
      let finalSampleSelection: SampleSelection | undefined = undefined
      
      if (sampleSelection.strategy === 'specific_ids') {
        if (selectedSampleIds.size === 0) {
          alert('请至少选择一个样本')
          return
        }
        finalSampleSelection = {
          strategy: 'specific_ids',
          sample_ids: Array.from(selectedSampleIds),
        }
      } else if (sampleSelection.strategy === 'random') {
        finalSampleSelection = {
          strategy: 'random',
          sample_size: sampleSelection.sample_size || 10,
          random_seed: sampleSelection.random_seed,
        }
      } else if (sampleSelection.strategy === 'all') {
        finalSampleSelection = {
          strategy: 'all',
        }
      }
      
      const response = await startEvaluation({
        dataset_name: formData.dataset_name,
        rag_name: formData.rag_name,
        eval_type: formData.eval_type,
        model_info: {
          llm_model_id: formData.llm_model_id,
          embedding_model_id: formData.embedding_model_id,
        },
        sample_selection: finalSampleSelection,
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

  const handleDeleteTask = async (task: any) => {
    setDeleteConfirmTask(task)
  }

  const confirmDeleteTask = async () => {
    if (!deleteConfirmTask) return
    
    setDeleting(true)
    try {
      await deleteEvaluationTask(deleteConfirmTask.task_id)
      
      // Remove from tasks list
      setTasks(prev => prev.filter(t => t.task_id !== deleteConfirmTask.task_id))
      
      // Clear selected task if it was deleted
      if (selectedTask?.task_id === deleteConfirmTask.task_id) {
        setSelectedTask(null)
        setPolling(false)
      }
      
      setDeleteConfirmTask(null)
    } catch (error: any) {
      const errorDetail = error.response?.data?.detail
      let errorMessage = '删除任务失败'
      
      if (typeof errorDetail === 'object' && errorDetail.error) {
        errorMessage = `${errorDetail.error}: ${errorDetail.details || ''}`
      } else if (typeof errorDetail === 'string') {
        errorMessage = errorDetail
      } else if (error.message) {
        errorMessage = error.message
      }
      
      alert(errorMessage)
    } finally {
      setDeleting(false)
    }
  }

  const cancelDeleteTask = () => {
    setDeleteConfirmTask(null)
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

              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">评测类型</label>
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

              {/* Sample Selection Section */}
              <div className="border-t pt-2 mt-2">
                <label className="block text-xs font-medium text-gray-700 mb-2">样本选择策略</label>
                <div className="space-y-2">
                  <div className="flex items-center space-x-2">
                    <input
                      type="radio"
                      id="strategy-all"
                      name="strategy"
                      value="all"
                      checked={sampleSelection.strategy === 'all'}
                      onChange={(e) => setSampleSelection({ ...sampleSelection, strategy: e.target.value as any })}
                      className="w-4 h-4"
                    />
                    <label htmlFor="strategy-all" className="text-sm text-gray-700">全部样本</label>
                  </div>
                  
                  <div className="space-y-1">
                    <div className="flex items-center space-x-2">
                      <input
                        type="radio"
                        id="strategy-random"
                        name="strategy"
                        value="random"
                        checked={sampleSelection.strategy === 'random'}
                        onChange={(e) => setSampleSelection({ ...sampleSelection, strategy: e.target.value as any })}
                        className="w-4 h-4"
                      />
                      <label htmlFor="strategy-random" className="text-sm text-gray-700">随机采样</label>
                    </div>
                    {sampleSelection.strategy === 'random' && (
                      <div className="ml-6 grid grid-cols-2 gap-2">
                        <div>
                          <label className="block text-xs text-gray-600 mb-1">样本数</label>
                          <input
                            type="number"
                            value={sampleSelection.sample_size || 10}
                            onChange={(e) => setSampleSelection({ ...sampleSelection, sample_size: parseInt(e.target.value) })}
                            className="w-full px-2 py-1 text-sm border rounded"
                            min="1"
                            max="1000"
                          />
                        </div>
                        <div>
                          <label className="block text-xs text-gray-600 mb-1">随机种子（可选）</label>
                          <input
                            type="number"
                            value={sampleSelection.random_seed || ''}
                            onChange={(e) => setSampleSelection({ ...sampleSelection, random_seed: e.target.value ? parseInt(e.target.value) : undefined })}
                            className="w-full px-2 py-1 text-sm border rounded"
                            placeholder="留空随机"
                          />
                        </div>
                      </div>
                    )}
                  </div>
                  
                  <div className="space-y-1">
                    <div className="flex items-center space-x-2">
                      <input
                        type="radio"
                        id="strategy-specific"
                        name="strategy"
                        value="specific_ids"
                        checked={sampleSelection.strategy === 'specific_ids'}
                        onChange={(e) => {
                          setSampleSelection({ ...sampleSelection, strategy: e.target.value as any })
                          setShowSamplePreview(false)
                          setDatasetSamples([])
                        }}
                        className="w-4 h-4"
                      />
                      <label htmlFor="strategy-specific" className="text-sm text-gray-700">指定样本</label>
                    </div>
                    {sampleSelection.strategy === 'specific_ids' && (
                      <div className="ml-6 space-y-2">
                        {!showSamplePreview && (
                          <button
                            type="button"
                            onClick={handleLoadSamples}
                            disabled={loadingSamples || !formData.dataset_name}
                            className="w-full px-3 py-2 bg-green-500 text-white text-sm rounded hover:bg-green-600 disabled:opacity-50 flex items-center justify-center space-x-2"
                          >
                            {loadingSamples ? (
                              <>
                                <Loader2 className="w-4 h-4 animate-spin" />
                                <span>加载中...</span>
                              </>
                            ) : (
                              <>
                                <Eye className="w-4 h-4" />
                                <span>加载并预览样本</span>
                              </>
                            )}
                          </button>
                        )}
                        
                        {showSamplePreview && datasetSamples.length > 0 && (
                          <div className="border rounded p-3 bg-gray-50">
                            <div className="flex justify-between items-center mb-2">
                              <div className="flex items-center space-x-3">
                                <span className="text-xs font-semibold">样本列表 ({datasetSamples.length})</span>
                                <span className="text-xs text-gray-600">
                                  已选择: {selectedSampleIds.size} / {datasetSamples.length}
                                </span>
                              </div>
                              <button
                                type="button"
                                onClick={handleToggleAllSamples}
                                className="text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded hover:bg-blue-200"
                              >
                                {selectedSampleIds.size === datasetSamples.length ? '取消全选' : '全选'}
                              </button>
                            </div>
                            
                            <div className="max-h-64 overflow-y-auto space-y-1.5">
                              {datasetSamples.map((sample: any) => {
                                const sampleId = sample.id
                                const displayId = sampleId.length > 8 ? `${sampleId.substring(0, 8)}...` : sampleId
                                return (
                                  <div
                                    key={sampleId}
                                    className={`p-2 rounded border cursor-pointer transition-colors ${
                                      selectedSampleIds.has(sampleId)
                                        ? 'border-blue-500 bg-blue-50'
                                        : 'border-gray-200 bg-white hover:border-gray-300'
                                    }`}
                                    onClick={() => handleToggleSample(sampleId)}
                                  >
                                    <div className="flex items-start space-x-2">
                                      <input
                                        type="checkbox"
                                        checked={selectedSampleIds.has(sampleId)}
                                        onChange={() => handleToggleSample(sampleId)}
                                        className="mt-0.5"
                                        onClick={(e) => e.stopPropagation()}
                                      />
                                      <div className="flex-1 min-w-0">
                                        <div 
                                          className="text-xs font-mono text-gray-500 mb-0.5"
                                          title={sampleId}
                                        >
                                          {displayId}
                                        </div>
                                        <p className="text-xs text-gray-700 line-clamp-2">
                                          {sample.user_input || sample.question || sample.query || '无问题内容'}
                                        </p>
                                      </div>
                                    </div>
                                  </div>
                                )
                              })}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
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
                <div
                  key={task.task_id}
                  className={`relative rounded-lg border-2 transition-colors ${
                    selectedTask?.task_id === task.task_id
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <button
                    onClick={() => handleSelectTask(task)}
                    className="w-full text-left p-3"
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
                      {task.sample_info && (
                        <div className="flex items-center space-x-1 mt-1 text-gray-500">
                          <Users className="w-3 h-3" />
                          <span>
                            {task.sample_info.selected_samples} 样本
                            {task.sample_info.selection_strategy === 'specific_ids' && ' (指定)'}
                            {task.sample_info.selection_strategy === 'random' && ' (随机)'}
                            {task.sample_info.selection_strategy === 'all' && ' (全部)'}
                          </span>
                        </div>
                      )}
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
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      handleDeleteTask(task)
                    }}
                    className="absolute top-2 right-2 p-1.5 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded transition-colors"
                    title="删除任务"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* 右侧：任务详情 */}
      <div className="lg:col-span-2">
        {selectedTask ? (
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">任务详情</h3>
              <button
                onClick={() => handleDeleteTask(selectedTask)}
                className="flex items-center space-x-2 px-3 py-2 text-sm text-red-600 hover:bg-red-50 rounded-lg transition-colors"
              >
                <Trash2 className="w-4 h-4" />
                <span>删除任务</span>
              </button>
            </div>
            
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

              {/* Sample Information */}
              {selectedTask.sample_info && (
                <div className="border border-gray-200 rounded-lg p-4 bg-gray-50">
                  <h4 className="font-semibold text-gray-900 mb-3 flex items-center">
                    <Users className="w-4 h-4 mr-2" />
                    样本信息
                  </h4>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                    <div>
                      <div className="text-xs text-gray-500">选择策略</div>
                      <div className="text-sm font-medium">
                        {selectedTask.sample_info.selection_strategy === 'specific_ids' && '指定样本ID'}
                        {selectedTask.sample_info.selection_strategy === 'random' && '随机采样'}
                        {selectedTask.sample_info.selection_strategy === 'all' && '全部样本'}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-500">数据集总样本数</div>
                      <div className="text-sm font-medium">{selectedTask.sample_info.total_samples}</div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-500">选中样本数</div>
                      <div className="text-sm font-medium">{selectedTask.sample_info.selected_samples}</div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-500">已完成</div>
                      <div className="text-sm font-medium text-green-600">
                        {selectedTask.sample_info.completed_samples}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-500">失败</div>
                      <div className="text-sm font-medium text-red-600">
                        {selectedTask.sample_info.failed_samples}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-500">进行中</div>
                      <div className="text-sm font-medium text-blue-600">
                        {selectedTask.sample_info.selected_samples - 
                         selectedTask.sample_info.completed_samples - 
                         selectedTask.sample_info.failed_samples}
                      </div>
                    </div>
                  </div>
                  {selectedTask.sample_info.sample_ids && selectedTask.sample_info.sample_ids.length > 0 && (
                    <div className="mt-3">
                      <div className="text-xs text-gray-500 mb-1">样本ID列表</div>
                      <div className="text-xs bg-white p-2 rounded border max-h-24 overflow-y-auto font-mono">
                        {selectedTask.sample_info.sample_ids.join(', ')}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Checkpoint Information */}
              {selectedTask.checkpoint_info && selectedTask.checkpoint_info.has_checkpoint && (
                <div className="border border-blue-200 rounded-lg p-4 bg-blue-50">
                  <h4 className="font-semibold text-blue-900 mb-3">检查点信息</h4>
                  <div className="space-y-2 text-sm">
                    <div>
                      <span className="text-blue-700">已完成阶段: </span>
                      <span className="text-blue-900 font-medium">
                        {selectedTask.checkpoint_info.completed_stages.join(', ')}
                      </span>
                    </div>
                    {selectedTask.checkpoint_info.current_stage && (
                      <div>
                        <span className="text-blue-700">当前阶段: </span>
                        <span className="text-blue-900 font-medium">
                          {selectedTask.checkpoint_info.current_stage}
                        </span>
                      </div>
                    )}
                    {selectedTask.checkpoint_info.last_checkpoint_at && (
                      <div>
                        <span className="text-blue-700">最后保存: </span>
                        <span className="text-blue-900 font-medium">
                          {new Date(selectedTask.checkpoint_info.last_checkpoint_at).toLocaleString('zh-CN')}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              )}

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

      {/* Delete Confirmation Dialog */}
      {deleteConfirmTask && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl p-6 max-w-md w-full mx-4">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              <Trash2 className="w-5 h-5 mr-2 text-red-600" />
              确认删除任务
            </h3>
            <p className="text-gray-600 mb-4">
              确定要删除此评测任务吗？此操作无法撤销。
            </p>
            <div className="bg-gray-50 rounded p-3 mb-4 text-sm">
              <div className="font-medium text-gray-900 mb-1">任务信息：</div>
              <div className="text-gray-600">
                <div>任务ID: {deleteConfirmTask.task_id}</div>
                <div>数据集: {deleteConfirmTask.request?.dataset_name}</div>
                <div>RAG系统: {deleteConfirmTask.request?.rag_name}</div>
                <div>状态: {getStatusText(deleteConfirmTask.status)}</div>
              </div>
            </div>
            {deleteConfirmTask.status === 'running' && (
              <div className="bg-yellow-50 border border-yellow-200 rounded p-3 mb-4 text-sm text-yellow-800">
                <strong>注意：</strong> 此任务正在运行中。删除操作不会停止正在运行的进程，仅删除任务记录。
              </div>
            )}
            <div className="flex space-x-3">
              <button
                onClick={cancelDeleteTask}
                disabled={deleting}
                className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 disabled:opacity-50"
              >
                取消
              </button>
              <button
                onClick={confirmDeleteTask}
                disabled={deleting}
                className="flex-1 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 flex items-center justify-center"
              >
                {deleting ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    删除中...
                  </>
                ) : (
                  '确认删除'
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
