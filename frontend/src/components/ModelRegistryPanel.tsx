import { useState, useEffect } from 'react'
import { registerModel, listModels, deleteModel, updateModel, ModelInfo } from '../api/client'
import { Database, Plus, Edit2, Trash2, Eye, EyeOff } from 'lucide-react'

export default function ModelRegistryPanel() {
  const [llmModels, setLlmModels] = useState<ModelInfo[]>([])
  const [embeddingModels, setEmbeddingModels] = useState<ModelInfo[]>([])
  const [showForm, setShowForm] = useState(false)
  const [editingModel, setEditingModel] = useState<ModelInfo | null>(null)
  const [showApiKeys, setShowApiKeys] = useState<{ [key: string]: boolean }>({})
  
  const [formData, setFormData] = useState<ModelInfo>({
    model_id: '',
    model_name: '',
    model_type: 'llm',
    base_url: '',
    api_key: '',
    description: '',
  })

  useEffect(() => {
    loadModels()
  }, [])

  const loadModels = async () => {
    try {
      const response = await listModels()
      setLlmModels(response.data.llm_models)
      setEmbeddingModels(response.data.embedding_models)
    } catch (error) {
      console.error('加载模型列表失败:', error)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    try {
      if (editingModel) {
        await updateModel(editingModel.model_id, formData)
      } else {
        await registerModel(formData)
      }
      
      await loadModels()
      resetForm()
    } catch (error: any) {
      alert('操作失败: ' + (error.response?.data?.detail || error.message))
    }
  }

  const handleEdit = (model: ModelInfo) => {
    setEditingModel(model)
    setFormData(model)
    setShowForm(true)
  }

  const handleDelete = async (model_id: string) => {
    if (!confirm('确定要删除这个模型吗？')) return
    
    try {
      await deleteModel(model_id)
      await loadModels()
    } catch (error: any) {
      alert('删除失败: ' + (error.response?.data?.detail || error.message))
    }
  }

  const resetForm = () => {
    setFormData({
      model_id: '',
      model_name: '',
      model_type: 'llm',
      base_url: '',
      api_key: '',
      description: '',
    })
    setEditingModel(null)
    setShowForm(false)
  }

  const toggleApiKeyVisibility = (model_id: string) => {
    setShowApiKeys(prev => ({ ...prev, [model_id]: !prev[model_id] }))
  }

  const maskApiKey = (apiKey: string) => {
    if (apiKey.length <= 8) return '***'
    return apiKey.substring(0, 4) + '***' + apiKey.substring(apiKey.length - 4)
  }

  const ModelCard = ({ model }: { model: ModelInfo }) => (
    <div className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
      <div className="flex justify-between items-start mb-2">
        <div className="flex-1">
          <h4 className="font-semibold text-gray-900">{model.model_id}</h4>
          <p className="text-sm text-gray-600">{model.model_name}</p>
          {model.description && (
            <p className="text-xs text-gray-500 mt-1">{model.description}</p>
          )}
        </div>
        <div className="flex space-x-2">
          <button
            onClick={() => handleEdit(model)}
            className="p-1 text-blue-600 hover:bg-blue-50 rounded"
            title="编辑"
          >
            <Edit2 className="w-4 h-4" />
          </button>
          <button
            onClick={() => handleDelete(model.model_id)}
            className="p-1 text-red-600 hover:bg-red-50 rounded"
            title="删除"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>
      
      {model.base_url && (
        <div className="text-xs text-gray-500 mb-1">
          <span className="font-medium">Base URL:</span> {model.base_url}
        </div>
      )}
      
      <div className="flex items-center space-x-2 text-xs text-gray-500">
        <span className="font-medium">API Key:</span>
        <span className="font-mono">
          {showApiKeys[model.model_id] ? model.api_key : maskApiKey(model.api_key)}
        </span>
        <button
          onClick={() => toggleApiKeyVisibility(model.model_id)}
          className="p-1 hover:bg-gray-100 rounded"
        >
          {showApiKeys[model.model_id] ? (
            <EyeOff className="w-3 h-3" />
          ) : (
            <Eye className="w-3 h-3" />
          )}
        </button>
      </div>
    </div>
  )

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-semibold flex items-center">
            <Database className="w-5 h-5 mr-2" />
            模型仓库
          </h2>
          <button
            onClick={() => setShowForm(!showForm)}
            className="flex items-center space-x-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
          >
            <Plus className="w-4 h-4" />
            <span>{editingModel ? '取消编辑' : '注册模型'}</span>
          </button>
        </div>

        {showForm && (
          <form onSubmit={handleSubmit} className="mb-6 p-4 bg-gray-50 rounded-lg space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  模型ID <span className="text-red-500">*</span>
                </label>
                <input
                  type="text"
                  value={formData.model_id}
                  onChange={(e) => setFormData({ ...formData, model_id: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                  placeholder="例如: gpt-3.5-turbo-default"
                  required
                  disabled={!!editingModel}
                />
                <p className="text-xs text-gray-500 mt-1">唯一标识，创建后不可修改</p>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  模型类型 <span className="text-red-500">*</span>
                </label>
                <select
                  value={formData.model_type}
                  onChange={(e) => setFormData({ ...formData, model_type: e.target.value as 'llm' | 'embedding' })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                  required
                >
                  <option value="llm">LLM (语言模型)</option>
                  <option value="embedding">Embedding (嵌入模型)</option>
                </select>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                模型名称 <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                value={formData.model_name}
                onChange={(e) => setFormData({ ...formData, model_name: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                placeholder="例如: gpt-3.5-turbo"
                required
              />
              <p className="text-xs text-gray-500 mt-1">实际调用的模型名称</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Base URL
              </label>
              <input
                type="text"
                value={formData.base_url}
                onChange={(e) => setFormData({ ...formData, base_url: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                placeholder="例如: https://api.openai.com/v1 (可选)"
              />
              <p className="text-xs text-gray-500 mt-1">留空使用默认值</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                API Key <span className="text-red-500">*</span>
              </label>
              <input
                type="password"
                value={formData.api_key}
                onChange={(e) => setFormData({ ...formData, api_key: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                placeholder="sk-..."
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                描述
              </label>
              <input
                type="text"
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                placeholder="模型用途说明（可选）"
              />
            </div>

            <div className="flex space-x-3">
              <button
                type="submit"
                className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
              >
                {editingModel ? '更新' : '注册'}
              </button>
              <button
                type="button"
                onClick={resetForm}
                className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300"
              >
                取消
              </button>
            </div>
          </form>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="text-sm font-semibold text-gray-700 mb-3">
              LLM模型 ({llmModels.length})
            </h3>
            <div className="space-y-3">
              {llmModels.length === 0 ? (
                <p className="text-sm text-gray-500 text-center py-4">暂无LLM模型</p>
              ) : (
                llmModels.map(model => <ModelCard key={model.model_id} model={model} />)
              )}
            </div>
          </div>

          <div>
            <h3 className="text-sm font-semibold text-gray-700 mb-3">
              Embedding模型 ({embeddingModels.length})
            </h3>
            <div className="space-y-3">
              {embeddingModels.length === 0 ? (
                <p className="text-sm text-gray-500 text-center py-4">暂无Embedding模型</p>
              ) : (
                embeddingModels.map(model => <ModelCard key={model.model_id} model={model} />)
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
