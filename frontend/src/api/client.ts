import axios from 'axios'

const API_BASE_URL = '/api'

const client = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export interface DatasetInfo {
  name: string
  subset?: string
}

export interface DatasetStats {
  dataset_name: string
  subset?: string
  record_count: number
  avg_input_length: number
  avg_reference_length: number
  avg_contexts_per_record: number
  corpus_count: number
}

export interface RAGConfig {
  top_k?: number
  temperature?: number
  max_length?: number
}

export interface ModelInfo {
  model_id: string
  model_name: string
  model_type: 'llm' | 'embedding'
  base_url?: string
  api_key: string
  description?: string
}

export interface ModelInfoConfig {
  llm_model_id: string
  embedding_model_id: string
}

export interface CreateRAGRequest {
  name: string
  rag_type?: 'baseline' | 'api'
  model_info: ModelInfoConfig
  rag_config?: RAGConfig
  api_config?: {
    retrieval_endpoint?: string
    generation_endpoint?: string
    api_key?: string
  }
}

export interface IndexDocumentsRequest {
  rag_name: string
  dataset_name: string
  subset?: string
  document_ids?: string[]
}

export interface QueryRequest {
  rag_name: string
  query: string
  top_k?: number
}

export interface EvaluateRequest {
  dataset_name: string
  subset?: string
  rag_name: string
  eval_type?: 'e2e' | 'retrieval' | 'generation'
  sample_size?: number
  model_info: ModelInfoConfig
}

export interface IndexStatus {
  has_index: boolean
  document_count?: number
  embedding_dimension?: number
  created_at?: string
  updated_at?: string
  total_size_bytes?: number
}

export interface TaskStatus {
  task_id: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  current_stage?: string
  result?: any
  error?: string
  created_at: string
  updated_at: string
}

// Model Registry APIs
export const registerModel = (data: ModelInfo) =>
  client.post('/models/register', data)

export const listModels = () =>
  client.get<{ llm_models: ModelInfo[], embedding_models: ModelInfo[], total: number }>('/models/list')

export const getModel = (model_id: string) =>
  client.get<ModelInfo>(`/models/${encodeURIComponent(model_id)}`)

export const deleteModel = (model_id: string) =>
  client.delete(`/models/${encodeURIComponent(model_id)}`)

export const updateModel = (model_id: string, data: ModelInfo) =>
  client.put(`/models/${encodeURIComponent(model_id)}`, data)

// Dataset APIs
export const listDatasets = () => client.get<string[]>('/datasets')

export const getDatasetStats = (data: DatasetInfo) =>
  client.post<DatasetStats>('/datasets/stats', data)

export const sampleDataset = (data: DatasetInfo, n: number = 5) =>
  client.post('/datasets/sample', data, { params: { n } })

export const previewCorpus = (data: DatasetInfo, limit: number = 100) =>
  client.post('/datasets/corpus/preview', data, { params: { limit } })

// RAG APIs
export const createRAG = (data: CreateRAGRequest) =>
  client.post('/rag/create', data)

export const listRAGs = () => client.get('/rag/list')

export const deleteRAG = (rag_name: string) =>
  client.delete(`/rag/${encodeURIComponent(rag_name)}`)

export const indexDocuments = (data: IndexDocumentsRequest) =>
  client.post('/rag/index', data)

export const queryRAG = (data: QueryRequest) =>
  client.post('/rag/query', data)

export const getRAGIndexStatus = (rag_name: string) =>
  client.get<IndexStatus>(`/rag/${encodeURIComponent(rag_name)}/index/status`)

// Evaluation APIs
export const startEvaluation = (data: EvaluateRequest) =>
  client.post('/evaluate/start', data)

export const getEvaluationStatus = (task_id: string) =>
  client.get<TaskStatus>(`/evaluate/status/${task_id}`)

export const listEvaluationTasks = () =>
  client.get('/evaluate/tasks')

export default client
