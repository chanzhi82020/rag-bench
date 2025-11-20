import { useState, useEffect } from 'react'
import { listDatasets, getDatasetStats, sampleDataset, DatasetStats } from '../api/client'
import { Loader2, Database, FileText, ChevronDown, ChevronRight } from 'lucide-react'

export default function DatasetPanel() {
  const [datasets, setDatasets] = useState<string[]>([])
  const [selectedDataset, setSelectedDataset] = useState<string>('')
  const [stats, setStats] = useState<DatasetStats | null>(null)
  const [samples, setSamples] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [expandedContexts, setExpandedContexts] = useState<Set<string>>(new Set())

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
    try {
      const [statsRes, samplesRes] = await Promise.all([
        getDatasetStats({ name }),
        sampleDataset({ name }, 3)
      ])
      setStats(statsRes.data)
      setSamples(samplesRes.data)
    } catch (error) {
      console.error('加载数据集信息失败:', error)
    } finally {
      setLoading(false)
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

      {samples && !loading && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <FileText className="w-5 h-5 mr-2" />
            数据样本
          </h3>
          <div className="space-y-4">
            {samples.samples.map((sample: any, idx: number) => (
              <div key={idx} className="border border-gray-200 rounded-lg p-4">
                <div className="mb-2">
                  <span className="text-xs font-semibold text-gray-500 uppercase">问题</span>
                  <p className="text-sm text-gray-900 mt-1">{sample.user_input}</p>
                </div>
                <div className="mb-2">
                  <span className="text-xs font-semibold text-gray-500 uppercase">参考答案</span>
                  <p className="text-sm text-gray-700 mt-1">{sample.reference}</p>
                </div>
                <div>
                  <span className="text-xs font-semibold text-gray-500 uppercase">
                    参考上下文 ({sample.reference_contexts.length})
                  </span>
                  <div className="mt-1 space-y-2">
                    {sample.reference_contexts.map((ctx: string, i: number) => {
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
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
