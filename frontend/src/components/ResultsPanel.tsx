import { useState, useEffect } from 'react'
import { listEvaluationTasks } from '../api/client'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { RefreshCw, TrendingUp } from 'lucide-react'

export default function ResultsPanel() {
  const [tasks, setTasks] = useState<any[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    loadTasks()
  }, [])

  const loadTasks = async () => {
    setLoading(true)
    try {
      const response = await listEvaluationTasks()
      const completedTasks = response.data.tasks.filter((t: any) => t.status === 'completed')
      setTasks(completedTasks)
    } catch (error) {
      console.error('加载任务列表失败:', error)
    } finally {
      setLoading(false)
    }
  }

  const getChartData = () => {
    if (tasks.length === 0) return []
    
    // 提取所有指标名称
    const allMetrics = new Set<string>()
    tasks.forEach(task => {
      if (task.result?.metrics) {
        Object.keys(task.result.metrics).forEach(key => allMetrics.add(key))
      }
    })

    // 为每个指标创建数据点
    return Array.from(allMetrics).map(metric => {
      const dataPoint: any = { metric }
      tasks.forEach((task, idx) => {
        const value = task.result?.metrics?.[metric]
        if (typeof value === 'number') {
          dataPoint[`task_${idx}`] = value
        }
      })
      return dataPoint
    })
  }

  const chartData = getChartData()

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-semibold flex items-center">
            <TrendingUp className="w-5 h-5 mr-2" />
            评测结果
          </h2>
          <button
            onClick={loadTasks}
            disabled={loading}
            className="flex items-center space-x-2 px-3 py-2 text-sm bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            <span>刷新</span>
          </button>
        </div>

        {tasks.length === 0 ? (
          <div className="text-center py-12 text-gray-500">
            暂无已完成的评测任务
          </div>
        ) : (
          <>
            <div className="mb-6">
              <h3 className="text-sm font-semibold text-gray-700 mb-3">指标对比</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="metric" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  {tasks.map((_, idx) => (
                    <Bar
                      key={idx}
                      dataKey={`task_${idx}`}
                      fill={`hsl(${(idx * 360) / tasks.length}, 70%, 50%)`}
                      name={`任务 ${idx + 1}`}
                    />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="space-y-4">
              <h3 className="text-sm font-semibold text-gray-700">任务详情</h3>
              {tasks.map((task, idx) => (
                <div key={task.task_id} className="border border-gray-200 rounded-lg p-4">
                  <div className="flex justify-between items-start mb-3">
                    <div>
                      <div className="font-medium text-gray-900">任务 {idx + 1}</div>
                      <div className="text-xs text-gray-500 mt-1">
                        ID: {task.task_id}
                      </div>
                      <div className="text-xs text-gray-500">
                        创建时间: {new Date(task.created_at).toLocaleString('zh-CN')}
                      </div>
                    </div>
                    <span className="px-2 py-1 text-xs font-medium bg-green-100 text-green-800 rounded">
                      已完成
                    </span>
                  </div>

                  {task.result && (
                    <div>
                      <div className="text-sm text-gray-600 mb-2">
                        评测类型: {task.result.eval_type} | 样本数: {task.result.sample_count}
                      </div>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        {Object.entries(task.result.metrics).map(([key, value]: [string, any]) => (
                          <div key={key} className="bg-gray-50 p-2 rounded">
                            <div className="text-xs text-gray-500">{key}</div>
                            <div className="text-sm font-semibold text-gray-900">
                              {typeof value === 'number' ? value.toFixed(4) : value}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
