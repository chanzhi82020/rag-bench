import { useState } from 'react'
import { Database, Cpu, BarChart3, Settings, Server } from 'lucide-react'
import DatasetPanel from './components/DatasetPanel'
import ModelRegistryPanel from './components/ModelRegistryPanel'
import RAGPanel from './components/RAGPanel'
import EvaluationPanel from './components/EvaluationPanel'
import ResultsPanel from './components/ResultsPanel'

type Tab = 'datasets' | 'models' | 'rag' | 'evaluation' | 'results'

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('datasets')

  const tabs = [
    { id: 'datasets' as Tab, name: '数据集', icon: Database },
    { id: 'models' as Tab, name: '模型仓库', icon: Server },
    { id: 'rag' as Tab, name: 'RAG系统', icon: Cpu },
    { id: 'evaluation' as Tab, name: '评测', icon: Settings },
    { id: 'results' as Tab, name: '结果', icon: BarChart3 },
  ]

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <h1 className="text-2xl font-bold text-gray-900">RAG Benchmark</h1>
          <p className="text-sm text-gray-500 mt-1">RAG系统评测平台</p>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {tabs.map((tab) => {
              const Icon = tab.icon
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`
                    flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm
                    ${activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    }
                  `}
                >
                  <Icon className="w-5 h-5" />
                  <span>{tab.name}</span>
                </button>
              )
            })}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'datasets' && <DatasetPanel />}
        {activeTab === 'models' && <ModelRegistryPanel />}
        {activeTab === 'rag' && <RAGPanel />}
        {activeTab === 'evaluation' && <EvaluationPanel />}
        {activeTab === 'results' && <ResultsPanel />}
      </main>
    </div>
  )
}

export default App
