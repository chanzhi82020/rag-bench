import { SearchX } from 'lucide-react'

export interface EmptySearchResultsProps {
  searchQuery: string
  onClearSearch?: () => void
}

/**
 * Component to display when search returns no results
 */
export default function EmptySearchResults({
  searchQuery,
  onClearSearch,
}: EmptySearchResultsProps) {
  return (
    <div className="flex flex-col items-center justify-center py-12 px-4">
      <div className="bg-gray-100 rounded-full p-4 mb-4">
        <SearchX className="w-12 h-12 text-gray-400" />
      </div>
      
      <h3 className="text-lg font-semibold text-gray-900 mb-2">
        未找到结果
      </h3>
      
      <p className="text-sm text-gray-600 text-center mb-4 max-w-md">
        没有找到与 "<span className="font-medium text-gray-900">{searchQuery}</span>" 匹配的结果
      </p>
      
      <div className="text-sm text-gray-500 space-y-2 max-w-md">
        <p className="font-medium">建议：</p>
        <ul className="list-disc list-inside space-y-1 text-left">
          <li>检查搜索词的拼写</li>
          <li>尝试使用不同的关键词</li>
          <li>使用更通用的搜索词</li>
          <li>减少搜索词的数量</li>
        </ul>
      </div>
      
      {onClearSearch && (
        <button
          onClick={onClearSearch}
          className="
            mt-6 px-4 py-2 rounded-lg
            bg-blue-500 text-white text-sm font-medium
            hover:bg-blue-600 transition-colors
          "
        >
          清除搜索
        </button>
      )}
    </div>
  )
}
