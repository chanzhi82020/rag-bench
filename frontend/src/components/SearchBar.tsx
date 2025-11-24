import { useState, useEffect, useRef } from 'react'
import { Search, X } from 'lucide-react'

export interface SearchBarProps {
  value: string
  onChange: (value: string) => void
  onSearch: (value: string) => void
  placeholder?: string
  debounceMs?: number
}

export default function SearchBar({
  value,
  onChange,
  onSearch,
  placeholder = '搜索...',
  debounceMs = 300,
}: SearchBarProps) {
  const [localValue, setLocalValue] = useState(value)
  const debounceTimerRef = useRef<number | null>(null)

  // Sync local value with prop value
  useEffect(() => {
    setLocalValue(value)
  }, [value])

  // Debounced search
  useEffect(() => {
    // Clear existing timer
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current)
    }

    // Set new timer
    debounceTimerRef.current = setTimeout(() => {
      if (localValue !== value) {
        onChange(localValue)
        onSearch(localValue)
      }
    }, debounceMs)

    // Cleanup
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current)
      }
    }
  }, [localValue, debounceMs, onChange, onSearch, value])

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setLocalValue(e.target.value)
  }

  const handleClear = () => {
    setLocalValue('')
    onChange('')
    onSearch('')
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    // Clear debounce timer and search immediately
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current)
    }
    onChange(localValue)
    onSearch(localValue)
  }

  return (
    <form onSubmit={handleSubmit} className="relative">
      <div className="relative">
        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
          <Search className="h-5 w-5 text-gray-400" />
        </div>
        <input
          type="text"
          value={localValue}
          onChange={handleInputChange}
          placeholder={placeholder}
          className="
            block w-full pl-10 pr-20 py-2 
            border border-gray-300 rounded-lg
            text-sm text-gray-900 placeholder-gray-500
            focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
            transition-colors
          "
        />
        <div className="absolute inset-y-0 right-0 flex items-center gap-1 pr-2">
          {localValue && (
            <button
              type="button"
              onClick={handleClear}
              className="
                p-1 rounded-md text-gray-400 hover:text-gray-600 hover:bg-gray-100
                transition-colors
              "
              aria-label="清除搜索"
            >
              <X className="h-4 w-4" />
            </button>
          )}
          <button
            type="submit"
            className="
              px-3 py-1 rounded-md text-sm font-medium
              bg-blue-500 text-white hover:bg-blue-600
              transition-colors
            "
          >
            搜索
          </button>
        </div>
      </div>
    </form>
  )
}
