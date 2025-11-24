/**
 * Example integration of Search components
 * This file demonstrates how to use SearchBar, HighlightedText, and EmptySearchResults together
 */

import { useState } from 'react'
import SearchBar from './SearchBar'
import HighlightedText from './HighlightedText'
import EmptySearchResults from './EmptySearchResults'

export default function SearchExample() {
  const [searchQuery, setSearchQuery] = useState('')
  
  // Example data
  const sampleData = [
    { id: 1, text: 'What is the capital of France?' },
    { id: 2, text: 'How does photosynthesis work?' },
    { id: 3, text: 'What are the benefits of exercise?' },
  ]

  // Filter data based on search query (case-insensitive)
  const filteredData = searchQuery.trim()
    ? sampleData.filter(item =>
        item.text.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : sampleData

  const handleSearch = (value: string) => {
    console.log('Searching for:', value)
  }

  const handleClearSearch = () => {
    setSearchQuery('')
  }

  return (
    <div className="space-y-4">
      <SearchBar
        value={searchQuery}
        onChange={setSearchQuery}
        onSearch={handleSearch}
        placeholder="搜索问题..."
      />

      {filteredData.length === 0 && searchQuery.trim() ? (
        <EmptySearchResults
          searchQuery={searchQuery}
          onClearSearch={handleClearSearch}
        />
      ) : (
        <div className="space-y-2">
          {filteredData.map(item => (
            <div key={item.id} className="p-4 border border-gray-200 rounded-lg">
              <HighlightedText
                text={item.text}
                searchQuery={searchQuery}
                className="text-sm text-gray-900"
              />
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
