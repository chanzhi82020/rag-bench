import { useMemo } from 'react'

export interface HighlightedTextProps {
  text: string
  searchQuery: string
  className?: string
  highlightClassName?: string
}

/**
 * Component that highlights matching text in a case-insensitive manner
 */
export default function HighlightedText({
  text,
  searchQuery,
  className = '',
  highlightClassName = 'bg-yellow-200 font-medium',
}: HighlightedTextProps) {
  const highlightedContent = useMemo(() => {
    // If no search query, return the original text
    if (!searchQuery || !searchQuery.trim()) {
      return <span className={className}>{text}</span>
    }

    // Escape special regex characters in the search query
    const escapedQuery = searchQuery.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

    // Create case-insensitive regex
    const regex = new RegExp(`(${escapedQuery})`, 'gi')

    // Split text by matches
    const parts = text.split(regex)

    // Return highlighted parts
    return (
      <span className={className}>
        {parts.map((part, index) => {
          // Check if this part matches the search query (case-insensitive)
          const isMatch = part.toLowerCase() === searchQuery.toLowerCase()
          
          return isMatch ? (
            <mark key={index} className={highlightClassName}>
              {part}
            </mark>
          ) : (
            <span key={index}>{part}</span>
          )
        })}
      </span>
    )
  }, [text, searchQuery, className, highlightClassName])

  return highlightedContent
}
