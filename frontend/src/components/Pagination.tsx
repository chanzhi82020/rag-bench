import { ChevronLeft, ChevronRight } from 'lucide-react'

export interface PaginationProps {
  currentPage: number
  totalPages: number
  pageSize: number
  totalCount: number
  onPageChange: (page: number) => void
  onPageSizeChange: (size: number) => void
}

export default function Pagination({
  currentPage,
  totalPages,
  pageSize,
  totalCount,
  onPageChange,
  onPageSizeChange,
}: PaginationProps) {
  // Calculate the range of items being displayed
  const startItem = totalCount === 0 ? 0 : (currentPage - 1) * pageSize + 1
  const endItem = Math.min(currentPage * pageSize, totalCount)

  // Generate page numbers to display
  const getPageNumbers = () => {
    const pages: (number | string)[] = []
    const maxPagesToShow = 7

    if (totalPages <= maxPagesToShow) {
      // Show all pages if total is small
      for (let i = 1; i <= totalPages; i++) {
        pages.push(i)
      }
    } else {
      // Always show first page
      pages.push(1)

      if (currentPage > 3) {
        pages.push('...')
      }

      // Show pages around current page
      const start = Math.max(2, currentPage - 1)
      const end = Math.min(totalPages - 1, currentPage + 1)

      for (let i = start; i <= end; i++) {
        pages.push(i)
      }

      if (currentPage < totalPages - 2) {
        pages.push('...')
      }

      // Always show last page
      if (totalPages > 1) {
        pages.push(totalPages)
      }
    }

    return pages
  }

  const handlePrevious = () => {
    if (currentPage > 1) {
      onPageChange(currentPage - 1)
    }
  }

  const handleNext = () => {
    if (currentPage < totalPages) {
      onPageChange(currentPage + 1)
    }
  }

  const handlePageClick = (page: number | string) => {
    if (typeof page === 'number') {
      onPageChange(page)
    }
  }

  const handlePageSizeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newSize = parseInt(e.target.value, 10)
    onPageSizeChange(newSize)
  }

  return (
    <div className="flex flex-col sm:flex-row items-center justify-between gap-4 py-4">
      {/* Pagination info display */}
      <div className="text-sm text-gray-700">
        显示 <span className="font-medium">{startItem}</span> 到{' '}
        <span className="font-medium">{endItem}</span> 共{' '}
        <span className="font-medium">{totalCount}</span> 条记录
        {totalPages > 0 && (
          <>
            {' '}(第 <span className="font-medium">{currentPage}</span> 页，共{' '}
            <span className="font-medium">{totalPages}</span> 页)
          </>
        )}
      </div>

      <div className="flex items-center gap-4">
        {/* Page size selector */}
        <div className="flex items-center gap-2">
          <label htmlFor="pageSize" className="text-sm text-gray-700">
            每页显示:
          </label>
          <select
            id="pageSize"
            value={pageSize}
            onChange={handlePageSizeChange}
            className="border border-gray-300 rounded-md px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value={10}>10</option>
            <option value={20}>20</option>
            <option value={50}>50</option>
            <option value={100}>100</option>
          </select>
        </div>

        {/* Page navigation controls */}
        <div className="flex items-center gap-1">
          {/* Previous button */}
          <button
            onClick={handlePrevious}
            disabled={currentPage === 1}
            className={`
              p-2 rounded-md border transition-colors
              ${
                currentPage === 1
                  ? 'border-gray-200 text-gray-400 cursor-not-allowed'
                  : 'border-gray-300 text-gray-700 hover:bg-gray-50'
              }
            `}
            aria-label="上一页"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>

          {/* Page numbers */}
          {getPageNumbers().map((page, index) => (
            <button
              key={index}
              onClick={() => handlePageClick(page)}
              disabled={page === '...'}
              className={`
                min-w-[2.5rem] px-3 py-2 rounded-md text-sm font-medium transition-colors
                ${
                  page === currentPage
                    ? 'bg-blue-500 text-white border border-blue-500'
                    : page === '...'
                    ? 'text-gray-400 cursor-default'
                    : 'border border-gray-300 text-gray-700 hover:bg-gray-50'
                }
              `}
            >
              {page}
            </button>
          ))}

          {/* Next button */}
          <button
            onClick={handleNext}
            disabled={currentPage === totalPages || totalPages === 0}
            className={`
              p-2 rounded-md border transition-colors
              ${
                currentPage === totalPages || totalPages === 0
                  ? 'border-gray-200 text-gray-400 cursor-not-allowed'
                  : 'border-gray-300 text-gray-700 hover:bg-gray-50'
              }
            `}
            aria-label="下一页"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  )
}
