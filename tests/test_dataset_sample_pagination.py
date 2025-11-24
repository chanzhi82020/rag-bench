"""Tests for pagination and search functionality in POST /datasets/sample endpoint

Tests verify pagination parameters, search functionality, and response format
according to Requirements 1.1, 4.1, 5.1, 5.2, 5.3, 5.5
"""

import pytest
from fastapi.testclient import TestClient

from rag_benchmark.api.main import app


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


class TestPaginationParameters:
    """Tests for pagination parameter validation (Requirements 1.1, 5.1, 5.5)"""
    
    def test_default_pagination_parameters(self, client):
        """Test that default pagination parameters are applied correctly"""
        response = client.post(
            "/datasets/sample",
            json={"name": "xquad", "subset": None}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify default values
        assert data["page"] == 1
        assert data["page_size"] == 20
        assert "total_count" in data
        assert "total_pages" in data
    
    def test_custom_page_size(self, client):
        """Test that custom page_size is respected"""
        for page_size in [10, 20, 50, 100]:
            response = client.post(
                "/datasets/sample",
                json={"name": "xquad", "subset": None},
                params={"page": 1, "page_size": page_size}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["page_size"] == page_size
            assert len(data["samples"]) <= page_size
    
    def test_invalid_page_size(self, client):
        """Test that invalid page_size values are rejected (Requirement 5.5)"""
        invalid_sizes = [5, 15, 25, 200]
        
        for page_size in invalid_sizes:
            response = client.post(
                "/datasets/sample",
                json={"name": "xquad", "subset": None},
                params={"page": 1, "page_size": page_size}
            )
            
            assert response.status_code == 400
            data = response.json()
            assert "error" in data["detail"]
            assert "page_size" in data["detail"]["details"]
    
    def test_invalid_page_number(self, client):
        """Test that page < 1 is rejected (Requirement 5.5)"""
        response = client.post(
            "/datasets/sample",
            json={"name": "xquad", "subset": None},
            params={"page": 0, "page_size": 20}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data["detail"]
        assert "page" in data["detail"]["details"]
    
    def test_pagination_across_pages(self, client):
        """Test that pagination returns different samples across pages"""
        # Get first page
        response1 = client.post(
            "/datasets/sample",
            json={"name": "xquad", "subset": None},
            params={"page": 1, "page_size": 10}
        )
        
        assert response1.status_code == 200
        data1 = response1.json()
        
        # Get second page
        response2 = client.post(
            "/datasets/sample",
            json={"name": "xquad", "subset": None},
            params={"page": 2, "page_size": 10}
        )
        
        assert response2.status_code == 200
        data2 = response2.json()
        
        # Verify different samples
        ids1 = {s["id"] for s in data1["samples"]}
        ids2 = {s["id"] for s in data2["samples"]}
        
        # Pages should have different samples (no overlap)
        assert len(ids1.intersection(ids2)) == 0


class TestResponseFormat:
    """Tests for response format (Requirements 1.5, 5.2)"""
    
    def test_response_includes_pagination_metadata(self, client):
        """Test that response includes all required pagination metadata"""
        response = client.post(
            "/datasets/sample",
            json={"name": "xquad", "subset": None},
            params={"page": 1, "page_size": 10}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify all required fields
        required_fields = [
            "dataset_name", "subset", "total_count",
            "page", "page_size", "total_pages", "samples"
        ]
        for field in required_fields:
            assert field in data, f"Response must include '{field}' field"
    
    def test_total_pages_calculation(self, client):
        """Test that total_pages is calculated correctly"""
        response = client.post(
            "/datasets/sample",
            json={"name": "xquad", "subset": None},
            params={"page": 1, "page_size": 10}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Calculate expected total pages
        expected_total_pages = (data["total_count"] + data["page_size"] - 1) // data["page_size"]
        assert data["total_pages"] == expected_total_pages
    
    def test_samples_include_reference_context_ids(self, client):
        """Test that samples include reference_context_ids field"""
        response = client.post(
            "/datasets/sample",
            json={"name": "xquad", "subset": None},
            params={"page": 1, "page_size": 10}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify each sample has reference_context_ids
        for sample in data["samples"]:
            assert "reference_context_ids" in sample


class TestSearchFunctionality:
    """Tests for search functionality (Requirements 4.1, 5.3)"""
    
    def test_search_filters_samples(self, client):
        """Test that search parameter filters samples by user_input"""
        # Get all samples first
        response_all = client.post(
            "/datasets/sample",
            json={"name": "xquad", "subset": None},
            params={"page": 1, "page_size": 100}
        )
        
        assert response_all.status_code == 200
        all_data = response_all.json()
        total_count_all = all_data["total_count"]
        
        # Search for a specific term (use a common word)
        response_search = client.post(
            "/datasets/sample",
            json={"name": "xquad", "subset": None},
            params={"page": 1, "page_size": 100, "search": "the"}
        )
        
        assert response_search.status_code == 200
        search_data = response_search.json()
        
        # Search should return fewer or equal samples
        assert search_data["total_count"] <= total_count_all
        
        # Verify all returned samples contain the search term
        for sample in search_data["samples"]:
            assert "the" in sample["user_input"].lower()
    
    def test_search_case_insensitive(self, client):
        """Test that search is case-insensitive (Requirement 4.4)"""
        # Search with lowercase
        response_lower = client.post(
            "/datasets/sample",
            json={"name": "xquad", "subset": None},
            params={"page": 1, "page_size": 100, "search": "what"}
        )
        
        # Search with uppercase
        response_upper = client.post(
            "/datasets/sample",
            json={"name": "xquad", "subset": None},
            params={"page": 1, "page_size": 100, "search": "WHAT"}
        )
        
        assert response_lower.status_code == 200
        assert response_upper.status_code == 200
        
        # Should return same count
        assert response_lower.json()["total_count"] == response_upper.json()["total_count"]
    
    def test_search_with_pagination(self, client):
        """Test that search works correctly with pagination"""
        response = client.post(
            "/datasets/sample",
            json={"name": "xquad", "subset": None},
            params={"page": 1, "page_size": 10, "search": "what"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify pagination metadata is present
        assert "total_count" in data
        assert "total_pages" in data
        assert data["page"] == 1
        assert data["page_size"] == 10
        
        # Verify samples match search
        for sample in data["samples"]:
            assert "what" in sample["user_input"].lower()
    
    def test_empty_search_results(self, client):
        """Test that empty search results are handled correctly (Requirement 4.3)"""
        # Search for a term that likely doesn't exist
        response = client.post(
            "/datasets/sample",
            json={"name": "xquad", "subset": None},
            params={"page": 1, "page_size": 20, "search": "xyzabc123nonexistent"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return empty results
        assert data["total_count"] == 0
        assert data["total_pages"] == 0
        assert len(data["samples"]) == 0


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""
    
    def test_last_page_partial_results(self, client):
        """Test that last page can have fewer samples than page_size"""
        # Get total count first
        response = client.post(
            "/datasets/sample",
            json={"name": "xquad", "subset": None},
            params={"page": 1, "page_size": 10}
        )
        
        assert response.status_code == 200
        data = response.json()
        total_pages = data["total_pages"]
        
        # Get last page
        response_last = client.post(
            "/datasets/sample",
            json={"name": "xquad", "subset": None},
            params={"page": total_pages, "page_size": 10}
        )
        
        assert response_last.status_code == 200
        last_data = response_last.json()
        
        # Last page should have <= page_size samples
        assert len(last_data["samples"]) <= last_data["page_size"]
    
    def test_page_beyond_total_pages(self, client):
        """Test requesting a page beyond total_pages returns empty results"""
        # Get total pages first
        response = client.post(
            "/datasets/sample",
            json={"name": "xquad", "subset": None},
            params={"page": 1, "page_size": 10}
        )
        
        assert response.status_code == 200
        data = response.json()
        total_pages = data["total_pages"]
        
        # Request page beyond total - should return 400 error
        response_beyond = client.post(
            "/datasets/sample",
            json={"name": "xquad", "subset": None},
            params={"page": total_pages + 10, "page_size": 10}
        )
        
        assert response_beyond.status_code == 400
        error_data = response_beyond.json()
        
        # Should return error with helpful message
        assert "detail" in error_data
        assert "error" in error_data["detail"]
        assert "exceeds total pages" in error_data["detail"]["details"]
        assert error_data["detail"]["total_pages"] == total_pages
