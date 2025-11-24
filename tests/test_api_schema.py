"""Test that API schema is properly generated with new models"""

from rag_benchmark.api.main import app


def test_openapi_schema_includes_sample_selection():
    """Test that OpenAPI schema includes SampleSelection model"""
    schema = app.openapi()
    
    # Check that SampleSelection is in the components/schemas
    assert "SampleSelection" in schema["components"]["schemas"]
    
    # Check SampleSelection properties
    sample_selection_schema = schema["components"]["schemas"]["SampleSelection"]
    assert "strategy" in sample_selection_schema["properties"]
    assert "sample_ids" in sample_selection_schema["properties"]
    assert "sample_size" in sample_selection_schema["properties"]
    assert "random_seed" in sample_selection_schema["properties"]
    
    # Check that strategy is required
    assert "strategy" in sample_selection_schema["required"]


def test_openapi_schema_evaluate_request_has_sample_selection():
    """Test that EvaluateRequest includes sample_selection field"""
    schema = app.openapi()
    
    # Check that EvaluateRequest is in the components/schemas
    assert "EvaluateRequest" in schema["components"]["schemas"]
    
    # Check EvaluateRequest has sample_selection property
    evaluate_request_schema = schema["components"]["schemas"]["EvaluateRequest"]
    assert "sample_selection" in evaluate_request_schema["properties"]
    
    # Check that sample_selection references SampleSelection
    sample_selection_prop = evaluate_request_schema["properties"]["sample_selection"]
    # It should be anyOf with SampleSelection and null (since it's Optional)
    if "anyOf" in sample_selection_prop:
        refs = [item.get("$ref") for item in sample_selection_prop["anyOf"] if "$ref" in item]
        assert any("SampleSelection" in ref for ref in refs)


def test_openapi_schema_has_descriptions():
    """Test that models have proper descriptions"""
    schema = app.openapi()
    
    # Check SampleSelection has description
    sample_selection_schema = schema["components"]["schemas"]["SampleSelection"]
    assert "description" in sample_selection_schema or "title" in sample_selection_schema
    
    # Check EvaluateRequest has description
    evaluate_request_schema = schema["components"]["schemas"]["EvaluateRequest"]
    assert "description" in evaluate_request_schema or "title" in evaluate_request_schema


def test_evaluate_endpoint_accepts_sample_selection():
    """Test that /evaluate/start endpoint accepts sample_selection in request body"""
    schema = app.openapi()
    
    # Find the /evaluate/start endpoint
    evaluate_start_path = schema["paths"]["/evaluate/start"]
    post_operation = evaluate_start_path["post"]
    
    # Check request body schema
    request_body = post_operation["requestBody"]
    content = request_body["content"]["application/json"]
    schema_ref = content["schema"]["$ref"]
    
    # Should reference EvaluateRequest
    assert "EvaluateRequest" in schema_ref
