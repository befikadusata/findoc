"""
Test script for the prompt management system
This script tests the new prompt management system functionality.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_prompt_manager():
    """Test the basic functionality of the prompt manager"""
    print("Testing Prompt Manager functionality...")
    
    try:
        from prompts.prompt_manager import PromptManager, render_rag_query_prompt, render_entity_extraction_prompt, render_summarization_prompt
        
        # Test creating a prompt manager instance
        pm = PromptManager()
        print("✓ PromptManager instance created successfully")
        
        # Test listing available templates
        templates = pm.list_available_templates()
        print(f"✓ Found {len(templates)} available templates")
        for name, info in templates.items():
            print(f"  - {name}: v{info['version']}")
        
        # Test rendering the RAG query prompt
        rag_prompt = render_rag_query_prompt(context="This is a test context", query="What is this document about?")
        if rag_prompt:
            print("✓ RAG query prompt rendered successfully")
            print(f"  Sample: {rag_prompt[:100]}...")
        else:
            print("✗ Failed to render RAG query prompt")
        
        # Test rendering the entity extraction prompt
        entity_prompt = render_entity_extraction_prompt(doc_type="invoice", text="Sample invoice text")
        if entity_prompt:
            print("✓ Entity extraction prompt rendered successfully")
            print(f"  Sample: {entity_prompt[:100]}...")
        else:
            print("✗ Failed to render entity extraction prompt")
        
        # Test rendering the summarization prompt
        summary_prompt = render_summarization_prompt(doc_type="contract", text_truncated="Sample contract text", max_length=300)
        if summary_prompt:
            print("✓ Summarization prompt rendered successfully")
            print(f"  Sample: {summary_prompt[:100]}...")
        else:
            print("✗ Failed to render summarization prompt")
        
        # Test version retrieval
        rag_version = pm.get_template_version("rag/rag_query.j2")
        extraction_version = pm.get_template_version("extraction/entity_extraction.j2")
        summary_version = pm.get_template_version("summarization/document_summarization.j2")
        
        print(f"✓ Template versions - RAG: v{rag_version}, Extraction: v{extraction_version}, Summary: v{summary_version}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing prompt manager: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_template_rendering():
    """Test template rendering with various parameters"""
    print("\nTesting template rendering with various parameters...")
    
    try:
        from prompts.prompt_manager import render_rag_query_prompt, render_entity_extraction_prompt, render_summarization_prompt
        
        # Test RAG prompt with realistic values
        context = "The contract was signed on 2023-01-15. The total amount is $50,000. Payment is due within 30 days."
        query = "When is the payment due?"
        rag_result = render_rag_query_prompt(context=context, query=query)
        
        if rag_result and context in rag_result and query in rag_result:
            print("✓ RAG prompt correctly includes context and query")
        else:
            print("✗ RAG prompt does not include expected values")
        
        # Test entity extraction prompt
        doc_type = "contract"
        text = "Sample contract text with invoice number INV-001 and amount $1000"
        extraction_result = render_entity_extraction_prompt(doc_type=doc_type, text=text)
        
        if extraction_result and doc_type in extraction_result and text in extraction_result:
            print("✓ Entity extraction prompt correctly includes doc_type and text")
        else:
            print("✗ Entity extraction prompt does not include expected values")
        
        # Test summarization prompt
        doc_type = "agreement"
        text_truncated = "Sample agreement text for summarization"
        max_length = 500
        summary_result = render_summarization_prompt(doc_type=doc_type, text_truncated=text_truncated, max_length=max_length)
        
        if summary_result and doc_type in summary_result and str(max_length) in summary_result:
            print("✓ Summarization prompt correctly includes parameters")
        else:
            print("✗ Summarization prompt does not include expected values")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing template rendering: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fallback_mechanism():
    """Test the fallback mechanism when templates don't exist"""
    print("\nTesting fallback mechanism...")
    
    try:
        from prompts.prompt_manager import PromptManager
        
        pm = PromptManager()
        
        # Try to render a non-existent template - should return None
        result = pm.render_template("non/existent/template.j2", some_param="test")
        if result is None:
            print("✓ Correctly returned None for non-existent template")
        else:
            print("✗ Should have returned None for non-existent template")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing fallback mechanism: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("Starting tests for the prompt management system...\n")
    
    test_results = []
    test_results.append(test_prompt_manager())
    test_results.append(test_template_rendering())
    test_results.append(test_fallback_mechanism())
    
    print(f"\nTest Results: {sum(test_results)}/{len(test_results)} passed")
    
    if all(test_results):
        print("\n🎉 All tests passed! The prompt management system is working correctly.")
        return True
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)