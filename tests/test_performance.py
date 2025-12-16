import sys
import os
import time
import pytest

# Add the src directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from app import load_documents, RAGAssistant

@pytest.fixture(scope="module")
def documents():
    start = time.time()
    docs = load_documents()
    end = time.time()
    print(f"\nload_documents took {end - start:.4f} seconds")
    assert len(docs) > 0, "No documents loaded."
    return docs

@pytest.fixture(scope="module")
def assistant(documents):
    assistant = RAGAssistant()
    start = time.time()
    assistant.add_documents(documents)
    end = time.time()
    print(f"add_documents took {end - start:.4f} seconds")
    return assistant

@pytest.mark.parametrize("question", [
    "What is artificial intelligence?",
    "What are Machine Learning and MLOps?",
    "Who is home minister of India?"
])
def test_assistant_invoke_avg_performance(assistant, question):
    times = []
    runs = 3  # Number of times to run for averaging
    for index in range(runs):
        start = time.time()
        response = assistant.invoke(question)
        end = time.time()
        times.append(end - start)
        print(response)
        assert isinstance(response, str)
        assert len(response) > 0
        if index < runs - 1:
            time.sleep(15)   # To avoid rate limit error
        
    avg_time = sum(times) / runs
    print(f"Avg time for assistant.invoke('{question[:30]}...') over {runs} runs: {avg_time:.4f} seconds")