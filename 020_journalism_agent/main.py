import os
import pprint
from pathlib import Path
from langchain_community.document_loaders.pdf import PyMuPDFLoader

from src.workflow.graph import create_workflow

def clean_page_content(page_content: str) -> str:
    page_content = page_content.replace("\n", " ")
    page_content = page_content.replace("\t", " ")
    return page_content

def main():
    print("Initializing Journalism Agent Workflow...")
    app = create_workflow()
    
    data_path = Path(__file__).resolve().parent.parent / "data"
    file_path = data_path / "Sample AI Generated Article.pdf"
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return
        
    print(f"Loading article from {file_path}...")
    pages = []
    try:
        loader = PyMuPDFLoader(str(file_path))
        for page in loader.lazy_load():
            pages.append(page)
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return

    formatted_pages = []
    for page in pages:
        page_content = clean_page_content(page.page_content)
        formatted_pages.append(page_content)

    full_article_text = " ".join(formatted_pages)
    
    query = input("\nEnter your query (e.g. 'Can you provide a full report on this article?'):\n> ")
    if not query.strip():
        query = "Can you provide a full report on this article?"
    
    print("\nRunning workflow...\n")
    
    initial_state = {
        "current_query": query,
        "article_text": full_article_text,
        "chunks": [],
        "actions": []
    }
    
    try:
        final_state = app.invoke(initial_state)
        
        print("\n" + "="*50)
        print("WORKFLOW COMPLETED. RESULTS:")
        print("="*50)
        
        if final_state.get("summary_result"):
            print("\n[SUMMARY]")
            print(final_state["summary_result"])
            
        if final_state.get("fact_check_result"):
            print("\n[FACT CHECKING]")
            for item in final_state["fact_check_result"]:
                print(f"- Statement: {item.get('statement')}")
                print(f"  Status: {item.get('status')}")
                print(f"  Explanation: {item.get('explanation')}")
                
        if final_state.get("tone_analysis_result"):
            print("\n[TONE ANALYSIS]")
            for item in final_state["tone_analysis_result"]:
                print(item)
                
        if final_state.get("quote_extraction_result"):
            print("\n[QUOTE EXTRACTION]")
            for item in final_state["quote_extraction_result"]:
                print(item)
                
        if final_state.get("grammar_and_bias_review_result"):
            print("\n[GRAMMAR & BIAS REVIEW]")
            for item in final_state["grammar_and_bias_review_result"]:
                print(item)
                
    except Exception as e:
        print(f"\nError executing workflow: {e}")

if __name__ == "__main__":
    main()
