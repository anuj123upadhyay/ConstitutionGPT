import requests
import json
from typing import Dict, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import GITHUB_TOKEN, GITHUB_MODEL_ENDPOINT


def construct_prompt(question: str, context: str) -> str:
    """Construct prompt with context and question"""
    prompt_template = """
You are a constitutional law expert specializing in the Constitution of India. Your task is to provide accurate, precise answers based on the provided constitutional context.

GUIDELINES:
- Answer ALL questions that can be answered from the provided constitutional context
- Provide lawyer-like precision with specific Article/Section references when available
- Keep responses concise, factual, and authoritative
- Cite relevant constitutional provisions and legal terminology
- Only respond "I don't have sufficient constitutional context to answer this question accurately" if the question is COMPLETELY unrelated to the provided context
- For constitutional questions where you have partial context, provide what you can and indicate what additional information might be needed

Constitutional Context:
{context}

Question: {question}

Constitutional Analysis:
"""
    return prompt_template.format(context=context, question=question)




def generate_response(prompt: str, context: str = "") -> str:
    """Generate response using GitHub Marketplace model"""
    try:
        # Construct the full prompt with context
        full_prompt = construct_prompt(prompt, context)
        
        headers = {
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Try different payload formats for GitHub Models
        payload = {
            "model": "gpt-4o-mini",  # or whatever model you're using
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions about the Constitution of India based on the provided context."
                },
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        print(f"Making request to: {GITHUB_MODEL_ENDPOINT}")
        print(f"Token starts with: {GITHUB_TOKEN[:10]}..." if GITHUB_TOKEN else "No token")
        
        # Make API call
        response = requests.post(
            GITHUB_MODEL_ENDPOINT,
            headers=headers,
            data=json.dumps(payload),
            timeout=30
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            # Extract response (adjust based on your model's response format)
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            error_msg = f"Error: Failed to get response from GitHub model. Status: {response.status_code}"
            print(f"Error details: {response.text}")
            return error_msg
            
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        print(error_msg)
        return error_msg
