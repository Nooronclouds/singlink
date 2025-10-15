import requests
import os
import json
from typing import List

class AITranscriber:
    def __init__(self):
        self.api_key = os.getenv("HF_API_KEY", "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")
        # Using a better model for text generation
        self.api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        
    def transcribe_signs(self, words: List[str]) -> str:
        """
        Takes detected sign words and converts them to coherent sentences using AI
        """
        if not words or len(words) == 0:
            return "No words detected yet"
        
        # If only one word, just capitalize it
        if len(words) == 1:
            return f"{words[0].capitalize()}."
        
        # Create a better prompt for the AI
        word_list = ", ".join(words)
        prompt = f"""[INST] You are a sign language interpreter. Convert these sign language words into a natural, grammatically correct English sentence. Only output the sentence, nothing else.

Sign words: {word_list}

Natural sentence: [/INST]"""
        
        try:
            if not self.api_key:
                print("Warning: No HF_API_KEY found, using fallback")
                return self.fallback_transcription(words)
            
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 100,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "do_sample": True,
                    "return_full_text": False
                },
                "options": {
                    "wait_for_model": True
                }
            }
            
            response = requests.post(
                self.api_url, 
                headers=headers, 
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get('generated_text', '').strip()
                    
                    # Clean up the response
                    if generated_text:
                        # Remove any remaining prompt text
                        generated_text = generated_text.replace(prompt, '').strip()
                        
                        # Ensure it ends with punctuation
                        if not generated_text.endswith(('.', '!', '?')):
                            generated_text += '.'
                        
                        # Capitalize first letter
                        if generated_text:
                            generated_text = generated_text[0].upper() + generated_text[1:]
                        
                        return generated_text
                
                # If AI didn't return good text, use fallback
                return self.fallback_transcription(words)
            else:
                print(f"AI API error: {response.status_code} - {response.text}")
                return self.fallback_transcription(words)
                
        except requests.exceptions.Timeout:
            print("AI API timeout, using fallback")
            return self.fallback_transcription(words)
        except Exception as e:
            print(f"AI Transcription error: {e}")
            return self.fallback_transcription(words)
    
    def fallback_transcription(self, words: List[str]) -> str:
        """
        Simple rule-based sentence formation when AI fails
        """
        if not words:
            return "No words detected yet"
        
        # Basic grammar rules for common sign patterns
        sentence = []
        
        for i, word in enumerate(words):
            word = word.lower()
            
            # Handle personal pronouns
            if word in ['i', 'me']:
                word = 'I'
            
            # Capitalize first word
            if i == 0:
                word = word.capitalize()
            
            sentence.append(word)
        
        # Join words
        result = ' '.join(sentence)
        
        # Add period if not present
        if not result.endswith(('.', '!', '?')):
            result += '.'
        
        return result


# Alternative: Using free GPT-like API (if HuggingFace doesn't work)
class AITranscriberAlternative:
    """
    Alternative transcriber using free APIs
    """
    def __init__(self):
        pass
    
    def transcribe_signs(self, words: List[str]) -> str:
        """
        Smarter fallback with grammar rules
        """
        if not words or len(words) == 0:
            return "No words detected yet"
        
        if len(words) == 1:
            return f"{words[0].capitalize()}."
        
        # Apply basic ASL to English grammar conversion
        sentence = self._asl_to_english(words)
        return sentence
    
    def _asl_to_english(self, words: List[str]) -> str:
        """
        Apply ASL to English grammar rules
        """
        words = [w.lower() for w in words]
        result = []
        
        # Handle common patterns
        if 'i' in words or 'me' in words:
            # Replace with proper pronoun
            words = ['I' if w in ['i', 'me'] else w for w in words]
        
        # Add articles for common nouns
        for i, word in enumerate(words):
            if word in ['home', 'house', 'work'] and i > 0:
                if words[i-1] not in ['the', 'a', 'my']:
                    result.append('the')
            result.append(word)
        
        # Handle verb tenses (basic)
        sentence = ' '.join(result)
        
        # Capitalize first letter
        sentence = sentence[0].upper() + sentence[1:] if sentence else sentence
        
        # Add punctuation
        if not sentence.endswith(('.', '!', '?')):
            # Add question mark for questions
            if any(q in words for q in ['who', 'what', 'where', 'when', 'why', 'how']):
                sentence += '?'
            else:
                sentence += '.'
        
        return sentence


# Usage:
if __name__ == "__main__":
    # Test the transcriber
    transcriber = AITranscriber()
    
    test_cases = [
        ["hello"],
        ["i", "hello"],
        ["thank", "you"],
        ["please", "help"],
        ["i", "work", "home"],
        ["bye", "i", "home"]
    ]
    
    for words in test_cases:
        result = transcriber.transcribe_signs(words)
        print(f"{words} → {result}")
