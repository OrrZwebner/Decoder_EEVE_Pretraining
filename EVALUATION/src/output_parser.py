"""
Output parser for cleaning model translations
"""

import json
import re
from langchain.schema import BaseOutputParser


class TranslationOutputParser(BaseOutputParser):
    """
    Parser to extract translation from JSON output
    Expected format: {"translation": "translated text here"}
    """
    
    def __init__(self, config=None):
        """
        Initialize parser
        
        Args:
            config: Configuration dictionary (not used, kept for compatibility)
        """
        pass
    
    def parse(self, output: str) -> str:
        """
        Parse JSON output and extract translation
        
        Args:
            output: Raw model output string
            
        Returns:
            Cleaned translation string
        """
        try:
            # Try to extract JSON from output
            json_str = self._extract_json(output)
            
            # Parse JSON
            data = json.loads(json_str)
            
            # Extract translation
            if 'translation' in data:
                translation = data['translation']
                return translation.strip()
            else:
                # Fallback: return raw output if no translation key
                return self._fallback_parse(output)
                
        except (json.JSONDecodeError, Exception) as e:
            # Fallback to simple parsing if JSON fails
            return self._fallback_parse(output)
    
    def _extract_json(self, output: str) -> str:
        """
        Extract JSON string from output (handles cases with extra text)
        
        Args:
            output: Raw output string
            
        Returns:
            JSON string
        """
        # Try to find JSON object in the output
        # Look for pattern: {...}
        match = re.search(r'\{[^{}]*"translation"[^{}]*\}', output, re.DOTALL)
        if match:
            return match.group(0)
        
        # If no match, try to extract everything between first { and last }
        first_brace = output.find('{')
        last_brace = output.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return output[first_brace:last_brace+1]
        
        # Return as-is if no braces found
        return output
    
    def _fallback_parse(self, output: str) -> str:
        """
        Fallback parser when JSON parsing fails
        
        Args:
            output: Raw output string
            
        Returns:
            Cleaned translation string
        """
        # Remove common prompt echoes
        if "translation" in output.lower():
            parts = output.lower().split("translation")
            if len(parts) > 1:
                # Take everything after "translation:"
                output = output[output.lower().find("translation") + len("translation"):]
                output = output.lstrip(':').strip()
        
        # Remove JSON artifacts
        output = output.strip().strip('{').strip('}').strip()
        output = output.replace('"translation":', '').strip()
        output = output.strip('"').strip("'").strip()
        
        # Take only first line (avoid explanations)
        lines = output.split('\n')
        output = lines[0].strip() if lines else output.strip()
        
        return output
    
    @property
    def _type(self) -> str:
        """Return parser type"""
        return "translation_json_parser"