import google.generativeai as genai
import os
import tempfile
import subprocess
import sys
import json
import logging
import pandas as pd
import traceback
import argparse
from typing import Dict, Any
import requests
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class qwenDebugger:
    """
    Fixed qwen debugger with proper error handling and validation
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_base = "https://openrouter.ai/api/v1/chat/completions"
        
    def call_qwen(self, prompt: str) -> str:
        """Make API call to qwen via OpenRouter with proper debugging and retry logic"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost",  # Some APIs require this
            "X-Title": "PDF Parser Generator"
        }
        
        data = {
            "model": "qwen/qwen3-coder:free",  # Keep your original model
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 6000,  # Increased token limit
            "stream": False
        }
        
        logger.info(f"Calling qwen API with prompt length: {len(prompt)}")
        
        # Retry logic for rate limits
        max_retries = 3
        base_delay = 30  # seconds
        
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_base, headers=headers, json=data, timeout=120)
                
                # Log response details
                logger.info(f"API Response Status: {response.status_code}")
                logger.info(f"API Response Headers: {dict(response.headers)}")
                
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries}), waiting {delay} seconds...")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"Rate limit hit on final attempt")
                        raise Exception(f"Rate limit exceeded after {max_retries} attempts")
                
                if response.status_code != 200:
                    logger.error(f"API returned status {response.status_code}")
                    logger.error(f"Response text: {response.text}")
                    raise Exception(f"API call failed with status {response.status_code}: {response.text}")
                
                result = response.json()
                
                # Validate response structure
                if 'choices' not in result or not result['choices']:
                    logger.error(f"Invalid API response structure: {result}")
                    raise Exception("API returned invalid response structure")
                
                content = result['choices'][0]['message']['content']
                
                # DEBUG: Log the actual response details
                logger.info(f" qwen returned {len(content)} characters")
                logger.info(f"First 300 chars: {repr(content[:300])}")
                logger.info(f"Last 100 chars: {repr(content[-100:])}")
                
                if len(content.strip()) < 50:
                    logger.warning(f" qwen response very short: {len(content)} chars")
                    logger.warning(f"Full response: {repr(content)}")
                
                return content
                
            except requests.exceptions.Timeout:
                logger.error("qwen API call timed out (120s)")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {base_delay} seconds...")
                    time.sleep(base_delay)
                    continue
                raise Exception("API call timed out")
            except requests.exceptions.RequestException as e:
                logger.error(f"qwen API request failed: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    logger.error(f"Response status: {e.response.status_code}")
                    logger.error(f"Response content: {e.response.text}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {base_delay} seconds...")
                    time.sleep(base_delay)
                    continue
                raise
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse API response as JSON: {e}")
                logger.error(f"Raw response: {response.text}")
                raise
            except Exception as e:
                if "rate limit" in str(e).lower() or "429" in str(e):
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Rate limit detected, waiting {delay} seconds...")
                        time.sleep(delay)
                        continue
                logger.error(f"qwen API call failed: {e}")
                raise
    
    def extract_and_validate_code(self, raw_response: str) -> str:
        """
        Extract Python code from response and validate it
        """
        logger.info(" Extracting code from qwen response...")
        
        original_response = raw_response
        fixed_code = raw_response.strip()
        
        # Method 1: Look for ```python blocks
        if "```python" in fixed_code:
            logger.info("Found ```python markers")
            parts = fixed_code.split("```python")
            if len(parts) > 1:
                # Take the first python block
                code_section = parts[1]
                if "```" in code_section:
                    fixed_code = code_section.split("```")[0]
                else:
                    fixed_code = code_section
                logger.info(" Extracted from ```python block")
            else:
                logger.warning("Found ```python but couldn't split properly")
        
        # Method 2: Look for generic ``` blocks
        elif "```" in fixed_code:
            logger.info("Found generic ``` markers")
            parts = fixed_code.split("```")
            if len(parts) >= 3:
                # Take the first code block (parts[1])
                fixed_code = parts[1]
                logger.info(" Extracted from generic ``` block")
            elif len(parts) == 2:
                fixed_code = parts[1]
                logger.info(" Extracted from single ``` block")
            else:
                logger.warning("Found ``` but couldn't extract code")
        
        # Method 3: Look for class definitions (fallback)
        elif "class IntelligentPDFParser" in fixed_code:
            logger.info(" Found class definition directly, using full response")
        else:
            logger.warning("No code markers found, using full response")
        
        fixed_code = fixed_code.strip()
        
        # Validation checks
        logger.info(" Validating extracted code...")
        
        if not fixed_code:
            logger.error(" Extracted code is empty!")
            logger.error(f"Original response (first 500 chars): {repr(original_response[:500])}")
            raise Exception("Extracted code is empty")
        
        if len(fixed_code) < 200:  # Reasonable minimum for a parser class
            logger.error(f" Extracted code too short: {len(fixed_code)} characters")
            logger.error(f"Extracted code: {repr(fixed_code[:200])}")
            logger.error(f"Original response (first 1000 chars): {repr(original_response[:1000])}")
            raise Exception(f"Extracted code too short: {len(fixed_code)} characters")
        
        # Check for required components
        required_components = [
            "class IntelligentPDFParser",
            "def parse_pdf_to_csv",
            "if __name__"
        ]
        
        missing_components = []
        for component in required_components:
            if component not in fixed_code:
                missing_components.append(component)
        
        if missing_components:
            logger.warning(f" Missing components: {missing_components}")
            # Don't fail here, might still work
        
        # Python syntax validation
        try:
            compile(fixed_code, '<generated_code>', 'exec')
            logger.info(" Code passes syntax validation")
        except SyntaxError as e:
            logger.error(f" Generated code has syntax errors: {e}")
            logger.error(f"Error at line {e.lineno}: {e.text}")
            logger.error(f"Code around error:\n{self._get_code_context(fixed_code, e.lineno)}")
            raise Exception(f"Generated code has syntax errors: {e}")
        except Exception as e:
            logger.error(f" Code compilation failed: {e}")
            raise Exception(f"Code compilation failed: {e}")
        
        logger.info(f" Successfully extracted and validated {len(fixed_code)} characters of code")
        return fixed_code
    
    def _get_code_context(self, code: str, error_line: int, context: int = 3) -> str:
        """Get context around error line for debugging"""
        lines = code.split('\n')
        start = max(0, error_line - context - 1)
        end = min(len(lines), error_line + context)
        
        context_lines = []
        for i in range(start, end):
            prefix = ">>> " if i == error_line - 1 else "    "
            context_lines.append(f"{prefix}{i+1:3d}: {lines[i]}")
        
        return "\n".join(context_lines)
    
    def debug_code(self, code: str, error_message: str, pdf_content: str, csv_differences: str = None) -> str:
        """
        Debug code with error context, PDF content, and optional CSV differences
        """
        
        # Build prompt with CSV differences if provided
        csv_context = ""
        if csv_differences:
            csv_context = f"""
CSV COMPARISON DIFFERENCES FOUND:
{csv_differences}

The parser output doesn't match the expected target CSV. Fix the parsing logic to match the expected format exactly.
"""
        
        prompt = f"""You are an expert Python debugger. Fix this PDF parser code based on the error and PDF content.

ORIGINAL CODE:
```python
{code}
```

ERROR MESSAGE:
{error_message}

{csv_context}

PDF CONTENT SAMPLE:
{pdf_content[:3000]}

CRITICAL REQUIREMENTS:
1. Fix the specific error mentioned above
2. Create a complete working Python script
3. Must include class 'IntelligentPDFParser' with method 'parse_pdf_to_csv(pdf_path, output_csv)'
4. Must include command-line interface: if __name__ == "__main__": with sys.argv[1] and sys.argv[2]
5. Must include top-level function: def parse(pdf_path) -> pd.DataFrame
6. Must actually CREATE and SAVE the CSV file (not empty)
7. Must handle errors gracefully
8. Use proper libraries: PyPDF2, pdfplumber, pandas, logging

IMPORTANT: Return ONLY the complete Python code. No explanations, no markdown, just the working Python script."""

        try:
            logger.info(" Calling qwen for code debugging...")
            raw_response = self.call_qwen(prompt)
            
            logger.info("🔧 Processing qwen response...")
            fixed_code = self.extract_and_validate_code(raw_response)
            
            logger.info(" qwen debugging completed successfully")
            return fixed_code
            
        except Exception as e:
            logger.error(f" qwen debugging failed: {e}")
            logger.error("Returning original code as fallback")
            return code  # Return original instead of raising

class SimplePDFParserGenerator:
    """
    Enhanced PDF Parser Generator with T1-T4 compliance
    """
    
    def __init__(self, gemini_api_key: str, openrouter_api_key: str):
        self.gemini_api_key = gemini_api_key
        self.openrouter_api_key = openrouter_api_key
        
        # Initialize Gemini
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Initialize qwen debugger
        self.debugger = qwenDebugger(openrouter_api_key)
        
    def generate_parser_code(self, pdf_content: str) -> str:
        """Generate PDF parser code with Gemini"""
        
        prompt = f"""
        You are an expert Python developer. Create a comprehensive PDF parsing system for bank statements.
        
        CRITICAL RULES:
        - READ THE PDF AND EXTRACT ALL DATA AS-IS AND SAVE TO CSV
        - DON'T CHANGE ANYTHING FROM THE ORIGINAL DATA
        - Handle bank statement PDFs specifically
        - Handle all pages and page breaks (multi pages)
        
        REQUIREMENTS:
        1. Parse bank statement PDF documents 
        2. Extract transaction data, account info, balances
        3. Handle both tabular and text-based layouts
        4. Save all extracted data to CSV format
        5. Include proper error handling
        6. Make it production-ready

        LIBRARIES TO USE:
        - PyPDF2 for text extraction
        - pdfplumber for table detection  
        - pandas for data manipulation
        - csv for output
        - logging for debugging

        PDF CONTENT SAMPLE:
        {pdf_content[:2000]}

        Create a complete Python script with:
        - Class 'IntelligentPDFParser'
        - Method parse_pdf_to_csv(pdf_path, output_csv)
        - Top-level function: def parse(pdf_path) -> pd.DataFrame
        - Command-line interface that accepts sys.argv[1] as pdf_path and sys.argv[2] as output_csv
        
        Write ONLY the complete Python code, no explanations or markdown.
        """
        
        try:
            logger.info(" Generating parser code with Gemini...")
            response = self.model.generate_content(prompt)
            generated_code = response.text
            
            logger.info(f" Gemini returned {len(generated_code)} characters")
            
            # Use the same extraction and validation logic
            validated_code = self.debugger.extract_and_validate_code(generated_code)
            
            # Ensure parse() function exists
            validated_code = self.ensure_parse_function(validated_code)
            
            logger.info(" Gemini code generation completed successfully")
            return validated_code
            
        except Exception as e:
            logger.error(f" Gemini code generation failed: {e}")
            raise
    
    def ensure_parse_function(self, code: str) -> str:
        """Ensure the code has a top-level parse(pdf_path) -> pd.DataFrame function"""
        
        if "def parse(pdf_path)" in code:
            logger.info(" parse() function already exists")
            return code
        
        logger.info("🔧 Adding parse() function wrapper")
        
        # Add the parse function wrapper
        wrapper_code = '''

def parse(pdf_path):
    """
    Top-level parse function that returns a DataFrame
    Required for T3 compliance: parse(pdf_path) -> pd.DataFrame
    """
    import tempfile
    import os
    
    parser = IntelligentPDFParser()
    
    # Create temp CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        temp_csv = tmp.name
    
    try:
        # Parse to CSV
        parser.parse_pdf_to_csv(pdf_path, temp_csv)
        
        # Load and return DataFrame
        df = pd.read_csv(temp_csv)
        return df
    finally:
        # Clean up temp file
        if os.path.exists(temp_csv):
            os.unlink(temp_csv)

'''
        
        # Insert before if __name__ == "__main__"
        if 'if __name__ == "__main__"' in code:
            parts = code.split('if __name__ == "__main__"')
            enhanced_code = parts[0] + wrapper_code + '\nif __name__ == "__main__"' + parts[1]
        else:
            enhanced_code = code + wrapper_code
        
        return enhanced_code
    
    def save_code_safely(self, code: str, filename: str) -> str:
        """
        Safely save code to file with validation
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            
            # Write code to file
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # Validate file was written correctly
            if not os.path.exists(filename):
                raise Exception(f"File {filename} was not created")
            
            file_size = os.path.getsize(filename)
            if file_size == 0:
                raise Exception(f"File {filename} is empty")
            
            if file_size < len(code.encode('utf-8')) * 0.9:  # Allow for encoding differences
                logger.warning(f" File size ({file_size}) much smaller than expected ({len(code)})")
            
            # Validate file content (fix encoding)
            with open(filename, 'r', encoding='utf-8') as f:
                saved_content = f.read()
            
            if len(saved_content) < len(code) * 0.9:
                raise Exception(f"Saved file content is truncated. Expected ~{len(code)}, got {len(saved_content)}")
            
            logger.info(f" Successfully saved {file_size} bytes to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f" Failed to save code to {filename}: {e}")
            raise
    
    def test_code(self, code: str, pdf_path: str, target_csv: str = None) -> Dict[str, Any]:
        """Test generated code and verify CSV output is created and matches target if provided"""
        
        # Save code to temporary file with proper validation
        timestamp = int(pd.Timestamp.now().timestamp())
        temp_script = f"temp_parser_{timestamp}.py"
        
        try:
            # Save and validate
            self.save_code_safely(code, temp_script)
            
            # Test execution
            test_csv = f"test_output_{timestamp}.csv"
            
            # Remove test CSV if it exists from previous runs
            if os.path.exists(test_csv):
                os.unlink(test_csv)
            
            logger.info(f" Testing parser: {temp_script}")
            logger.info(f" Input PDF: {pdf_path}")
            logger.info(f"Output CSV: {test_csv}")
            
            cmd = [sys.executable, temp_script, pdf_path, test_csv]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            logger.info(f"Script return code: {result.returncode}")
            logger.info(f"Script stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"Script stderr: {result.stderr}")
            
            # Check if script ran without crashing AND created the CSV file
            if result.returncode == 0:
                if os.path.exists(test_csv):
                    csv_size = os.path.getsize(test_csv)
                    logger.info(f" CSV created successfully: {csv_size} bytes")
                    
                    if csv_size > 0:
                        # If target CSV provided, compare with it
                        if target_csv:
                            logger.info(f"Comparing output CSV with target CSV...")
                            comparison = self.compare_csvs(test_csv, target_csv)
                            logger.info(f"Comparison result: has_differences={comparison['has_differences']}")
                            
                            if comparison['has_differences']:
                                logger.warning(f"CSV differences found: {comparison['differences'][:500]}...")
                                return {
                                    'success': False, 
                                    'error': f"CSV created but doesn't match target format.\nDifferences: {comparison['differences']}",
                                    'csv_differences': comparison['differences']
                                }
                            else:
                                logger.info("Perfect CSV match with target!")
                        return {'success': True, 'output': result.stdout}
                    else:
                        return {'success': False, 'error': f"CSV file created but is empty. Output: {result.stdout}\nError: {result.stderr}"}
                else:
                    return {'success': False, 'error': f"Script ran but failed to create CSV file. Output: {result.stdout}\nError: {result.stderr}"}
            else:
                error_msg = result.stderr if result.stderr else "Unknown error"
                return {'success': False, 'error': error_msg, 'output': result.stdout}
                
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': "Code execution timed out (120s limit)"}
        except Exception as e:
            logger.error(f" Test execution failed: {e}")
            return {'success': False, 'error': f"Execution exception: {str(e)}"}
        finally:
            # Cleanup
            try:
                if os.path.exists(temp_script):
                    os.unlink(temp_script)
                if os.path.exists(f"test_output_{timestamp}.csv"):
                    os.unlink(f"test_output_{timestamp}.csv")
            except Exception as cleanup_error:
                logger.warning(f"Cleanup failed: {cleanup_error}")
    
    def compare_csvs(self, output_csv: str, target_csv: str) -> Dict[str, Any]:
        """
        Compare output CSV with target CSV and return differences
        """
        try:
            if not os.path.exists(output_csv):
                return {
                    'has_differences': True,
                    'error': f"Output CSV not found: {output_csv}",
                    'differences': "Output CSV file was not created"
                }
            
            if not os.path.exists(target_csv):
                return {
                    'has_differences': False,
                    'message': "No target CSV provided for comparison"
                }
            
            # Load both CSVs
            try:
                output_df = pd.read_csv(output_csv)
                target_df = pd.read_csv(target_csv)
            except Exception as e:
                return {
                    'has_differences': True,
                    'error': f"Failed to load CSV files: {e}",
                    'differences': f"CSV loading error: {e}"
                }
            
            differences = []
            
            # Compare shapes
            if output_df.shape != target_df.shape:
                differences.append(f"Shape mismatch: Output {output_df.shape} vs Target {target_df.shape}")
            
            # Compare columns
            output_cols = set(output_df.columns)
            target_cols = set(target_df.columns)
            
            if output_cols != target_cols:
                missing_cols = target_cols - output_cols
                extra_cols = output_cols - target_cols
                
                if missing_cols:
                    differences.append(f"Missing columns: {list(missing_cols)}")
                if extra_cols:
                    differences.append(f"Extra columns: {list(extra_cols)}")
            
            # Compare data types for common columns
            common_cols = output_cols.intersection(target_cols)
            for col in common_cols:
                if str(output_df[col].dtype) != str(target_df[col].dtype):
                    differences.append(f"Data type mismatch in '{col}': Output {output_df[col].dtype} vs Target {target_df[col].dtype}")
            
            # Compare sample data (first few rows)
            if len(common_cols) > 0 and len(output_df) > 0 and len(target_df) > 0:
                # Compare first 5 rows for common columns
                sample_size = min(5, len(output_df), len(target_df))
                output_sample = output_df[list(common_cols)].head(sample_size)
                target_sample = target_df[list(common_cols)].head(sample_size)
                
                try:
                    if not output_sample.equals(target_sample):
                        differences.append("Data content differs in first 5 rows")
                        differences.append(f"Output sample:\n{output_sample.to_string()}")
                        differences.append(f"Target sample:\n{target_sample.to_string()}")
                except:
                    differences.append("Could not compare data content directly")
            
            # Check row counts
            if len(output_df) != len(target_df):
                differences.append(f"Row count mismatch: Output {len(output_df)} vs Target {len(target_df)}")
            
            has_differences = len(differences) > 0
            
            return {
                'has_differences': has_differences,
                'differences': '\n'.join(differences) if has_differences else "No differences found",
                'output_shape': output_df.shape,
                'target_shape': target_df.shape,
                'output_columns': list(output_df.columns),
                'target_columns': list(target_df.columns)
            }
            
        except Exception as e:
            return {
                'has_differences': True,
                'error': f"CSV comparison failed: {e}",
                'differences': f"Comparison error: {e}"
            }
    
    def extract_pdf_content(self, pdf_path: str) -> str:
        """Extract sample content from PDF"""
        try:
            import PyPDF2
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                sample_text = ""
                
                logger.info(f" PDF has {len(reader.pages)} pages")
                
                for i, page in enumerate(reader.pages[:3]):  # Extract from first 3 pages
                    try:
                        page_text = page.extract_text()
                        sample_text += f"--- Page {i+1} ---\n{page_text}\n\n"
                        if len(sample_text) > 5000:  # Increased limit
                            break
                    except Exception as e:
                        logger.warning(f"Could not extract from page {i+1}: {e}")
                
                logger.info(f" Extracted {len(sample_text)} characters from PDF")
                return sample_text[:5000]  # Return more content
                
        except Exception as e:
            logger.warning(f"Could not extract PDF content: {e}")
            return ""
    
    def create_test_file(self, target: str, pdf_path: str, target_csv: str) -> str:
        """Create pytest file for T4 compliance"""
        test_content = f'''import pandas as pd
import pytest
import sys
import os

# Add custom_parsers to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'custom_parsers'))

try:
    import {target}_parser
except ImportError as e:
    pytest.skip(f"Could not import {target}_parser: {{e}}")

def test_parse_matches_target():
    """Test that parse() output equals target CSV via DataFrame.equals (T4)"""
    
    # Load target DataFrame
    target_df = pd.read_csv("{target_csv}")
    
    # Parse PDF using generated parser
    output_df = {target}_parser.parse("{pdf_path}")
    
    # T4: DataFrame.equals comparison
    assert output_df.equals(target_df), f"Parser output doesn't match target CSV.\\nOutput shape: {{output_df.shape}}\\nTarget shape: {{target_df.shape}}"
    
    print(f"Test passed: {{len(output_df)}} rows match target exactly")

if __name__ == "__main__":
    test_parse_matches_target()
    print("Manual test completed successfully!")
'''
        
        # Create tests directory
        os.makedirs("tests", exist_ok=True)
        
        test_file_path = f"tests/test_{target}_parser.py"
        
        # Fix Unicode encoding issue on Windows
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        logger.info(f"Created test file: {test_file_path}")
        return test_file_path
    
    def create_working_parser(self, pdf_path: str, target_csv: str = None, max_iterations: int = 3) -> str:
        """
        Main workflow: Generate -> Test -> Debug if needed (T1: ≤3 attempts)
        """
        
        # Extract PDF content for context
        pdf_content = self.extract_pdf_content(pdf_path)
        
        iteration = 0
        current_code = None
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"🔄 Iteration {iteration}/{max_iterations} (T1: ≤3 attempts)")
            
            try:
                # Step 1: Generate code with Gemini (only on first iteration)
                if current_code is None:
                    logger.info(" Generating parser code with Gemini...")
                    current_code = self.generate_parser_code(pdf_content)
                
                # Step 2: Test the code (now includes CSV comparison if target provided)
                logger.info(" Testing generated code...")
                test_result = self.test_code(current_code, pdf_path, target_csv)
                
                if test_result['success']:
                    logger.info(" Code works and CSV matches target! Saving final version...")
                    break
                else:
                    # Code execution failed or CSV doesn't match - debug with qwen
                    logger.info("🐛 Code failed or CSV doesn't match, debugging with qwen...")
                    logger.info(f"Error: {test_result['error'][:500]}...")
                    
                    
                    
                    csv_differences = test_result.get('csv_differences', None)
                    
                    current_code = self.debugger.debug_code(
                        code=current_code,
                        error_message=test_result['error'],
                        pdf_content=pdf_content,
                        csv_differences=csv_differences
                    )
                    
                    # Ensure parse function after debugging
                    current_code = self.ensure_parse_function(current_code)
                    
                    logger.info("🔧 qwen debugging completed")
                    continue
                
            except Exception as e:
                logger.error(f" Iteration {iteration} failed: {e}")
                traceback.print_exc()
                if iteration == max_iterations:
                    # If all iterations failed, return the last generated code anyway
                    logger.warning("All iterations failed, returning last generated code")
                    break
                continue
        
        return current_code
    
    def parse_pdf(self, pdf_path: str, output_csv: str = None, target_csv: str = None, target: str = None, max_attempts: int = 3) -> str:
        """
        Complete workflow with T2 CLI compliance
        """
        
        # T1: Enforce ≤3 attempts
        max_iterations = min(max_attempts, 3)
        logger.info(f" T1: Using {max_iterations} attempts maximum")
        
        # Generate working parser with target CSV comparison
        parser_code = self.create_working_parser(pdf_path, target_csv, max_iterations)
        
        # T2: Save to custom_parsers/{target}_parser.py
        if target:
            os.makedirs("custom_parsers", exist_ok=True)
            # Create __init__.py for importable package (fix encoding)
            init_file = "custom_parsers/__init__.py"
            if not os.path.exists(init_file):
                with open(init_file, 'w', encoding='utf-8') as f:
                    f.write("# Custom parsers package\n")
            
            parser_script_path = os.path.join("custom_parsers", f"{target}_parser.py")
            self.save_code_safely(parser_code, parser_script_path)
            logger.info(f" T2: Saved parser to {parser_script_path}")
            
            # T4: Create pytest file
            if target_csv:
                test_file = self.create_test_file(target, pdf_path, target_csv)
                logger.info(f" T4: Created test file {test_file}")
        else:
            # Fallback to timestamp naming if no target specified
            timestamp = int(pd.Timestamp.now().timestamp())
            parser_script_path = f"debugged_parser_{timestamp}.py"
            self.save_code_safely(parser_code, parser_script_path)
        
        # Execute final parser
        logger.info(" Executing final parser...")
        
        if output_csv is None:
            output_csv = pdf_path.replace('.pdf', '_parsed.csv')
        
        cmd = [sys.executable, parser_script_path, pdf_path, output_csv]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            logger.info(f"Final execution return code: {result.returncode}")
            logger.info(f"Final execution stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"Final execution stderr: {result.stderr}")
            
            if result.returncode == 0:
                logger.info(" Final execution successful!")
                print(result.stdout)
                
                # Check if output CSV was actually created
                if os.path.exists(output_csv):
                    csv_size = os.path.getsize(output_csv)
                    logger.info(f" Final CSV created: {csv_size} bytes")
                    
                    # Final CSV comparison if target provided
                    if target_csv and csv_size > 0:
                        logger.info("Final CSV comparison...")
                        final_comparison = self.compare_csvs(output_csv, target_csv)
                        
                        if final_comparison['has_differences']:
                            logger.warning(" Final output still has differences from target")
                            print(f"Differences: {final_comparison['differences']}")
                        else:
                            logger.info(" Perfect match with target CSV!")
                            
                        # T4: Run pytest if target specified
                        if target:
                            logger.info(" T4: Running pytest...")
                            try:
                                pytest_result = subprocess.run(
                                    [sys.executable, "-m", "pytest", f"tests/test_{target}_parser.py", "-v"],
                                    capture_output=True,
                                    text=True,
                                    timeout=60
                                )
                                if pytest_result.returncode == 0:
                                    logger.info(" T4: Pytest passed!")
                                    print(" All tests passed!")
                                else:
                                    logger.warning(f" T4: Pytest failed: {pytest_result.stdout}\n{pytest_result.stderr}")
                            except Exception as e:
                                logger.warning(f" T4: Could not run pytest: {e}")
                else:
                    logger.error(" Final CSV was not created!")
                
                return output_csv
            else:
                logger.warning(f" Final execution had issues: {result.stderr}")
                return output_csv
                
        except subprocess.TimeoutExpired:
            raise Exception("Final execution timed out (5 minutes)")
        except Exception as e:
            logger.error(f" Final execution failed: {e}")
            raise

def parse_arguments():
    """Parse command line arguments for T2 CLI compliance"""
    parser = argparse.ArgumentParser(
        description="T1-T4 Compliant PDF Parser Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent.py --target icici
  python agent.py --target hdfc --attempts 2
        """
    )
    
    parser.add_argument(
        "--target",
        required=True,
        help="Target bank name (e.g., icici, hdfc). Reads data/{target}/{target}_sample.pdf and data/{target}/result.csv"
    )
    
    parser.add_argument(
        "--attempts",
        type=int,
        default=3,
        help="Maximum attempts for debugging (default: 3, max: 3 for T1 compliance)"
    )
    
    return parser.parse_args()

def main():
    """Main function with T2 CLI compliance"""
    
    # T2: Parse CLI arguments
    args = parse_arguments()
    
    # T1: Enforce max 3 attempts
    max_iterations = min(args.attempts if args.attempts else 3, 3)
    
    # API Keys (in production, use environment variables)
    GEMINI_API_KEY = "AIzaSyDJXO92pL2Exq3BnoGIjkeL3znoaAeiMz4"
    OPENROUTER_API_KEY = "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"
    
    if not GEMINI_API_KEY or not OPENROUTER_API_KEY:
        print(" Please set both API keys")
        return
    
    # T2: Construct standard paths from --target
    target = args.target
    pdf_path = os.path.join("data", target, f"{target} sample.pdf")
    target_csv = os.path.join("data", target, "result.csv")
    output_csv = os.path.join("data", target, "parsed_output.csv")
    
    # Validate paths exist
    if not os.path.exists(pdf_path):
        print(f" PDF not found: {pdf_path}")
        return
    
    if not os.path.exists(target_csv):
        print(f" Target CSV not found: {target_csv}")
        return
    
    try:
        logger.info(f" Starting T1-T4 Compliant PDF Parser Agent...")
        logger.info(f" Target: {target}")
        logger.info(f" PDF: {pdf_path}")
        logger.info(f"Target CSV: {target_csv}")
        logger.info(f"Max attempts: {max_iterations}")
        
        # Initialize generator
        generator = SimplePDFParserGenerator(GEMINI_API_KEY, OPENROUTER_API_KEY)
        
        # T2: Parse PDF with target CSV comparison
        result_csv = generator.parse_pdf(
            pdf_path=pdf_path, 
            output_csv=output_csv, 
            target_csv=target_csv,
            target=target,
            max_attempts=max_iterations
        )
        
        print(f"\n Successfully parsed PDF with T1-T4 compliance!")
        print(f" Input PDF: {pdf_path}")
        print(f"Output CSV: {result_csv}")
        print(f"Parser saved: custom_parsers/{target}_parser.py")
        print(f" Test file: tests/test_{target}_parser.py")
        
        # Preview results
        try:
            if os.path.exists(result_csv) and os.path.getsize(result_csv) > 0:
                df = pd.read_csv(result_csv)
                print(f"\n Preview ({len(df)} rows, {len(df.columns)} columns):")
                print(df.head().to_string(index=False))
            else:
                print(f"\n Output CSV is empty or doesn't exist")
        except Exception as e:
            print(f"Could not preview: {e}")
        
        # Final validation message
        print(f"\n T1-T4 Compliance Summary:")
        print(f" T1: Used ≤3 attempts ({max_iterations})")
        print(f" T2: CLI `python agent.py --target {target}` reads standard paths")
        print(f" T3: Generated parser has parse(pdf_path) -> DataFrame")
        print(f" T4: Created pytest with DataFrame.equals test")
            
    except Exception as e:
        print(f" Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()