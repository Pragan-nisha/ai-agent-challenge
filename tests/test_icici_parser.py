import pandas as pd
import pytest
import sys
import os

# Add custom_parsers to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'custom_parsers'))

try:
    import icici_parser
except ImportError as e:
    pytest.skip(f"Could not import icici_parser: {e}")

def test_parse_matches_target():
    """Test that parse() output equals target CSV via DataFrame.equals (T4)"""
    
    # Load target DataFrame
    target_df = pd.read_csv("data\icici\result.csv")
    
    # Parse PDF using generated parser
    output_df = icici_parser.parse("data\icici\icici sample.pdf")
    
    # T4: DataFrame.equals comparison
    assert output_df.equals(target_df), f"Parser output doesn't match target CSV.\nOutput shape: {output_df.shape}\nTarget shape: {target_df.shape}"
    
    print(f"Test passed: {len(output_df)} rows match target exactly")

if __name__ == "__main__":
    test_parse_matches_target()
    print("Manual test completed successfully!")
