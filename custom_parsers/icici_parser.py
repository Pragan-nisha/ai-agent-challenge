import sys
import os
import logging
import pandas as pd
import PyPDF2
import pdfplumber
import re
from typing import List, Tuple

class IntelligentPDFParser:
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def parse_pdf_to_csv(self, pdf_path, output_csv):
        try:
            all_data = []
            
            # Try to extract tables first
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        table = page.extract_table()
                        if table:
                            # Skip header row (first row) for all pages except the first
                            start_row = 1 if all_data else 0
                            all_data.extend(table[start_row:])
                            
            except Exception as e:
                logging.warning(f"Table extraction failed: {e}. Attempting text parsing.")
                
            # If no table data extracted, try text extraction
            if not all_data:
                with open(pdf_path, 'rb') as pdf_file:
                    reader = PyPDF2.PdfReader(pdf_file)
                    num_pages = len(reader.pages)
                    combined_text = ""
                    for page_num in range(num_pages):
                        page = reader.pages[page_num]
                        text = page.extract_text()
                        combined_text += text + "\n"
                
                # Parse text lines
                lines = combined_text.splitlines()
                all_data = []
                for line in lines:
                    # Match date pattern at start of line
                    if re.match(r'\d{2}-\d{2}-\d{4}', line):
                        # Split line into components
                        parts = line.split()
                        if len(parts) >= 4:  # Minimum expected parts
                            all_data.append(parts)
            
            if not all_data:
                logging.error("No data extracted from PDF")
                return
                
            # Create DataFrame
            if all_data:
                # Check if first row looks like a header
                if all_data[0][0] == "Date" and all_data[0][1] == "Description":
                    # Remove header row
                    all_data = all_data[1:]
                
                # Ensure we have the right number of columns
                processed_data = []
                for row in all_data:
                    if len(row) >= 5:
                        # Combine description parts if needed
                        date = row[0]
                        debit_amt = row[-3]
                        credit_amt = row[-2]
                        balance = row[-1]
                        description_parts = row[1:-3]
                        description = " ".join(description_parts)
                        processed_data.append([date, description, debit_amt, credit_amt, balance])
                    elif len(row) == 5:
                        processed_data.append(row)
                
                # Create DataFrame with correct column names
                df = pd.DataFrame(processed_data, columns=['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance'])
                
                # Convert Balance column to float
                df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce')
                df['Debit Amt'] = pd.to_numeric(df['Debit Amt'], errors='coerce')
                df['Credit Amt'] = pd.to_numeric(df['Credit Amt'], errors='coerce')
                
                # Save to CSV
                df.to_csv(output_csv, index=False)
                logging.info(f"Data successfully extracted and saved to {output_csv}")
            else:
                # Create empty DataFrame with headers if no data
                df = pd.DataFrame(columns=['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance'])
                df.to_csv(output_csv, index=False)
                logging.info(f"Empty CSV created at {output_csv}")

        except FileNotFoundError:
            logging.error(f"Error: PDF file not found at {pdf_path}")
        except PyPDF2.errors.PdfReadError:
            logging.error(f"Error: Could not read PDF file at {pdf_path}")
        except Exception as e:
            logging.exception(f"An unexpected error occurred: {e}")

def parse(pdf_path) -> pd.DataFrame:
    parser = IntelligentPDFParser()
    try:
        temp_csv = "temp_output.csv"
        parser.parse_pdf_to_csv(pdf_path, temp_csv)
        if os.path.exists(temp_csv):
            df = pd.read_csv(temp_csv)
            os.remove(temp_csv)
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error during parsing: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <pdf_path> <output_csv>")
        sys.exit(1)
    pdf_path = sys.argv[1]
    output_csv = sys.argv[2]
    parser = IntelligentPDFParser()
    parser.parse_pdf_to_csv(pdf_path, output_csv)