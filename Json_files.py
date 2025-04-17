import re
import fitz  # PyMuPDF
import json

# Define regex patterns for financial numbers
patterns = {
    "Revenue": r"Revenue[s]?[^\d]*([\d,.]+)\s*(billion|million)?",
    "Operating Income": r"Operating income[^\d]*([\d,.]+)\s*(billion|million)?",
    "Net Income": r"Net income[^\d]*([\d,.]+)\s*(billion|million)?",
    "EPS": r"Diluted EPS[^\d]*([\d.]+)"
}

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(pdf_file)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# Function to extract financial data safely
def extract_financial_metrics(text, patterns):
    extracted_data = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            groups = match.groups()  # Get all groups safely
            value = groups[0].replace(",", "") if groups[0] else None  # First group (number)
            unit = groups[1].lower() if len(groups) > 1 and groups[1] else ""  # Second group (unit)

            # Convert to numeric values
            try:
                value = float(value)
                if unit == "billion":
                    value *= 1e9
                elif unit == "million":
                    value *= 1e6
            except ValueError:
                value = None  # Handle unexpected formatting
        else:
            value = None  # If no match found

        extracted_data[key] = value

    return extracted_data

# Function to extract and save financial data
def extract_and_save_financial_data(pdf_file, output_file):
    raw_text = extract_text_from_pdf(pdf_file)
    financial_data = extract_financial_metrics(raw_text, patterns)

    # Save extracted data as JSON
    with open(output_file, 'w') as json_file:
        json.dump(financial_data, json_file)

# Example usage (for Google)
pdf_path = '/Users/adwaitaboralkar/Downloads/googleEarnings.pdf'  # Replace with the actual path
output_file = 'google_financial_data.json'  # The output JSON file
extract_and_save_financial_data(pdf_path, output_file)

import re
import fitz  # PyMuPDF
import json

# Define regex patterns for financial numbers
patterns = {
    "Revenue": r"Revenue[s]?[^\d]*([\d,.]+)\s*(billion|million)?",
    "Operating Income": r"Operating income[^\d]*([\d,.]+)\s*(billion|million)?",
    "Net Income": r"Net income[^\d]*([\d,.]+)\s*(billion|million)?",
    "EPS": r"Diluted EPS[^\d]*([\d.]+)"
}

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(pdf_file)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# Function to extract financial data safely
def extract_financial_metrics(text, patterns):
    extracted_data = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            groups = match.groups()  # Get all groups safely
            value = groups[0].replace(",", "") if groups[0] else None  # First group (number)
            unit = groups[1].lower() if len(groups) > 1 and groups[1] else ""  # Second group (unit)

            # Convert to numeric values
            try:
                value = float(value)
                if unit == "billion":
                    value *= 1e9
                elif unit == "million":
                    value *= 1e6
            except ValueError:
                value = None  # Handle unexpected formatting
        else:
            value = None  # If no match found

        extracted_data[key] = value

    return extracted_data

# Main function to process the PDF and save as JSON
def extract_and_save_financial_data(pdf_path, output_file):
    raw_text = extract_text_from_pdf(pdf_path)
    financial_data = extract_financial_metrics(raw_text, patterns)

    # Save the extracted data as JSON
    with open(output_file, "w") as f:
        json.dump(financial_data, f, indent=4)

df_path = '/Users/adwaitaboralkar/Downloads/googleEarnings.pdf'  # Update with actual path
output_file = 'google_financial_data.json'  # Output file for Tesla data
extract_and_save_financial_data(pdf_path, output_file)
print(output_file)

pdf_path = '/Users/adwaitaboralkar/Downloads/Tesla2024.pdf'  # Update with actual path
output_file = 'tesla_financial_data.json'  # Output file for Tesla data
extract_and_save_financial_data(pdf_path, output_file)
print(output_file)

pdf_path = '/Users/adwaitaboralkar/Downloads/BerkshireHathaway.pdf'  # Update with actual path
output_file = 'berkshire_hathaway_financial_data.json'  # Output file for Berkshire Hathaway data
extract_and_save_financial_data(pdf_path, output_file)
print(output_file)

pdf_path = '/Users/adwaitaboralkar/Downloads/OpendoorTechnologies.pdf'  # Replace with actual path to the Opendoor Technologies PDF
output_file = 'opendoor_technologies_financial_data.json'  # Output file for Opendoor Technologies data
extract_and_save_financial_data(pdf_path, output_file)
print(output_file)
