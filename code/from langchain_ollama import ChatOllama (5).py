import pandas as pd
from langchain_ollama import ChatOllama
import arabic_reshaper
from bidi.algorithm import get_display
import re

# Load the LLM model
llm = ChatOllama(
    model="llama3.2",  # Ensure this is the correct model name
    temperature=0.2,    # Adjust for relevance and consistency
    max_tokens=1024     # Define token limit for output
)

# Define the key point extraction prompt template
key_point_prompt_template = """
يرجى قراءة النص التالي واستخراج النقاط الرئيسية فقط (مثل الشكوى الطبية أو الأعراض أو أي تفاصيل مهمة).
إذا كان بالإمكان استخراج العمر، يرجى تضمينه.
إذا لم يتمكن النموذج من الاستخراج، يرجى ترك النص كما هو دون أي تعديل.

النص: {text}
"""

# Function to reshape Arabic text for display in the terminal
def reshape_arabic_text_for_terminal(text):
    reshaped_text = arabic_reshaper.reshape(text)  # Reshape for proper Arabic text
    bidi_text = get_display(reshaped_text)  # Handle bidirectional text
    return bidi_text

# Function to extract key points from Arabic text
def extract_key_points(text):
    # Construct the prompt
    prompt = key_point_prompt_template.format(text=text)

    # Construct the messages for ChatOllama
    messages = [
        {"role": "system", "content": "You are an assistant specializing in extracting key points from Arabic text."},
        {"role": "user", "content": prompt}
    ]

    try:
        # Generate the key points using llm.invoke() method
        result = llm.invoke(messages)

        # Debug: Print raw model output
        print("Raw Output:", result)

        # Extract the content from the model's response
        if hasattr(result, "content") and result.content:
            extracted_text = result.content.strip()

            # Remove the phrase "النصوص المستخرجة" if it appears in the result
            extracted_text = extracted_text.replace("النصوص المستخرجة", "").strip()

            # Remove any unnecessary hyphens from the text
            extracted_text = extracted_text.replace("-", "").strip()

            # Return the extracted text or the original text if nothing meaningful was extracted
            return extracted_text if extracted_text else text

        # If no key points were extracted, return the original text
        return text
    except Exception as e:
        print(f"Error while extracting key points: {e}")
        return text

# Function to process the dataset and save the results
def process_dataset(input_excel_path, output_excel_path):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(input_excel_path)

    # Ensure the "Post" column exists
    if "Post" not in df.columns:
        print("Error: The 'Post' column is missing in the dataset.")
        return

    # Create a new column to store the extracted key points
    df['Extracted_Key_Points'] = None

    # Process each row in the "Post" column
    for index, row in df.iterrows():
        post_text = row['Post']
        
        if not post_text:
            continue
        
        # Extract the key points from the text
        key_points = extract_key_points(post_text)

        # Print the reshaped output for terminal display
        reshaped_output = reshape_arabic_text_for_terminal(key_points)
        print(f"Processing Row {index + 1}: {reshape_arabic_text_for_terminal(post_text)}")
        print(f"Extracted Key Points -> {reshaped_output}\n")

        # Store the extracted key points in the new column without reshaping
        df.at[index, 'Extracted_Key_Points2'] = key_points 

    # Save the augmented dataset to a new Excel file
    df.to_excel(output_excel_path, index=False)
    print(f"Augmented dataset saved to {output_excel_path}")

# Example usage
if __name__ == "__main__":
    input_excel_path = r"c:\Users\Lenovo\Downloads\filtered_data_with_selected_columns2.xlsx"  # Replace with your actual input file path
    output_excel_path = r"c:\Users\Lenovo\Downloads\filtered_data_with_selected_columns2.xlsx"  # Path to save the output dataset

    process_dataset(input_excel_path, output_excel_path)