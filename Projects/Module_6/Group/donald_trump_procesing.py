import re
import os

def clean_text(text):
    # Remove [tags]
    text = re.sub(r'\[.*?\]', '', text)
    
    # Remove common speaker labels
    text = re.sub(r'\b(The President|President Trump|Donald J\. Trump|Mr\. Trump|Governor [A-Z][a-z]+|Chairwoman [A-Z][a-z]+|Vice President [A-Z][a-z]+):', '', text, flags=re.IGNORECASE)
    
    # Remove generic NAME: format
    text = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+:', '', text)  # e.g. Kristi Noem:
    text = re.sub(r'\b[A-Z][a-z]+:', '', text)  # e.g. Trump:

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def clean_file(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        raw = f.read()
    cleaned = clean_text(raw)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned)
    print(f"✅ Cleaned file saved to: {output_path}")

# Clean the tweet file
clean_file("C:\\Users\\trist\\OneDrive\\BYUI\\BYUI_2025_Spring\\CSE 450\\module_6\\trump_data\\trump_tweets_cleaned.txt", "trump_tweets_final.txt")

# Clean the speech file
clean_file("C:\\Users\\trist\\OneDrive\\BYUI\\BYUI_2025_Spring\\CSE 450\\module_6\\trump_data\\trump_speeches_clean.txt", "trump_speeches_final.txt")
