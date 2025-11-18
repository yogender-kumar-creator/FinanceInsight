import re
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load your fine-tuned NER model and tokenizer
MODEL_NAME = "path/to/your/finbert-ner-model"  # update with your model path or HuggingFace model ID
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

# Initialize NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Predefined regex patterns for custom financial entities (expand as needed)
REGEX_PATTERNS = {
    "P/E ratio": re.compile(r'\bP/?E\s?ratio\s?(is|of)?\s?(\d+(\.\d+)?)', re.IGNORECASE),
    "EPS": re.compile(r'\b(EPS|earnings per share)\s?(is|of)?\s?(\d+(\.\d+)?)', re.IGNORECASE),
    "Dividend yield": re.compile(r'\bdividend yield\s?(is|of)?\s?(\d+(\.\d+)?%|\d+(\.\d+)? percent)', re.IGNORECASE),
    "Market Cap": re.compile(r'\bmarket capitalization|market cap\s?(is|of)?\s?(\$?\d+(\.\d+)?(B|M|bn|mn)?)', re.IGNORECASE),
    # Add more patterns as needed...
}

def extract_entities_ner(text):
    """
    Extract entities using the fine-tuned NER model.
    Returns a dictionary of recognized entities grouped by entity label.
    """
    ner_results = ner_pipeline(text)
    entities = {}
    for ent in ner_results:
        label = ent['entity_group']
        if label not in entities:
            entities[label] = []
        entities[label].append(ent['word'])
    return entities

def extract_entities_regex(text, user_entities):
    """
    Extract entities from text using regex patterns for user-defined entities.
    user_entities: list of entity names that user wants to extract.
    Returns dictionary of entity -> matched string.
    """
    extractions = {}
    for entity in user_entities:
        pattern = REGEX_PATTERNS.get(entity)
        if pattern:
            match = pattern.search(text)
            if match:
                # Group 2 or 3 usually contains the numeric value
                # Some regex patterns capture multiple groups; adjust accordingly
                for i in range(2, match.lastindex + 1):
                    if match.group(i):
                        extractions[entity] = match.group(i)
                        break
    return extractions

def user_defined_entity_extraction(texts, user_entities):
    """
    Extract entities specified by the user from a list of texts.
    Combines NER model outputs and regex-based extraction.
    Returns list of dictionaries for each text input.
    """
    results = []
    for text in texts:
        result = {}

        # Step 1: Extract with NER model
        ner_entities = extract_entities_ner(text)

        # Step 2: Map user entities to NER extracted entities if possible
        # Normalize keys to lowercase for matching flexibility
        ner_flat = {}
        for k, v in ner_entities.items():
            ner_flat[k.lower()] = ' '.join(v)

        for entity in user_entities:
            entity_lower = entity.lower()
            if entity_lower in ner_flat:
                result[entity] = ner_flat[entity_lower]

        # Step 3: For entities not found in NER, try regex-based extraction
        missing_entities = [e for e in user_entities if e not in result]
        regex_extracted = extract_entities_regex(text, missing_entities)
        result.update(regex_extracted)

        results.append(result)
    return results

# Example Usage
if __name__ == "__main__":
    # Sample texts to extract entities from
    sample_texts = [
        "Company ABC reported an EPS of 3.45 and a P/E ratio of 18.7 in the latest quarter.",
        "The market capitalization reached $12.5B as of the fiscal year end.",
        "Dividend yield is currently 2.5 percent, showing steady growth."
    ]

    # Entities user wants to extract
    user_requested_entities = ["EPS", "P/E ratio", "Market Cap", "Dividend yield"]

    # Extract entities
    extracted_data = user_defined_entity_extraction(sample_texts, user_requested_entities)

    # Pretty print results
    print(json.dumps(extracted_data, indent=4))

    # Save results to file (optional)
    with open("extracted_financial_entities.json", "w") as f:
        json.dump(extracted_data, f, indent=4)

