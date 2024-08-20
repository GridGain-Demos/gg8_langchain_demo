import json
import csv
import time
import argparse
from utils import initialize_embeddings_model

def load_json_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def generate_vector(text, embeddings):
    time.sleep(20)  # Simulating rate limiting
    return embeddings.embed_query(text)

def main(api_key):
    # Initialize embeddings model
    embeddings = initialize_embeddings_model(api_key)

    # Load data from JSON files
    specs_data = load_json_data("data/laptop_specs.json")
    reviews_data = load_json_data("data/laptop_reviews.json")

    # Prepare the data
    data = []
    for laptop_id, specs in specs_data.items():
        print(f"Processing laptop {laptop_id}")
        
        # Combine specs and reviews for content
        content = f"Name: {specs['name']}, RAM: {specs['ram']}, GPU: {specs['gpu']}, CPU: {specs['cpu']}, Storage: {specs['storage']}. " \
                  f"Review: {reviews_data[laptop_id]}"

        content_vector = generate_vector(content, embeddings)
        
        data.append({
            'id': laptop_id,
            'content': content,
            'contentVector': content_vector
        })
        print(f"Completed processing laptop {laptop_id}")

    # Write to CSV
    with open('data/laptops.csv', 'w', newline='') as csvfile:
        fieldnames = ['id', 'content', 'content_vector']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        
        writer.writeheader()
        for row in data:
            writer.writerow(row)

    print("CSV file 'laptops.csv' has been created successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Laptop CSV Data Generator")
    parser.add_argument("--use_api_key", help="The API key to be used")
    args = parser.parse_args()

    api_key = args.use_api_key or input("\nPlease provide your OpenAI API key: ")

    main(api_key)