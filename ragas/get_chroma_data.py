import chromadb
from chromadb.config import Settings

# Connect to your selected database (the one highlighted in red: db_20250410-061055)
client = chromadb.PersistentClient(
    path="/home/xwh/AutoRAG/experiments/db_resources/db_20250410-061055"
)

# List all collections in the database
collections = client.list_collections()
print(f"Collections in database: {collections}")

# For each collection, get and print the data
for collection_info in collections:
    collection_name = collection_info.name
    collection = client.get_collection(collection_name)
    
    # Get all items in the collection
    results = collection.get()
    
    print(f"\nCollection: {collection_name}")
    print(f"Number of items: {len(results['ids']) if 'ids' in results else 0}")
    
    # Print sample data (first 5 items)
    if 'documents' in results and results['documents']:
        print("\nSample documents:")
        for i in range(min(5, len(results['documents']))):
            print(f"ID: {results['ids'][i]}")
            # print(f"Document: {results['documents'][i][:200]}...")  # Print first 200 chars
            print(results)
            if 'metadatas' in results and results['metadatas'] and results['metadatas'][i]:
                print(f"Metadata: {results['metadatas'][i]}")
            print("-" * 80)