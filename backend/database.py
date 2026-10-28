
import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")

if not MONGODB_URI:
    raise ValueError("MONGODB_URI is missing from the .env file")

client = MongoClient(
    MONGODB_URI,
    serverSelectionTimeoutMS=10000,
)

db = client["page_indexing_db"]

documents_collection = db["documents"]
pages_collection = db["pages"]
page_index_trees_collection = db["page_index_trees"]

def test_connection():
    client.admin.command("ping")
    print("MongoDB Atlas connected successfully!") 
