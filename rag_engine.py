from pathlib import Path

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from sample_products import SAMPLE_PRODUCTS

# Description para1 in rag_engine.py notes file
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = str(BASE_DIR/ "products_vectorstore")
COLLECTION_NAME = "products"
ENCODER_MODEL = "sentence-transformers/all-MiniLM-l6-v2"

class RAGPricePredictor:
    def __init__(self,collection_name = COLLECTION_NAME):
        # SentenceTransformer changes text into vectors
        self.encoder = SentenceTransformer(ENCODER_MODEL)
        # Chroma stores the product vectors locally in this project folder
        self.client = chromadb.PersistentClient(path=DB_PATH)

        # Create or open our product collection.
        self.collection = self.client.get_or_create_collection(
            name = COLLECTION_NAME,
            embedding_function= None,
        )
        # embedding_function = None tells ChromaDb: "Don't convert text to vectors for me. I will do it myself. It means your self.encoder handles their creation instead of Chroma doing that automatically."
        
        # If the dataset is empty, add our small begineer dataset automatically
        if self.collection.count() == 0:
            if collection_name == COLLECTION_NAME:
                self._add_sample_products()
            else:
                raise ValueError(
                    f"Collection '{collection_name}' is empty. "
                    "Run setup_rag.py for it or set RAG_COLLECTION = products."
                )

    def _add_sample_products(self):
        """Put the small starter product list into ChromaDB."""

        documents = [item["description"] for item in SAMPLE_PRODUCTS]

        # Convert every description into a vector.
        embeddings = self.encoder.encode(documents).astype(float).tolist()

        metadatas = [
            {
                "category": item["category"],
                "price":float(item["price"]),
            }
            for item in SAMPLE_PRODUCTS
        ]  

        # i changes each time, so every product gets a different ID.
        ids = [f"sample_{i}" for i in range(len(SAMPLE_PRODUCTS))]

        self.collection.add(
            ids = ids,
            documents = documents,
            embeddings = embeddings,
            metadatas = metadatas,
        )

    def find_similar_products(self, description, number_of_results = 5):
        """
        Find products whose descriptions are close to the user's description.
        """
        description = (description or "").strip()
        if not description:
            raise ValueError("Please enter a product description.")
        if number_of_results < 1:
            raise ValueError("number_of_results must be at least 1.")
        count = self.collection.count()
        if count == 0:
            return []
        # The query must also become a vector
        query_vector = self.encoder.encode([description]).astype(float).tolist()

        results = self.collection.query(
            query_embeddings = query_vector,
            n_results = number_of_results,
            include = ["documents", "metadatas","distances"],
        )


        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        similar_products = []

        for document, metadata, distance in zip(
            documents,metadatas,distances
        ):
            similar_products.append(
                {
                    "description":document,
                    "price":float(metadata["price"]),
                    "category": metadata.get("category",""),
                    "distance": float(distance)
                }
            )
       # print("Similar",similar_products)
        return similar_products


    def estimate_price_locally(self, description):
        """
        Begineer-friendly estimate:
        use a weighted average of the 5 retrieved product prices. A smaller vector distance means "more similar", so we give it more weight.
        """

        similar_products = self.find_similar_products(description)

        prices = np.array(
            [item["price"] for item in similar_products],
            dtype = float,
        )

        distances = np.array(
            [item["distance"] for item in similar_products],
            dtype = float
        )
        # Smaller distance gives a larger weight. Add 0.05 so we never divide by zero
        weights = 1.0/ (distances + 0.05)

        estimated_price = float(np.average(prices,weights=weights))

        return round(estimated_price,2), similar_products