from pathlib import Path

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from sample_products import SAMPLE_PRODUCTS

DB_PATH = "products_vectorstore"
COLLECTION_NAME = "products"
ENCODER_MODEL = "sentence-transformers/all-MiniLM-l6-v2"

class RAGPricePredictor:
    def __init__(self):
        # SentenceTransformer changes text into vectors
        self.encoder = SentenceTransformer(ENCODER_MODEL)
        # Chroma stores the product vectors locally in this project folder
        self.client = chromadb.PersistentClient(path=DB_PATH)

        # Create or open our product collection.
        self.collection = self.client.get_or_create_collection(
            name = COLLECTION_NAME
        )

        # If the dataset is empty, add our small begineer dataset automatically
        if self.collection.count() == 0:
            self._add_sample_products()

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

        ids = [f"sample_{id}" for i in range(len(SAMPLE_PRODUCTS))]

        self.collection.add(
            ids = ids,
            documents = documents,
            embeddings = embeddings,
            metadatas = metadatas,
        )

    def find_similar_products(self, description, number_of_results):
        """
        Find products whose descriptions are close to the user's description.
        """

        # The query must also become a vector
        query_vector = self.encoder.encode([description]).astype(float).tolist()

        results = self.collection.query(
            query_embeddings = query_vector,
            n_results = number_of_results,
            include = ["documents", "metadatas","distances"],
        )
        similar_products = []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

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
            [item["distances"] for item in similar_products],
            dtype = float
        )
        # Add 0.05 so we never divide by zero
        weights = 1.0/ (distances + 0.05)

        estimated_price = float(np.average(prices,weights=weights))

        return round(estimated_price,2), similar_products