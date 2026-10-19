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
        # print("Documents",documents)
        # raise SystemExit("debug stop: documents printed.")
        # Convert every description into a vector.
        embeddings = self.encoder.encode(documents).astype(float).tolist()
        # embeddings er maddome documents ta vector akare save hobe.
        metadatas = [
            {
                "category": item["category"],
                "price":float(item["price"]),
            }
            for item in SAMPLE_PRODUCTS
        ]  
        # print("Metadata",metadatas)
        # i changes each time, so every product gets a different ID.
        ids = [f"sample_{i}" for i in range(len(SAMPLE_PRODUCTS))]
        # ids =  ['sample_0', 'sample_1',..] 
        #print("ID",ids)
        #raise SystemExit("id")
        self.collection.add(
            ids = ids,
            documents = documents,
            embeddings = embeddings,
            metadatas = metadatas,
        )
        # stored_products = self.collection.get(
        #     include=["embeddings","metadatas"]
        # )
        # print(stored_products)
        # raise SystemExit("storeProducts")
    '''
    Chroma saves each product as a separate record, identified by its ID. It uses the matching positions in your Python lists to assemble each record.
    Suppose your lists contain:

    ids = ["sample_0", "sample_1"]

    documents = ["iPhone 15", "iPhone 15 Pro"]

    embeddings = [
        [0.21, 0.35],  # Example vector for iPhone 15
        [0.25, 0.38]   # Example vector for iPhone 15 Pro
    ]

    metadatas = [
        {"price": 799.0},
        {"price": 1099.0}
    ]

    When you call:

    self.collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    Chroma associates them like this:

    Input position	Stored ID	Description	Vector	Metadata
    Index 0 from every list	sample_0	iPhone 15	[0.21, 0.35]	{"price": 799.0}
    Index 1 from every list	sample_1	iPhone 15 Pro	[0.25, 0.38]	{"price": 1099.0}
    The result is two product records. The four Python lists are the input format for the database write.
        
        '''

    def find_similar_products(self, description, number_of_results = 5):
        """
        Find products whose descriptions are close to the user's description.
        """
        description = (description or "").strip()
        #raise RuntimeError(description,"FromRagEngineProducts")
        if not description:
            raise ValueError("Please enter a product description.")
        if number_of_results < 1:
            raise ValueError("number_of_results must be at least 1.")
        count = self.collection.count()
        #raise RuntimeError(count,"Count")
        if count == 0:
            return []
        # The query must also become a vector
        query_vector = self.encoder.encode([description]).astype(float).tolist()
        #raise RuntimeError(query_vector,"User given value converts to vector")
        
        # so, in results i give query_vector of user input, and it searches and returns similar 5 products to me .
        results = self.collection.query(
            query_embeddings = query_vector,
            n_results = number_of_results,
            include = ["documents", "metadatas","distances"],
        )
        #raise RuntimeError(results,"Results")


        '''
        You can see the same product names, but the list structure is different. The [0] removes one outer list.
        Using two products to keep the example short:

        Without [0]:
        documents = results["documents"]

        documents contains:
        [    [        "Apple iPhone 15 Pro 256GB smartphone",        "Apple iPhone 15 128GB smartphone"    ]]

        This is one list inside another list.

        With [0]:
        documents = results["documents"][0]

        documents contains:
        [    "Apple iPhone 15 Pro 256GB smartphone",    "Apple iPhone 15 128GB smartphone"]


        This is a list of product descriptions.
        The product names are unchanged. Look carefully at the brackets: [[...]] versus [...].
        '''
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        similar_products = []
        #raise RuntimeError(documents,"Documents")


        '''
        After both iterations, your list contains:
        similar_products = 
        [
        {"description": "Apple iPhone 15 Pro 256GB smartphone","price": 1099.0,"category": "Electronics","distance": 0.749992},    
        {"description": "Apple iPhone 15 128GB smartphone", "price": 799.0, "category": "Electronics","distance": 0.822220}
        ]
        With your actual five matches, this loop creates five dictionaries inside the list, preserving their order.
        '''
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
        #raise RuntimeError("Similar",similar_products)
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
        #raise RuntimeError(prices,"prices")
        distances = np.array(
            [item["distance"] for item in similar_products],
            dtype = float
        )
        # Smaller distance gives a larger weight. Add 0.05 so we never divide by zero
        weights = 1.0/ (distances + 0.05)

        estimated_price = float(np.average(prices,weights=weights))

        return round(estimated_price,2), similar_products