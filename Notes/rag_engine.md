# Discuss `_add_sample_products` — explained for a total beginner

This function's only job is: **"Take the hardcoded list of products from `sample_products.py` and store them inside ChromaDB so we can search them later."**

It runs automatically only **once** — the first time the app starts and the database is empty. After that, `if self.collection.count() == 0:` is `False`, so it's skipped.

Let's go line by line.

---

## 1. The function name starts with `_`

```python
def _add_sample_products(self):
```

The leading underscore is a Python **convention** that means: _"this is an internal helper — please don't call it from outside the class."_ It still works if you call it, but it's a polite signal saying "private, for internal use only."

---

## 2. Pull out the descriptions

```python
documents = [item["description"] for item in SAMPLE_PRODUCTS]
```

`SAMPLE_PRODUCTS` is a list of 25 dictionaries that looks like:

```python
{"description": "Apple iPhone 15 128GB smartphone", "category": "Electronics", "price": 799.00}
```

This line is a **list comprehension** — a compact `for` loop. It says:

> "For every `item` in `SAMPLE_PRODUCTS`, grab `item["description"]`, and collect all of them into a new list called `documents`."

After this line, `documents` looks like:

```python
[
  "Apple iPhone 15 128GB smartphone",
  "Apple iPhone 15 Pro 256GB smartphone",
  "Samsung Galaxy S24 128GB Android smartphone",
  ...
]
```

These plain text strings are what the search engine will actually compare against.

---

## 3. Turn each description into a vector

```python
embeddings = self.encoder.encode(documents).astype(float).tolist()
```

Computers can't compare "meaning" of text directly — they compare **numbers**. So `self.encoder` (a `SentenceTransformer` model) reads every description and produces a **vector**: a long list of ~384 numbers that captures the _meaning_ of that sentence.

- Similar sentences (e.g. two iPhone descriptions) → similar numbers.
- Unrelated sentences (iPhone vs. guitar) → very different numbers.

So if `documents` has 25 items, `embeddings` becomes a list of 25 vectors:

```python
[
  [0.12, -0.44, 0.07, ...],   # vector for iPhone 15
  [0.11, -0.41, 0.09, ...],   # vector for iPhone 15 Pro (similar!)
  [0.09, -0.38, 0.01, ...],   # vector for Galaxy S24
  ...
]
```

`.astype(float).tolist()` is just housekeeping — it ensures the numbers are plain Python floats in plain lists so ChromaDB can store them.

---

## 4. Build the metadata list

```python
metadatas = [
    {
        "category": item["category"],
        "price": float(item["price"]),
    }
    for item in SAMPLE_PRODUCTS
]
```

**Metadata** = extra info stored _alongside_ each description. ChromaDB requires metadata to be simple values (strings, numbers, booleans) — not nested objects.

This is also a list comprehension. It creates one small dict per product:

```python
[
  {"category": "Electronics", "price": 799.0},
  {"category": "Electronics", "price": 1099.0},
  ...
]
```

Later, when you search, ChromaDB returns both the matched description **and** this metadata — that's how `find_similar_products` gets the prices out.

---

## 5. Create the IDs ⚠️ (this line has a bug)

```python
ids = [f"sample_{id}" for i in range(len(SAMPLE_PRODUCTS))]
```

ChromaDB requires **every item to have a unique ID**. The intent here is clearly `sample_0, sample_1, sample_2, ...`.

But there's a mistake: the code uses **`id`** instead of **`i`**.

- `i` is the loop variable you defined — `0, 1, 2, ...`
- `id` is a **built-in Python function** (returns the memory address of an object)

So instead of `sample_0, sample_1, ...`, you get `sample_<built-in function id>` repeated 25 times — 25 **identical** IDs. ChromaDB will reject that with a duplicate-ID error.

**Fix:**

```python
ids = [f"sample_{i}" for i in range(len(SAMPLE_PRODUCTS))]
```

Even simpler and cleaner:

```python
ids = [f"sample_{i}" for i, _ in enumerate(SAMPLE_PRODUCTS)]
```

---

## 6. Store everything in the database

```python
self.collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
)
```

This is the actual save. ChromaDB expects these four lists to be **parallel** — meaning `ids[0]`, `documents[0]`, `embeddings[0]`, and `metadatas[0]` all describe the **same product**.

So after this call, ChromaDB holds 25 rows that look like:

| id       | document                               | embedding          | metadata                                  |
| -------- | -------------------------------------- | ------------------ | ----------------------------------------- |
| sample_0 | "Apple iPhone 15 128GB smartphone"     | [0.12, -0.44, ...] | {"category":"Electronics","price":799.0}  |
| sample_1 | "Apple iPhone 15 Pro 256GB smartphone" | [0.11, -0.41, ...] | {"category":"Electronics","price":1099.0} |
| ...      | ...                                    | ...                | ...                                       |

And because the client was created with `PersistentClient(path=DB_PATH)`, all of this is saved to disk in the `products_vectorstore` folder. Next time you run the app, `collection.count()` will be `25`, so this function is skipped and the data is loaded from disk instead.

---

## The whole thing in one sentence

> `_add_sample_products` takes the 25 hardcoded products, converts each description into a numeric "meaning vector," packs the category and price as metadata, and writes all four parallel lists into ChromaDB so future searches can find similar products by meaning.

**One action item:** fix the `id` → `i` typo on the `ids = ...` line, otherwise the very first run will crash with a duplicate-ID error.

**`estimate_price_locally()` calculates an estimated price using the prices of similar products. Products that are more similar have a greater influence on the estimate.**

Imagine you want to estimate the price of some wireless headphones. You look at five comparable products, check their prices, and give more importance to the closest matches. Your function does this calculation.

Here is your code:

```python
def estimate_price_locally(self, description):
    """
    Beginner-friendly estimate:
    use a weighted average of the 5 retrieved product prices.

    A smaller vector distance means "more similar",
    so we give it more weight.
    """

    similar_products = self.find_similar_products(description)

    prices = np.array(
        [item["price"] for item in similar_products],
        dtype=float,
    )

    distances = np.array(
        [item["distance"] for item in similar_products],
        dtype=float,
    )

    # Add 0.05 so we never divide by zero.
    weights = 1.0 / (distances + 0.05)

    estimated_price = float(
        np.average(prices, weights=weights)
    )

    return round(estimated_price, 2), similar_products
```

Let’s follow one example through the whole function.

---

The first line defines the method:

```python
def estimate_price_locally(self, description):
```

- **`self`** refers to the current predictor object.
- **`description`** contains the product description supplied by the caller.

For example:

```python
price, matches = rag.estimate_price_locally(
    "Wireless headphones with noise cancellation"
)
```

Inside the method, `description` now contains:

```python
"Wireless headphones with noise cancellation"
```

The triple-quoted text below the definition is a **docstring** explaining the method’s purpose.

---

**First, the function finds similar products.**

```python
similar_products = self.find_similar_products(description)
```

This calls the search method you have been learning about.

It passes your description to `find_similar_products()`, which searches the database and returns matching product information.

Because the call does not specify a result count, it uses that method’s default:

```python
number_of_results=5
```

For this explanation, imagine the search returns these five products. **These products, prices, and distances are invented examples to make the arithmetic easy.**

| Similar product | Price | Distance |
| --------------- | ----: | -------: |
| Headphones A    |  $100 |     0.05 |
| Headphones B    |  $120 |     0.15 |
| Headphones C    |  $150 |     0.45 |
| Headphones D    |  $180 |     0.45 |
| Headphones E    |  $200 |     0.95 |

A smaller distance means a closer match under the database’s distance measure.

Therefore:

- Headphones A are the closest match.
- Headphones B are the next closest.
- Headphones E are the least similar among these five.

`similar_products` is a **list of dictionaries**. Its first item would look like:

```python
{
    "description": "Headphones A",
    "price": 100.0,
    "category": "Electronics",
    "distance": 0.05
}
```

The function now has the comparison data it needs.

---

**Next, it collects the prices into a numerical array.**

```python
prices = np.array(
    [item["price"] for item in similar_products],
    dtype=float,
)
```

Start with the inner expression:

```python
[item["price"] for item in similar_products]
```

This means:

> “Visit every matched product, take its price, and put those prices into a list.”

For our example, the result is:

```python
[100.0, 120.0, 150.0, 180.0, 200.0]
```

The longer version would be:

```python
price_list = []

for item in similar_products:
    price_list.append(item["price"])
```

Then:

```python
np.array(price_list, dtype=float)
```

converts that list into a **NumPy array**.

`np` is the short name created by this import elsewhere in your file:

```python
import numpy as np
```

NumPy helps Python perform numerical calculations on groups of values.

`dtype=float` specifies that the array should contain floating-point numbers, such as `100.0`.

After this block, `prices` contains the five prices in the same order as the matched products.

---

**The function collects their distances in the same way.**

```python
distances = np.array(
    [item["distance"] for item in similar_products],
    dtype=float,
)
```

This time, it extracts the `"distance"` value from each product.

Using our example, the distances are:

```python
[0.05, 0.15, 0.45, 0.45, 0.95]
```

The two arrays stay aligned:

| Position | Price | Distance | Product |
| -------: | ----: | -------: | ------- |
|      `0` | 100.0 |     0.05 | A       |
|      `1` | 120.0 |     0.15 | B       |
|      `2` | 150.0 |     0.45 | C       |
|      `3` | 180.0 |     0.45 | D       |
|      `4` | 200.0 |     0.95 | E       |

This alignment is important: each price must be paired with the distance of the same product.

---

**Now the function decides how much influence each product should have.**

```python
weights = 1.0 / (distances + 0.05)
```

A **weight** is a number indicating how strongly an item influences the final calculation.

Think of asking several people to estimate a price. You might give more importance to someone who knows the exact type of product. Your code gives more importance to products with closer descriptions.

The rule is:

$$
\text{Weight}=\frac{1}{\text{Distance}+0.05}
$$

For Headphones A:

```python
distance = 0.05

weight = 1.0 / (0.05 + 0.05)
```

This becomes:

```python
weight = 1.0 / 0.10
```

So:

```python
weight = 10.0
```

For Headphones E:

```python
weight = 1.0 / (0.95 + 0.05)
```

This becomes:

```python
weight = 1.0 / 1.00
```

So:

```python
weight = 1.0
```

The complete calculation is:

| Product | Distance | Distance + 0.05 | Weight |
| ------- | -------: | --------------: | -----: |
| A       |     0.05 |            0.10 |     10 |
| B       |     0.15 |            0.20 |      5 |
| C       |     0.45 |            0.50 |      2 |
| D       |     0.45 |            0.50 |      2 |
| E       |     0.95 |            1.00 |      1 |

**Product A gets ten times the weight of Product E.** Its price therefore has much more influence on the estimate.

Because `distances` is a NumPy array, this one line:

```python
weights = 1.0 / (distances + 0.05)
```

performs the calculation for **every distance**. You do not need to write a separate loop.

The resulting weights are:

```python
[10.0, 5.0, 2.0, 2.0, 1.0]
```

Why add `0.05`?

A distance could be zero. Without the adjustment, the calculation would attempt:

```python
1.0 / 0.0
```

Adding `0.05` keeps the denominator nonzero:

```python
1.0 / (0.0 + 0.05)
```

which gives a finite weight of `20`.

The value `0.05` is a design choice in your code. It also affects how strongly the closest products are weighted.

---

**Next, the function calculates the weighted average price.**

```python
estimated_price = float(
    np.average(prices, weights=weights)
)
```

A weighted average follows this rule:

> Multiply each price by its weight, add those amounts, and divide by the total weight.

For our example:

| Product   | Price | Weight | Price × weight |
| --------- | ----: | -----: | -------------: |
| A         |   100 |     10 |          1,000 |
| B         |   120 |      5 |            600 |
| C         |   150 |      2 |            300 |
| D         |   180 |      2 |            360 |
| E         |   200 |      1 |            200 |
| **Total** |       | **20** |      **2,460** |

Therefore:

$$
\text{Estimated price}
=
\frac{2460}{20}
=
123
$$

The estimated price in this illustration is **$123**.

An ordinary average of the five prices would be $150. Your weighted calculation produces $123 because the closest matches—A and B—have lower prices and greater influence.

In the code:

```python
np.average(prices, weights=weights)
```

performs that weighted-average calculation.

The surrounding:

```python
float(...)
```

converts the NumPy result into an ordinary Python floating-point number.

After this line:

```python
estimated_price = 123.0
```

---

**Finally, the method returns both the estimate and the comparison products.**

```python
return round(estimated_price, 2), similar_products
```

The first part:

```python
round(estimated_price, 2)
```

rounds the estimate to two decimal places.

For example:

```python
round(123.4567, 2)
```

produces:

```python
123.46
```

The second part:

```python
similar_products
```

returns the original list of retrieved products. This lets the application show which examples contributed to the estimate.

The comma combines the two values into a **tuple**:

```python
(estimated_price, similar_products)
```

That is why the caller can write:

```python
price, matches = rag.estimate_price_locally(
    "Wireless headphones with noise cancellation"
)
```

Python assigns the returned values in order:

| Variable  | Receives                        |
| --------- | ------------------------------- |
| `price`   | The rounded estimated price     |
| `matches` | The list of comparison products |

Using our illustrative data, you could display:

```python
print(f"Estimated price: ${price:.2f}")
```

Output:

```text
Estimated price: $123.00
```

Here, `:.2f` controls the display so it always shows two decimal places. The returned `matches` list remains available to display the products used for that estimate.

# Description para1

**`BASE_DIR` is the folder containing your Python file. `DB_PATH` is the path to the folder where Chroma stores your database.**

Suppose your `rag_engine.py` file is here:

```text
/Users/hasan/pricePredictor/rag_engine.py
```

Then these variables would contain:

| Variable   | Value                                              |
| ---------- | -------------------------------------------------- |
| `BASE_DIR` | `/Users/hasan/pricePredictor`                      |
| `DB_PATH`  | `/Users/hasan/pricePredictor/products_vectorstore` |

Let’s break down the first line:

```python
BASE_DIR = Path(__file__).resolve().parent
```

| Part             | What it does                                                                  |
| ---------------- | ----------------------------------------------------------------------------- |
| `__file__`       | Gives the path of the Python file containing this code—here, `rag_engine.py`. |
| `Path(__file__)` | Creates a Python path object so you can work with that location.              |
| `.resolve()`     | Gets the full, absolute path and resolves symbolic links.                     |
| `.parent`        | Gets the containing folder, leaving out the filename.                         |

So `BASE_DIR` holds the location of the folder containing `rag_engine.py`.

Now the second line:

```python
DB_PATH = str(BASE_DIR / "products_vectorstore")
```

DB_PATH is the folder where ChromaDB stores its database files.
This starts with `BASE_DIR` and adds the folder name `products_vectorstore`.

- **`/` joins paths here.** Because `BASE_DIR` is a `Path` object, this does not mean division.
- **`str()` converts the resulting path into ordinary text**, which you pass to Chroma.

Your code then uses that location:

```python
self.client = chromadb.PersistentClient(path=DB_PATH)
```

Chroma stores its database files there, including your product descriptions, vectors, and metadata. This allows the stored data to remain available after you stop and restart the app.

**Why build the path this way?** The database location stays beside `rag_engine.py`, even if you launch the program from a different terminal folder.
What are COLLECTION_NAME and ENCODER_MODEL?
These are just plain string variables (constants) that hold names. They make your code easier to read and change later, because instead of writing the same long string in ten different places, you write it once at the top and reuse the variable name.

Let's look at each one.

1. COLLECTION_NAME = "products"
   python
   COLLECTION_NAME = "products"
   What is a "collection"?
   In ChromaDB, a collection is like a table in a normal database, or a folder inside a cabinet.

Your ChromaDB database (the folder products_vectorstore) can hold many collections. Right now, your app only uses one — the one named "products".

Where is it used?
Look at **init**:

python
def **init**(self, collection_name=COLLECTION_NAME):
...
self.collection = self.client.get_or_create_collection(
name=collection_name,
...
)
This says:

"Open the collection called 'products'. If it doesn't exist yet, create it."

Why use a variable instead of typing "products" directly?
Because later in **init**, this check happens:

python
if collection_name == COLLECTION_NAME:
self.\_add_sample_products()
else:
raise ValueError(...)
So COLLECTION_NAME is a named label you can compare against. If you ever wanted to rename the collection from "products" to "store_products", you'd only change it in one place — the top of the file.

In one sentence: COLLECTION_NAME is the name of the "table" inside ChromaDB where your product vectors are stored.

2. ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
   python
   ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
   What is it?
   This is the name of a pre-trained AI model hosted on Hugging Face (a website where people share AI models). The model's full name is all-MiniLM-L6-v2.

The name breaks down like this:

Part Meaning
all Trained on "all" kinds of general text (not just one topic)
MiniLM A small, fast version of a language model
L6 6 layers deep (small = fast to run)
v2 Version 2 of the model
What does this model do?
It takes a sentence (like "Wireless noise-cancelling headphones") and converts it into a vector — a list of ~384 numbers that captures the meaning of that sentence.

Two sentences that mean similar things (e.g. "wireless headphones" and "Bluetooth earbuds") will produce vectors that are close together.

Where is it used?
In **init**:

python
self.encoder = SentenceTransformer(ENCODER_MODEL)
And later, both \_add_sample_products and find_similar_products call self.encoder.encode(...) to turn text into vectors.

In one sentence: ENCODER_MODEL is the name of the AI model that converts text (like a product description) into vectors (numbers), so ChromaDB can compare them by meaning.
