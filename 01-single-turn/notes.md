# Notes: Main Issues

## load_data.py

**1. Real filename in the `def` line**

Wrong: `def load_pdf("Catalog.pdf"):`

`def` takes a placeholder name, not the real value. The real filename is passed when you call the function.

Fix: `def load_pdf_data(file_path):`, then call `load_pdf_data("Catalog.pdf")`.

**2. Importing a file runs it**

Wrong: `from load_data import text`, with loading and printing at the bottom of `load_data.py`.

Importing ran that file, so the PDF loaded once on import and again in my own code.

Fix:
- Import the function: `from load_data import load_pdf_data`
- Put test lines under `if __name__ == "__main__":`

Rule for every file: imports and functions at the top, tests under the guard at the bottom.

## embeddings.py

**3. API key passed without its label**

Wrong: `client = OpenAI(openai_api_key)`

Error: `TypeError: OpenAI.__init__() takes 1 positional argument but 2 were given`

Python secretly passes `self` as the first unlabeled argument, so my key counted as a second one. The `OpenAI` class only has room for one unlabeled value.

Fix: `client = OpenAI(api_key=openai_api_key)`

Rule: most library classes want their inputs labeled, like `name="Rex"` or `api_key=...`. Check the docs for the label names.

**4. Wrong path to the numbers in the response**

Wrong: `for embedding in response.data.embedding:`

Error: `AttributeError`, because `response.data` is a list and a list has no `.embedding`.

Also wrong, with no error: `for embedding in response.data[0].embedding:`. It only reads the first text and builds one flat list of numbers.

Only each Embedding object inside the list has `.embedding`. The `for` loop hands me the next object each round, so I never write `[0]`.

Fix:
- `for item in response.data:`
- `embedding_list.append(item.embedding)`

Rule: the inner loop goes inside the outer loop, because `response` is replaced every round.

## pinecone_upsert.py

**5. Pinecone rejected the upload**

Wrong: nothing in my code. I called `index.upsert(vectors=batch)` on an index I had created in the Pinecone dashboard.

Error: `[400] This index has a document schema, so writes must go through the documents API.`

Pinecone now has two kinds of index, and each accepts only one kind of upload. The kind is fixed when the index is created:
- Vector index (classic): `index.upsert(vectors=...)`
- Document index (new): `index.documents.upsert(...)`

Indexes made in the dashboard are the new document kind, so my classic upsert was refused. My friend's tutorial is older than this change.

Fix:
- Create a classic vector index in code, once, in its own script: `client.create_index(name=..., dimension=1024, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))`
- Change only the name in `client.Index("ragpipeline-vectors")`. The upload code stays the same.

Rule: read the error message, because it named the cause. When a tutorial's code fails, check the current docs.