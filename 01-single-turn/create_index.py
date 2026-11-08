# ============================================================
# create_index.py
# ============================================================
#
# WHAT THIS FILE DOES
#   Creates a new, empty, classic vector index on Pinecone's servers.
#   Run it ONCE. A second run fails with "already exists".
#   pinecone_upsert.py then connects to this index and fills it.
#
# CONTENTS
#   1. The major issue: Pinecone rejected my upload
#   2. The create_index call, part by part
#   3. Code
#
#
# ============================================================
# 1. THE MAJOR ISSUE: PINECONE REJECTED MY UPLOAD
# ============================================================
#
# THE ERROR (when I ran index.upsert(vectors=batch))
#   [400] This index has a document schema, so writes must go through the
#   documents API.
#
# IN SIMPLE WORDS
#   Pinecone said: "this index is the new type, and the new type does not
#   accept the upload method you used."
#
#   Pinecone now has two kinds of index. Each accepts only ONE kind of upload:
#
#     vector index (classic)  ->  index.upsert(vectors=...)
#     document index (new)    ->  index.documents.upsert(...)
#
#   The kind is fixed when the index is created, and it cannot be changed after.
#   Indexes made in the Pinecone dashboard are the new document kind, so my
#   classic upsert was refused.
#
# WHAT WAS NOT WRONG
#   My code was fine. My friend's tutorial is older than this change.
#
# THE FIX
#   Create a classic vector index in code (this file), then connect to it:
#     client.create_index(...)                  <- this file, run once
#     index = client.Index("ragpipeline-vectors")  <- in pinecone_upsert.py
#   The upload code did not need to change. Only the index name did.
#
# TERMS
#   control plane = managing indexes (create, list, delete)   <- this file
#   data plane    = working with the data inside one index (upsert, query)
#
# WHAT TO REMEMBER
#   - Read the error message. It named the cause.
#   - When a tutorial's code fails, check the current docs.
#   - Libraries like LangChain's PineconeVectorStore may not write to the
#     new document indexes (a GitHub issue reported this), so classic is safer.
#
#
# ============================================================
# 2. THE create_index CALL, PART BY PART
# ============================================================
#
#   client.create_index(...)
#     client         my Pinecone object (the waiter for my account)
#     .create_index  a method (an action): build a new index
#
#   name="ragpipeline-vectors"
#     the index's name: lowercase letters, numbers and hyphens only.
#     I use this exact name later in client.Index("ragpipeline-vectors").
#
#   dimension=1024
#     how many numbers one vector has.
#     Must equal the vectors from embeddings.py, or the upload fails.
#
#   metric="cosine"
#     how Pinecone measures how close two vectors are.
#
#   spec=ServerlessSpec(cloud="aws", region="us-east-1")
#     spec = "specification": where and how Pinecone runs the index.
#     ServerlessSpec is a class from the pinecone library. Calling it builds
#     an object that holds the settings, the same pattern as Pinecone(...).
#       serverless  Pinecone manages the machines and capacity, not me
#       cloud       which company's data centers run it ("aws", lowercase)
#       region      which data center location ("us-east-1")
#     Cloud and region do not change how my code uses the index.
#     In one sentence: "Build this index on Amazon's servers in us-east-1,
#     managed by Pinecone."
#
#   print(client.list_indexes())
#     shows my indexes. The new name should appear. The layout may look
#     different in newer SDK versions, so I look for the name.


# ============================================================
# CODE
# ============================================================

from dotenv import load_dotenv  # function that loads the .env file
import os  # lets me read environment variables with os.getenv
from pinecone import Pinecone, ServerlessSpec  # the client class and the settings class

load_dotenv()  # load .env first, so getenv can find the key

if __name__ == "__main__":  # runs only when I run THIS file directly
    pinecone_api_key = os.getenv("PINECONE_API_KEY")  # read the key as a string

    client = Pinecone(api_key=pinecone_api_key)  # my connection to the Pinecone account

    client.create_index(  # build the new classic vector index (run once)
        name="ragpipeline-vectors",
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    print(client.list_indexes())  # check that the new index exists