# ============================================================
# pinecone_upsert.py
# ============================================================
#
# WHAT THIS FILE DOES
#   Stores my chunks and their embeddings in a Pinecone index.
#   Each record = an ID + the vector + the original chunk text.
#
#   chunks + embeddings (from embedding.py) -> THIS FILE -> Pinecone index
#
# CONTENTS
#   1. Setup: .env, os, load_dotenv, pip
#   2. Pinecone concepts: client, API, key, index
#   3. The two kinds of index (a major issue)
#   4. The function: inputs, records, batches
#   5. How to test
#   6. Known weaknesses
#
#
# ============================================================
# 1. SETUP: .env, os, load_dotenv, pip
# ============================================================
#
#   .env          text file of NAME=value lines. Holds secrets (API keys).
#                 Never in my code, never on GitHub (.gitignore).
#   load_dotenv() loads those lines into the program's environment variables.
#                 Must run BEFORE os.getenv.
#   os.getenv("NAME")  reads one value, as a string. The name is in quotes
#                 because getenv needs it as text. A wrong name returns None
#                 (no crash), so check early: print(key is not None)
#
#   ENVIRONMENT VARIABLE vs PYTHON VARIABLE
#     Python variable:       made in my code (x = 5), any type, read as x
#     Environment variable:  lives outside my code, always a string,
#                            read with os.getenv("NAME")
#     Usual flow: read it once, keep it in a Python variable.
#
#   WHY import os: getenv lives inside the os module (built into Python).
#   module  = one file of code I can import (os, load_data.py)
#   library = a bundle of modules someone else wrote (openai, pinecone)
#
#   pip install downloads a library into my Python environment.
#     - install once, import in every file that uses it
#     - install while the venv is active
#     - install name can differ from import name (python-dotenv -> dotenv)
#     - ModuleNotFoundError = not installed in this environment
#
#
# ============================================================
# 2. PINECONE CONCEPTS
# ============================================================
#
#   Pinecone(api_key=...)   Pinecone is a CLASS from the pinecone library.
#                           Calling it builds an object, client = my connection
#                           to my Pinecone account (same pattern as OpenAI(...)).
#                           Create it ONCE and reuse it for every request.
#
#   API KEY   my ID and password. Pinecone checks it on every request:
#             valid key -> request accepted and tied to my account.
#
#   API       the set of requests a server accepts, with the exact format
#             for each one (the menu). client is not the API. client is
#             an object in my program that talks to it (the waiter).
#
#       my code -> client (waiter) -> Pinecone's servers (kitchen)
#       my code <- client (waiter) <- response comes back to the line that asked
#
#   INDEX     the database inside Pinecone that stores vectors and searches them.
#             Its settings are fixed at creation (vector size 1024, metric cosine).
#
#       Pinecone account  (what client connects to)
#       +-- index "ragpipeline-vectors"  (one database)
#             +-- record: id + vector + chunk text
#             +-- record: id + vector + chunk text
#
#   client.Index("name")   returns a Python OBJECT that points at the real
#                          database on Pinecone's servers. It does not create
#                          anything. The name must match exactly.
#
#   WHY upsert is called on index, not client
#     client = the account. index = one database. upsert must say WHICH database.
#     client: list_indexes(), create_index(...), Index("name")   (manage indexes)
#     index:  upsert(...), query(...), describe_index_stats()    (use the data)
#
#   BANK PICTURE
#     client = the bank branch. index = one drawer. index.upsert(...) = put
#     items in that drawer. A method can return an object (Index returns one).
#
#   CONTROL PLANE = managing indexes (create, list, delete)  -> client
#   DATA PLANE    = working with data inside one index       -> index
#
#
# ============================================================
# 3. THE TWO KINDS OF INDEX (MAJOR ISSUE)
# ============================================================
#
#   ERROR I GOT: [400] This index has a document schema, so writes must go
#   through the documents API.
#
#   In simple words: Pinecone now has two kinds of index, and each accepts
#   only ONE kind of upload. The kind is fixed when the index is created.
#
#     vector index (classic)  ->  index.upsert(vectors=...)      <- what I use
#     document index (new)    ->  index.documents.upsert(...)
#
#   Indexes made in the dashboard are the new document kind, so my upsert was
#   refused. My code was fine. The fix was creating a classic vector index
#   in code (see create_index.py) and changing only the name in client.Index(...).
#
#
# ============================================================
# 4. THE FUNCTION
# ============================================================
#
#   batch_based_upserting(index, embedding_list, chunk_list, batch_size=100)
#
#     index           the Pinecone index object to store the records in
#     embedding_list  list of vectors, one per chunk (from embedding.py)
#     chunk_list      list of the original text chunks (from word_chunking.py)
#     batch_size      records per upload request. Has a default, so it is OPTIONAL.
#
#   Parameters with a default are optional when calling. Parameters without one
#   are required, and they must come first in the def line.
#
#   THE KEY RULE: position i in one list belongs to position i in the other.
#     chunk_list       ["text 0",       "text 1",       ...]
#     embedding_list   [[1024 numbers], [1024 numbers], ...]
#                           i = 0            i = 1
#   Both lists must be the same length. If the order breaks, search returns
#   the WRONG text with no error.
#
#   STEP 1: BUILD ONE RECORD PER CHUNK
#     records = []                      empty list that collects the records
#     for i in range(0, len(embedding_list)):    i = each position, step 1
#         records.append({
#             "id": f"chunk-{i}",             unique STRING per chunk
#             "values": embedding_list[i],    the vector for this chunk
#             "metadata": {"text": chunk_list[i]},   the original text
#         })
#     f"chunk-{i}": the f lets {i} be replaced by its value ("chunk-0", ...).
#     Upsert = update + insert. Same ID again overwrites the whole record.
#     Why store the text: Pinecone searches by vector and returns records.
#     Without the text in metadata, a search returns numbers and no words.
#
#   STEP 2: UPLOAD IN BATCHES (second loop, same indent as the first)
#     for i in range(0, len(records), batch_size):   i = where each batch starts
#         batch = records[i : i + batch_size]        cut out one batch
#         index.upsert(vectors=batch)                send it to Pinecone
#     The first loop BUILDS all records. The second loop SENDS them, after.
#     Per Pinecone's docs: max 2MB or 1000 records per request, metadata
#     max 40KB per record. Batches of 100 are far below that.
#     Batching reasons: see the notes in embedding.py (size limit, time, rate limit).
#
#   STEP 3: return len(records)
#     number of records uploaded. Outside both loops, so it runs once at the end.
#
#
# ============================================================
# 5. HOW TO TEST
# ============================================================
#
#   The test block under  if __name__ == "__main__":  runs only when I run
#   THIS file directly. The connection lines sit inside it, so importing this
#   file does no network work.
#
#   Test: 2 chunks -> embed them -> upsert -> print the results.
#     print(result)                         -> 2
#     print(index.describe_index_stats())   -> dimension=1024, vector count 2
#   As far as I know, the count can lag a few seconds. Print again if it shows 0.
#   The test records use IDs chunk-0 and chunk-1. The real upload overwrites them.
#
#
# ============================================================
# 6. KNOWN WEAKNESSES (left for later)
# ============================================================
#
#   - IDs like "chunk-0" collide if I upload a second document: its chunk-0
#     silently overwrites the first one's. Fix: include the document name,
#     for example "catalog-chunk-0".
#   - No check that embedding_list and chunk_list have the same length.
#   - No error handling. If one batch fails, the earlier batches are already
#     uploaded and the rest are not. Fix: retries, and tracking progress.
#   - The index name is written directly in the code.
#   - The newer Documents API (index.documents.upsert) is the other path. It
#     needs a schema and a different record format.


# ============================================================
# CODE
# ============================================================

import os  # lets me read environment variables with os.getenv
from dotenv import load_dotenv  # function that loads the .env file
from pinecone import Pinecone  # class (blueprint) for the Pinecone client

load_dotenv()  # load .env first, so getenv can find the key


def batch_based_upserting(index, embedding_list, chunk_list, batch_size=100):
    records = []  # one record per chunk, built first, sent afterwards

    for i in range(0, len(embedding_list)):  # i = position in both lists
        records.append(
            {
                "id": f"chunk-{i}",  # unique string ID
                "values": embedding_list[i],  # the vector for chunk i
                "metadata": {"text": chunk_list[i]},  # the original text of chunk i
            }
        )

    for i in range(0, len(records), batch_size):  # i = where each batch starts
        batch = records[i : i + batch_size]  # cut out one batch

        index.upsert(vectors=batch)  # store this batch in Pinecone

    return len(records)  # outside both loops: runs once, at the end


if __name__ == "__main__":
    pinecone_api_key = os.getenv("PINECONE_API_KEY")  # read the key as a string

    client = Pinecone(api_key=pinecone_api_key)  # my connection to the account

    index = client.Index("ragpipeline-vectors")  # point at my classic vector index

    test_chunks = ["Hi there", "Bye now"]

    from embedding import batch_based_embeddings  # my embeddings file (embedding.py)

    test_embedding = batch_based_embeddings(test_chunks)  # one vector per chunk

    result = batch_based_upserting(index, test_embedding, test_chunks)  # batch_size uses its default

    print(result)  # expected: 2
    print(index.describe_index_stats())  # expected: vector count 2 (may lag a few seconds)