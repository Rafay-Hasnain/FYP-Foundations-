# ============================================================
# embeddings.py
# ============================================================
#
# WHAT THIS FILE DOES
#   Turns text chunks into embedding vectors using OpenAI.
#   Input:  a list of strings (chunks)
#   Output: a list of vectors, in the SAME ORDER as the chunks
#
#   PDF text -> word_based_chunking -> chunks -> THIS FILE -> embeddings -> Pinecone
#
# CONTENTS
#   1. Concepts: embedding, dimensions, order
#   2. Secrets: .env, load_dotenv, os.getenv
#   3. Why send chunks in batches
#   4. Classes and objects: how the client is built
#   5. Attribute, method, parameter, argument
#   6. The API call and the response object
#   7. The loops inside the function
#   8. How to test
#   9. Known weaknesses
#
#
# ============================================================
# 1. CONCEPTS
# ============================================================
#
# EMBEDDING
#   A list of numbers (a vector) that represents the MEANING of a text.
#   Similar meaning -> similar numbers. This makes search by meaning possible.
#   The model runs on OpenAI's servers, not on my machine, and costs a small
#   amount per token. So don't embed the same catalog again and again while testing.
#
#       "Hi there"  ->  [0.02, -0.13, 0.08, ...]
#
# DIMENSIONS
#   = how many numbers are in ONE vector.
#   As far as I know, text-embedding-3-small gives 1536 numbers by default,
#   and the dimensions= input asks for fewer (check the OpenAI docs to confirm).
#
#   Pinecone's index has a fixed vector size, set when the index was created.
#   Mine is 1024. Every vector must have exactly 1024 numbers, or the upload fails.
#   The number must be the SAME in the embedding call and in the Pinecone index.
#
#       Pinecone index: 1024   <- chosen when the index was created
#       Embedding:      1536   <- does not fit, upload fails
#       Embedding:      1024   <- fits, works
#
# PINECONE INDEX
#   The database that stores the vectors and searches through them.
#   Each stored item = an ID + the vector + the original chunk text (metadata).
#
# ORDER IS CRITICAL
#   embeddings[0] belongs to chunks[0], embeddings[1] to chunks[1], and so on.
#   If the order is ever scrambled, search returns the WRONG text with no error.
#   That is why each chunk's text is stored next to its vector in Pinecone.
#
#
# ============================================================
# 2. SECRETS: .env, load_dotenv, os.getenv
# ============================================================
#
#   - Secrets (API keys) live in the .env file, NEVER in the code.
#   - .env goes in .gitignore so it never reaches GitHub.
#   - load_dotenv() reads the .env file and makes its values available to Python.
#   - os.getenv("NAME") reads one value, as a string.
#
#   Why the name is in quotes (a string):
#     getenv needs the NAME as plain text to look it up.
#       os.getenv("OPENAI_API_KEY")   quotes = plain text, looked up outside my code
#       os.getenv(OPENAI_API_KEY)     no quotes = Python searches for a variable in
#                                     my code, finds none, NameError
#
#   A wrong name inside the quotes does NOT crash. getenv returns None, and the
#   problem only shows up later as a confusing "no API key" error.
#   So check early:  print(openai_api_key is not None)   -> should be True
#   (Never print the key itself. Secrets end up in logs and screenshots.)
#
#   Order matters: load_dotenv() must run BEFORE os.getenv(...).
#   A function name without () is only a reference. load_dotenv does nothing,
#   load_dotenv() runs it.
#
#
# ============================================================
# 3. WHY SEND CHUNKS IN BATCHES
# ============================================================
#
#   all at once      -> exceeds the API's request size limit
#   one at a time    -> 2,000 chunks = 2,000 requests (slow, hits the rate limit)
#   batches of ~100  -> 2,000 chunks = 20 requests (the good middle)
#
#   Request size rules out sending everything at once.
#   Rate limit and total time rule out one chunk per request.
#
#   Every request is a round trip (me -> OpenAI -> me) with fixed waiting time.
#   Too many requests too fast -> the API rejects them (HTTP 429, "too many
#   requests") and the pipeline breaks partway through.
#
#   NOT a reason for batching: "a huge chunk can't fit in one vector".
#   That is about CHUNKING. Chunking decides the size of each piece,
#   batching decides how many pieces go in one request.
#
#
# ============================================================
# 4. CLASSES AND OBJECTS: HOW THE CLIENT IS BUILT
# ============================================================
#
#   class  = blueprint (capital first letter: OpenAI, Embedding, Dog)
#   object = a real thing built from a class (client, rex)
#
#   from openai import OpenAI     only brings the blueprint. Nothing is connected yet.
#   client = OpenAI(api_key=...)  calls the class and builds the object.
#
#   Same idea as load_dotenv: importing does nothing until I call with ().
#
#   WHY api_key= NEEDS ITS LABEL
#     Python secretly passes self (the object being built) as the first,
#     unlabeled argument. That uses up the one unlabeled slot.
#       OpenAI(openai_api_key)           TypeError: takes 1 positional argument
#                                        but 2 were given
#       OpenAI(api_key=openai_api_key)   correct
#     RULE: most library classes want their inputs labeled. Check the docs
#     for the label names. Don't guess.
#
#   WHAT SELF IS
#     self = "this particular object". Python passes it in automatically, so I
#     never write it when calling. One class can build many objects, and self
#     tells each method which object's data to use:
#       rex.bark()   -> Python runs bark(self=rex)    -> Rex says woof
#       fido.bark()  -> Python runs bark(self=fido)   -> Fido says woof
#
#   "Inherit" is a different idea: one CLASS getting features from another CLASS.
#   Building an object is not inheritance. The object simply HAS what its class gives it.
#
#   Everything in Python is an object: lists, strings and numbers too.
#   type([1, 2]) shows <class 'list'>.
#
#
# ============================================================
# 5. ATTRIBUTE, METHOD, PARAMETER, ARGUMENT
# ============================================================
#
#   attribute = something an object HAS (data). No parentheses.   rex.name
#   method    = something an object DOES (an action). Parentheses. rex.bark()
#   parameter = the input SLOT in a definition.                    def f(name):
#   argument  = the VALUE I pass in when calling.                  f("Rex")
#
#   class Dog:
#       def __init__(self, name):    # name = PARAMETER (an empty slot)
#           self.name = name         # now it is also an ATTRIBUTE (stored on the object)
#       def bark(self):              # METHOD
#           print(self.name, "says woof")
#
#   rex = Dog("Rex")                 # "Rex" = ARGUMENT, fills the slot
#   rex.name                         # attribute
#   rex.bark()                       # method
#
#   __init__ = the start-up method. It runs automatically when I write Dog("Rex").
#
#   RULE: no parentheses = a thing. Parentheses = an action.
#
#   Examples from this file:
#     response.data                    attribute (the list in the response)
#     item.embedding                   attribute (the numbers in an Embedding)
#     client.embeddings                attribute (the embeddings tool group)
#     embedding_list.append(x)         method
#     client.embeddings.create(...)    method (sends the request to OpenAI)
#
#   An attribute can hold another object, and that object has its own methods:
#     client                 object (built from OpenAI)
#      └ .embeddings         attribute of client; its value is another object
#         └ .create(...)     method of THAT object
#   (That client.embeddings is an object is my inference. To check:
#    print(type(client.embeddings)))
#
#   Two similar names, do not mix them up:
#     client.embeddings    (with s) belongs to client
#     item.embedding       (no s) belongs to each Embedding object in the response
#
#
# ============================================================
# 6. THE API CALL AND THE RESPONSE OBJECT
# ============================================================
#
#   response = client.embeddings.create(input=batch, model=..., dimensions=...)
#
#     input=       the list of texts to embed (a whole batch, not one at a time)
#     model=       which embedding model, as a string
#     dimensions=  how many numbers per vector. Must match the Pinecone index (1024)
#
#   The result is a response object. Reading a printed object:
#     Each  Name(  opens an OBJECT.   Each  label=  names an ATTRIBUTE inside it.
#
#   STRUCTURE (boxes inside boxes)
#
# ┌────────────────────────────────────────────────────────┐
# │ response                                               │
# │                                                        │
# │  ┌──────────────────────────────────────────────────┐  │
# │  │ response.data  (a list, one per text)            │  │
# │  │                                                  │  │
# │  │  ┌────────────────────┐  ┌────────────────────┐  │  │
# │  │  │ data[0]            │  │ data[1]            │  │  │
# │  │  │ Embedding, text 1  │  │ Embedding, text 2  │  │  │
# │  │  │                    │  │                    │  │  │
# │  │  │  ┌──────────────┐  │  │  ┌──────────────┐  │  │  │
# │  │  │  │ .embedding   │  │  │  │ .embedding   │  │  │  │
# │  │  │  │ the numbers  │  │  │  │ the numbers  │  │  │  │
# │  │  │  └──────────────┘  │  │  └──────────────┘  │  │  │
# │  │  └────────────────────┘  └────────────────────┘  │  │
# │  └──────────────────────────────────────────────────┘  │
# └────────────────────────────────────────────────────────┘
#
#   HOW TO REACH EACH PART (each dot or [ ] = one box deeper)
#     response                      the whole response object
#     response.data                 the list of Embedding objects
#     response.data[0]              the first Embedding object
#     response.data[0].embedding    the list of numbers
#
#   1 text sent = 1 Embedding object. 100 texts sent = 100 objects,
#   in the same order as the batch.
#
#   CLASSES vs OBJECTS HERE
#     CreateEmbeddingResponse and Embedding are two SEPARATE classes.
#     Only the OBJECTS nest: the response object holds a list of Embedding objects.
#     Embedding (capital E) = the class.  embedding (small e) = an attribute.
#
#   NEW API? Print the response once, read it with the rule above, and
#   check the docs for the response fields.
#
#
# ============================================================
# 7. THE LOOPS INSIDE THE FUNCTION
# ============================================================
#
#   OUTER LOOP:  for i in range(0, len(chunks), batch_size)
#     range(start, stop, step). The third slot is how much i grows each round.
#     With 250 chunks and batch_size=100:  i = 0, 100, 200
#       i = 0    chunks[0:100]     100 chunks
#       i = 100  chunks[100:200]   100 chunks
#       i = 200  chunks[200:300]   50 chunks (a slice stops at the end of the list
#                                  without crashing, so the last batch is just shorter)
#     i is a POSITION: where the batch begins.
#
#   SLICE:  chunks[i : i + batch_size]
#     "from i, up to i + batch_size". The number after : is a position, not a length.
#
#   INNER LOOP:  for item in response.data
#     The loop hands me the NEXT object each round, so I never write [0], [1], [2].
#     item stands for "the Embedding object I am holding this round".
#     Wrong:  response.data.embedding        (a list has no .embedding)
#     Wrong:  response.data[0].embedding     (only the first object, then a flat list of numbers)
#     Right:  item.embedding                 (the numbers of the current object)
#
#   WHY THE INNER LOOP IS NESTED INSIDE THE OUTER LOOP
#     response is replaced every round, so it must be collected before then.
#     If the inner loop is not nested, it runs after all rounds and sees only the
#     LAST batch. Earlier batches are lost silently, with no error.
#
#   WHY return IS OUTSIDE BOTH LOOPS
#     Inside the loop, the function would return after round 1 and skip the rest.
#
#   embedding_list ends up as a list of lists, one entry per chunk:
#     [ [1024 numbers], [1024 numbers], ... ]
#
#   Parameter with a default: batch_size=100 means the caller may leave it out.
#   The trailing comma after the last argument of a multi-line call is optional,
#   but common style. It makes adding a new argument later a one-line change.
#
#
# ============================================================
# 8. HOW TO TEST
# ============================================================
#
#   The test block under  if __name__ == "__main__":  runs only when I run THIS
#   file directly, not when another file imports it.
#
#   With 2 test chunks and batch_size=1:
#     len(result)      -> 2      (entries in the list, one per chunk, NOT the numbers inside)
#     len(result[0])   -> 1024   (numbers in the first embedding; must equal the Pinecone dimension)
#
#   If the second number is not 1024, dimensions= is not working.
#
#
# ============================================================
# 9. KNOWN WEAKNESSES (left for later)
# ============================================================
#
#   - No error handling. If one batch fails (for example a rate limit), all
#     progress is lost. Professional fix: retries with backoff.
#   - An empty string in chunks can make the API call fail.
#   - client is used from outside the function. It works, but passing client
#     in as a parameter is cleaner.
#   - dimensions=1024 is written directly in the code. A named constant would be
#     easier to keep in sync with the Pinecone index.


# ============================================================
# CODE
# ============================================================

import os  # lets me read environment variables with os.getenv
from dotenv import load_dotenv  # function that loads the .env file
from openai import OpenAI  # the class (blueprint) for the OpenAI client

load_dotenv()  # load the .env file first, so getenv can find the key

openai_api_key = os.getenv("OPENAI_API_KEY")  # read the key as a string (None if the name is wrong)

client = OpenAI(api_key=openai_api_key)  # build the client object, input labeled with api_key=


def batch_based_embeddings(chunks, batch_size=100):
    embedding_list = []  # collects one vector per chunk, in order

    for i in range(0, len(chunks), batch_size):  # i = where each batch starts
        batch = chunks[i : i + batch_size]  # cut out one batch of chunks

        response = client.embeddings.create(  # send the batch to OpenAI
            input=batch,
            model="text-embedding-3-small",
            dimensions=1024,  # must match the Pinecone index
        )

        for item in response.data:  # item = one Embedding object per round
            embedding_list.append(item.embedding)  # keep only its numbers

    return embedding_list  # outside both loops: runs once, at the end


if __name__ == "__main__":
    test_chunks = ["Hi there", "Bye now"]
    result = batch_based_embeddings(test_chunks, 1)
    print(len(result))  # expected: 2
    print(len(result[0]))  # expected: 1024