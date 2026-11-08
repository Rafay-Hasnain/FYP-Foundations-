# ============================================================
# word_chunking.py
# ============================================================
#
# WHAT THIS FILE DOES
#   Splits one big text into smaller pieces (chunks) of N words each,
#   where neighbouring chunks share some words (overlap).
#
#   PDF text -> THIS FILE -> chunks -> embeddings.py -> Pinecone
#
# CONTENTS
#   1. Why we chunk
#   2. How the function works, line by line
#   3. How the loop moves (overlap)
#   4. How the list keeps the order
#   5. How to test
#   6. Known weaknesses
#
#
# ============================================================
# 1. WHY WE CHUNK
# ============================================================
#
# THE PROBLEM
#   I want to ask "What is the warranty on part X?" and get back the right
#   piece of the catalog. The system searches my text, and in RAG the search
#   works on embeddings.
#
# WHAT AN EMBEDDING IS
#   A list of numbers (a vector) that represents the meaning of a text.
#   Similar meaning -> similar numbers, and that is how search finds text.
#
# WHY ONE EMBEDDING FOR THE WHOLE PDF DOES NOT WORK
#   1. Input limit:    an embedding model accepts only a limited amount of text
#                      at once. 407 pages is far over it, so it would be rejected
#                      or cut off.
#   2. Vague meaning:  one vector for 407 pages is a blurry average of everything,
#                      so a question about one specific part would not match it well.
#   3. Useless result: even a match would return the entire catalog, when I want
#                      the one paragraph that answers the question.
#
# THE FIX IS CHUNKING
#   Split the text into small pieces (for example 100 words each) and make one
#   embedding per piece. Each vector now covers one focused topic, and a search
#   returns only the few chunks closest to the question.
#
#   Big text -> chunks -> one embedding per chunk -> search returns the best chunks
#
# WHY OVERLAP
#   If a sentence like "tighten the bolt to 25 Nm" is cut between two chunks,
#   neither chunk has the full meaning. Overlap repeats a few words in both
#   chunks, so information at the boundary is not lost.
#
#
# ============================================================
# 2. HOW THE FUNCTION WORKS, LINE BY LINE
# ============================================================
#
# def word_based_chunking(text_from_load_data, chunk_size, overlap):
#     inputs: the full text, words per chunk, words shared between chunks
#
# words = text_from_load_data.split()
#     "Computer Science requires 132 credits"
#                    | .split()
#                    v
#     ["Computer", "Science", "requires", "132", "credits"]
#     one string becomes a list of words
#
# chunk_list = []
#     empty list that collects the finished chunks
#
# start = 0
#     position of the first word of the current chunk
#
# step_size = chunk_size - overlap
#     how far start moves each round: 100 - 20 = 80
#
# while start < len(words):
#     keep going until start passes the last word
#
# chunk = words[start : start + chunk_size]
#     slice out chunk_size words (a list of words).
#     A slice stops at the end of the list without crashing,
#     so the last chunk is simply shorter.
#
# chunk_text = " ".join(chunk)
#     list of words -> one string with spaces between the words
#     (the reverse of .split())
#
# chunk_list.append(chunk_text)
#     save the chunk. append always adds to the END of the list.
#
# start += step_size
#     move forward 80 words (not 100), so 20 words repeat = the overlap
#
# return chunk_list
#     hand back all the chunks (outside the loop, so it runs once at the end)
#
#
# ============================================================
# 3. HOW THE LOOP MOVES (chunk_size=100, overlap=20, step=80)
# ============================================================
#
#   words:  0 ........ 80 ........ 160 ........ 240
#
#   round 1: start=0    -> words[0:100]     |----chunk 0----|
#   round 2: start=80   -> words[80:180]              |----chunk 1----|
#   round 3: start=160  -> words[160:260]                       |----chunk 2----|
#
#   Overlap: words 80-99 appear in chunk 0 AND chunk 1.
#
#   Example with 10 words, chunk_size=4, overlap=1 (step = 3):
#     chunk 1: w1 w2 w3 w4
#     chunk 2: w4 w5 w6 w7
#     chunk 3: w7 w8 w9 w10
#
#
# ============================================================
# 4. HOW THE LIST KEEPS THE ORDER
# ============================================================
#
#   The list does not "know" anything. It keeps items in the order they were
#   appended, and Python gives each item a number (index) automatically.
#
#   round 1 -> [ "The Clockmaker ..." ]
#   round 2 -> [ "The Clockmaker ...", "of his clocks ..." ]
#   round 3 -> [ "The Clockmaker ...", "of his clocks ...", "he would peer ..." ]
#
#   chunk_list:      [ chunk A , chunk B , chunk C ]
#   index:              0         1         2
#
#   So test_obj[0], [1], [2] are the first, second and third chunk.
#   Indexes start at 0, so the last index is always length - 1
#   (14 chunks -> last index 13).
#
#   Printing the whole list:  [ ] = the list, " " = each chunk is a string,
#   , = separates one chunk from the next.
#
#
# ============================================================
# 5. HOW TO TEST
# ============================================================
#
#   The test block under  if __name__ == "__main__":  runs only when I run THIS
#   file directly, not when another file imports word_based_chunking.
#
#   OVERLAP CHECK (from my story.pdf output):
#     The end of chunk 1 and the start of chunk 2 share this phrase:
#       "of his clocks, hundreds of them, all ticking slightly out of step
#        with one another, like rain on a tin"
#     It is 20 words, which matches my input overlap = 20.
#
#   WORD COUNT CHECK:
#     print(len(text.split()))   counts the words in the original text.
#
#
# ============================================================
# 6. KNOWN WEAKNESSES (left for later)
# ============================================================
#
#   - overlap >= chunk_size makes step_size 0 or negative, so start never moves
#     forward and the loop never ends. A guard before the loop would prevent it.
#   - If the leftover at the end is no longer than the overlap, the last chunk
#     is entirely inside the previous one (redundant). Example: 100 words with
#     chunk_size=100, overlap=20 gives chunk 0 plus a 20-word chunk 1 that is a
#     copy of the end of chunk 0.
#   - The test block assumes at least 3 chunks. A shorter text raises IndexError.
#   - Word-based splitting ignores sentence and page boundaries, so it can cut a
#     repair step in half. Professional systems use recursive or
#     structure-aware splitters.
#   - .split() collapses all whitespace and line breaks, so layout such as tables
#     and lists is lost.


# ============================================================
# CODE
# ============================================================

from load_data import load_pdf_data  # my PDF loading function (used only in the test)


def word_based_chunking(text_from_load_data, chunk_size, overlap):
    words = text_from_load_data.split()  # one string -> list of words

    chunk_list = []  # collects the finished chunks, in order

    start = 0  # position of the first word of the current chunk

    step_size = chunk_size - overlap  # how far start moves each round

    while start < len(words):  # stop once start passes the last word
        chunk = words[start : start + chunk_size]  # slice out one chunk of words

        chunk_text = " ".join(chunk)  # list of words -> one string

        chunk_list.append(chunk_text)  # save the chunk at the end of the list

        start += step_size  # move forward, leaving "overlap" words repeated

    return chunk_list  # outside the loop: runs once, at the end


if __name__ == "__main__":
    text = load_pdf_data("story.pdf")  # full text of the PDF

    test_obj = word_based_chunking(text, 100, 20)  # list of chunks

    print("ALL CHUNKS:")
    print(test_obj)

    print("\nCHUNK 1:")
    print(test_obj[0])

    print("\nCHUNK 2:")
    print(test_obj[1])

    print("\nCHUNK 3:")
    print(test_obj[2])