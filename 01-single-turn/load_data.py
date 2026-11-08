# ============================================================
# load_data.py
# ============================================================
#
# WHAT THIS FILE DOES
#   Turns a PDF file into one plain string that Python can work with.
#
#   Catalog.pdf -> THIS FILE -> one big string -> word_chunking.py -> chunks
#
# CONTENTS
#   1. Goal and steps
#   2. Line by line
#   3. Why "rb" (binary mode)
#   4. The test block
#   5. Known weaknesses
#
#
# ============================================================
# 1. GOAL AND STEPS
# ============================================================
#
#   Loading = turning a source file into plain text Python can work with.
#
#   1. Import PyPDF2 (installed with: pip install PyPDF2)
#   2. Open the PDF file in "read binary" mode: "rb"
#   3. Create a PdfReader object from that file
#   4. Loop through reader.pages
#   5. For each page, call .extract_text() to get its text
#   6. Build up one big string of all the text combined
#   7. Close the file
#
#
# ============================================================
# 2. LINE BY LINE
# ============================================================
#
# def load_pdf_data(file_path):
#   file_path is the placeholder (a parameter). The real filename
#   ("Catalog.pdf") only appears when I call the function (an argument).
#
# file = open(file_path, "rb")
#   opens the PDF as raw bytes. See section 3 for why "rb".
#
# pdf_reader = PyPDF2.PdfReader(file)
#   PdfReader takes the open file and works out where each page is.
#   The result is an object whose .pages attribute is a list of page objects.
#
# pdf_text = ""
#   empty string that will collect all the text.
#
# for page in pdf_reader.pages:
#   takes the pages one at a time (the loop hands me the next page each round).
#
# page_text = page.extract_text()
#   pulls the text out of ONE page (a method: parentheses = an action).
#
# if page_text:
#   only continues if the page has text. A page with no extractable text
#   (for example a scanned image) gives an empty result, which counts as False.
#
# pdf_text += page_text + "\n"
#   adds the page text to the big string, plus a line break, so the last word
#   of one page does not stick to the first word of the next.
#   (+= means: take what is there and add to it.)
#
# file.close()
#   frees the resources the operating system holds for the file. Prevents
#   "too many open files" errors, locked files I can't delete, and lost data
#   when writing.
#
# return pdf_text
#   sends the finished string back to whoever called the function.
#   Outside the loop, so it runs once at the end.
#
#
# ============================================================
# 3. WHY "rb" (BINARY MODE)
# ============================================================
#
#   "rb" = r (read only) + b (binary: raw bytes, no text decoding)
#
#   Text mode "r":    bytes on disk -> decoded into letters -> Python gets a string
#   Binary mode "rb": bytes on disk -> untouched            -> Python gets bytes
#
#   .txt file:   [72][105][33]               "r"  -> "Hi!"      works
#   PDF file:    [37][80][68][70][143][162]  "r"  -> UnicodeDecodeError / damaged data
#                                            "rb" -> raw bytes  works
#
#   A PDF is not made of letters. Its page content, images and fonts are stored
#   compressed, mixed with layout data. PdfReader reads the raw bytes, and
#   extract_text() decompresses a page and rebuilds the words.
#
#   compressed   = the same information stored in fewer bytes (unreadable)
#   decompressed = rebuilt into the original readable form
#
#     aaaaaaaaaabbbbbbbbbb  --compress-->  10a 10b  --decompress-->  aaaaaaaaaabbbbbbbbbb
#
#   RULE:
#     plain text (.txt .csv .json .py)          -> "r"
#     everything else (pdf, image, audio, zip)  -> "rb"
#
#
# ============================================================
# 4. THE TEST BLOCK
# ============================================================
#
#   if __name__ == "__main__":
#     means: run these lines only if I started THIS file myself.
#     If another file imports me, skip them.
#
#     Run load_data.py directly       -> the lines run (that is my test)
#     Import it from word_chunking    -> the lines are skipped (no extra load)
#
#   Why: importing a file runs all of its top-level lines. Without the guard,
#   importing load_data.py would load the PDF and print it every time.
#
#
# ============================================================
# 5. KNOWN WEAKNESSES (left for later)
# ============================================================
#
#   - If an error happens before file.close(), the file stays open. The
#     professional fix is:  with open(file_path, "rb") as file:
#     which closes the file automatically.
#   - PyPDF2 is old. As far as I know it is no longer developed, and its
#     successor is called pypdf. Check the current docs before a new project.
#   - Scanned PDFs (pictures of pages) have no text to extract, so they need
#     OCR. Those pages are silently skipped by the "if page_text" check.
#   - A wrong file_path raises FileNotFoundError. Nothing catches it yet.
#   - Building one big string with += is fine for this catalog. Collecting the
#     pages in a list and joining them is the more efficient pattern.
#   - Tables and layout are flattened into plain text.


# ============================================================
# CODE
# ============================================================

import PyPDF2  # library that reads PDF files


def load_pdf_data(file_path):
    """
    Read a PDF and return the text of all its pages as one string.

    file_path: path to the PDF file, e.g. "Catalog.pdf".
    Returns an empty string if no page contains extractable text.
    """

    file = open(file_path, "rb")  # open the PDF as raw bytes

    pdf_reader = PyPDF2.PdfReader(file)  # object that knows where each page is

    pdf_text = ""  # collects the text of all pages

    for page in pdf_reader.pages:  # one page per round
        page_text = page.extract_text()  # pull the text out of this page

        if page_text:  # skip pages with no extractable text
            pdf_text += page_text + "\n"  # add it, with a line break after each page

    file.close()  # release the file
    return pdf_text  # outside the loop: runs once, at the end


if __name__ == "__main__":
    text = load_pdf_data("story.pdf")  # test with the story PDF
    print(text)