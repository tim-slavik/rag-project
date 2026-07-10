import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
from docx import Document
import pdfplumber
import re
from io import BytesIO

def gather_links(config):
    os.makedirs(config.get('output_folder'), exist_ok=True)

    session = requests.Session()
    if config.get('User-Agent'):
        session.headers.update({"User-Agent": config.get('User-Agent')})

    page = session.get(config.get('base_url'))
    page.raise_for_status()

    # parse the page with BeautifulSoup
    soup = BeautifulSoup(page.text, 'html.parser')

    # find all desired doc links and remove duplicates
    article_links = set(list([
        urljoin(config.get('base_url'), href)
        for href in (a.get("href") for a in soup.find_all("a"))
        if href and any(href.lower().endswith(ext) for ext in config.get('doc_types_to_download'))
    ]))

    return article_links

def get_file_type(filename):
    if filename.lower().endswith(".pdf"):
        return "pdf"
    if filename.lower().endswith(".docx"):
        return "docx"
    return None

def extract_docx_text(url):
    time.sleep(1)
    response = requests.get(url)
    response.raise_for_status()

    file_obj = BytesIO(response.content)
    doc = Document(file_obj)

    return "\n".join(para.text for para in doc.paragraphs)


def extract_pdf_text(url):
    time.sleep(1)
    response = requests.get(url)
    response.raise_for_status()

    file_obj = BytesIO(response.content)

    text = ""
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def process_links(article_links : list):
    all_text = []

    for link in article_links:
        print(f"Processing: {link}")
        try:

            filetype = get_file_type(link)

            if filetype == "pdf":
                text = extract_pdf_text(link)
            elif filetype == "docx":
                text = extract_docx_text(link)
            else:
                print(f"Skipping unknown file type: {link}")
                continue

            all_text.append(text)
        except Exception as e:
            print(f"Error processing {link}: {e}")
    
    return all_text

def remove_text_before_colon(docs: list[str]) -> str:
    cleaned_docs = []
    for text in docs:
        idx = text.find(" : ")
        if idx == -1:
            cleaned_docs.append(text)
        else:
            cleaned_docs.append(text[idx + len(" : "):])
    
    return cleaned_docs


def clean_text(text :list[str]):
    cleaned_docs = []
    for s in text:
        # Remove escape sequences like \n, \t
        s = s.replace("\n", "").replace("\t", "")

        # Collapse multiple spaces into one
        s = re.sub(r"\s+", " ", s)

        # Strip leading/trailing whitespace
        s = s.strip()

        cleaned_docs.append(s)
    return cleaned_docs

def get_prepared_docs(config: dict):
    
    links = gather_links(config)
    raw_text = process_links(links)
    no_title_text = remove_text_before_colon(raw_text)
    prepared_text = clean_text(no_title_text)   

    return prepared_text
    
    
