import requests
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
from pypdf import PdfReader
from io import BytesIO

def get_all_links(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    links = []
    for link in soup.find_all("a"):
        links.append(link.get("href"))
    return links

def get_all_events():
    event_links = get_all_links("https://aclanthology.org/")

    events = []

    for link in event_links:
        if link.startswith("/events"):
            match = re.match("/events/(?P<eventname>\w+)-(?P<eventyear>\d+)/?", link)
            events.append({"name": match.group("eventname"), "year": match.group("eventyear")})
    
    return events

def get_all_pdf_links(events, after_year=2000):
    pdf_links = []

    for event in tqdm(events, total=len(events)):
        if int(event["year"]) >= after_year:
            links = get_all_links("https://aclanthology.org/events/" + event["name"] + "-" + event["year"] + "/")
            for link in links:
                if link.endswith(".pdf"):
                    pdf_links.append(link)


def process_pdf_links(pdf_links):
    with open("processed_pdf_links.txt", "r") as f:
        processed_pdf_links = f.read().splitlines()

    def save_link(filepath, link):
        with open(filepath, "a") as f:
            f.write(link + "\n")

    def save_processed_pdf_link(pdf_link):
        save_link("processed_pdf_links.txt", pdf_link)

    def save_potential_pdf_link(pdf_link):
        save_link("potential_pdf_links.txt", pdf_link)

    def save_failed_pdf_link(pdf_link):
        save_link("failed_pdf_links.txt", pdf_link)

    for pdf_link in tqdm(pdf_links, total=len(pdf_links), desc="Processing PDFs"):
        if pdf_link not in processed_pdf_links:
            try:
                pdf = requests.get(pdf_link)
                reader = PdfReader(BytesIO(pdf.content))
                number_of_pages = len(reader.pages)

                if number_of_pages < 100:
                    for page_num in range(number_of_pages):
                        page = reader.pages[page_num]
                        text = page.extract_text()
                        if ("commonsense" in text.lower() or "world knowledge" in text.lower()) and "error analysis" in text.lower():
                            save_potential_pdf_link(pdf_link)
                            break
            except Exception as e:
                save_failed_pdf_link(pdf_link)
            
            processed_pdf_links.append(pdf_link)
            save_processed_pdf_link(pdf_link)

if __name__ == "__main__":
    # events = get_all_events()
    # pdf_links = get_all_pdf_links(events)
    with open("pdf_links_2000.txt", "r") as f:
        pdf_links = f.read().splitlines()
    process_pdf_links(pdf_links)