import requests
from minsearch import Index

def load_faq_data():
    """
    Load FAQ data from the faq url.
    """
    docs_url = 'https://datatalks.club/faq/json/courses.json'
    response = requests.get(docs_url)
    courses_raw = response.json()

    documents = []
    url_prefix='https://datatalks.club/faq'

    for course in courses_raw:
        print(f'Processing course: {course}')

        course_url = f'{url_prefix}{course["path"]}'
        course_response = requests.get(course_url)
        course_data = course_response.json()

        documents.extend(course_data)

    return documents

def build_index(documents):
    """
    Build the index from the documents.
    """
    index = Index(
        text_fields=['question', 'section', 'answer'],
        keyword_fields=['course']
    )
    index.fit(documents)
    return index