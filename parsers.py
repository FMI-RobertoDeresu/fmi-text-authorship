import pandas as pd
import re


def parse_federalist_papers(file_path):
    with open(file_path) as f:
        content = f.read()

    dataset = pd.DataFrame(columns=['no', 'author', 'content'])

    regex_paper_start = 'federalist?.\sno?.\s\d{1,2}'
    regex_paper_end = '\n\n\n\n\n|\*There\sare\stwo\sslightly'
    regex_paper_author = 'HAMILTON\sAND\sMADISON|HAMILTON\sOR\sMADISON|HAMILTON|JAY|MADISON'

    paper_start_matches = list(re.finditer(regex_paper_start, content, re.IGNORECASE))
    for paper_start_match in paper_start_matches:
        # paper
        paper_start = paper_start_match.span()[0]

        paper_end_match = re.search(regex_paper_end, content[paper_start:], re.IGNORECASE)
        paper_end = paper_start + paper_end_match.span()[0]

        # paper no
        paper_no_start = paper_start
        paper_no_end = paper_start_match.span()[1]

        # author
        paper_author_match = re.search(regex_paper_author, content[paper_start:])
        paper_author_start = paper_start + paper_author_match.span()[0]
        paper_author_end = paper_start + paper_author_match.span()[1]

        # content
        paper_content_start = paper_author_end
        paper_content_end = paper_end

        paper_data = {
            'no': content[paper_no_start:paper_no_end],
            'author': content[paper_author_start:paper_author_end],
            'content': re.sub(r"\s+", " ", content[paper_content_start:paper_content_end])
        }

        dataset = dataset.append(paper_data, ignore_index=True)

    return dataset
