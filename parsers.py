import os
import pandas as pd
import re
import numpy as np
from input import test
import json


def parse_test():
    dataset = pd.DataFrame(columns=['no', 'author', 'content'])

    for item in test.input:
        line_data = {
            'no': item["no"],
            'author': item['author'],
            'content': re.sub(r"\s+", " ", item['content'])
        }

        dataset = dataset.append(line_data, ignore_index=True)

    return dataset


def parse_federalist_papers(file_path):
    csv_file_path = os.path.join(os.path.dirname(file_path), "federalist_papers.csv")
    if os.path.exists(csv_file_path):
        dataset = pd.read_csv(csv_file_path, index_col=0, dtype=np.str)
        return dataset

    with open(file_path, encoding="utf8") as f:
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

        no = content[paper_no_start:paper_no_end]
        author = content[paper_author_start:paper_author_end]
        content = re.sub(r"\s+", " ", content[paper_content_start:paper_content_end])

        paper_data = {
            'no': no,
            'author': author,
            'content': content
        }

        dataset = dataset.append(paper_data, ignore_index=True)

    dataset.to_csv(csv_file_path)
    return dataset


def parse_pan_11_dataset(path):
    csv_file_path = os.path.join(os.path.dirname(path), os.path.basename(path) + ".csv")
    if os.path.exists(csv_file_path):
        dataset = pd.read_csv(csv_file_path, index_col=0, dtype=np.str)
        return dataset

    dataset = parse_pan_dataset(path)
    dataset.to_csv(csv_file_path)
    return dataset


def parse_pan_12_dataset(path):
    datasets = {}
    dirs = [os.path.join(path, x) for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
    for dir_path in dirs:
        dir_name = os.path.basename(dir_path)
        csv_file_path = os.path.join(path, dir_name + ".csv")
        if os.path.exists(csv_file_path):
            dataset = pd.read_csv(csv_file_path, index_col=0, dtype=np.str)
            datasets[dir_name] = dataset
            continue

        dataset = parse_pan_dataset(dir_path)
        dataset.to_csv(csv_file_path)
        datasets[dir_name] = dataset

    return datasets


def parse_pan_dataset(path):
    dataset = pd.DataFrame(columns=['no', 'author', 'content', 'true-author'])
    with open(os.path.join(path, "ground-truth.json"), encoding="utf8") as f:
        ground_truth = json.load(f)["ground-truth"]

    dirs = [os.path.join(path, x) for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
    for dir_path in dirs:
        files = os.listdir(dir_path)
        for file_name in files:
            no = re.search('\d+', file_name).group()
            author = os.path.basename(dir_path)

            with open(os.path.join(dir_path, file_name), encoding="utf8") as f:
                content = f.read()
                content = re.sub(r"<NAME/>", "", content)
                content = re.sub(r"\s+", " ", content)

            if file_name.startswith("unknown"):
                true_author = list(filter(lambda x: x["unknown-text"] == file_name, ground_truth))[0]["true-author"]
                author_code = int(re.search("\d+", true_author).group())

                # if code is 0, skip it
                if author_code == 0:
                    continue
            else:
                true_author = author

            data = {
                'no': no,
                'author': author,
                'content': content,
                'true-author': true_author
            }

            dataset = dataset.append(data, ignore_index=True)

    return dataset
