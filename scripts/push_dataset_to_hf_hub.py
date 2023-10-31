"""Requirements:

- HF_TOKEN environment variable set to your HuggingFace token
- HF Datasets installed
- Jinja2 installed
- Git LFS installed
"""

import argparse
import os

import datasets
import detectors
import parse_arxiv
from datasets import Dataset
from huggingface_hub.hf_api import create_repo
from huggingface_hub.repocard import DatasetCard, DatasetCardData


def main(
    dataset: Dataset,
    dataset_name,
    pretty_dataset_name,
    license,
    hf_id="detectors",
    private=False,
    **kwargs,
):
    token = os.environ["HF_TOKEN"]
    n = len(dataset)
    size_categories = "n<1K" if n < 1000 else "1K<n<10K" if n < 10000 else "10K<n<100K" if n < 100000 else "n>100K"
    original_dataset_url = kwargs.get("url", None)
    original_paper_url = kwargs.get("original_paper_url", None)
    paper = kwargs.get("paper", None)
    authors = kwargs.get("authors", None)
    original_authors = kwargs.get("original_authors", None)
    original_citation_bibtex = kwargs.get("original_citation_bibtex", "")
    citation_bibtex = kwargs.get("citation_bibtex", "")
    paperswithcode_id = kwargs.get("paperswithcode_id", None)
    repo = None
    demo = None
    curators = "Eduardo Dadalto"
    dataset_card_authors = curators
    dataset_card_contact = "https://huggingface.co/edadaltocg"
    direct_use = (
        "This dataset is intended to be used as an ouf-of-distribution dataset for image classification benchmarks."
    )
    curation_rationale_section = """The goal in curating and sharing this dataset to the HuggingFace Hub is to accelerate research and promote reproducibility in generalized Out-of-Distribution (OOD) detection.

Check the python library [detectors](https://github.com/edadaltocg/detectors) if you are interested in OOD detection."""
    out_of_scope_use = "This dataset is not annotated."
    personal_and_sensitive_information = "Please check original paper for details on the dataset."
    bias_risks_limitations = "Please check original paper for details on the dataset."

    citation_bibtex = (
        """@software{detectors2023,
author = {Eduardo Dadalto},
title = {Detectors: a Python Library for Generalized Out-Of-Distribution Detection},
url = {https://github.com/edadaltocg/detectors},
doi = {https://doi.org/10.5281/zenodo.7883596},
month = {5},
year = {2023}
}"""
        + "\n\n"
        + citation_bibtex
        or "" + "\n\n" + original_citation_bibtex
        or ""
    ).strip()

    card_data = DatasetCardData(
        task_categories=["image-classification"],
        pretty_name=pretty_dataset_name,
        license=license,
        size_categories=size_categories,
        paperswithcode_id=paperswithcode_id if len(paperswithcode_id) > 1 else None,
    )
    dataset_card = DatasetCard.from_template(
        card_data=card_data,
        template_path=os.path.join("templates", "DATASET_CARD_TEMPLATE.md"),
        paper=paper,
        original_paper_url=original_paper_url,
        demo=demo,
        repo=repo,
        dataset_card_authors=dataset_card_authors,
        dataset_card_contact=dataset_card_contact,
        curators=curators,
        direct_use=direct_use,
        curation_rationale_section=curation_rationale_section,
        out_of_scope_use=out_of_scope_use,
        personal_and_sensitive_information=personal_and_sensitive_information,
        bias_risks_limitations=bias_risks_limitations,
        citation_bibtex=citation_bibtex,
        original_dataset_url=original_dataset_url,
        authors=authors,
        original_authors=original_authors,
    )

    repo_id = f"{dataset_name}-ood"
    tag = f"{hf_id}/{repo_id}"
    create_repo(repo_id=tag, exist_ok=True, token=token, private=private, repo_type="dataset")
    dataset_card.push_to_hub(tag, token=token)
    dataset.push_to_hub(tag, token=token)

    # test
    dataset = datasets.load_dataset(tag, split="train")
    assert len(dataset) == n
    for x in dataset:
        print(x)
        assert "image" in x
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="ssb_easy")
    parser.add_argument("--pretty_dataset_name", type=str, default=None)
    parser.add_argument("--paperswithcode_id", type=str, default=None)

    parser.add_argument("--hf_id", type=str, default=None)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    dataset = detectors.create_dataset(args.dataset_name)

    try:
        arxiv = dataset.paper_url
        parsed = parse_arxiv.main(arxiv)
    except AttributeError:
        parsed = {"link": None, "authors": None, "bibtext": None}

    try:
        license = dataset.license
    except:
        license = "unknown"

    try:
        url = dataset.url
    except:
        url = None

    try:
        original_paper_url = dataset.original_paper_url
        original_parsed = parse_arxiv.main(original_paper_url)
    except:
        original_parsed = {"link": None, "authors": None, "bibtext": None}

    kwargs = {
        "original_paper_url": original_parsed["link"],
        "paper": parsed["link"],
        "original_authors": original_parsed["authors"],
        "authors": parsed["authors"],
        "citation_bibtex": parsed["bibtext"],
        "original_citation_bibtex": original_parsed["bibtext"],
        "url": url,
        "paperswithcode_id": args.paperswithcode_id,
    }

    def gen():
        for x, y in dataset:
            yield {"image": x}

    dataset = Dataset.from_generator(gen)

    main(dataset, args.dataset_name, args.pretty_dataset_name, license=license, private=args.private, **kwargs)
