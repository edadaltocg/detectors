---
# For reference on dataset card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/datasetcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/datasets-cards
{{ card_data }}
---

# Dataset Card for {{ pretty_name | default("Dataset Name", true) }} for OOD Detection

<!-- Provide a quick summary of the dataset. -->

{{ dataset_summary | default("", true) }}

## Dataset Details

### Dataset Description

<!-- Provide a longer summary of what this dataset is. -->

{{ dataset_description | default("", true) }}

- **Original Dataset Authors**: {{ original_authors | default("[More Information Needed]", true)}}
- **OOD Split Authors:** {{ authors | default("[More Information Needed]", true)}}
- **Shared by:** {{ curators | default("[More Information Needed]", true)}}
- **License:** {{ license | default("[More Information Needed]", true)}}

### Dataset Sources

<!-- Provide the basic links for the dataset. -->

- **Original Dataset Paper:** {{ original_paper_url | default("[More Information Needed]", true)}}
- **First OOD Application Paper:** {{ paper | default("[More Information Needed]", true)}}


### Direct Use

<!-- This section describes suitable use cases for the dataset. -->

{{ direct_use | default("[More Information Needed]", true)}}

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the dataset will not work well for. -->

{{ out_of_scope_use | default("[More Information Needed]", true)}}


### Curation Rationale

<!-- Motivation for the creation of this dataset. -->

{{ curation_rationale_section | default("[More Information Needed]", true)}}

### Personal and Sensitive Information

<!-- State whether the dataset contains data that might be considered personal, sensitive, or private (e.g., data that reveals addresses, uniquely identifiable names or aliases, racial or ethnic origins, sexual orientations, religious beliefs, political opinions, financial or health data, etc.). If efforts were made to anonymize the data, describe the anonymization process. -->

{{ personal_and_sensitive_information | default("[More Information Needed]", true)}}

### Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

{{ bias_risks_limitations | default("[More Information Needed]", true)}}

## Citation

<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

```bibtex
{{ citation_bibtex | default("[More Information Needed]", true)}}
```

## Dataset Card Authors

{{ dataset_card_authors | default("[More Information Needed]", true)}}

## Dataset Card Contact

{{ dataset_card_contact | default("[More Information Needed]", true)}}