# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PMC Open Access Subset."""

import datetime
from functools import cache

import datasets
import fsspec
import pandas as pd

_CITATION = """\
PMC Open Access Subset [Internet]. Bethesda (MD): National Library of Medicine. 2003 - [cited YEAR MONTH DAY]. Available from https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/
"""

_DESCRIPTION = """\
The PMC Open Access Subset includes more than 3.4 million journal articles and preprints that are made available under
license terms that allow reuse. 

Not all articles in PMC are available for text mining and other reuse, many have copyright protection, however articles
in the PMC Open Access Subset are made available under Creative Commons or similar licenses that generally allow more
liberal redistribution and reuse than a traditional copyrighted work. 

The PMC Open Access Subset is one part of the PMC Article Datasets
"""

_HOMEPAGE = "https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/"

_LICENSE = """\
Within the PMC Open Access Subset, there are three groupings based on available license terms:
- Commercial Use Allowed - CC0, CC BY, CC BY-SA, CC BY-ND licenses;
- Non-Commercial Use Only - CC BY-NC, CC BY-NC-SA, CC BY-NC-ND licenses; and
- Other - no machine-readable Creative Commons license, no license, or a custom license.
"""

_URL = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/{subset}/txt/"
_SUBSETS = {
    "commercial": "oa_comm",
    "non_commercial": "oa_noncomm",
    "other": "oa_other",
}


@cache
def request_data_urls():
    fs = fsspec.filesystem("https")
    result = {}
    for subset, subset_url in _SUBSETS.items():
        urls = fs.ls(_URL.format(subset=subset_url), detail=False)
        baseline_urls = [
            url for url in urls for filename in url.split("/")[-1:] if filename.startswith(f"{subset_url}_txt.PMC")
        ]
        baseline_date = parse_date(baseline_urls[0])
        baseline_file_list_urls = [url for url in baseline_urls if url.endswith(".csv")]
        baseline_archive_urls = [url for url in baseline_urls if url.endswith(".tar.gz")]
        incremental_urls = [
            url for url in urls for filename in url.split("/")[-1:] if filename.startswith(f"{subset_url}_txt.incr.")
        ]
        incremental_file_list_urls = [url for url in incremental_urls if url.endswith(".csv")]
        incremental_archive_urls = [url for url in incremental_urls if url.endswith(".tar.gz")]
        result["baseline_date"] = baseline_date
        result[subset] = {
            "baseline_urls": list(zip(baseline_file_list_urls, baseline_archive_urls, strict=False)),
            "incremental_urls": list(zip(incremental_file_list_urls, incremental_archive_urls, strict=False)),
        }
    return result


def parse_date(url):
    return url.split("/")[-1].split(".")[-3]


class OpenAccessConfig(datasets.BuilderConfig):
    """BuilderConfig for the PMC Open Access Subset."""

    def __init__(self, date=None, subsets="all", **kwargs):
        """BuilderConfig for the PMC Open Access Subset.

        Args:
            date (`str`, default BASELINE_DATE) : Up to date, in ISO format. Pass 'latest' for latest date.
            subsets (`str` or `list[str]`, default 'all'): List of subsets to load. Possible values are 'all' or any combination
                of {'commercial', 'non_commercial', 'other'}.
            **kwargs: Keyword arguments forwarded to `BuilderConfig`.
        """
        if date is None:
            date = request_data_urls()["baseline_date"]
        date = datetime.date.today().isoformat() if date == "latest" else date
        subsets = [subsets] if isinstance(subsets, str) else subsets
        subsets_name = "+".join(subsets)
        name = f"{date}.{subsets_name}"
        super().__init__(name=name, **kwargs)
        self.subsets = subsets if subsets_name != "all" else list(_SUBSETS.keys())
        self.date = date


class OpenAccess(datasets.GeneratorBasedBuilder):
    """PMC Open Access Subset."""

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIG_CLASS = OpenAccessConfig
    BUILDER_CONFIGS = [OpenAccessConfig(subsets="all")] + [OpenAccessConfig(subsets=subset) for subset in _SUBSETS]
    DEFAULT_CONFIG_NAME = f"{request_data_urls()['baseline_date']}.all"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "pmid": datasets.Value("string"),
                    "accession_id": datasets.Value("string"),
                    "license": datasets.Value("string"),
                    "last_updated": datasets.Value("string"),
                    "retracted": datasets.Value("string"),
                    "citation": datasets.Value("string"),
                }
            ),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        urls = request_data_urls()
        date = datetime.date.fromisoformat(self.config.date)
        paths = []
        for subset in self.config.subsets:
            # Baselines
            baseline_urls = urls[subset]["baseline_urls"]
            # Incremental
            incremental_urls = [
                url_pair
                for url_pair in urls[subset]["incremental_urls"]
                if datetime.date.fromisoformat(parse_date(url_pair[0])) <= date
            ]
            paths += dl_manager.download(baseline_urls + incremental_urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "paths": [(file_list, dl_manager.iter_archive(archive)) for file_list, archive in paths],
                },
            ),
        ]

    def _generate_examples(self, paths):
        key = 0
        for file_list, archive in paths:
            file_list_data = pd.read_csv(file_list, index_col="Article File").to_dict(orient="index")
            for path, file in archive:
                data = file_list_data.pop(path)
                content = file.read()
                try:
                    text = content.decode("utf-8").strip()
                except UnicodeDecodeError:
                    text = content.decode("latin-1").strip()
                data = {
                    "text": text,
                    "pmid": data["PMID"],
                    "accession_id": data["AccessionID"],
                    "license": data["License"],
                    "last_updated": data["LastUpdated (YYYY-MM-DD HH:MM:SS)"],
                    "retracted": data["Retracted"],
                    "citation": data["Article Citation"],
                }
                yield key, data
                key += 1


if __name__ == "__main__":
    ds = load_dataset("pmc/open_access", "wikitext-103-v1")
    print(ds)