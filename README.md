# CRoW: Benchmarking Commonsense Reasoning in Real-World Tasks

[Paper](https://aclanthology.org/2023.emnlp-main.607) | [Website](https://mete.is/crow) | [Leaderboard](https://mete.is/crow/leaderboard) | [Download data](https://mete.is/crow/tasks)

CRoW is a multi-task benchmark to evaluate commonsense reasoning ability of NLP systems in solving real-world tasks where this ability is required.

This repo contains the code used to build [CRoW benchmark](https://mete.is/crow) and evaluate models on it. If you would like to download the data for this benchmark and evaluate your own models on it, please check out the [Tasks](https://mete.is/crow/tasks) section. We also keep an active [leaderboard](https://mete.is/crow/leaderboard) for this benchmark and you can contribute to it by following the [Getting Started](https://mete.is/crow/getting-started) guide.

For more information on this benchmark, check the [website](https://mete.is/crow).

## Citation
```
@inproceedings{ismayilzada-etal-2023-crow,
    title = "{CR}o{W}: Benchmarking Commonsense Reasoning in Real-World Tasks",
    author = "Ismayilzada, Mete  and
      Paul, Debjit  and
      Montariol, Syrielle  and
      Geva, Mor  and
      Bosselut, Antoine",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.607",
    pages = "9785--9821",
    abstract = "Recent efforts in natural language processing (NLP) commonsense reasoning research have yielded a considerable number of new datasets and benchmarks. However, most of these datasets formulate commonsense reasoning challenges in artificial scenarios that are not reflective of the tasks which real-world NLP systems are designed to solve. In this work, we present CRoW, a manually-curated, multi-task benchmark that evaluates the ability of models to apply commonsense reasoning in the context of six real-world NLP tasks. CRoW is constructed using a multi-stage data collection pipeline that rewrites examples from existing datasets using commonsense-violating perturbations. We use CRoW to study how NLP systems perform across different dimensions of commonsense knowledge, such as physical, temporal, and social reasoning. We find a significant performance gap when NLP systems are evaluated on CRoW compared to humans, showcasing that commonsense reasoning is far from being solved in real-world task settings. We make our dataset and leaderboard available to the research community.",
}
```