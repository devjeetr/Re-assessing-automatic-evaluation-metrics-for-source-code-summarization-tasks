## Replication Package for FSE 2021 Paper "Reassessing Automatic Evaluation Metrics for Code Summarization Tasks"

This package contains data and scripts that can be used to replicate our paper
`"Reassessing automatic evaluation metrics for code summarization tasks." Proceedings of the 29th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering. 2021. Roy, Devjeet, Sarah Fakhoury, and Venera Arnaoudova.`

[Link to preprint](https://sarahfakhoury.com/2021-FSE-Summarization-Metrics.pdf)

### Data Description

- `data/assessments.csv` contains cleaned human assessments of automatically generated summaries used for analysis
- `data/demographics.csv` contains demographic information for the participants.
- `data/pairwise_model_comparisons.csv` contains all pairwise model comparisons (including human & metric statistical significance) for all metrics. Models 1-5 correspond to original models from Haque et al. and 6-105 correspond to synthetic models.

### Metric Calculation Details

We use official implementations for all metrics as much as possible. The only exceptions are SentBLEU (so we
can use Smoothing Method #5 from Chen & Cherry) and Rouge (Java implementation provided challenges in caching results for randomization tests).

| Metric      | Package                                                        | Notes                      |
| ----------- | -------------------------------------------------------------- | -------------------------- |
| Corpus BLEU | [sacrebleu](https://github.com/mjpost/sacrebleu)               | default setup              |
| Rouge       | [rouge](https://github.com/pltrdy/rouge)                       | default setup              |
| BERTScore   | [bertscore](https://github.com/Tiiiger/bert_score)             | default (en/roberta-large) |
| METEOR      | [official meteor 1.55](https://www.cs.cmu.edu/~alavie/METEOR/) | default setup              |
| SentBLEU    | [NLTK](https://pypi.org/project/nltk/)                         | smoothing method 5         |
| chrF        | [chrF](https://github.com/m-popovic/chrF)                      | default setup              |

### Kendall's TAU

The implementation for Kendall's TAU can be found in `scripts/kendalls_tau.py`. Please refer to [Stanchevet al.][1] for details and rationale.

[1]: https://aclanthology.org/2020.wmt-1.103/

### Citation

```
@inproceedings{roy2021reassessing,
  title={Reassessing automatic evaluation metrics for code summarization tasks},
  author={Roy, Devjeet and Fakhoury, Sarah and Arnaoudova, Venera},
  booktitle={Proceedings of the 29th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering},
  pages={1105--1116},
  year={2021}
}
```
