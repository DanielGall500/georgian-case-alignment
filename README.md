## Targeted Syntactic Evaluation of Georgian Case Alignment

This repository contains the code used to run the evaluations of models on the [Georgian Case Alignment](huggingface.co/datasets/DanielGallagherIRE/georgian-case-alignment) dataset.
The corresponding paper can be found in the ACL Anthology [here](https://aclanthology.org/2026.loreslm-1.23/).
It was presented at LoResLM co-located with EACL 2026 and was nominated for the Best Paper Award.
Much of the creation and evaluation of the dataset uses code from the repository [Grew-TSE](https://github.com/DanielGall500/Grew-TSE).

#### Citation
```
@inproceedings{gallagher-heyer-2026-targeted,
    title = "Targeted Syntactic Evaluation of Language Models on {G}eorgian Case Alignment",
    author = "Gallagher, Daniel  and
      Heyer, Gerhard",
    editor = "Hettiarachchi, Hansi  and
      Ranasinghe, Tharindu  and
      Plum, Alistair  and
      Rayson, Paul  and
      Mitkov, Ruslan  and
      Gaber, Mohamed  and
      Premasiri, Damith  and
      Tan, Fiona Anting  and
      Uyangodage, Lasitha",
      booktitle = "Proceedings of the Second Workshop on Language Models for Low-Resource Languages ({L}o{R}es{LM} 2026)",
      month = mar,
      year = "2026",
      address = "Rabat, Morocco",
      publisher = "Association for Computational Linguistics",
      url = "https://aclanthology.org/2026.loreslm-1.23/",
      doi = "10.18653/v1/2026.loreslm-1.23",
      pages = "259--270",
      ISBN = "979-8-89176-377-7",
      abstract = "This paper evaluates the performance of transformer-based language models on split-ergative case alignment in Georgian, a particularly rare system for assigning grammatical cases to mark argument roles. We focus on subject and object marking determined through various permutations of nominative, ergative, and dative noun forms. A treebank-based approach for the generation of minimal pairs using the Grew query language is implemented. We create a dataset of 370 syntactic tests made up of seven tasks containing 50-70 samples each, where three noun forms are tested in any given sample. Five encoder- and two decoder-only models are evaluated with word- and/or sentence-level accuracy metrics. Regardless of the specific syntactic makeup, models performed worst in assigning the ergative case correctly and strongest in assigning the nominative case correctly. Performance correlated with the overall frequency distribution of the three forms (NOM {\ensuremath{>}} DAT {\ensuremath{>}} ERG). Though data scarcity is a known issue for low-resource languages, we show that the highly specific role of the ergative along with a lack of available training data likely contributes to poor performance on this case. The dataset is made publicly available and the methodology provides an interesting avenue for future syntactic evaluations of languages where benchmarks are limited."
}
```

#### Acknowledgements
Part of this work was conducted within the [CORAL project](https://coral-nlp.github.io) funded by the German Federal Ministry of Research, Technology, and Space (BMFTR) under the grant number 16IS24077A. Responsibility for the content of this publication lies with the authors.
