# Improved Cross-Lingual Question Retrieval for Community Question Answering

This repository contains the data and code to reproduce the results of our paper: 
https://dl.acm.org/citation.cfm?id=3308558.3313502

Please use the following citation:

```
@article{rueckle:WWW:2019,
  title = {Improved Cross-Lingual Question Retrieval for Community Question Answering},
  author = {R{\"u}ckl{\'e}, Andreas and Swarnkar, Krishnkant and Gurevych, Iryna},
  publisher = {ACM},
  booktitle = {The World Wide Web Conference (WWW 2019)},
  pages = {3179--3186},
  year = {2019},
  location = {San Francisco, California, USA},
  doi = {10.1145/3308558.3313502},
  url = {http://doi.acm.org/10.1145/3308558.3313502},
}
```

> **Abstract:** We perform cross-lingual question retrieval in community question answering (cQA), i.e., we retrieve similar questions for queries that are given in another language. The standard approach to cross-lingual information retrieval, which is to automatically translate the query to the target language and continue with a monolingual retrieval model, typically falls short in cQA due to translation errors. This is even more the case for specialized domains such as in technical cQA, which we explore in this work. To remedy, we propose two extensions to this approach that improve cross-lingual question retrieval: (1) we enhance an NMT model with monolingual cQA data to improve the translation quality, and (2) we improve the robustness of a state-of-the-art neural question retrieval model to common translation errors by adding back-translations during training. Our results show that we achieve substantial improvements over the baseline approach and considerably close the gap to a setup where we have access to an external commercial machine translation service (i.e., Google Translate), which is often not the case in many practical scenarios. 




Contact person: Andreas Rücklé

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/


> This repository contains experimental software and is published for the sole purpose of giving additional background 
  details on the respective publication. 


## Data

The data is available on our [public fileserver](http://public.ukp.informatik.tu-darmstadt.de/www19-xling-qr).

  - _askubuntu-human-translations_: This folder contains the human translations of the AskUbuntu dev/test queries
  - _backtranslations-nmt-training_: Contains the parallel sentences obtained by translating titles from AskUbuntu and StackOverflow (java+python splits) to German using the standard en->de Transformer model
  - _rcnn-data_
    - _StackExchange-Monolingual_: The StackOverflow dataset (monolingual)
    - _StackExchange-Monolingual-Paraphrases_: The StackOverflow dataset with paraphrases obtained by backtranslating titles of query questions from en to de and back to en (GT)
    - _AskUbuntu-Monolingual-Paraphrases_: The AskUbuntu dataset with paraphrases
    - _Translations_
      - _AskUbuntu-de-en(GT)_, _AskUbuntu-de-en(TR-CQA)_, etc.: Titles of questions that were translated from German back to English using GT, TR-CQA, etc.


## Usage

The source code of our RCNN adaptation is available here: [RCNN-adaptation](./RCNN-adaptation)

