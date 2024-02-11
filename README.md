# TATA: Stance Detection via Topic-Agnostic and Topic-Aware Embeddings
Github Repository for TATA: Stance Detection via Topic-Agnostic and Topic-Aware Embeddings

Full Paper: https://www.hanshanley.com/files/tata.pdf

Stance detection is important for understanding different attitudes and beliefs on the Internet. However, given that a passage's stance toward a given topic is often highly dependent on that topic, building a stance detection model that generalizes to unseen topics is difficult. In this work, we propose using contrastive learning as well as an unlabeled dataset of news articles that cover a variety of different topics to train topic-agnostic/TAG and topic-aware/TAW embeddings for use in downstream stance detection. Combining these embeddings in our full TATA model, we achieve state-of-the-art performance across several public stance detection datasets (0.771-score on the Zero-shot VAST dataset). 


## Topic-Aware (TAW) Dataset 
Within this work, utilizing a dataset of [news articles from 3,074 news websites](https://arxiv.org/abs/2305.09820), the [MPNet model](https://huggingface.co/sentence-transformers/all-mpnet-base-v2), the [Parrot paraphrase](https://github.com/PrithivirajDamodaran/Parrot_Paraphraser), and [Flan-T5-XXL](https://huggingface.co/google/flan-t5-xxl), we extract and pair paragraphs with similar topics from different websites for use in training a Topic-Aware (TAW) encoding model. To request the extended dataset of 238,228 (where there are no more than 1,000 paragraphs from any one given website), please fill out this Google form. To request the unfiltered dataset of 984539 (an unrestricted number of paragraphs from any given website), please fill out this [Google form](https://forms.gle/hbiJ11ipbsmmXPfC6). This dataset may only be utilized for research purchases, the copyright of the articles within this dataset belongs to the respective websites. 

## Topic-Agnostic (TAG) Dataset
In order to initially train a dataset of topic-agnostic encoding layer for use in our stance detection model, we extended the [original VAST dataset](https://github.com/emilyallaway/zero-shot-stance) using [the Dipper Paraphraser](https://huggingface.co/kalpeshk2011/dipper-paraphraser-xxl). You can download the extended VAST/TAG dataset, at the following link. 

## Citing the paper
If you use the code or datasets from this apper, you can cite us with the following BibTex entry:
```
@inproceedings{hanley2023tata,
    title={{TATA}: Stance Detection via Topic-Agnostic and Topic-Aware Embeddings},
    author={Hanley, Hans W. A. and Durumeric, Zakir},
    booktitle={The 2023 Conference on Empirical Methods in Natural Language Processing},
    year={2023},
    url={https://openreview.net/forum?id=J9Vx7eTuWb}
  }
```
