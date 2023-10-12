# sigir_2023_uCTRL

This code is the implementation of the paper "uCTRL: Unbiased Contrastive Representation Learning via Alignment and Uniformity for Collaborative Filtering" in SIGIR 2023 (https://dl.acm.org/doi/abs/10.1145/3539618.3592076)

Please refer to the "uctrl.py" code in "/recbole/model/general_recommender/" for detailed model implementation.


We conducted experiments by modifying the code from "https://recbole.io/".
Since we assumed that the training data was biased, we utilized unbiased evaluation for validation.

Unbiased evaluation:
Longqi Yang et al., Unbiased offline recommender evaluation for missing-not-at-random implicit feedback, RecSys 2018. (https://dl.acm.org/doi/10.1145/3240323.3240355)
