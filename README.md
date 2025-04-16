# FedEFC: Federated Learning Using Enhanced Forward Correction Against Noisy Labels

### Abstract
Federated Learning (FL) is a powerful framework for privacy-preserving distributed learning. It enables multiple clients to collaboratively train a global model without sharing raw data. However, handling noisy labels in FL remains a major challenge due to heterogeneous data distributions and communication constraints, which can severely degrade model performance. To address this issue, we propose FedEFC, a novel method designed to tackle the impact of noisy labels in FL. FedEFC mitigates this issue through two key techniques: (1) prestopping, which prevents overfitting to mislabeled data by dynamically halting training at an optimal point, and (2) loss correction, which adjusts model updates to account for label noise. In particular, we develop an effective loss correction tailored to the unique challenges of FL, including data heterogeneity and decentralized training. Furthermore, we provide a theoretical analysis, leveraging the composite proper loss property, to demonstrate that the FL objective function under noisy label distributions can be aligned with the clean label distribution. Extensive experimental results validate the effectiveness of our approach, showing that it consistently outperforms existing FL techniques in mitigating the impact of noisy labels, particularly under heterogeneous data settings (e.g., achieving up to 41.64% relative performance improvement over the existing loss correction method).


### Citation

Seunghun Yu, Jin-Hyun Ahn, and Joonhyuk Kang. FedEFC: Federated Learning Using Enhanced Forward Correction Against Noisy Labels. *arXiv preprint arXiv:2504.05615*, 2025.


```
@article{yu2025fedefcfederatedlearningusing,
      title={{FedEFC}: Federated Learning Using Enhanced Forward Correction Against Noisy Labels}, 
      author={Seunghun Yu and Jin-Hyun Ahn and Joonhyuk Kang},
      year={2025},
      journal={arXiv preprint arXiv:2504.05615},
      eprint={2504.05615},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
}
```
