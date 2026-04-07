# MILE: Multi-Expert Ensemble with Instance Selection for Multi-Class Imbalanced Learning

This repository implements MILE as presented in the IEEE TEVC 2026 paper. MILE is a skill-diverse expert learning method for multi-class imbalanced learning.

<img width="1882" height="816" alt="image" src="https://github.com/user-attachments/assets/bc127676-f544-4b82-aef2-a67476c61e2f" />

# Abstract
Classification tasks often encounter imbalanced datasets, where skewed class distributions bias models toward the majority class, resulting in poor performance for the minority class. This issue becomes even more challenging in multi-class imbalanced datasets. Existing methods for addressing class imbalance often prioritize improving the classification performance of the minority class at the expense of the majority class. To tackle this issue, this paper proposes a skill-diverse expert learning strategy, which performs multi-objective evolutionary sampling from imbalanced data to obtain representative high-quality instance subsets. Three expert objective functions, acting as experts simulating different class distributions, are designed to evaluate the quality of the instance subsets. A constraint is proposed for each expert objective function to ensure that the instance subsets achieve better classification performance than using the full training set. Finally, an ensemble strategy is used to combine classifiers trained on diverse subsets of instances for prediction.
Compared to state-of-the-art data-level and ensemble learning-based methods, the experimental results show that the proposed method delivers the best overall classification performance across 22 imbalanced datasets.

# Acknowledge
Please kindly cite this paper in your publications if it helps your research:
```
@article{chen2026multi,
title={MILE: Multi-expert ensemble with instance selection for multi-class imbalanced learning},
author={Zhai, Shichao and Jiao, Ruwang and Xue, Bing and Nojima, Yusuke and Zhang, Mengjie},
journal={IEEE Transactions on Evolutionary Computation},
year={2026},
publisher={IEEE}
}
```

# License
This code is released under the MIT License.

# Contact
If you face any difficulty with the implementation, please refer to: sc_zhai@163.com
