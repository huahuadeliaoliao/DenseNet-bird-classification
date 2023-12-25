# DenseNet-bird-classification
## [Datasets](https://drive.google.com/file/d/1CWwA4M7bZiKvsZ2sF7_4Vc3f3YPK7zcS/view?usp=sharing)
## Performance on the test set

|      | Desnet-121 | *Densenet-121withSE | Densenet-201withECA |
| :--- | :---: | :---: | :---: |
| accuracy | 0.79 | *0.67 | 0.85 |
| loss | 0.81 | *1.45 | 0.71 |

## Directions for continued improvement
1. Add more data enhancements.
2. Since DenseNet121,169 and 201 all have the same k-value, there is not much difference in the performance of these three models on the current dataset, and a DenseNet model with a larger k-value such as DenseNet161 or Wide-DenseNet-BC can be considered.

## *Note
The addition of the SE attention mechanism DenseNet model requires a lot of computational resources, due to my time and equipment constraints so the results I got about DenseNet121withSE do not accurately represent the effect of adding the SE attention mechanism.
In the related[papers](https://www.sciencedirect.com/science/article/abs/pii/S195903182100141X)I checked, it is said that the SE module has a great improvement on the DenseNet model, so if your device has enough computational resources, you should be able to get good results from the DenseNet model using the SE mechanism by adjusting the parameters, and if your device doesn't have enough computational resources, then I would recommend you to use the ECA attention mechanism.