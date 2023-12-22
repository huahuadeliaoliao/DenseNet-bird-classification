# DenseNet-bird-classification
# 使用DenseNet模型对200种鸟类图像进行分类
## 在测试集上的表现

|      | Desnet-121 | *Densenet-121withSE | Densenet-201withECA |
| :--- | :---: | :---: | :---: |
| accuracy | 0.79 | *0.67 | 0.85 |
| loss | 0.81 | *1.45 | 0.71 |

## 可以继续改进的方向
1. 加入更多的数据增强
2. 由于DenseNet121,169和201的k值都相同,这三个模型在当前数据集上的表现差异并不大,可以考虑使用k值更大的DenseNet模型，如DenseNet161或Wide-DenseNet-BC

## *备注
加入了SE注意力机制DenseNet模型需要大量的计算资源，由于我的时间和设备限制所以我得到的关于DenseNet121withSE的结果并不能准确表示加入了SE注意力机制的效果。在我查询的相关[论文](https://www.sciencedirect.com/science/article/abs/pii/S195903182100141X)中表示SE模块对DenseNet模型有很大的提升,所以如果你的设备有足够的计算资源，通过调整参数应该可以从使用了SE机制的DenseNet模型上得到很好的效果,如果你的设备不具备足够的计算资源，那么我推荐你使用ECA注意力机制。