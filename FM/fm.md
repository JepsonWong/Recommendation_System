# FM

[推荐系统召回四模型之：全能的FM模型](https://zhuanlan.zhihu.com/p/58160982)

* MF和FM的联系：MF即协同过滤；MF模型是FM模型的特例；FM模型可以看作是MF模型的进一步拓展；FM继承了MF的特征embedding化表达这个优点，同时引入了更多Side information作为特征，将更多特征及Side information embedding化融入FM模型中；
* 计算效率优化：线性时间复杂度；化简之后发现，二阶特征组合等价于将FM的所有特征项的embedding向量累加，之后求内积；
* 如何利用FM模型做统一的召回模型
* FM模型能否将召回和排序阶段一体化
* 优秀的开源实现：https://github.com/Angel-ML/angel

