# DeepFM

[推荐系统遇上深度学习(三)--DeepFM模型理论和实践](https://www.jianshu.com/p/6f1c2643d31b)

动机：低阶组合特征或者高阶组合特征很重要。FM因为计算复杂度的原因一般都只用到了二阶特征组合；**对于高阶的特征组合可以通过多层的神经网络即DNN去解决**。

实现：

**把低阶特征组合单独建模，然后融合高阶特征组合**。DeepFM包含两部分：**神经网络部分与因子分解机部分**，分别负责低阶特征的提取和高阶特征的提取，这两部分共享同样的输入。

如果某一个离散特征有多个取值，也是要将其变换成定长的embedding。可以采用embedding\_lookup\_sparse函数提供的方法。
[推荐系统遇上深度学习(四)--多值离散特征的embedding解决方案](https://www.jianshu.com/p/4a7525c018b2)

代码：https://github.com/princewen/tensorflow_practice/tree/master/recommendation/Basic-DeepFM-model https://github.com/ChenglongChen/tensorflow-DeepFM 
