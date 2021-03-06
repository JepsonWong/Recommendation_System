# Recommendation\_System

**个性化推荐**

[（好文，没看完）《搜索与推荐中的深度学习匹配》之推荐篇](https://zhuanlan.zhihu.com/p/45849695)

[今日头条推荐算法原理全文详解](http://lusongsong.com/info/post/9829.html)

[国内外在推荐系统领域的发展现状？](https://www.zhihu.com/question/29531839/answer/554371884)

[微信“看一看”推荐的框架设计](https://zhuanlan.zhihu.com/p/111342449)

## 1. 基础工作

[用户画像阶段性总结](http://rongzijing.win/index.php/archives/182/)

评价指标：Precision，Recall，F1-score，AUC，Logloss，Normalized Logloss，Calicration。

特征处理：[推荐系统算法FM、FFM使用时，连续性特征，是直接作为输入，还是经过离散化后one-hot处理呢？](https://www.zhihu.com/question/328925143/answer/714773812) [连续特征的离散化：在什么情况下将连续的特征离散化之后可以获得更好的效果？](https://www.zhihu.com/question/31989952/answer/54184582) [特征组合&特征交叉 (Feature Crosses)](https://segmentfault.com/a/1190000014799038) [百面机器学习3---组合特征与选择](https://blog.csdn.net/qq_20880939/article/details/98729327)

* 连续性特征：离散化后one-hot处理（树模型、特征hash分桶）
* 离散性特征：one-hot处理

离散特征的好处：

* 增加模型建模能力：离散特征通常比连续特征能表征更多复杂信息；
* 减少特征异常造成的模型鲁棒性风险；
* 减少算法框架实现的复杂度，1001:0.1 这类特征可以离散化表示为100001，而去掉：0.1，其中算wx的时候，也可以变为求和，而不是乘之后求和，大大减少样本存储以及网络传输开销，也减少了计算复杂度；
* 离散化后通常特征处理的逻辑会比较简单，特征拼接、分桶加上交叉，适合标准化，来完成线上的特征组装，达到online learning的需求；

特征存储方式：libsvm [LibSVM格式简介](https://www.cnblogs.com/luozeng/p/11028029.html) [libsvm使用的数据格式](https://www.cnblogs.com/petali/archive/2008/11/06/1328454.html)

## 2. 召回

[Embedding在深度推荐系统中的3大应用方向](https://zhuanlan.zhihu.com/p/67218758)

* 在推荐系统中，如果面对新的Item，能否在不重新训练Embedding的前提下得到这个新Item的Embedding向量？大家有什么实践经验分享吗？

### 2.1 向量化：Item2vec、User2Vec召回算法

Item2vec
* 把item视为word，用户的行为序列视为一个集合，item间的共现为正样本，并按照item的频率分布进行负样本采样。
* 缺点是相似度的计算还只是利用到了item共现信息，（1）忽略了user行为序列信息；（2）没有建模用户对不同item的喜欢程度高低。

[万物皆Embedding，从经典的word2vec到深度学习基本操作item2vec](https://zhuanlan.zhihu.com/p/53194407)

[DNN论文分享 - Item2vec: Neural Item Embedding for Collaborative Filtering](https://zhuanlan.zhihu.com/p/24339183)

[word2vec doc2vec paragraph2vec topic2vec prodct2vec——paper 笔记](https://blog.csdn.net/wang2008start/article/details/78775289?utm_source=blogxgwz1)

[E-commerce in Your Inbox:Product Recommendations at Scale-----产品推荐（prod2vec和user2vec)](http://www.cnblogs.com/Lee-yl/p/9833279.html)

[User2Vec? representing a user based on the docs they consume](https://stackoverflow.com/questions/46426380/user2vec-representing-a-user-based-on-the-docs-they-consume)

### 2.2 基于人口统计学的推荐（基于用户）

根据用户年龄、性别等人口统计学来推荐。 [推荐系统简介](https://zhuanlan.zhihu.com/p/146359940)

### 2.3 基于协同过滤的推荐算法

[推荐算法之协同过滤算法(Collaborative Filtering，简称CF)](http://rongzijing.win/index.php/archives/40/)

协同过滤：完全根据历史行为（比如购买行为或者评分行为）来定义用户和商品之间的相关性，再以这个相关性来给用户相应的推荐。算法的三个假设：
* 相似的用户会喜欢相似的商品
* 用户之间的相似度可以用商品来衡量
* 商品的相似度可以通过哪些用户购买来衡量


1. 基于内存的算法（Memory-based approach）
* 基于用户的协同过滤算法（User-based，CF）：利用用户之间的相似性来推荐用户感兴趣的信息。一般适用于新闻推荐这种物品变化很快的场景。
  * 存在稀疏性的问题，即在系统使用初期由于系统资源还未获足够多的评价，很难利用这些评价来发现相似的用户。
  * 存在可扩展性的问题，即随着系统用户和资源的增多，系统的性能会越来越差。
* 基于物品的协同过滤算法（Item-based，CF）：利用item之间的相似性来推荐和用户之前浏览过的item相似的其他item。一般适用于电商这种物品较为固定的场景。

2. 基于模型的算法（Model-based approach）
* 隐语义模型（latent semantic models，LFM），矩阵分解（Matrix Factorization，MF）技术是实现隐语义模型使用最广泛的一种方法。**基于矩阵分解的推荐算法的核心假设是用隐语义（隐变量）来表达用户和物品，他们的乘积关系就成为了原始的元素。** [机器学习(5) 推荐 矩阵分解(Matrix Factorization)](https://blog.csdn.net/GZHermit/article/details/73920755) [推荐算法中的MF, PMF, BPMF](https://blog.csdn.net/shulixu/article/details/75349138) [矩阵分解(MF,SVD)和协同过滤(CF)](https://blog.csdn.net/u012151283/article/details/77716085) [推荐算法之矩阵分解](https://www.biaodianfu.com/matrix-factorization.html) （隐语义模型通过机器学习方法从用户-物品评分矩阵中分解为两个低阶矩阵，表示对用户兴趣和物品的隐含分类特征，通过隐含特征预测用户评分） **矩阵分解算法如何训练？**

3. 混合模型（Hybrid models）

### 2.4 基于内容过滤的推荐算法

根据信息资源与用户兴趣的相似性来推荐商品，通过计算用户兴趣模型和商品特征向量之间的向量相似性，主动将相似度高的商品发送给该模型的客户。

### 2.5 扩展推荐

基于用户兴趣点、内容类别等扩展。

### 2.6 基于新热的推荐算法

将当前的热门item进行推荐。可用于冷启动的策略之一和召回的策略之一。

一种简单方法：
* 统计出指定时间窗口的商品的曝光量、点击量，从而计算出曝光点击比，即CTR，根据CTR大小作为热门程度的量化指标；当然也可以参考其它指标，如下载展示比、播放展示比等等。
* 可以从所有商品中，按照CTR排序，选出top k个，作为召回结果。
* 可以根据商品的标签信息，统计出该标签下所有商品，并按照CTR排序，得出召回结果。

### 2.7 基于环境特征的推荐

基于地域、时间、场景等推荐。

### 2.8 社会化推荐

根据用户的社会关系，推荐朋友的朋友中与用户“志同道合”的朋友给用户，或者将与用户“志同道合”的朋友们感兴趣的项目推荐给用户，这种推荐算法主要应用于像QQ、微信和Facebook之类的社交软件中。

## 3. 排序

涉及一个item、item对、item序列的打分任务；涉及LR、GBDT、RNN、DNN模型。

[Learning to Rank简介（没看）](https://www.cnblogs.com/bentuwuying/p/6681943.html)

[谷歌、阿里、微软等10大深度学习CTR模型最全演化图谱【推荐、广告、搜索领域】](https://zhuanlan.zhihu.com/p/63186101)

[前深度学习时代CTR预估模型的演化之路](https://zhuanlan.zhihu.com/p/61154299)

[从FM推演各深度CTR预估模型(附代码)](https://zhuanlan.zhihu.com/p/109131795)

### 3.1 point-wise(single)

* 协同过滤（Collaborative filtering）
* 基于内容的推荐（Content based）
* 混合方法（Hybird method）
* GBDT+LR [腾讯大数据：CTR预估中GBDT与LR融合方案](http://www.cbdio.com/BigData/2015-08/27/content_3750170.htm) [Practical Lessons from Predicting Clicks on Ads at Facebook](http://quinonero.net/Publications/predicting-clicks-facebook.pdf)
* DNN+GBDT [Model Ensemble for Click Prediction in Bing Search Ads 2017](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/04/main-1.pdf)
* DNN-based：Google的wide and deep模型、deepfm模型、xDeepFM模型、youtube的视频推荐系统（特征层面的创新）

  * FM系列：[FM系列算法解读（FM+FFM+DeepFM）](https://blog.csdn.net/hiwallace/article/details/81333604)
  * FNN：FM初始化Embedding，然后接多层前馈神经网络。
  * DCN：**显示学习高阶特征交互**。
  * xDeepFM：FNN、Deep Crossing、Wide and Deep中的Deep部分使用前馈神经网络来建模高阶交互（这种交互是建立在bit-wise上的，也就是说同一个隐藏层中的元素也会相互影响）。PNN、DeepFM会考虑bit-wise和vector-wise上的交互。本文提出了一种**显示的、vector-wise的高阶交互学习方法，可以有效捕获有限阶数**。 [xDeepFM介绍](http://d0evi1.com/xdeepfm/)

参考论文：

* Item-Based Collaborative Filtering Recommendation Algorithms
* Matrix factorization techniques for recommender systems
* Wide & Deep Learning for Recommender Systems
* Deep Neural Networks for YouTube Recommendations

缺点是完全从单文档的分类角度计算，没有考虑文档之间的相对顺序。

比较代表性的有LR、XGBoost。

### 3.2 pair-wise(pair)

从排序序列中抽出“数据对”，每个数据对都有代表它们相对关系的标签。然后我们使用标签数据来训练一个分类模型，再用分类模型做排序。

比较代表性的有RankSVM、LambdaMART。

[在个性化推荐问题中， GBRank和RankingSVM哪个效果更好一些？](https://www.zhihu.com/question/29271787)

[learning to rank 算法总结之pairwise](https://www.jianshu.com/p/235756fbf6b6)

* Sortnet
* RankNet

缺点：

* 当有多个顺序的类别时，只知道good or bad，**不知道好多少和差多少**。
* 当文章较多时，**会产生大量的pairs**。
* 对噪声的label 非常敏感，一个错误的label，**可能会导致分类结果中有较多的错误**。
* 大多数没有考虑结果中文档对的顺序，文档对的顺序也没有在损失函数中体现。

### 3.3 list-wise(list)

不同于pair-wise方法，list-wise方法将整个结果作为一个训练实例。

例如将整个推荐列表/用户短期行为通过RNN来进行建模，这种做法的好处是不用显式的构造session级特征，能够捕获用户的**短时兴趣**对当前排序结果的影响。

参考论文：

* Improved Recurrent Neural Networks for Session-based Recommendations
* Personalizing Session-based Recommendations with Hierarchical Recurrent Neural Networks

比较代表性的有ListNet。

损失函数：

* 和评价指标相关：
* 和评价指标不相关：

### 3.4 一些问题以及解决方案

一些问题：

* 多样性问题：在top-k的推荐系统中非常重要。
* item间的关联性：推荐的k个item之间不应该相互独立应该是相互关联的。比如某些相关的结果同时出现，会显著提升CTR。同时将高CTR的item放在list的前面，在一些情况下不如将高CTR的item分散到list的不同位置中。
* position bias：信息流推荐的position bias与搜索引擎的position bias有着本质区别，cascade click model等模型的假设在信息流推荐系统中很难满足。
* 全局最优的问题：一个一个挑选item组成list并不能保证全局最优；他只能考虑到上文信息，考虑不到下文信息。
* 实时性问题：[天下武功，唯快不破，论推荐系统的「 实时性」](https://zhuanlan.zhihu.com/p/74813776) [如何增强推荐系统模型更新的「实时性」？](https://zhuanlan.zhihu.com/p/75597761)

[什么是position bias（没看）](https://www.google.com.hk/search?safe=strict&ei=tBcIXK3EOou9rQHT3KD4Bw&q=%E4%BB%80%E4%B9%88%E6%98%AFposition+bias&oq=%E4%BB%80%E4%B9%88%E6%98%AFposition+bias&gs_l=psy-ab.3...17177.19764..19910...0.0..0.199.1857.0j10......0....1..gws-wiz.......0i71.DjuOs9x7x9w)

[什么是cascade click model假设（没看）](https://www.google.com.hk/search?safe=strict&biw=1412&bih=697&ei=ZhgIXOa0J8vd9QON-oXIDw&q=%E4%BB%80%E4%B9%88%E6%98%AFcascade+click+model%E5%81%87%E8%AE%BE&oq=%E4%BB%80%E4%B9%88%E6%98%AFcascade+click+model%E5%81%87%E8%AE%BE&gs_l=psy-ab.3..33i160l3.4499.10202..10365...1.0..0.232.3490.0j12j6......0....1..gws-wiz.zBqVi2vGHn8)

[Position bias & Click Model（没看）](https://haorenhao.github.io/2016/11/26/ClickModel/)

[点击模型（没看）](https://baike.baidu.com/item/%E7%82%B9%E5%87%BB%E6%A8%A1%E5%9E%8B/13677663?fr=aladdin)

解决方案：

* 在排序模型中加入context特征（context特征就是前面位置的主题、类别、子类别等信息），使得模型在考虑CTR信息的时候也会**考虑上下文信息**。同时也加入**针对position bias的处理**。
* 两阶段的list-wise排序框架。该框架将这个排序阶段分为“list生成”和“list评估”两个阶段。从而使得我们的框架能够从list-wise的维度来考虑排序，这个过程中，综合**考虑了item之间的关联、position bias以及list全局信息**。
* list生成，set-to-sequence模型+强化学习的方法，将list评估模型用于帮助list生成模型训练，提升list生成的候选的效果。

[线下AUC与线上CTR不一致问题](http://rongzijing.win/index.php/archives/199/)

## 4. 推荐系统和知识图谱

[推荐算法不够精准？让知识图谱来解决](https://www.msra.cn/zh-cn/news/features/embedding-knowledge-graph-in-recommendation-system-i)

[如何将知识图谱特征学习应用到推荐系统？](https://www.msra.cn/zh-cn/news/features/embedding-knowledge-graph-in-recommendation-system-ii)

## ABTest

[一文搞懂AB Testing的分层分流](http://www.woshipm.com/pd/1080730.html/comment-page-1)

[谷歌是如何高效做AB实验的](https://zhuanlan.zhihu.com/p/121792914)

[Abtest那些事](https://www.sohu.com/a/149827453_650581)

[一文看懂AB测试](https://zhuanlan.zhihu.com/p/108916194)

对照/实验、域/层/桶

## 博客

[Ad-papers](http://wzhe.me/Ad-papers/)

## 参考

[美团 深度学习在文本领域的应用](https://tech.meituan.com/deep_learning_doc.html)

[关于个性化主页定制的新闻推荐算法研究](http://media.people.com.cn/n1/2016/0316/c402797-28203840.html)

[推荐系统中的点击率预估 – Advertising & Recommendation](http://itindex.net/detail/58640-%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F-%E7%82%B9%E5%87%BB%E7%8E%87-advertising)

[排序学习实践---ranknet方法](https://www.cnblogs.com/LBSer/p/4439542.html)

[pair-wise rank](http://www.doc88.com/p-6741372678601.html)

[【点击模型学习笔记】A survey on click modeling in web search 点击模型不同层次的假设](https://blog.csdn.net/xceman1997/article/details/29379369)
