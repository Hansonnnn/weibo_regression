## 新浪微博互动预测-挑战Baseline

https://tianchi.aliyun.com/competition/entrance/231574/information
该仓库用来记录我参加新浪预测比赛的思路及代码过程。

### Step 1 数据预处理
- 空值处理
- 测试集和训练集划分
- 数据探索

### Step 2 特征工程
拿到这个题目首先想到的是先从现有数据中抽象出特征，然后再用回归算法去预测结果。根据题目意思是需要根据用户及其原创微博在发表一天后的转发、评论、赞总数来预测用户后期的博文发表一天后的转发、评论、赞总数。抛析这句话就可以得知用户维度是非常重要的一个类型的特征。所以最后在特征工程方面，我分别以三种类型抽象特征，分别是时间特征维度，用户特征维度（通过每一用户发表博文的评论、点赞、转发数的特征抽象）以及博文特征维度。下面我将具体介绍抽象这三个维度的特征抽象的过程及部分代码。

#### 1.时间特征
对于博文发表时间，提取出以下几种特征。
- 转化为时间类型（datatime）
```
dataframe['time'] = dataframe['time'].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'))

```
- 发文时间段（按小时，24个时间段）
```
dataframe['hour_seg'] = dataframe['time'].dt.hour
```
- 发文时间是属于周几
```
dataframe['weekday'] = dataframe['time'].dt.weekday
```
- 发文时间是否是周末

```
dataframe['isWeekend'] = dataframe['weekday'].apply(lambda x: 1 if x == 5 or x == 6 else 0)

```
- 发文时间是否是节假日,博文数据来自2015-02-01至2015-07-31
```
dr = pd.date_range(start='2015-02-01', end='2015-07-31')
cal = calendar()
holidays = cal.holidays(start=dr.min(), end=dr.max())
dataframe['Holiday'] = dataframe['time'].dt.date.apply(lambda x: 1 if x in holidays else 0)
```
#### 2.用户特征

- 最大评论、点赞、转发数
```
dataframe.groupby(['uid'])[['forward_count']].max()
dataframe.groupby(['uid'])[['comment_count']].max()
dataframe.groupby(['uid'])[['like_count']].max()
```
- 最小评论、点赞、转发数
```
dataframe.groupby(['uid'])[['forward_count']].min()
dataframe.groupby(['uid'])[['comment_count']].min()
dataframe.groupby(['uid'])[['like_count']].min()
```
- 平均评论、点赞、转发数
```
dataframe.groupby(['uid'])[['forward_count']].mean()
dataframe.groupby(['uid'])[['comment_count']].mean()
dataframe.groupby(['uid'])[['like_count']].mean()
```
- 最大评论/点赞率、转发/点赞率
```
result = pd.concat(
[max_comment_count, max_forward_count, max_like_count, min_comment_count, min_forward_count, min_like_count,
mean_comment_count, mean_forward_count, mean_like_count], axis=1, join_axes=[max_comment_count.index],
)
```

```
result['max_comment_count'].div(result['max_like_count'])
result['max_forward_count'].div(result['max_like_count'])
```
- 最小评论/点赞率、转发/点赞率
```
result['min_comment_count'].div(result['min_like_count'])
result['min_forward_count'].div(result['min_like_count'])
```
- 平均评论/点赞率、转发/点赞率
```
result['mean_comment_count'].div(result['mean_like_count'])
result['mean_forward_count'].div(result['mean_like_count'])
```

#### 3. 博文特征
- 博文是否存在主题‘##’
```
topic_pattern = re.compile(r'^#.{1,}#')
dataframe['content'].apply(lambda x: 1 if topic_pattern.findall(x) else 0)

```
- 博文是否存在‘@’
```
dataframe['content'].apply(lambda x: 1 if '@' in x else 0)
```
- 博文是否存在引用或转发，‘【】’
```
re.compile('【.{1,}】')
dataframe['if_reference'] = dataframe['content'].apply(lambda x: 1 if reference_pattern.findall(x) else 0)
```
- 博文是否存在网页url
```
re.compile('[a-zA-z]+://[^\s]*')
dataframe['if_url'] = dataframe['content'].apply(lambda x: 1 if url_pattern.findall(x) else 0)
```
- 博文长度
```
dataframe['content'].apply(lambda x: len(x))
```
