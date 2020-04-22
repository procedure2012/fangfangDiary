# FangfangDiary
I used a few simple text mining techniques to analyses Fangfang' Diary. The code is very ugly but it successfully finished its task! (ﾟ∀ﾟ)

Here, I only give a simple result. For details, please check my blog[]().

## Summarazation

I abstracted the summarization of the whole diary and every single day using gensim. If you don't want to read the whole diary, you can find the summarization [here](https://github.com/procedure2012/fangfangDiary/blob/master/data/fangfangAbstract.txt).
## LDA Topics

I also used gensim to construct an LDA model and calculated the topic distribution on all media and Fangfang's diary.

|Topic1|Topic2|Topic3|Topic4|Topic5|Topic6|Topic7|
|--|--|--|--|--|--|--|
|传播      |特朗普    |医院|小区|公司|防控|复工|
|流感      |媒体      |方舱医院|宿舍|经济|物资|复产|
|疾病      |美国      |志愿者|村|工厂|捐赠|亿元|
|研究      |台湾      |去|意大利|下跌|人民|发行|
|传人      |政治      |患者|确诊|美元|总书记|无|
|动物      |文章      |收治|日本|汽车|抗击|码|
|专家      |批评      |病人|号|投资者|打赢|有序|
|症状      |领导人    |社区|乘客|指数|群众|通告|
|冠状病毒   |中共     |护士|伊朗|亿美元|阻击战|指挥部|
|传染      |中国政府  |床位|邮轮|增长|湖北|开学|
|李文亮     |危机     |医疗队|死亡|上涨|党中央|券商|
|科学家     |西方     |妈妈|韩国|月份|保卫战|防控|
|传染病     |总统     |吃|该国|影响|万元|温州|
|华南海鲜市场|指责     |发热|入境|美联储|保障|万亿元|
|感染者     |评论     |电话|周日|预期|捐款|证监会|
|实验室     |驱逐     |医护人员|钻石公主号|生产|工作|重点|
|疫苗       |外国     |院长 |穿上|市场|一线|加快|
|死亡率     |世界     |孩子    |东京|供应链|指导组|中证报|
|医生       |政府     |母亲    |新增|经济学家|众志成城|债|
|新型冠状病毒|中国外交部|家里   |花园|国债|慈善|融资|
|蝙蝠       |习近平    |门诊   |病毒检测|跌幅|干部|证券时报|
|野生动物   |言论      |记者   |周六|收益率|工作者|申报|
|海鲜       |写道      |病区   |阅读|收入|支援|监狱|
|传染性     |人权      |父母   |大邱|制造业|胜|上证|
|肺炎       |权力      |病房   |累计|分析师|同舟共济|新增|
|确认       |外交      |张     |周四|产品|部署|管理|
|检测       |体制      |穿     |奥运会|关税|力量|绿码|
|流行病学   |民主      |定点医院|回国|涨幅|考察|企业|
|公共卫生   |反对      |同事    |周三|股市|胜则|贷款|
|戴        |中国病毒   |没      |街|放缓|基金会|降准|

Here is the distribution on all media and Fangfang's Diary.

||topic1|topic2|topic3|topic4|topic5|topic6|topic7|
|--|--|--|--|--|--|--|--|
|人民网|14.0%| 3.0%| 27.4%| **0.2%**| 17.4%| **36.5%**| 1.5%|
|新华网|6.2%| 3.7%| 27.5%| **1.6%**| 17.8%| **37.9%**| 5.2%|
|环球网|8.1%| 12.6%| **24.7%**| **1.6%**| 19.8%| 21.0%| 12.0%|
|观察者网|4.4%| 13.5%| 20.2%| **1.3%**| **27.3%**| 23.9%| 9.1%|
|文汇网|2.4%| 16.1%| **43.1%**| **0.9%**| 11.1%| 18.7%| 7.7%|
|中国日报|2.1%| 4.7%| **40.1%**| **1.4%**| 10.6%| 36.0%| 5.0%|
|卫星通讯社|**35.2%**| 11.0%| **2.2%**| 11.4%| 8.0%| 11.1%| 20.2%|
|BBC|16.8%| 23.6%| 6.2%| 11.5%| 4.3%| **2.4%**| **35.1%**|
|德国之声|19.9%| 29.5%| 5.8%| 10.7%| 2.2%| **1.6%**| **30.1%**|
|华尔街日报|14.3%| 7.0%| 1.4%| **61.6%**| 9.6%| **0.7%**| 5.2%|
|纽约时报|**23.2%**| 22.3%| 8.6%| 9.0%| **0.5%**| 1.1%| 35.4%|
|**方方日记**|**0.06%**| 11.7%| **69.4%**| 0.3%| 2.1%| 0.4%| 15.9%|

## Doc2Vec

To find the similar documents. I used dov2vec. The following is the result.

|相似|文本|
|:--:|:--:|
|[德国之声 武汉日记：长歌当哭](https://www.dw.com/zh/%E6%AD%A6%E6%B1%89%E6%97%A5%E8%AE%B0%E9%95%BF%E6%AD%8C%E5%BD%93%E5%93%AD/a-52431933)|[文汇网 专访六六：我拒绝的，和我想写的](http://wenhui.whb.cn/zhuzhan/xinwen/20200309/331724.html)|
|[纽约时报 方方的武汉日记和一场政治风暴](https://cn.nytimes.com/china/20200415/coronavirus-china-fang-fang-author/)|[新华网 武汉民警一线抗疫日记：为他们平安，我们愿逆行而上](http://www.xinhuanet.com/politics/2020-01/31/c_1125515295.htm)|
|[环球网 疫情下的武汉人：珍惜“梗朋友”，不爱“阴倒搞”](https://society.huanqiu.com/article/9CaKrnKpON5)|[德国之声 长平观察：李文亮微博—“中国哭墙”下的抗议](https://www.dw.com/zh/%E9%95%BF%E5%B9%B3%E8%A7%82%E5%AF%9F%E6%9D%8E%E6%96%87%E4%BA%AE%E5%BE%AE%E5%8D%9A%E4%B8%AD%E5%9B%BD%E5%93%AD%E5%A2%99%E4%B8%8B%E7%9A%84%E6%8A%97%E8%AE%AE/a-52862102)|
|[德国之声 还要有多少李文亮才会让哨声嘹亮？](https://www.dw.com/zh/%E8%BF%98%E8%A6%81%E6%9C%89%E5%A4%9A%E5%B0%91%E6%9D%8E%E6%96%87%E4%BA%AE%E6%89%8D%E4%BC%9A%E8%AE%A9%E5%93%A8%E5%A3%B0%E5%98%B9%E4%BA%AE/a-52295308)|[环球网 原创条漫“自述”武汉 宛若“人在画中游”](https://cul.huanqiu.com/article/3xHQy52DrG1)|
|[德国之声 武汉日记：元宵节](https://www.dw.com/zh/%E6%AD%A6%E6%B1%89%E6%97%A5%E8%AE%B0%E5%85%83%E5%AE%B5%E8%8A%82/a-52311158)|[德国之声 武汉日记：度日如年](https://www.dw.com/zh/%E6%AD%A6%E6%B1%89%E6%97%A5%E8%AE%B0%E5%BA%A6%E6%97%A5%E5%A6%82%E5%B9%B4/a-52498004)|
|[纽约时报 “吹哨者”李文亮之死引众怒，中国网民发起反抗](https://cn.nytimes.com/china/20200208/china-coronavirus-doctor-death/)|[德国之声 暴力防疫引众怒 网友：回到红卫兵时代](https://www.dw.com/zh/%E6%9A%B4%E5%8A%9B%E9%98%B2%E7%96%AB%E5%BC%95%E4%BC%97%E6%80%92-%E7%BD%91%E5%8F%8B%E5%9B%9E%E5%88%B0%E7%BA%A2%E5%8D%AB%E5%85%B5%E6%97%B6%E4%BB%A3/a-52409071)|
