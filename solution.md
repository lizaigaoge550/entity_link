PAD O B I E S START END
0   1 2 3 4 5 6      7
* (0,0) : pad->pad
* (1,2) : O->B
* (1,5) : O->S
* (1,7) : O->END
* (1,0) : O->pad
* (2,3) : B->I
* (2,4) : B->E
* (3,3) : I->I
* (3,4) : I->E
* (4,1) : E->O
* (4,5) : E->S
* (4,7) : E->END
* (4,0) : E->pad
* (5,2) : S->B
* (5,1) : S->O
* (5,5) : S->S
* (5,7) : S->END
* (5,0) : S->pad
* (7,0) : END->pad

#任务
* detect mention
* linking entity


#Detect Mention
* 提取kb中的subject, object 匹配text中的mention

* mention 的entity cands:
    * 首先拆分text为n-gram, 得到一些列的wordlist
    * 然后用wordlist与kb中subject匹配
        * LCS的F值， 排序得到候选集。
        * 去标点，大小写
        
* 对于不在kb中的mention, 采用lstm-crf

* 去除S标签，在kb中查找S标记，直接在原文中打标

* 去除文章尾部标点符号


#Entity Linking
* 先选出候选集，规则：Rouge score
* entity: 
    * 摘要，当做context
    * Typing, 可以当做额外的特征
    * 关系： 
