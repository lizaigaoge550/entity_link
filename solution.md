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
