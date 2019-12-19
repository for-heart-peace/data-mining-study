import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


valid_row_data_path = 'C:\\sna_test_author_raw.json'
valid_pub_data_path = 'C:\\test_pub_sna.json'

# 合并数据
validate_pub_data = json.load(open(valid_pub_data_path, 'r', encoding='utf-8'))
validate_data = json.load(open(valid_row_data_path, 'r', encoding='utf-8'))
merge_data = {}
for author in validate_data: 
    validate_data[author] = [validate_pub_data[paper_id] for paper_id in validate_data[author]] 

# 验证集数据分析
authors = validate_data.keys()
papers_perauthor = [len(validate_data[author]) for author in validate_data]
print('同名作者数量：', len(authors))
print('涉及的论文数：', np.sum(papers_perauthor))
print('平均论文数量：', np.mean(papers_perauthor))
print('提供的论文数：',len(validate_pub_data))

# 绘制同名作者论文数量
#plt.figure(figsize=(20, 8), dpi=80)
#x = range(len(authors))

#plt.bar(x, papers_perauthor, width=0.8)
#plt.xticks(x, authors)
#plt.xticks(rotation=270) 
#plt.xlabel('测试集同名作者')
#plt.ylabel('测试集论文数量（篇）')
#for xl, yl in zip(x, papers_perauthor):
#    plt.text(xl, yl+0.3, str(yl), ha='center', va='bottom', fontsize=10.5) 
    
#plt.gca().hlines(np.mean(papers_perauthor),-1,50,linestyles='--',colors='red',label='平均值')
#plt.annotate(u"平均值", xy = (0, np.mean(papers_perauthor)), xytext = (0, 1400),arrowprops=dict(facecolor='red',shrink=0.1,width=2))

#plt.show()



# 查看某个同名作者论文情况
# 绘制该名称下论文数据情况
#fig = plt.figure(figsize=(20, 20), dpi=80)

#ax1 = fig.add_subplot(2,2,1)
#x = range(5)
#y = [len(papers), len(venue_dict), len(year_dict), len(keywords_dict), len(org_dict)]
#s = ['涉及论文数量', '涉及期刊数量', '涉及年份数量', '涉及关键字数量', '涉及机构数量']

#plt.bar(x, y, width=0.5)
#plt.xticks(x, s, rotation=270)  
#plt.xlabel('%s论文数据情况' % author)
#plt.ylabel('数量（个）')
#for xl, yl in zip(x, y):
#    plt.text(xl, yl+0.3, str(yl), ha='center', va='bottom', fontsize=10.5) 

#ax2 = fig.add_subplot(2,2,2)
#plt.bar(range(len(venue_dict)), venue_dict.values(), width=0.3)
#plt.xlabel('%s期刊数据情况' % author)
#plt.ylabel('数量（个）')

#ax3 = fig.add_subplot(2,2,3)
#plt.bar(range(len(year_dict)), year_dict.values(), width=0.5)
#plt.xticks(range(len(year_dict)), year_dict.keys(), rotation=270) 
#plt.xlabel('%s年份数据情况' % author)
#plt.ylabel('数量（个）')

#ax4 = fig.add_subplot(2,2,4)
#plt.bar(range(len(org_dict)), org_dict.values(), width=0.5)  
#plt.xlabel('%s机构数据情况' % author)
#plt.ylabel('数量（个）')
#plt.show()

# 查看论文作者名中是否包含消歧作者名

'''
作者名存在不一致的情况：
1、大小写
2、姓、名顺序不一致
3、下划线、横线
4、简写与不简写
5、姓名有三个字的表达: 名字是否分开
同理：机构的表达也存在不一致的情况
因此：需要对数据做相应的预处理统一表达
'''

import re
# 数据预处理

# 预处理名字
def precessname(name):   
    name = name.lower().replace(' ', '_')
    name = name.replace('.', '_')
    name = name.replace('-', '')
    name = re.sub(r"_{2,}", "_", name) 
   # name = name.replace('.', '_')
    return name

# 预处理机构,简写替换，
def preprocessorg(org):
    if org != "":
        org = org.replace('Sch.', 'School')
        org = org.replace('Dept.', 'Department')
        org = org.replace('Coll.', 'College')
        org = org.replace('Inst.', 'Institute')
        org = org.replace('Univ.', 'University')
        org = org.replace('Lab ', 'Laboratory ')
        org = org.replace('Lab.', 'Laboratory')
        org = org.replace('Natl.', 'National')
        org = org.replace('Comp.', 'Computer')
        org = org.replace('Sci.', 'Science')
        org = org.replace('Tech.', 'Technology')
        org = org.replace('Technol.', 'Technology')
        org = org.replace('Elec.', 'Electronic')
        org = org.replace('Engr.', 'Engineering')
        org = org.replace('Aca.', 'Academy')
        org = org.replace('Syst.', 'Systems')
        org = org.replace('Eng.', 'Engineering')
        org = org.replace('Res.', 'Research')
        org = org.replace('Appl.', 'Applied')
        org = org.replace('Chem.', 'Chemistry')
        org = org.replace('Prep.', 'Petrochemical')
        org = org.replace('Phys.', 'Physics')
        org = org.replace('Phys.', 'Physics')
        org = org.replace('Mech.', 'Mechanics')
        org = org.replace('Mat.', 'Material')
        org = org.replace('Cent.', 'Center')
        org = org.replace('Ctr.', 'Center')
        org = org.replace('Behav.', 'Behavior')
        org = org.replace('Atom.', 'Atomic')
        org = org.split(';')[0]  # 多个机构只取第一个
    return org

#正则去标点
def etl(content):
    content = re.sub("[\s+\.\!\/,;$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+", " ", content)
    content = re.sub(r" {2,}", " ", content)
    return content

def get_org(co_authors, author_name):
    for au in co_authors:
        name = precessname(au['name'])
        name = name.split('_')
        if ('_'.join(name) == author_name or '_'.join(name[::-1]) == author_name) and 'org' in au:
            return au['org']
    return ''


# 3. 无监督聚类（根据合作者和机构TFIDF进行相似度聚类） 线上得分：0.2637
import random
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
def findfa(i,fa):
    while(i!=fa[i]):
        i=fa[i]
    return i
def same(i,fa,j):
    while(i!=fa[i]):
        temp=i;
        i=fa[i]
        fa[temp]=j
    fa[i]=j
    
def disambiguate_by_cluster():
    res_dict = {}
    db=["heng_li"
"fang_chen"
"lin_zhou"
"y_luo"
"zhigang_chen"
"y_guo"
"g_li"
"jing_huang"
"atsushi_takeda"
"fei_gao"
"qi_li"
"bin_ren"
"yong_cao"
"jing_zhou"
"hong_jiang"
"ming_xu"
"hao_zhang"
"jue_wang"
"y_shi"
"ping_zhang"
"feng_zhou"
"dong_zhang"
"li_huang"
"shiyong_liu"
"min_yang"
"jie_zhou"
"h_y_wang"
"qing_li"
"d_wang"
"hongsheng_chen"
"hong_guo"
"gang_chen"
"yuanyuan_zhang"
"lin_ma"
"yao_zhang"
]
    hhhhh=0
    for author in validate_data:
        hhhhh=hhhhh+1
        print(author)
        coauther_orgs = []
        papers = validate_data[author]
        if len(papers) == 0:
            res_dict[author] = []
            continue
        print(len(papers))
        paper_dict = {}
        flag=0
        for db1 in db:
            if db1==author:
                flag=1
        if flag!=1:

           fa=[]
           for i in range(len(papers)-1):
               fa.append(i)
           hhh=0
           for i in range(len(papers)-1):
               for j in  range(i+1,len(papers)-1):
                     paper=papers[i]
                     paperj=papers[j]
                     if(len(paper['authors'])>50):
                         authors = random.sample(paper['authors'],50)
                         if hhh==0:
                             print(len(paper['authors']))
                         hhh=1
                     else:
                         authors = paper['authors'] 
                     
                     names = [precessname(paper_author['name']) for paper_author in authors]
                     orgs = [preprocessorg(paper_author['org']) for paper_author in authors if 'org' in paper_author]  
           
                     if(len(paperj['authors'])>50):
                         authorsj = random.sample(paperj['authors'],50)
                     else:
                         authorsj = paperj['authors']  
           
                     namesj = [precessname(paper_authorj['name']) for paper_authorj in authorsj]
                     orgsj = [preprocessorg(paper_authorj['org']) for paper_authorj in authorsj if 'org' in paper_authorj]  
           
                     keywords = paper["keywords"] if 'keywords' in paper else ''
                     venue = paper["venue"] if 'venue' in paper else ''
            
                     keywordsj = paperj["keywords"] if 'keywords' in paperj else ''
                     venuej = paperj["venue"] if 'venue' in paperj else ''
                     #print(venue)
                     #abstract = paper["abstract"] if 'abstract' in paper else ''
                     #coauther_orgs.append(etl(' '.join(names + orgs) + ' '+ abstract))   
                     #coauther_orgs.append(etl(' '.join(names + orgs)  +' '+venue)) 
                     cnt=0
                     if(venuej==venue and venue!=''):
                         cnt=cnt+6
                     flag=0
                     for keyw in keywords:
                         for keyj in keywordsj:
                             if keyw==keyj  and keyj!='':
                                 flag=flag+1     
                                 break
                     if cnt==6:
                         flag=0
                     if(flag>=2):
                        cnt=cnt+6
                     else:
                        cnt=cnt+flag*3
                     flag=0
                     flag1=0
                     for i1 in range(len(names)-1):
                         if(flag>2):
                             break
                         for j1 in range(len(namesj)-1):
                             if names[i1]==namesj[j1] and names[i1]!=author:
                                 flag=flag+1
                                 break
                     for i1 in range(len(orgs)-1):
                          if(flag1>2):
                              break
                          for j1 in range(len(orgsj)-1):
                              if orgs[i1]==orgsj[j1] and orgs[i1]!='':
                                  flag1=flag1+1
                                  break
                     
                     if flag==1:
                         cnt=cnt+4
                     if flag==2:
                         cnt=cnt+7
                     if flag==3:
                         cnt=cnt+10
                     #flag1=max(flag1-1,0)
                     if(flag1==1):
                         cnt=cnt+3
                     if(flag1==2):
                         cnt=cnt+4
                     if(flag1>=3):
                         cnt=cnt+7
                     RNG=8
                    
                     if(cnt>=RNG):
                        i1=findfa(i,fa)
                        #j1=findfa(j,fa)
                        #fa[i1]=j1
                        #fa[i]=j1
                        same(j,fa,i1)
           for i in  range(len(papers)-1):
               i1=findfa(i,fa)
               if str(i1) not in paper_dict:
                   paper_dict[str(i1)] = [papers[i]['id']]
               else:
                   paper_dict[str(i1)].append(papers[i]['id'])
           res_dict[author] = list(paper_dict.values())
          
        else :
           for paper in papers:
               authors = paper['authors'] 
               names = [precessname(paper_author['name']) for paper_author in authors]
               orgs = [preprocessorg(paper_author['org']) for paper_author in authors if 'org' in paper_author]  
               venue = paper["venue"] if 'venue' in paper else ''
               keywords = paper["keywords"] if 'keywords' in paper else ''
               coauther_orgs.append(etl( venue+' '+' '.join(keywords)+' '+abstract))          
           tfidf = TfidfVectorizer().fit_transform(coauther_orgs)

           # sim_mertric = pairwise_distances(tfidf, metric='cosine')
           
           clf = DBSCAN(metric='cosine')
           s = clf.fit_predict(tfidf)
           #每个样本所属的簇 
           for label, paper in zip(clf.labels_, papers):
               if str(label) not in paper_dict:
                   paper_dict[str(label)] = [paper['id']]
               else:
                   paper_dict[str(label)].append(paper['id'])
           for i in range(len(paper_dict)):
               fa[i]=i;
           
           res_dict[author] = list(paper_dict1.values())
           
    json.dump(res_dict, open('C:\\2.json', 'w', encoding='utf-8'), indent=4)

disambiguate_by_cluster()
