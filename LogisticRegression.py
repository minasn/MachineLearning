#_*_ coding:utf-8 _*_
import re
from scipy.sparse import lil_matrix,linalg
from scipy import linalg as la
import json
import nltk
import nltk.data
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem.lancaster import LancasterStemmer#Lancaster词干器
from nltk.corpus import stopwords
import numpy as np
import copy

LancasterStem=LancasterStemmer()

def preprocess(title):
	#分割成句子
	sents=sent_tokenize(title)
	s=''
	#句子内容清理，去掉数字标点和非字母字符
	for sentence in sents:
		#sents=re.sub('\s+',' ',title.strip())
		matchlabel=re.compile(r'<[^>]>')#去标签
		sentence=matchlabel.sub(' ',sentence)
		matchxml = re.compile(r'&lt|&gt|&amp|&quot|&copy')#去转义符
		sentence=matchxml.sub(' ',sentence)
		cleanLine=re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0123456789]',' ',sentence)
		wordInStr=nltk.word_tokenize(cleanLine) #将单句字符串分割成词
		
		#去停用词和长度小于2的词
		filtered_words=[word.lower() for word in wordInStr if word.lower() not in stopwords.words('english') and len(word)>1]
		for word in filtered_words: #词干化
			s=s+LancasterStem.stem(word)+' '
	return s

def sigmod(x,thata):
	return 1.0/(1+np.exp(-1*x*thata))

for querynum in range(50):
	#对每个query进行训练
	#训练集json内容为：article_id,title,body,score
	#测试集json内容为：article_id,title,body,index,score
	with open('E:\\courses\knowledgeanalysis\\regression\\documentsdivided\\train\\%s'%(str(201+querynum)+'.json'),'r',encoding='UTF-8')as f:
		load_train=json.load(f)

	#对每个doc的title进行预处理
	wordlist={}
	for trainnum in range(len(load_train)):
		load_train[trainnum]['title']=preprocess(load_train[trainnum]['title'])
		for word in re.split('\s+',load_train[trainnum]['title'].strip()):
			if word in wordlist:
				wordlist[word]+=1
			else:
				wordlist[word]=1
	keys=list(wordlist.keys())
	for key in keys:
		if wordlist[key]<2: #删除频次少的
			del wordlist[key]
	word2int=dict((c,i) for i,c in enumerate(wordlist))
	print('word2int size:%s'%len(word2int))
	wordlist_len=len(word2int)#词表大小
	A=lil_matrix((trainnum,wordlist_len+1))#训练doc数*词表大小的稀疏矩阵
	B=[]
	for i in range(trainnum):
		for word in re.split('\s+',load_train[i]['title'].strip()):
			if word in word2int:
				A[i,word2int[word]]=1
		B.append(float(load_train[i]['score'])/100)
	B=np.matrix(B)
	B=B.T
	a=A.todense()

	#SGD
	loop=0
	alpha=0.001
	thata=np.zeros(wordlist_len+1)#thata向量初始值
	thata=np.mat(thata)
	thata=thata.T
	print(thata.shape)
	err0=np.zeros(wordlist_len+1)
	err0=np.mat(err0)
	err0=err0.T
	epsilon=1e-4
	finish=0

	while loop<10000:
		for i in range(trainnum):
			diff=sigmod(a[i],thata)
			thata-=alpha*(diff[0,0]-B[i,0])*a[i].T
			if np.linalg.norm(thata-err0)<epsilon:
				finish=1
				break
			else:
				print(querynum,  loop, np.linalg.norm(thata-err0))
				loop+=1
				err0=copy.copy(thata)
		if finish==1:
			print('finish,',np.linalg.norm(thata-err0))
			break
	thata*=100


	#测试
	with open('E:\\courses\knowledgeanalysis\\regression\\documentsdivided\\test\\%s'%(str(201+querynum)+'.json'),'r',encoding='UTF-8')as f:
		load_test=json.load(f)
	#对测试集的每个doc的title进行预处理
	testscore=[]
	for testnum in range(len(load_test)):
		if load_test[testnum]['title']!='':
			load_test[testnum]['title']=preprocess(load_test[testnum]['title'])
		testscore.append(float(load_test[testnum]['score']))
	testA=lil_matrix((testnum,wordlist_len+1))#测试doc数*词表大小的稀疏矩阵
	testscore=np.matrix(testscore)#测试集的每项得分，用于计算MAE
	for i in range(testnum):
		for word in re.split('\s+',load_test[i]['title'].strip()):
			if word in word2int:
				testA[i,word2int[word]]=1
	resultmatrix=testA*thata
	s=''
	nMAE=0
	testdoccount=0
	for i in range(testnum):
		s=s+'%s Q0 %s %s %s Hiemstra_LM0.15_Bo1bfree_d_3_t_10\n'%(str(201+querynum),load_test[i]['article_id'],load_test[i]['index'],str(1.0/(1+np.exp(-1*resultmatrix[i,0]))))
		nMAE+=abs(testscore[0,i]-resultmatrix[i,0])
	testdoccount+=i
	with open('e:\\LogosticRegression.res','a',encoding='UTF-8') as f:
		f.write(s)
	print('finish %s  MAE=%s count=%s\n'%(str(querynum),str(nMAE/i),str(i)))