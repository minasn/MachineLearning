import re
from scipy.sparse import lil_matrix,linalg
from scipy import linalg as la
import numpy as np
import math
from sklearn.decomposition import PCA

trainlist=[]
trainlabels=[]
testlist=[]
testlabels=[]
index=[]
#提取训练集doc名称
with open('e:\\courses\\knowledgeanalysis\\regression\\train.res','r',encoding='UTF-8')as f:
	s=[]
	queryid=201
	line=f.readline()
	while line:		
		temp=re.split('\s+',line.strip())
		if temp[0]==str(queryid):
			s.append(temp[2])#article_id
			line=f.readline()
		else:
			trainlist.append(s[:200]+s[-200:])
			s=[]
			queryid+=1
	trainlist.append(s)
	
	label=[]
	for i in range(200):
		label.append(1)
	for i in range(200):
		label.append(0)
	for i in range(50):
		trainlabels.append(label)

#提取测试集doc名称
with open('e:\\courses\\knowledgeanalysis\\regression\\test.res','r',encoding='UTF-8')as f:
	s=[]
	score={}
	tempindex=[]
	queryid=201
	line=f.readline()
	while line:
		temp=re.split('\s+',line.strip())
		if temp[0]==str(queryid):
			s.append(temp[2])
			tempindex.append(temp[3])
			score[temp[2]]=float(temp[4])#分数
			line=f.readline()
		else:
			testlist.append(s[:50]+s[-50:])
			s=[]
			index.append(tempindex[:50]+tempindex[-50:])
			tempindex=[]
			queryid+=1
	testlist.append(s)
	index.append(tempindex)

	label=[]
	for i in range(50):
		label.append(1)
	for i in range(50):
		label.append(0)
	for i in range(50):
		testlabels.append(label)

def choosecenter(Dataset):#选取质心
	doccount=Dataset.shape[0]
	features=Dataset.shape[1]
	centroid=[]
	ones=np.array(([1 for i in range(doccount)]))
	for i in range(features):
		centroid.append(np.dot(Dataset[:,i],ones)/doccount) 
	return centroid


def kmeans(Dataset,k):
	doccount=Dataset.shape[0]
	features=Dataset.shape[1]
	centroid1=choosecenter(Dataset)#选取质心作为初始聚类中心
	maxdistance=0
	for i in range(doccount):
		distance=np.linalg.norm(centroid1-Dataset[i])
		if distance>maxdistance:
			maxdistance=distance
			centroid2=copy.copy(Dataset[i]) #选取离质心最远的点作为另一初始聚类中心
	cluster1=[]
	cluster2=[]
	err0,err1=[10,0]
	eps=0.1
	while(err0-err1>0.1):


	pass

for querynum in range(50):
	#词表大小
	word2int={}
	count=0
	trainvectors=[]
	for docnum in range(400):
		vector=[]
		with open('E:\\courses\\knowledgeanalysis\\cleanerdata\\titlevector\\%s\\%s.txt'%(201+querynum,trainlist[querynum][docnum]),'r',encoding='UTF-8')as f:
			line=f.readline()
			while line:
				if int(line) not in word2int:
					word2int[int(line)]=count
					vector.append(count)
					count+=1
				else:
					vector.append(word2int[int(line)])
				line=f.readline()
			trainvectors.append(vector)


	testvectors=[]
	for docnum in range(100):
		vector=[]
		with open('E:\\courses\\knowledgeanalysis\\cleanerdata\\titlevector\\%s\\%s.txt'%(201+querynum,testlist[querynum][docnum]),'r',encoding='UTF-8')as f:
			line=f.readline()
			while line:
				if int(line) in word2int:
					vector.append(word2int[int(line)])
				line=f.readline()
			testvectors.append(vector)

	samplemat=np.zeros((400,len(word2int)))
	samplelabels=list(range(len(word2int)))
	testmat=np.zeros((100,len(word2int)))
	
	for i in range(400):
		for col in trainvectors[i]:
			samplemat[i,col]+=1
	
	for i in range(100):
		for col in testvectors[i]:
			testmat[i,col]+=1

	

	nmae=0
	s=''
	for i in range(100):
		s=s+'%s Q0 %s %s %s Hiemstra_LM0.15_Bo1bfree_d_3_t_10\n'%(querynum+201,testlist[querynum][i],index[querynum][i],result[i])
		nmae+=abs(result[i]-testlabels[querynum][i])
	with open('E:\\courses\\knowledgeanalysis\\algorithm\\bayesian\\kmeans.res','a',encoding='UTF-8')as f:
		f.write(s)
	print('querynum:%s   mae=%s\n'%(querynum,nmae/100))
	with open('E:\\courses\\knowledgeanalysis\\algorithm\\bayesian\\kmeans.txt','a',encoding='UTF-8')as f:
		f.write('querynum:%s   mae=%s\n'%(querynum,nmae/100))