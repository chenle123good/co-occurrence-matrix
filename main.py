# coding:utf-8
import numpy as np
import pandas as pd
import jieba.analyse
import os
import xlrd
import pandas
import re
import math

r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'  # 用户也可以在此进行自定义过滤字符

# 获取关键词
def Get_file_keywords(dir):
    set_word = []  # 所有关键词的集合
    f_stop=open('./Data/stop_words','r',encoding='utf-8')
    stop_dic=[]
    for i in f_stop.readlines():
        stop_dic.append(i.strip())

    # 打开excel
    # wb = xlrd.open_workbook(dir)
    df=pandas.read_excel(dir, engine ="openpyxl")
    # line=[]
    line,all_line='',[]
    for row1,row2 in zip(df['态度'],df['建议']):  # 遍历文件夹下的每篇文章
        if row1!=row1 or row1=='无' or row1[:2] == '无无':
            row1=''
        else:
            row1 = re.sub(r1, '', row1)  # 去掉标点符号和数字
        if row2!=row2 or row2 == '无' or row2[:2] == '无无':
            row2 = ''
        else:
            row2 = re.sub(r1, '', row2)  # 去掉标点符号和数字


        words = ' '.join(jieba.lcut(row1 + row2))
        w=[]
        for l in words.split():
            if l not in stop_dic:
                w.append(l)
        if w!=[]:
            all_line.append(w)
        line=line+row1+row2

    # words = ' '.join(jieba.lcut(line))
    words = " ".join(jieba.analyse.extract_tags(sentence=line, topK=150, withWeight=False,allowPOS=()))  # TF-IDF分词
    w = []
    for l in words.split():
        if l not in stop_dic:
            w.append(l)
    words = w
    # data_array.append(words)

    for word in words:
        if word not in set_word:
            set_word.append(word)
    set_word = list(set(set_word))  # 所有关键词的集合
    list_word=[]
    for i in set_word:
        if i!='':
            list_word.append(i)
    return all_line, list_word



# 初始化矩阵
def build_matirx(set_word):
    edge = len(set_word) + 1  # 建立矩阵，矩阵的高度和宽度为关键词集合的长度+1
    # matrix = np.zeros((edge, edge), dtype=str)  # 另一种初始化方法
    matrix = [['' for j in range(edge)] for i in range(edge)]  # 初始化矩阵
    matrix[0][1:] = np.array(set_word)
    matrix = list(map(list, zip(*matrix)))
    matrix[0][1:] = np.array(set_word)  # 赋值矩阵的第一行与第一列
    return matrix


# 计算各个关键词的共现次数
def count_matrix(matrix, formated_data):
    for row in range(1, len(matrix)):
        # 遍历矩阵第一行，跳过下标为0的元素
        for col in range(1, len(matrix)):
            # 遍历矩阵第一列，跳过下标为0的元素
            # 实际上就是为了跳过matrix中下标为[0][0]的元素，因为[0][0]为空，不为关键词
            if matrix[0][row] == matrix[col][0]:
                # 如果取出的行关键词和取出的列关键词相同，则其对应的共现次数为0，即矩阵对角线为0
                matrix[col][row] = str(0)
            else:
                counter = 0  # 初始化计数器
                for ech in formated_data:
                    # 遍历格式化后的原始数据，让取出的行关键词和取出的列关键词进行组合，
                    # 再放到每条原始数据中查询
                    if matrix[0][row] in ech and matrix[col][0] in ech:
                        word1,word2=ech.index(matrix[0][row]),ech.index(matrix[col][0])
                        if word1+1==word2 or word1-1==word2:
                            counter += 1
                    else:
                        continue
                matrix[col][row] = str(counter)
    return matrix

def deal_data(matrix):
    all_node=[]
    same_node=[]
    node1,node2,number=[],[],[]
    for row in range(1, len(matrix)):
        # 遍历矩阵第一行，跳过下标为0的元素
        for col in range(1, len(matrix)):
            # 遍历矩阵第一列，跳过下标为0的元素
            # 实际上就是为了跳过matrix中下标为[0][0]的元素，因为[0][0]为空，不为关键词
            if matrix[0][row] == matrix[col][0]:
                same_node.append(matrix[0][col])
                continue
            else:

                # if matrix[0][col]+' '+matrix[row][0] not in all_node:
                node1.append(matrix[row][0])
                node2.append(matrix[0][col])
                number.append(matrix[row][col])
                all_node.append((matrix[row][0] + ' ' + matrix[0][col]))

    w = pd.DataFrame({'Source':node1,'Target':node2,'Weight':number})
    w.to_csv('./Data/edge.csv')

    f = pd.DataFrame({'Id': same_node, 'Label': same_node})
    f.to_csv('./Data/node.csv')




def main():

    formated_data, set_word = Get_file_keywords(r'./Data/评价文本.xlsx')
    print(set_word)
    print(formated_data)
    matrix = build_matirx(set_word)
    matrix = count_matrix(matrix, formated_data)

    deal_data(matrix)
    # data1 = pd.DataFrame(matrix)
    # data1.to_csv('data.csv', index=0, columns=None, encoding='utf_8_sig')


main()