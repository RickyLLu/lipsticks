f=open('dior.txt','r')
tmp2=f.readline().split('\t')
print tmp2[0].decode('gbk')
print type(tmp2[0].decode('gbk'))
