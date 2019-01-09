f=open('chanel.txt','r')
tmp2=f.readline().split('\t')
print tmp2[0].decode('utf8')
print type(tmp2[0])
