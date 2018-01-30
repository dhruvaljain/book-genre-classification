import sys

l=sys.argv[1]
with open(l) as f:
    lines = f.readlines()
i=1
li1=1
li2=5
for line in lines:
	
	for j in range(0,len(line)):
		
		if(line[j]=='+'):
			li1=i
			li2=i+4

	if i==li1 or i==li2:
		print line
	i=i+1

f.close

