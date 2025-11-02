import os
import shutil
# data/../../.. -> data/
cnt = 0
dir = 'DATA'
folders = [f for f in os.listdir(dir)]
for i in range(len(folders)):
    dir2 = dir + "/" + folders[i] #data/"100241"
    folders2 = [f for f in os.listdir(dir2)]
    for j in range (len(folders2)):
        dir3 = dir2 + "/" + folders2[j] #data/100241/"20240508"
        folders3 = [f for f in os.listdir(dir3)]
        for k in range(len(folders3)):
            dir4 = dir3 + "/" + folders3[k] #data/100241/20240508/"112048"
            folders4 = [f for f in os.listdir(dir4)]
            for l in range(len(folders4)):
                exFile=dir4 + "/" + folders4[l] #data/100241/20240508/112048/"ex140"
                cnt+=1
                shutil.move(exFile,"DATA/EXFILES/")

