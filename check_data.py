

import os

adenine = './list/adenine-old'
other = './list/other-old'
voxel_folder = './voxel_output/'

list_adenine = []
list_other = []
voxel_name_list = []
for filename in os.listdir(voxel_folder):
    if filename:
        voxel_name_list.append(filename[0:-4])

with open(adenine) as ad_in:
    for line in ad_in.readlines():
        temp = line.replace(' ', '').replace('\n', '')
        ttmp = temp.split('\t')
        tmp1 = ttmp[0].split('.')
        tmp2 = ttmp[1].split('.')
        aa = tmp1[0] + '_' + tmp2[1]

        res1 = any(aa in voxel for voxel in voxel_name_list)

        if res1:
            list_adenine.append(aa)
        else:
            print aa
    list_adenine.sort()
    list_adenine = list(set(list_adenine))
    print list_adenine
ad_in.close()

with open(other) as ot_in:
    for line in ot_in.readlines():
        temp = line.replace(' ', '').replace('\n', '')
        ttmp = temp.split('\t')

        tmp1 = ttmp[0].split('.')
        tmp2 = ttmp[1].split('.')
        aa = tmp1[0] + '_' + tmp2[1]

        res1 = any(aa in voxel for voxel in voxel_name_list)

        if res1:
            list_other.append(aa)
        else:
            print aa

    list_other.sort()
    list_other = list(set(list_other))
ot_in.close()

if os.path.exists("adenine"):
    os.remove("adenine")
with open("adenine", "w") as outf:
    for i in range(len(list_adenine)):
        outf.write('%s\n' % list_adenine[i])
outf.close()

if os.path.exists("other"):
    os.remove("other")
with open("other", "w") as outf:
    for i in range(len(list_other)):
        outf.write('%s\n' % list_other[i])
outf.close()
