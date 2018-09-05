import os
url = open('resource.txt','rt',encoding = 'utf-8')
url_image = url.readlines()
print(url_image)
m = len(url_image)
for i in range(m):
    print(url_image[i])
# url1 = open('src.txt','rt',encoding = 'utf-8')
# url_image = url1.readlines()
# # print(url_image)
# m = len(url_image)
# print(m)
# for i in range(m):
#     print(url_image[i])

