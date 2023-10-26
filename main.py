import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img_list = []
std_num = 10
filepath = 'D:/Jiwoon/dataset/RWF_2000_Dataset/train/Fight/'
savepath = 'D:/Jiwoon/dataset/data_4_fastflow/anomal_20frame/test/fight/'
framepath = 'D:/Jiwoon/dataset/data_4_fastflow/custom_frame/train/fight/'
video_list = os.listdir(filepath)
# print(video_list)
for v in video_list:
    video = cv2.VideoCapture(filepath+v) #'' 사이에 사용할 비디오 파일의 경로 및 이름을 넣어주도록 함

    if not video.isOpened():
        print("Could not Open :", filepath)
        exit(0)
    # length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    print(v, ' fps: ',fps)

    count = 0

    while (video.isOpened()):
        ret, image = video.read()
        if ret:
            if (int(video.get(1)) % 15 == 0):  # 앞서 불러온 fps 값을 사용하여 1초마다 추출

                if not os.path.exists(framepath+v[:-4]):
                    os.makedirs(framepath+v[:-4])
                img_list.append(image)
                print(framepath+v[:-4])
                cv2.imwrite(framepath + v[:-4] + "/fight_%d.png" % count, image)
                count+=1
                print('Saved frame number :', str(int(video.get(1))))
                # if len(img_list)==std_num:
                #     x_r = []
                #     x_g = []
                #     x_b = []
                #     for img in img_list:
                #         r, g, b = cv2.split(img)
                #         x_r.append(r)
                #         x_g.append(g)
                #         x_b.append(b)
                #         x_r_std = np.std(x_r, axis=0)
                #         x_g_std = np.std(x_g, axis=0)
                #         x_b_std = np.std(x_b, axis=0)
                #         result = cv2.merge((x_r_std, x_g_std, x_b_std))
                #         cv2.imwrite(savepath + "/val_%s_%d.png" % (v[:-4],count), result)
                #
                #     img_list.clear()
                #     count += 1
        else:
            break

    print(count)
    video.release()
    img_list.clear()



