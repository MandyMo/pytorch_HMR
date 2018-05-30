import cv2
import numpy as np


sa = cv2.imread('./mesh1.png')
sb = cv2.imread('./mesh2.png')
sc = cv2.imread('./mesh3.png')

da = dst_image = cv2.resize(sa, (300, 400), interpolation = cv2.INTER_CUBIC)
db = dst_image = cv2.resize(sb, (300, 400), interpolation = cv2.INTER_CUBIC)
dc = dst_image = cv2.resize(sc, (300, 400), interpolation = cv2.INTER_CUBIC)

d = np.zeros((400, 900, 3))
d[:, :300, :] = da[:, :, :]
d[:, 300:600,:] = db[:, :, :]
d[:, 600:, :] = dc[:, :, :]

cv2.imwrite('r.png', d)