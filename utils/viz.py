import cv2
import numpy as np


def label2rgb(label_np):
    print(label_np)
    label_color = np.argmax(label_np, axis=0)
    label_color = label_color / np.max(label_color) * 255
    print(label_color)
    n = label_color.astype(np.uint8)
    n = np.array(n)
    print(type(n))
    label_color = cv2.applyColorMap(n, 'jet')
    return label_color