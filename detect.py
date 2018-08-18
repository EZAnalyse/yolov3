#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import tensorflow.keras as keras
from PIL import Image, ImageDraw
import numpy as np
import cv2
import copy

ANCHORS = np.array([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]])
INDEX = [6, 7, 8], [3, 4, 5], [0, 1, 2]
CELL_SIZE = [32, 16, 8]


def grid_cell_left_up_coor():
    size = [13, 26, 52]
    CX = []
    CY = []
    for n in size:
        one = np.ones((n, n))
        num = np.arange(n)
        cx = one * num
        cy = cx.T
        CX.append(np.expand_dims(cx, 2))
        CY.append(np.expand_dims(cy, 2))
    return CX, CY


CX, CY = grid_cell_left_up_coor()


def resize_img(image, new_size=[416, 416]):
    # draw the image into a new canvas with new_size
    # the ratio of image.width/image.height can not change
    [delta_w, delta_h], [new_w, new_h], _ = convert_size(image.size, new_size)
    image = image.resize([new_w, new_h])
    # build canvas
    canvas = np.full(new_size + [3, ], 128, dtype='uint8')
    # draw the resized image into canvas
    canvas[delta_h:(delta_h + new_h), delta_w:(delta_h + new_w), :] = np.array(image)
    return canvas


def yolo_predict(image, yolo):
    data = resize_img(image)
    data = data / 255
    data = np.expand_dims(data, 0)
    result = yolo.predict(data)
    return result


def convert_size(image_size, input_size=(416, 416)):
    # draw the image into a new canvas with new_size
    # the ratio of image.width/image.height can not change
    inp_w, inp_h = input_size
    img_w, img_h = image_size
    ratio = min(inp_w / img_w, inp_h / img_h)
    new_w, new_h = new_size = [int(img_w * ratio), int(img_h * ratio)]
    delta_w, delta_h = (inp_w - new_w) // 2, (inp_h - new_h) // 2
    return [delta_w, delta_h], new_size, ratio


def convert_2_origin(boxes, image_size, input_size=(416, 416), threshold=0.6):
    res = []
    [delta_w, delta_h], _, ratio = convert_size(image_size, input_size)
    bx, by, bw, bh, conf, prob = boxes
    # convert the center to the original image
    bx -= delta_w
    by -= delta_h
    bx /= ratio
    by /= ratio
    # convert the width and height to the original image
    bw /= ratio
    bh /= ratio
    # select box to display
    w = h = len(bx)
    for i in range(h):
        for j in range(w):
            for k in range(3):
                txy = np.array([bx[i, j, k], by[i, j, k]])
                twh = np.array([bw[i, j, k], bh[i, j, k]])
                c = conf[i, j, k]
                p = prob[i, j, k]
                if max(c * p) > threshold:
                    x1, y1 = txy - twh / 2.
                    x2, y2 = txy + twh / 2.
                    res.append([x1, y1, x2, y2])
    return res


def parse_layer(box, layer):
    global ANCHORS, INDEX, CELL_SIZE, CX, CY
    box = box.reshape(box.shape[:2] + (3, -1))
    # center of input image
    tx = box[..., 0]
    ty = box[..., 1]
    cx = CX[layer]
    cy = CY[layer]
    bx = sigmoid(tx) + cx
    by = sigmoid(ty) + cy
    dw = dh = CELL_SIZE[layer]
    bx *= dw
    by *= dh
    # width and height of input image
    tw = box[..., 2]
    th = box[..., 3]
    anchors = ANCHORS[INDEX[layer]]
    pw = anchors[:, 0]
    ph = anchors[:, 1]
    bw = np.exp(tw) * pw
    bh = np.exp(th) * ph
    # confidence
    cb = box[..., 4]
    conf = sigmoid(cb)
    # class probability
    cp = box[..., 5:]
    prob = sigmoid(cp)
    return bx, by, bw, bh, conf, prob


def IOU(u, v):
    x1, y1, x2, y2 = u
    m1, n1, m2, n2 = v
    su = (x2 - x1) * (y2 - y1)
    sv = (m2 - m1) * (n2 - n1)
    a1 = max(x1, m1)
    b1 = max(y1, n1)
    a2 = min(x2, m2)
    b2 = min(y2, n2)
    if a1 < a2 and b1 < b2:
        s = (a2 - a1) * (b2 - b1)
        if max(s/su, s/sv) > 0.85:
            if su > sv:
                return u
            else:
                return v
        else:
            return  None
    else:
        return None


def del_repeat(boxes):
    B = copy.deepcopy(boxes)
    for i in range(len(boxes)):
        u = boxes[i]
        for j in range(i+1, len(boxes)):
            v = boxes[j]
            try:
                B.remove(IOU(u, v))
            except:
                continue
    return B


def parse_result(yolo_out, image, input_size=(416, 416), threshold=0.4):
    res = []
    for layer in range(3):
        box = yolo_out[layer][0]
        input_boxes = parse_layer(box, layer)
        res.extend(convert_2_origin(input_boxes, image.size, input_size, threshold))
    draw = ImageDraw.Draw(image)
    res = del_repeat(res)
    for b in res:
        draw.rectangle(b)
    return res


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def image(image_file):
    model_file = './yolo.h5'
    yolo_model = keras.models.load_model(model_file, compile=False)
    img = Image.open(image_file, 'r')
    img = img.convert('RGB')
    yolo_out = yolo_predict(img, yolo_model)
    parse_result(yolo_out, img)
    img.show()


def video(file):
    model_file = './yolo.h5'
    yolo_model = keras.models.load_model(model_file, compile=False)

    cap = cv2.VideoCapture(file)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Display the resulting frame
            img = Image.fromarray(frame)
            yolo_out = yolo_predict(img, yolo_model)
            parse_result(yolo_out, img)
            cv2.imshow('Frame', np.array(img))
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break  # Break the loop
        else:
            break  # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()


def main():
    # image_file = 'dog.jpg'
    # image(image_file)

    video_file = './HLX.MP4'
    video(video_file)


if __name__ == '__main__':
    main()
