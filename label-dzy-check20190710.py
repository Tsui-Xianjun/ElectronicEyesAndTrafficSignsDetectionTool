# #!/usr/bin/python
# # -*- coding: UTF-8 -*-
#
import os
from Tkinter import *
import tkFileDialog

from ctypes import *
import math
import random
import os
# import cv2
import time
# from PIL import Image
import csv
import shutil

import numpy as np
import sys

sys.path.append("./")
import cv2

import threading

# bp
fileList = []
# Picture name
picNameList = []
# Picture index in list
count = 0
list_plate = []
piliangchuli = 1

# 加载模型
# run = "20180822_ResNetW_1c_poi_plate+fixmodel"
run = "20181212_ResNetW_bp+fixmodel"

# self.run_dir = "D:/windowstrans/{}".format(run)
path1 = os.path.abspath('.')  # 获取当前脚本所在的路径
run_dir = "{}/{}".format(path1, run)
sys.path.append("D/label_dzy-checktool/")

# 设置gpu号
# 读取阈值
f = open("./cfg/gpunumber.txt")
line = f.readline()
# line=line[:-1]
gpunum = int(line)
f.close()

# 读取bp阈值
f = open("./cfg/threshold-bp.txt")
line = f.readline()
line = line[:-1]
thresh_bp = float(line)
f.close()


def loadmodel_bp(caffe, gpunum):
    caffe.set_device(gpunum)
    caffe.set_mode_gpu()  # or
    # caffe.set_mode_cpu()

    max_iter = 107180
    model_def = "{}/model/deploy.prototxt".format(run_dir)
    model_weights = "{}/model/{}_iter_{}.caffemodel".format(run_dir, run, max_iter)

    net1 = caffe.Net(model_def,  # defines the structure of the model
                     model_weights,  # contains the trained weights
                     caffe.TEST)  # use test mode (e.g., don't perform dropout)
    return net1


# set net to batch size of 1 and 1820x1820 inputs
def transformer_process(caffe, net1, image, image_resize_width, image_resize_height):
    net1.blobs['data'].reshape(1, 3, image_resize_width, image_resize_height)

    # input preprocessing: 'data' is the name of the input blob == net1.inputs[0]
    transformer = caffe.io.Transformer({'data': net1.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
    # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_raw_scale('data', 255)
    # the reference model has channels in BGR order instead of RGB
    transformer.set_channel_swap('data', (2, 1, 0))
    transformed_image = transformer.preprocess('data', image)
    return transformed_image


def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames


##bp

# 全局变量
filepath = "begin!!"


def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs) - 1


def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


def thread_it(func, *args):
    '''将函数打包进线程'''
    # 创建
    t = threading.Thread(target=func, args=args)
    # 守护
    t.setDaemon(True)
    # 启动
    t.start()
    # 阻塞--卡死界面
    # t.join()


def detect(net, meta, image, thresh=.85, hier_thresh=.5, nms=.45):
    # 读取阈值
    f = open("./cfg/threshold.txt")
    line = f.readline()
    line = line[:-1]
    thresh = float(line)
    f.close()
    #
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    letterbox = c_int(0)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum, letterbox)
    num = pnum[0]

    if (nms): do_nms_obj(dets, num, meta.classes, nms)

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                tmp = "dzy"
                # res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
                res.append((tmp, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res


def openFilePath(sOpenFilePath):
    # if v.get() is 0 or len(sOpenFilePath) is not 0:
    if v.get() is 0:
        return
    sOpenFilePath.append(tkFileDialog.askdirectory())

    # 实时更新text
    text1.delete(1.0, END)
    text1.insert(INSERT, sOpenFilePath[len(sOpenFilePath) - 1])
    text1.update()

    # print(sOpenFilePath)


# 运行程序--电子眼
def Work0(sOpenFilePath):
    net = load_net("./cfg/yolov3-voc.cfg", "./backup/yolov3-voc_v2_21000.weights", 0)
    meta = load_meta("./cfg/voc.data")

    data_dir = sOpenFilePath[len(sOpenFilePath) - 1].decode('gbk')

    # 自动搜索含有图像的文件夹路径
    num_total = 0
    g = os.walk(data_dir)
    for path, d, filelist in g:
        for filename in filelist:
            if filename.endswith('csv'):
                Isuandian = 0
                savedrownum = 0
                # #确定选择的文件是原始的轨迹FIT文件
                if filename.find('BP_DZY') >= 0 or filename.find('BP') >= 0 or filename.find('DZY') >= 0 or filename.find('CHANGE') >= 0:
                    # if filename.find('DZY') >= 0:
                    continue
                # 判断是否正常处理完
                # RootDir=os.path.dirname(path) ##获取上一层目录
                RootDir = path
                csv_file_Name_Change = RootDir + "//" + filename[:-4] + "_CHANGE.csv"
                if not os.path.exists(csv_file_Name_Change):  # 未处理过
                    IsDuandian = 0
                else:  # 之前处理过，需断点续处理
                    IsDuandian = 1

                # 获取图像路径
                tmp = path.split("\\")

                ##获取上一层目录
                # # RootDir=os.path.dirname(path)
                # RootDir = path

                # create csv-save file
                csv_file_Name = RootDir + "//" + filename[:-4] + "_DZY.csv"
                if not os.path.exists(csv_file_Name):
                    csv_reader_file1 = open(csv_file_Name, 'wb')
                    csv_writer = csv.writer(csv_reader_file1)
                else:  # 之前处理过
                    print("The file contains the trajectory that has been run before(" + filename + "),")
                    print("But I have to do it! ")
                    csv_reader_file1 = open(csv_file_Name, 'ab')
                    csv_writer = csv.writer(csv_reader_file1)
                    # continue

                # create csv-save file for original csv file change
                csv_file_Name_Change = RootDir + "//" + filename[:-4] + "_CHANGE.csv"
                if not os.path.exists(csv_file_Name_Change):
                    csv_reader_file2 = open(csv_file_Name_Change, 'wb')
                    csv_writer_Change = csv.writer(csv_reader_file2)
                else:  # 之前处理过，需断点续处理
                    print("The file contains the trajectory that has been run before(" + filename + "),")
                    print("But I have to do it! ")
                    # 获取断点
                    csv_reader_file_tmp = open(csv_file_Name_Change, 'r')
                    csv_reader_tmp = csv.reader(csv_reader_file_tmp)
                    savedrownum = sum(1 for row in csv_reader_tmp)
                    # 删除最后一行（防止半行）
                    # for row in csv_reader_file_tmp:
                    #     del row[savedrownum]
                    # savedrownum = savedrownum - 1
                    csv_reader_file_tmp.close()

                    csv_reader_file2 = open(csv_file_Name_Change, 'ab')
                    csv_writer_Change = csv.writer(csv_reader_file2)

                    # continue

                # 获取csv文件路径
                csv_file_Name_ori = RootDir + "//" + filename
                # 获取图像路径
                images_path = csv_file_Name_ori[:-7] + "IMG"
                if not os.path.exists(images_path):
                    print("The corresponding image(" + filename + ")file could not be found,please check it later!")
                    continue

                # 读取csv文件获取图像名称及点位
                image_ids = os.listdir(images_path)
                csv_reader_file3 = open(csv_file_Name_ori, 'r')
                csv_reader = csv.reader(csv_reader_file3)
                data_tmp = "tmp.jpg"
                objectnum = 0
                crop_size = 512
                idx = 0
                # 遍历路径开始执行程序
                #################遍历路径开始执行程序
                for line in csv_reader:
                    # print(line)
                    image_id = line[0]
                    idx = idx + 1
                    num_total = num_total + 1

                    # 断点处理
                    if idx <= savedrownum and IsDuandian is 1:
                        continue

                    # 保存第一行
                    if idx is 1 and not os.path.exists("{}/{}".format(images_path, image_id)):
                        csv_writer.writerow(line)

                    # 保存原始文件
                    if not os.path.exists("{}/{}".format(images_path, image_id)):
                        csv_writer_Change.writerow(line)

                    print("current image num of total: {}".format(num_total))
                    if not os.path.exists("{}/{}".format(images_path, image_id)):
                        continue
                    image = cv2.imread("{}/{}".format(images_path, image_id))
                    try:
                        image.shape
                    except:
                        print "Image {} is damaged image,skipped...".format(image_id)
                        continue
                    print "Image {}:".format(image_id)
                    im_save = cv2.imread("{}/{}".format(images_path, image_id))
                    print "current filepath:{}".format(path)
                    print "current image:{}/total imgaes:{}".format(idx, len(image_ids))
                    # subimage
                    y0 = 0
                    x0 = 0
                    time1 = 0.0
                    objectnum = 0
                    while (x0 + crop_size < int(image.shape[0] * 2 / 3)):
                        y0 = 0
                        while (y0 + crop_size < image.shape[1]):
                            subimage = image[x0:x0 + crop_size, y0:y0 + crop_size]
                            cv2.imwrite(data_tmp, subimage)
                            start = time.time()
                            r = detect(net, meta, data_tmp)
                            elapsed = time.time() - start
                            ##################################
                            for i in xrange(len(r)):
                                xmin = y0 + int(round(r[i][2][0] - r[i][2][2] / 2))
                                ymin = x0 + int(round(r[i][2][1] - r[i][2][3] / 2))
                                xmax = y0 + int(round(r[i][2][0] + r[i][2][2] / 2))
                                ymax = x0 + int(round(r[i][2][1] + r[i][2][3] / 2))
                                score = r[i][1]
                                label_name = r[0][0]

                                scale = 1
                                width = int((xmax - xmin) / scale)
                                height = int((ymax - ymin) / scale)
                                xoff = int(xmin / scale)
                                yoff = int(ymin / scale)

                                # labeling
                                if (yoff > image.shape[0] / 1) or width > 200 or height > 200:
                                    continue
                                if yoff > image.shape[0] * 3 / 4:
                                    continue
                                if yoff > 0:
                                    if yoff < 3:
                                        yoff = 3
                                    cv2.rectangle(image, (int(xoff), int(yoff)), (int(
                                        xoff + width), int(yoff + height)), (0, 0, 255), 3)
                                    display_txt = "\t{}\t[{}x{} @ {},{}]\t{}".format(
                                        label_name, width, height, xoff, yoff, "%.2f" % score)
                                    print display_txt
                                    objectnum = objectnum + 1
                                label_name1 = "{}:{}".format(label_name, "%.2f" % score)
                                if (yoff > 10):
                                    cv2.putText(image, label_name1, (int(xoff), int(
                                        yoff - 6)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0))
                                else:
                                    cv2.putText(image, label_name1, (int(xoff), int(
                                        yoff + 15)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0))
                            ##################################
                            y0 = y0 + int(crop_size / 100) * int(100 / 1.3)
                            time1 = time1 + elapsed
                        x0 = x0 + int(crop_size / 100) * int(100 / 1.3)
                    print "Inference Time: {} ms".format(int(1000 * time1))
                    print

                    ##show current result
                    # cv2.namedWindow("img", 2)
                    # cv2.imshow("img", image)
                    # cv2.waitKey(1)

                    # tmp1=os.path.dirname(path)  #上一级目录
                    tmp1 = path

                    # notargetresult="{}/{}".format(tmp1,"noTargeImages")
                    # if not os.path.exists(notargetresult):
                    # 		os.mkdir(notargetresult)

                    # checkresult = "{}/{}".format(tmp1, "checkImages")
                    # if not os.path.exists(checkresult):
                    # 	os.mkdir(checkresult)

                    # jianceresult = "{}/{}".format(checkresult, "location")
                    # if not os.path.exists(jianceresult):
                    # 	os.mkdir(jianceresult)

                    ##建立保存路径并保存
                    imgSaveFileName = filename[:-7] + "DZY"
                    checkimgpath = "{}/{}".format(tmp1, imgSaveFileName)
                    if not os.path.exists(checkimgpath):
                        os.mkdir(checkimgpath)

                    if objectnum > 0:
                        cv2.imwrite("{}/{}".format(checkimgpath, image_id), image)
                        # 保存csv
                        # extractLable=line.split(",")
                        extractLable = line
                        if len(extractLable) > 4:
                            extractLable[4] = '1'  # 无目标：0；电子眼：1；标牌：2；电子眼+标牌：1|2
                        csv_writer.writerow(extractLable)
                        csv_reader_file1.flush()
                        csv_writer_Change.writerow(extractLable)
                        csv_reader_file2.flush()
                    else:  # 保存无目标图像
                        # cv2.imwrite("{}/{}".format(checkimgpath, image_id), im_save)
                        shutil.copy("{}/{}".format(images_path, image_id),"{}/{}".format(checkimgpath, image_id))
                        # #保存ori cvs文件change
                        extractLable = line
                        if len(extractLable) > 4:
                            extractLable[4] = '0'  # 之前处理过的结果抹去
                        csv_writer_Change.writerow(extractLable)
                        csv_reader_file2.flush()

                # 更改原始csv文件
                # 删除文件
                csv_reader_file1.close()
                csv_reader_file2.close()
                csv_reader_file3.close()
                os.remove(csv_file_Name_ori)
                # 重命名文件
                shutil.move(csv_file_Name_Change, csv_file_Name_ori)
                break


# 运行程序--标牌
def Work1(sOpenFilePath):
    # Matplotlib elements used to draw the bounding rectangle
    import matplotlib.pyplot as plt
    # import numpy as np
    # caffe-ssd
    import math
    from google.protobuf import text_format
    import caffe
    from caffe.proto import caffe_pb2
    # matplotlib inline
    plt.rcParams['image.interpolation'] = 'nearest'

    net1 = loadmodel_bp(caffe, gpunum)
    # time.sleep(0.001)
    labelmap_file = "{}/labelmap.prototxt".format(run_dir)
    file = open(labelmap_file, 'r')
    labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(file.read()), labelmap)

    # 图像根目录
    data_dir = sOpenFilePath[len(sOpenFilePath) - 1].decode('gbk')

    # 自动搜索含有图像的文件夹路径
    num_total = 0
    g = os.walk(data_dir)
    for path, d, filelist in g:
        for filename in filelist:
            if filename.endswith('csv'):
                Isuandian = 0
                savedrownum = 0
                # 确定选择的文件是原始的轨迹FIT文件
                if filename.find('BP_DZY') >= 0 or filename.find('BP') >= 0 or filename.find('DZY') >= 0 or filename.find('CHANGE') >= 0:
                    # if filename.find('BP') >= 0:
                    continue
                # 判断是否正常处理完
                # RootDir=os.path.dirname(path) ##获取上一层目录
                RootDir = path
                csv_file_Name_Change = RootDir + "//" + filename[:-4] + "_CHANGE.csv"
                if not os.path.exists(csv_file_Name_Change):# 未处理过
                    IsDuandian = 0
                else:  # 之前处理过，需断点续处理
                    IsDuandian = 1

                # 获取图像路径
                tmp = path.split("\\")

                # ##获取上一层目录
                # # RootDir=os.path.dirname(path)
                # RootDir = path

                # create csv-save file
                csv_file_Name = RootDir + "//" + filename[:-4] + "_BP.csv"
                if not os.path.exists(csv_file_Name):
                    csv_reader_file1 = open(csv_file_Name, 'wb')
                    csv_writer = csv.writer(csv_reader_file1)
                else:  #之前处理过
                    print("The file contains the trajectory that has been run before(" + filename + "),")
                    print("But I have to do it! ")
                    csv_reader_file1 = open(csv_file_Name, 'ab')
                    csv_writer = csv.writer(csv_reader_file1)
                    # continue

                # create csv-save file for original csv file change
                csv_file_Name_Change = RootDir + "//" + filename[:-4] + "_CHANGE.csv"
                if not os.path.exists(csv_file_Name_Change):
                    csv_reader_file2 = open(csv_file_Name_Change, 'wb')
                    csv_writer_Change = csv.writer(csv_reader_file2)
                else:  # 之前处理过，需断点续处理
                    print("The file contains the trajectory that has been run before(" + filename + "),")
                    print("But I have to do it! ")
                    #获取断点
                    csv_reader_file_tmp = open(csv_file_Name_Change, 'rb')
                    csv_reader_tmp = csv.reader(csv_reader_file_tmp)
                    savedrownum = sum(1 for row in csv_reader_tmp)

                    # 删除最后一行（防止半行）
                    # csv_writer_file_tmp = open('tmp.csv', 'wb')
                    # csv_writer_tmp = csv.writer(csv_writer_file_tmp)
                    # for row in csv_reader_tmp:
                    #     del row[savedrownum-1]
                    #     csv_writer_tmp.writerow(row)
                    # savedrownum = savedrownum - 1
                    # csv_writer_file_tmp.close()
                    csv_reader_file_tmp.close()

                    csv_reader_file2 = open(csv_file_Name_Change, 'ab')
                    csv_writer_Change = csv.writer(csv_reader_file2)

                    # continue

                # 获取csv文件路径
                csv_file_Name_ori = RootDir + "//" + filename
                # 获取图像路径
                images_path = csv_file_Name_ori[:-7] + "IMG"
                if not os.path.exists(images_path):
                    print("The corresponding image(" + filename + ") file could not be found,please check it later!")
                    continue

                # 读取csv文件获取图像名称及点位
                image_ids = os.listdir(images_path)
                csv_reader_file3 = open(csv_file_Name_ori, 'r')
                csv_reader = csv.reader(csv_reader_file3)
                data_tmp = "tmp.jpg"
                objectnum = 0
                crop_size = 512
                idx = 0
                # 遍历路径开始执行程序
                #################遍历路径开始执行程序
                for line in csv_reader:
                    # print(line)
                    objectnum = 0
                    image_id = line[0]
                    idx = idx + 1
                    num_total = num_total + 1

                    #断点处理
                    if idx <= savedrownum and IsDuandian is 1:
                        continue

                    # 保存第一行
                    if idx is 1 and not os.path.exists("{}/{}".format(images_path, image_id)):
                        csv_writer.writerow(line)

                    # 保存原始文件
                    if not os.path.exists("{}/{}".format(images_path, image_id)):
                        csv_writer_Change.writerow(line)

                    print("current image num of total: {}".format(num_total))
                    if not os.path.exists("{}/{}".format(images_path, image_id)):
                        continue
                    image = cv2.imread("{}/{}".format(images_path, image_id))
                    try:
                        image.shape
                    except:
                        print "Image {} is damaged image,skipped...".format(image_id)
                        continue

                    print "Image {}:".format(image_id)
                    im_save = cv2.imread("{}/{}".format(images_path, image_id))
                    print "current filepath:{}".format(path)
                    print "current image:{}/total imgaes:{}".format(idx, len(image_ids))

                    img = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
                    image = img / 255.0
                    if image.shape[1] > 3000:
                        tmp111 = int(image.shape[1] / 3000) + 1
                        image_resize_width = int(image.shape[1] / tmp111)
                        image_resize_height = int(image.shape[0] / tmp111)
                    else:
                        image_resize_width = image.shape[1]
                        image_resize_height = image.shape[0]

                    transformed_image = transformer_process(caffe, net1, image, image_resize_width, image_resize_height)# width/height 对换20190808
                    net1.blobs['data'].data[...] = transformed_image

                    # Forward pass.
                    start = time.time()
                    detections = net1.forward()['detection_out']
                    elapsed = time.time() - start
                    print "Inference Time: {} ms".format(int(1000 * elapsed))
                    # print

                    # Parse the outputs.
                    det_label = detections[0, 0, :, 1]
                    det_conf = detections[0, 0, :, 2]
                    det_xmin = detections[0, 0, :, 3]
                    det_ymin = detections[0, 0, :, 4]
                    det_xmax = detections[0, 0, :, 5]
                    det_ymax = detections[0, 0, :, 6]

                    threshold = thresh_bp

                    top_indices = [i for i, conf in enumerate(det_conf) if conf >= threshold]

                    top_conf = det_conf[top_indices]
                    top_label_indices = det_label[top_indices].tolist()
                    top_labels = get_labelname(labelmap, top_label_indices)
                    top_xmin = det_xmin[top_indices]
                    top_ymin = det_ymin[top_indices]
                    top_xmax = det_xmax[top_indices]
                    top_ymax = det_ymax[top_indices]

                    for i in xrange(top_conf.shape[0]):
                        xmin = int(round(top_xmin[i] * image.shape[1]))
                        ymin = int(round(top_ymin[i] * image.shape[0]))
                        xmax = int(round(top_xmax[i] * image.shape[1]))
                        ymax = int(round(top_ymax[i] * image.shape[0]))
                        score = top_conf[i]
                        label = int(top_label_indices[i])
                        label_name = top_labels[i]
                        coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1

                        scale = float(image.shape[1]) / float(image_resize_width)
                        scale = 1
                        width = int((xmax - xmin) / scale)
                        height = int((ymax - ymin) / scale)
                        xoff = int(xmin / scale)
                        yoff = int(ymin / scale)
                        display_txt = "\t{}\t[{}x{} @ {},{}]\t{}".format(label_name, width, height, xoff, yoff,
                                                                         "%.2f" % score)
                        # print(display_txt)

                        # labeling denoise
                        # if (yoff < image.shape[0] / 4) or (yoff > image.shape[0] - 50):
                        #     continue
                        # if width/height<1.5 or width/height>4:
                        #    continue

                        # 显示数据
                        if (yoff >= 0):
                            if yoff < 3:
                                yoff = 3
                            cv2.rectangle(im_save, (int(xoff), int(yoff)), (int(
                                xoff + width), int(yoff + height)), (0, 0, 255), 3)
                            display_txt = "\t{}\t[{}x{} @ {},{}]\t{}".format(
                                label_name, width, height, xoff, yoff, "%.2f" % score)
                            print display_txt
                            objectnum = objectnum + 1
                        label_name1 = "{}:{}".format(label_name, "%.2f" % score)
                        if (yoff > 10):
                            cv2.putText(im_save, label_name1, (int(xoff), int(
                                yoff - 6)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0))
                        else:
                            cv2.putText(im_save, label_name1, (int(xoff), int(
                                yoff + 15)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0))

                    ##show current result
                    # cv2.namedWindow("img", 2)
                    # cv2.imshow("img", image)
                    # cv2.waitKey(1)

                    # tmp1=os.path.dirname(path)  #上一级目录
                    tmp1 = path

                    # notargetresult="{}/{}".format(tmp1,"noTargeImages")
                    # if not os.path.exists(notargetresult):
                    # 		os.mkdir(notargetresult)

                    # checkresult = "{}/{}".format(tmp1, "checkImages")
                    # if not os.path.exists(checkresult):
                    # 	os.mkdir(checkresult)

                    # jianceresult = "{}/{}".format(checkresult, "location")
                    # if not os.path.exists(jianceresult):
                    # 	os.mkdir(jianceresult)

                    ##建立保存路径并保存
                    imgSaveFileName = filename[:-7] + "BP"
                    checkimgpath = "{}/{}".format(tmp1, imgSaveFileName)
                    if not os.path.exists(checkimgpath):
                        os.mkdir(checkimgpath)

                    if objectnum > 0:
                        cv2.imwrite("{}/{}".format(checkimgpath, image_id), im_save)
                        # 保存csv
                        # extractLable=line.split(",")
                        extractLable = line
                        if len(extractLable) > 4:
                            extractLable[4] = '2'  # 无目标：0；电子眼：1；标牌：2；电子眼+标牌：1|2
                        csv_writer.writerow(extractLable)
                        csv_reader_file1.flush()
                        csv_writer_Change.writerow(extractLable)
                        csv_reader_file2.flush()
                    else:  # 保存无目标图像
                        # cv2.imwrite("{}/{}".format(checkimgpath, image_id), im_save)
                        shutil.copy("{}/{}".format(images_path, image_id), "{}/{}".format(checkimgpath, image_id))
                        # #保存ori cvs文件change
                        # csv_writer_Change.writerow(line)
                        extractLable = line
                        if len(extractLable) > 4:
                            extractLable[4] = '0'  # 如果存在之前结果，抹去
                        csv_writer_Change.writerow(extractLable)
                        csv_reader_file2.flush()

                # 更改原始csv文件
                # 删除文件
                csv_reader_file1.close()
                csv_reader_file2.close()
                csv_reader_file3.close()
                os.remove(csv_file_Name_ori)
                # 重命名文件
                shutil.move(csv_file_Name_Change, csv_file_Name_ori)
                break


# 运行程序--标牌+电子眼
def Work2(sOpenFilePath):
    # Matplotlib elements used to draw the bounding rectangle
    import matplotlib.pyplot as plt
    # import numpy as np
    # caffe-ssd
    import math
    from google.protobuf import text_format
    import caffe
    from caffe.proto import caffe_pb2
    # matplotlib inline
    plt.rcParams['image.interpolation'] = 'nearest'

    net1 = loadmodel_bp(caffe, gpunum)

    # 读取电子眼
    net2 = load_net("./cfg/yolov3-voc.cfg", "./backup/yolov3-voc_v2_21000.weights", 0)
    meta2 = load_meta("./cfg/voc.data")

    labelmap_file = "{}/labelmap.prototxt".format(run_dir)
    file = open(labelmap_file, 'r')
    labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(file.read()), labelmap)

    # 图像根目录
    data_dir = sOpenFilePath[len(sOpenFilePath) - 1].decode('gbk')

    # 自动搜索含有图像的文件夹路径
    num_total = 0
    g = os.walk(data_dir)
    for path, d, filelist in g:
        for filename in filelist:
            if filename.endswith('csv'):
                Isuandian = 0
                savedrownum = 0
                # # 确定选择的文件是原始的轨迹FIT文件
                if filename.find('BP_DZY') >= 0 or filename.find('BP') >= 0 or filename.find('DZY') >= 0 or filename.find('CHANGE') >= 0:
                    continue

                # 判断是否正常处理完
                # RootDir=os.path.dirname(path) ##获取上一层目录
                RootDir = path
                csv_file_Name_Change = RootDir + "//" + filename[:-4] + "_CHANGE.csv"
                if not os.path.exists(csv_file_Name_Change):  # 未处理过
                    IsDuandian = 0
                else:  # 之前处理过，需断点续处理
                    IsDuandian = 1

                # 获取图像路径
                tmp = path.split("\\")

                ##获取上一层目录
                # # RootDir=os.path.dirname(path)
                # RootDir = path

                # create csv-save file
                csv_file_Name = RootDir + "//" + filename[:-4] + "_BP_DZY.csv"
                if not os.path.exists(csv_file_Name):
                    csv_reader_file1 = open(csv_file_Name, 'wb')
                    csv_writer = csv.writer(csv_reader_file1)
                else:  # 之前处理过
                    print("The file contains the trajectory that has been run before(" + filename + "),")
                    print("But I have to do it! ")
                    csv_reader_file1 = open(csv_file_Name, 'ab')
                    csv_writer = csv.writer(csv_reader_file1)
                    # continue

                # create csv-save file for original csv file change
                csv_file_Name_Change = RootDir + "//" + filename[:-4] + "_CHANGE.csv"
                if not os.path.exists(csv_file_Name_Change):
                    csv_reader_file2 = open(csv_file_Name_Change, 'wb')
                    csv_writer_Change = csv.writer(csv_reader_file2)
                else:  # # 之前处理过，需断点续处理
                    print("The file contains the trajectory that has been run before(" + filename + "),")
                    print("But I have to do it! ")
                    # 获取断点
                    csv_reader_file_tmp = open(csv_file_Name_Change, 'r')
                    csv_reader_tmp = csv.reader(csv_reader_file_tmp)
                    savedrownum = sum(1 for row in csv_reader_tmp)
                    # 删除最后一行（防止半行）
                    # for row in csv_reader_file_tmp:
                    #     del row[savedrownum]
                    # savedrownum = savedrownum - 1
                    csv_reader_file_tmp.close()

                    csv_reader_file2 = open(csv_file_Name_Change, 'ab')
                    csv_writer_Change = csv.writer(csv_reader_file2)

                    # continue

                # 获取csv文件路径
                csv_file_Name_ori = RootDir + "//" + filename
                # 获取图像路径
                images_path = csv_file_Name_ori[:-7] + "IMG"
                if not os.path.exists(images_path):
                    print("The corresponding image(" + filename + ")file could not be found,please check it later!")
                    continue

                # 读取csv文件获取图像名称及点位
                image_ids = os.listdir(images_path)
                csv_reader_file3 = open(csv_file_Name_ori, 'r')
                csv_reader = csv.reader(csv_reader_file3)
                data_tmp = "tmp.jpg"
                objectnum = 0
                crop_size = 512
                idx = 0

                # 遍历路径开始执行程序
                #################遍历路径开始执行程序
                for line in csv_reader:
                    # print(line)
                    objectnum = 0
                    objectnum1 = 0
                    image_id = line[0]
                    idx = idx + 1
                    num_total = num_total + 1

                    # 断点处理
                    if idx <= savedrownum and IsDuandian is 1:
                        continue

                    # 保存第一行
                    if idx is 1 and not os.path.exists("{}/{}".format(images_path, image_id)):
                        csv_writer.writerow(line)

                    # 保存原始文件
                    if not os.path.exists("{}/{}".format(images_path, image_id)):
                        csv_writer_Change.writerow(line)

                    print("current image num of total: {}".format(num_total))
                    if not os.path.exists("{}/{}".format(images_path, image_id)):
                        continue
                    image = cv2.imread("{}/{}".format(images_path, image_id))
                    try:
                        image.shape
                    except:
                        print "Image {} is damaged image,skipped...".format(image_id)
                        continue

                    print "Image {}:".format(image_id)
                    im_save = cv2.imread("{}/{}".format(images_path, image_id))
                    print "current filepath:{}".format(path)
                    print "current image:{}/total imgaes:{}".format(idx, len(image_ids))

                    # 标牌检测部分
                    img = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
                    image1 = img / 255.0
                    # image_resize_width = image1.shape[1]
                    # image_resize_height = image1.shape[0]
                    if image.shape[1] > 3000:      #modified 20190808
                        tmp111 = int(image.shape[1] / 3000) + 1
                        image_resize_width = int(image.shape[1] / tmp111)
                        image_resize_height = int(image.shape[0] / tmp111)
                    else:
                        image_resize_width = image.shape[1]
                        image_resize_height = image.shape[0]

                    transformed_image = transformer_process(caffe, net1, image1,
                                                            image_resize_width, image_resize_height)#exchange w and h 20190808
                    net1.blobs['data'].data[...] = transformed_image

                    # Forward pass.
                    start = time.time()
                    detections = net1.forward()['detection_out']
                    elapsed = time.time() - start
                    print "Label Inference Time: {} ms".format(int(1000 * elapsed))

                    # Parse the outputs.
                    det_label = detections[0, 0, :, 1]
                    det_conf = detections[0, 0, :, 2]
                    det_xmin = detections[0, 0, :, 3]
                    det_ymin = detections[0, 0, :, 4]
                    det_xmax = detections[0, 0, :, 5]
                    det_ymax = detections[0, 0, :, 6]

                    threshold = thresh_bp

                    top_indices = [i for i, conf in enumerate(det_conf) if conf >= threshold]

                    top_conf = det_conf[top_indices]
                    top_label_indices = det_label[top_indices].tolist()
                    top_labels = get_labelname(labelmap, top_label_indices)
                    top_xmin = det_xmin[top_indices]
                    top_ymin = det_ymin[top_indices]
                    top_xmax = det_xmax[top_indices]
                    top_ymax = det_ymax[top_indices]

                    for i in xrange(top_conf.shape[0]):
                        xmin = int(round(top_xmin[i] * image1.shape[1]))
                        ymin = int(round(top_ymin[i] * image1.shape[0]))
                        xmax = int(round(top_xmax[i] * image1.shape[1]))
                        ymax = int(round(top_ymax[i] * image1.shape[0]))
                        score = top_conf[i]
                        label = int(top_label_indices[i])
                        label_name = top_labels[i]
                        coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1

                        scale = float(image1.shape[1]) / float(image_resize_width)
                        scale = 1
                        width = int((xmax - xmin) / scale)
                        height = int((ymax - ymin) / scale)
                        xoff = int(xmin / scale)
                        yoff = int(ymin / scale)
                        display_txt = "\t{}\t[{}x{} @ {},{}]\t{}".format(label_name, width, height, xoff, yoff,
                                                                         "%.2f" % score)
                        # print(display_txt)

                        # labeling denoise
                        # if (yoff < image1.shape[0] / 4) or (yoff > image1.shape[0] - 50):
                        #     continue
                        # if width/height<1.5 or width/height>4:
                        #    continue

                        # 显示数据
                        if (yoff >= 0):
                            if yoff < 3:
                                yoff = 3
                            cv2.rectangle(im_save, (int(xoff), int(yoff)), (int(
                                xoff + width), int(yoff + height)), (0, 0, 255), 3)
                            display_txt = "\t{}\t[{}x{} @ {},{}]\t{}".format(
                                label_name, width, height, xoff, yoff, "%.2f" % score)
                            print display_txt
                            objectnum = objectnum + 1
                        label_name1 = "{}:{}".format(label_name, "%.2f" % score)
                        if (yoff > 10):
                            cv2.putText(im_save, label_name1, (int(xoff), int(
                                yoff - 6)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0))
                        else:
                            cv2.putText(im_save, label_name1, (int(xoff), int(
                                yoff + 15)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0))
                    ##标牌检测部分end

                    # 电子眼检测部分start
                    # subimage
                    y0 = 0
                    x0 = 0
                    time1 = 0.0
                    while (x0 + crop_size < int(image.shape[0] * 2 / 3)):
                        y0 = 0
                        while (y0 + crop_size < image.shape[1]):
                            subimage = image[x0:x0 + crop_size, y0:y0 + crop_size]
                            cv2.imwrite(data_tmp, subimage)
                            start = time.time()
                            r = detect(net2, meta2, data_tmp)
                            elapsed = time.time() - start
                            ##################################
                            for i in xrange(len(r)):
                                xmin = y0 + int(round(r[i][2][0] - r[i][2][2] / 2))
                                ymin = x0 + int(round(r[i][2][1] - r[i][2][3] / 2))
                                xmax = y0 + int(round(r[i][2][0] + r[i][2][2] / 2))
                                ymax = x0 + int(round(r[i][2][1] + r[i][2][3] / 2))
                                score = r[i][1]
                                label_name = r[0][0]

                                scale = 1
                                width = int((xmax - xmin) / scale)
                                height = int((ymax - ymin) / scale)
                                xoff = int(xmin / scale)
                                yoff = int(ymin / scale)

                                # labeling
                                if (yoff > image.shape[0] / 1) or width > 200 or height > 200:
                                    continue
                                if yoff > image.shape[0] * 3 / 4:
                                    continue
                                if yoff > 0:
                                    if yoff < 3:
                                        yoff = 3
                                    cv2.rectangle(im_save, (int(xoff), int(yoff)), (int(
                                        xoff + width), int(yoff + height)), (0, 0, 255), 3)
                                    display_txt = "\t{}\t[{}x{} @ {},{}]\t{}".format(
                                        label_name, width, height, xoff, yoff, "%.2f" % score)
                                    print display_txt
                                    objectnum1 = objectnum1 + 1
                                label_name1 = "{}:{}".format(label_name, "%.2f" % score)
                                if (yoff > 10):
                                    cv2.putText(im_save, label_name1, (int(xoff), int(
                                        yoff - 6)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0))
                                else:
                                    cv2.putText(im_save, label_name1, (int(xoff), int(
                                        yoff + 15)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0))
                            ##################################
                            y0 = y0 + int(crop_size / 100) * int(100 / 1.3)
                            time1 = time1 + elapsed
                        x0 = x0 + int(crop_size / 100) * int(100 / 1.3)
                    print "Roadcamera Inference Time: {} ms".format(int(1000 * time1))
                    print
                    ##电子眼检测部分end

                    ##show current result
                    # cv2.namedWindow("img", 2)
                    # cv2.imshow("img", image)
                    # cv2.waitKey(1)

                    # tmp1=os.path.dirname(path)  #上一级目录
                    tmp1 = path

                    # notargetresult="{}/{}".format(tmp1,"noTargeImages")
                    # if not os.path.exists(notargetresult):
                    # 		os.mkdir(notargetresult)

                    # checkresult = "{}/{}".format(tmp1, "checkImages")
                    # if not os.path.exists(checkresult):
                    # 	os.mkdir(checkresult)

                    # jianceresult = "{}/{}".format(checkresult, "location")
                    # if not os.path.exists(jianceresult):
                    # 	os.mkdir(jianceresult)

                    ##建立保存路径并保存
                    imgSaveFileName = filename[:-7] + "BP_DZY"
                    checkimgpath = "{}/{}".format(tmp1, imgSaveFileName)
                    if not os.path.exists(checkimgpath):
                        os.mkdir(checkimgpath)

                    if objectnum > 0 or objectnum1 > 0:
                        cv2.imwrite("{}/{}".format(checkimgpath, image_id), im_save)
                        # 保存csv
                        # extractLable=line.split(",")
                        extractLable = line
                        if len(extractLable) > 4:
                            if objectnum > 0 and objectnum1 > 0:
                                extractLable[4] = '1|2'  # 无目标：0；电子眼：1；标牌：2；电子眼+标牌：1|2
                            if objectnum > 0 and objectnum1 is 0:
                                extractLable[4] = '1'
                            if objectnum is 0 and objectnum1 > 0:
                                extractLable[4] = '2'
                        csv_writer.writerow(extractLable)
                        csv_reader_file1.flush()
                        csv_writer_Change.writerow(extractLable)
                        csv_reader_file2.flush()
                    else:  # 保存无目标图像
                        # cv2.imwrite("{}/{}".format(checkimgpath, image_id), im_save)
                        shutil.copy("{}/{}".format(images_path, image_id), "{}/{}".format(checkimgpath, image_id))
                        # #保存ori cvs文件change
                        # csv_writer_Change.writerow(line)
                        extractLable = line
                        if len(extractLable) > 4:
                            extractLable[4] = '0'  # 如果存在之前结果，抹去
                        csv_writer_Change.writerow(extractLable)
                        csv_reader_file2.flush()

                # 更改原始csv文件
                # 删除文件
                csv_reader_file1.close()
                csv_reader_file2.close()
                csv_reader_file3.close()
                os.remove(csv_file_Name_ori)
                # 重命名文件
                shutil.move(csv_file_Name_Change, csv_file_Name_ori)
                break


def Work(sOpenFilePath):
    if v.get() is 0 or len(sOpenFilePath) is 0:
        return
    if v.get() is 1:
        Work1(sOpenFilePath)
    if v.get() is 2:
        Work0(sOpenFilePath)
    if v.get() is 3:
        Work2(sOpenFilePath)


if __name__ == "__main__":
    # lib = CDLL("D:/label_dzy-checktool/yolo_cpp_dll.dll", RTLD_GLOBAL)
    # os.chdir('D:\\darknet-master\\darknet-master\\dabao\\cameracheck_yolov3_windows_nogpu_20181210\\')
    # os.chdir('D:\\tmp-label-dzy-check-cxfreeze20190530\\')

    lib = CDLL("./yolo_cpp_dll.dll", RTLD_GLOBAL)
    #
    ###########################
    # lib=winDLL("./yolo_cpp_dll.dll")
    ###########################
    # lib = CDLL("./libdarknet.so", RTLD_GLOBAL)

    lib.network_width.argtypes = [c_void_p]
    lib.network_width.restype = c_int
    lib.network_height.argtypes = [c_void_p]
    lib.network_height.restype = c_int

    predict = lib.network_predict
    predict.argtypes = [c_void_p, POINTER(c_float)]
    predict.restype = POINTER(c_float)

    # 设置gpu号
    # 读取阈值
    set_gpu = lib.cuda_set_device(gpunum)

    make_image = lib.make_image
    make_image.argtypes = [c_int, c_int, c_int]
    make_image.restype = IMAGE

    get_network_boxes = lib.get_network_boxes
    get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int),
                                  c_int]
    get_network_boxes.restype = POINTER(DETECTION)

    make_network_boxes = lib.make_network_boxes
    make_network_boxes.argtypes = [c_void_p, c_float, POINTER(c_int)]
    make_network_boxes.restype = POINTER(DETECTION)

    free_detections = lib.free_detections
    free_detections.argtypes = [POINTER(DETECTION), c_int]

    free_ptrs = lib.free_ptrs
    free_ptrs.argtypes = [POINTER(c_void_p), c_int]

    network_predict = lib.network_predict
    network_predict.argtypes = [c_void_p, POINTER(c_float)]

    reset_rnn = lib.reset_rnn
    reset_rnn.argtypes = [c_void_p]

    load_net = lib.load_network
    load_net.argtypes = [c_char_p, c_char_p, c_int]
    load_net.restype = c_void_p

    do_nms_obj = lib.do_nms_obj
    do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

    do_nms_sort = lib.do_nms_sort
    do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

    free_image = lib.free_image
    free_image.argtypes = [IMAGE]

    letterbox_image = lib.letterbox_image
    letterbox_image.argtypes = [IMAGE, c_int, c_int]
    letterbox_image.restype = IMAGE

    load_meta = lib.get_metadata
    lib.get_metadata.argtypes = [c_char_p]
    lib.get_metadata.restype = METADATA

    load_image = lib.load_image_color
    load_image.argtypes = [c_char_p, c_int, c_int]
    load_image.restype = IMAGE

    rgbgr_image = lib.rgbgr_image
    rgbgr_image.argtypes = [IMAGE]

    predict_image = lib.network_predict_image
    predict_image.argtypes = [c_void_p, IMAGE]
    predict_image.restype = POINTER(c_float)

    sOpenFilePath = []
    # 初始化Tk()
    myWindow = Tk()

    # 设置标题
    myWindow.title('自动筛选工具')
    myWindow.geometry('350x120+500+200')
    myWindow.iconbitmap('./title.ico')

    # 单选框
    v = IntVar()
    Radiobutton(myWindow, text='标牌', variable=v, value=1).grid(row=0, column=0)
    Radiobutton(myWindow, text='电子眼', variable=v, value=2).grid(row=0, column=1)
    Radiobutton(myWindow, text='标牌+电子眼', variable=v, value=3).grid(row=0, column=2)

    # 复选框
    # CheckVar1 = IntVar()
    # CheckVar2 = IntVar()
    # C1 = Checkbutton(myWindow, text="电子眼", variable=CheckVar1, \
    #                  onvalue=1, offvalue=0, height=1, \
    #                  width=5).grid(row=0,column=0)
    # C2 = Checkbutton(myWindow, text="标牌", variable=CheckVar2, \
    #                  onvalue=1, offvalue=0, height=1, \
    #                  width=5).grid(row=0,column=1)

    # 标签控件布局
    Label(myWindow, text="gpu号:", width=5).grid(row=0, column=7)

    text1 = Text(myWindow, width=35, height=1)
    text1.grid(row=1, column=1, columnspan=8)

    # Entry控件布局
    e = 0
    entry1 = Entry(myWindow, state='normal', fg='green', textvariable=e, width=1)
    entry1.grid(row=0, column=8)
    f = open("./cfg/gpunumber.txt")
    line = f.readline()
    gpunum = int(line)
    entry1.insert(1, gpunum)
    entry1['state'] = 'disabled'

    b1 = Button(myWindow, text='打开目录>>', state='normal', command=lambda: openFilePath(sOpenFilePath)).grid(row=1,
                                                                                                           column=0,
                                                                                                           sticky=W,
                                                                                                           padx=5,
                                                                                                           pady=5)

    # Quit按钮退出；Run按钮打印计算结果
    Button(myWindow, text='退  出', width=10, height=1, state='normal', fg='green', command=myWindow.quit).grid(row=2,
                                                                                                              column=1,
                                                                                                              sticky=W,
                                                                                                              padx=5,
                                                                                                              pady=5)

    Button(myWindow, text='运  行', width=10, height=1, state='normal', fg='red',
           command=lambda: Work(sOpenFilePath)).grid(row=2, column=2, sticky=W, padx=5, pady=5)
    # Button(myWindow, text='运  行', width=10, height=1, state='normal', fg='red',
    #        command=lambda: thread_it(Work, sOpenFilePath)).grid(row=2, column=2, sticky=W, padx=5, pady=5)

    # 进入消息循环
    myWindow.mainloop()
