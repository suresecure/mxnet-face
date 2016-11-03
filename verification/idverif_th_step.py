#!/usr/bin/env python
from __future__ import division

import argparse
import logging
import os
import sys

import cv2
import mxnet as mx
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score

from lightened_cnn import lightened_cnn_b_feature

import time

ctx = mx.cpu()

def load_pairs(pairs_path):
    print("...Reading pairs.")
    pairs = []
    with open(pairs_path, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    assert(len(pairs) == 6000)
    return np.array(pairs)

def load_exector(model_prefix, epoch, size):
    _, model_args, model_auxs = mx.model.load_checkpoint(model_prefix, epoch)
    symbol = lightened_cnn_b_feature()
    return symbol, model_args, model_auxs

def pairs_info(pair, suffix):
    if len(pair) == 3:
        name1 = "{}/{}_{}.{}".format(pair[0], pair[0], pair[1].zfill(4), suffix)
        name2 = "{}/{}_{}.{}".format(pair[0], pair[0], pair[2].zfill(4), suffix)
        same = 1
    elif len(pair) == 4:
        name1 = "{}/{}_{}.{}".format(pair[0], pair[0], pair[1].zfill(4), suffix)
        name2 = "{}/{}_{}.{}".format(pair[2], pair[2], pair[3].zfill(4), suffix)
        same = 0
    else:
        raise Exception(
            "Unexpected pair length: {}".format(len(pair)))
    return (name1, name2, same)

def read2img(root, name1, name2, size, ctx):
    pair_arr = np.zeros((2, 1, size, size), dtype=float)
    img1 = np.expand_dims(cv2.imread(os.path.join(root, name1), 0), axis=0)
    img2 = np.expand_dims(cv2.imread(os.path.join(root, name2), 0), axis=0)
    assert(img1.shape == img2.shape == (1, size, size))
    pair_arr[0][:] = img1/255.0
    pair_arr[1][:] = img2/255.0
    return pair_arr

def readimg(root, name, size, ctx):
    pair_arr = np.zeros((1, 1, size, size), dtype=float)
    img = np.expand_dims(cv2.imread(os.path.join(root, name), 0), axis=0)
    assert(img.shape == (1, size, size))
    pair_arr[0][:] = img/255.0
    return pair_arr

def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[3]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = accuracy_score(y_true, y_predict)
    return accuracy

def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold

def acc(idfile, normalfile):
    print("...Computing accuracy.")
    # thresholds = np.arange(-1.0, 1.0, 0.005)
    # accuracy = []
    # thd = []
    idinfos = []
    normalinfos = []
    distinfos = []
    with open(idfile, "r") as idf:
        for line in idf.readlines():
            lsplit = line.split()
            filename = lsplit[0]
            idnum = filename[:filename.find('-')]
            idfeat = np.fromstring(' '.join(lsplit[1:]), sep=' ')
            # print(idnum)
            idinfos.append((idnum, idfeat))

    with open(normalfile, "r") as f:
        for line in f.readlines():
            lsplit = line.split()
            filename = lsplit[0]
            idnum = filename[:filename.find('-')]
            picnum = filename[filename.find('-')+1:filename.find('.')]
            idfeat = np.fromstring(' '.join(lsplit[1:]), sep=' ')
            # print(idnum)
            # print(picnum)
            normalinfos.append((idnum, picnum, idfeat))

    with open('dis.txt', 'w') as f:
        for idinfo in idinfos:
            for ninfo in normalinfos:
                gt = idinfo[0] == ninfo[0]
                dist = np.asscalar(np.dot(idinfo[1], ninfo[2])/np.linalg.norm(idinfo[1])/np.linalg.norm(ninfo[2]))
                # f.write(str(gt)+' '+idinfo[0]+' '+ninfo[0]+' '+ninfo[1]+' '+str(dist)+'\n')
                distinfos.append((gt, dist, idinfo[0], ninfo[0], ninfo[1]))

    print "%d pairs in total"%(len(distinfos))
    print "-----------------------------"
    print('th_dist\twrong_reject\tp_wr\twrong_accept\tp_wa')

    for thd in np.arange(0.3, 0.7, 0.01):
        wrong_reject = 0
        wrong_accept = 0
        correct_accept = 0
        correct_reject = 0
        wrong_reject_ids = []
        if True:
            for ddinfo in distinfos:
                if ddinfo[0] and ddinfo[1]<thd:
                    wrong_reject+=1
                    wrong_reject_ids.append(ddinfo[2]+' '+ddinfo[4]+'\n')
                elif ddinfo[0] and ddinfo[1]>=thd:
                    correct_accept += 1
                elif ddinfo[1] and ddinfo[1] < thd:
                    correct_reject += 1
                elif ddinfo[1] and ddinfo[1]>=thd:
                    wrong_accept+=1
                    # print('wrong_accept:'+ddinfo[2]+' '+ddinfo[3]+' '+ddinfo[4])
            print(str(thd)+'\t'
                  +str(wrong_reject)+'\t'
                  +str(wrong_reject/(wrong_reject+correct_accept)*100) +'\t'
                  +str(wrong_accept)+'\t'
                  +str(wrong_accept/(wrong_accept+correct_reject)*100))
            # if(wrong_reject<=20):
                # print(str(thd)+'\t'+str(wrong_reject)+'\t'+str(wrong_accept))
                # print wrong_reject_ids



        # with open(normalfile, 'r') as normalf:
            # with open('dis.txt', 'r') as disf:
                # idinfo = idf.readline()
                # featid = idf
        # predicts = f.readlines()
        # predicts = np.array(map(lambda line:line.strip('\n').split(), predicts))
        # for idx, (train, test) in enumerate(folds):
            # logging.info("processing fold {}...".format(idx))
            # best_thresh = find_best_threshold(thresholds, predicts[train])
            # accuracy.append(eval_acc(best_thresh, predicts[test]))
            # thd.append(best_thresh)

            # # dis = np.dot(output[0], output[1])/np.linalg.norm(output[0])/np.linalg.norm(output[1])
            # # f.write(name1 + '\t' + name2 + '\t' + str(dis) + '\t' + str(same) + '\n')
    # return accuracy,thd

def get_predict_file(args):
    # assert(os.path.exists(args.lfw_align))
    # pairs = load_pairs(args.pairs)
    _, model_args, model_auxs = mx.model.load_checkpoint(args.model_prefix, args.epoch)
    symbol = lightened_cnn_b_feature()
    start = time.clock()
    print("processing id image")
    with open("id.txt", "w") as f:
        count_id = 0
        for fname in os.listdir(args.idimage_align):
            print(fname)
            model_args['data'] = mx.nd.array(readimg(args.idimage_align, fname, args.size, ctx), ctx)
            exector = symbol.bind(ctx, model_args ,args_grad=None, grad_req="null", aux_states=model_auxs)
            exector.forward(is_train=False)
            exector.outputs[0].wait_to_read()
            output = exector.outputs[0].asnumpy()
            f.write(fname + ' ')
            output.tofile(f, ' ')
            f.write('\n')
            count_id += 1

    print("processing normal image")

    with open("normal.txt", "w") as f:
        count_normal = 0
        for fname in os.listdir(args.normalimage_align):
            print(fname)
            model_args['data'] = mx.nd.array(readimg(args.normalimage_align, fname, args.size, ctx), ctx)
            exector = symbol.bind(ctx, model_args ,args_grad=None, grad_req="null", aux_states=model_auxs)
            exector.forward(is_train=False)
            exector.outputs[0].wait_to_read()
            output = exector.outputs[0].asnumpy()
            f.write(fname + ' ')
            output.tofile(f, ' ')
            f.write('\n')
            count_normal += 1

    end = time.clock()
    print("count_id: %d; count_normal: %d" % (count_id, count_normal))
    print("Predict time: %f s" % (end - start))
    # print("Predict time per image: %f ms" % (end - start)/(count_id + count_normal))

    # with open(args.predict_file, 'w') as f:
        # for pair in pairs:
            # name1, name2, same = pairs_info(pair, args.suffix)
            # logging.info("processing name1:{} <---> name2:{}".format(name1, name2))
            # model_args['data'] = mx.nd.array(read2img(args.lfw_align, name1, name2, args.size, ctx), ctx)
            # exector = symbol.bind(ctx, model_args ,args_grad=None, grad_req="null", aux_states=model_auxs)
            # exector.forward(is_train=False)
            # exector.outputs[0].wait_to_read()
            # output = exector.outputs[0].asnumpy()
            # dis = np.dot(output[0], output[1])/np.linalg.norm(output[0])/np.linalg.norm(output[1])
            # f.write(name1 + '\t' + name2 + '\t' + str(dis) + '\t' + str(same) + '\n')

def print_result():
    # accuracy, threshold = acc('id.txt', 'normal.txt')
    acc('id.txt', 'normal.txt')
    # logging.info("10-fold accuracy is:\n{}\n".format(accuracy))
    # logging.info("10-fold threshold is:\n{}\n".format(threshold))
    # logging.info("mean threshold is:%.4f\n", np.mean(threshold))
    # logging.info("mean is:%.4f, var is:%.4f", np.mean(accuracy), np.std(accuracy))

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--pairs', type=str, default="./pairs.txt",
                        # help='Location of the LFW pairs file from http://vis-www.cs.umass.edu/lfw/pairs.txt')
    parser.add_argument('--idimage-align', type=str,
                        help='The directory of img-align, which contains the aligned lfw images')
    parser.add_argument('--normalimage-align', type=str,
                        help='The directory of img-align, which contains the aligned lfw images')
    parser.add_argument('--suffix', type=str, default="jpg",
                        help='The type of image')
    parser.add_argument('--size', type=int, default=128,
                        help='the image size of lfw aligned image, only support squre size')
    parser.add_argument('--model-prefix', default='../model/lightened_cnn/lightened_cnn',
                        help='The trained model to get feature')
    parser.add_argument('--epoch', type=int, default=165,
                        help='The epoch number of model')
    # parser.add_argument('--predict-file', type=str, default='./predict.txt',
                        # help='The file which contains similarity distance of every pair image given in pairs.txt')
    args = parser.parse_args()
    logging.info(args)
    # if not os.path.isfile(args.pairs):
        # logging.info("Error: LFW pairs (--lfwPairs) file not found.")
        # logging.info("Download from http://vis-www.cs.umass.edu/lfw/pairs.txt.")
        # logging.info("Default location:", "./pairs.txt")
        # sys.exit(-1)
    # print("Loading embeddings done")
    # if not os.path.exists(args.lfw_align):
        # logging.info("Error: lfw dataset not aligned.")
        # logging.info("Please use ./utils/align_face.py to align lfw firstly")
        # sys.exit(-1)
    if not os.path.isfile('id.txt') or not os.path.isfile('normal.txt'):
        logging.info("begin generate the predict.txt.")
        get_predict_file(args)
        logging.info("predict.txt has benn generated")
    print_result()
# python idverif.py --idimage-align /home/mythxcq/work/lightenface/id-align/idpic --normalimage-align /home/mythxcq/work/lightenface/id-align/normalpic --model-prefix ../model/lightened_cnn/lightened_cnn --epoch 166
if __name__ == '__main__':
    main()
