from utils import *
import inceptionv3
import tqdm
import time
import tensorflow as tf
import numpy as np
from albumentations import augmentations
from scipy.fftpack import dct, idct
import cv2
import random

# This file contains the defense methods compared in the paper.
# The FD algorithm's source code is from:
#   https://github.com/zihaoliu123/Feature-Distillation-DNN-Oriented-JPEG-Compression-Against-Adversarial-Examples/blob/master/utils/feature_distillation.py
# The FD algorithm is refer to the paper:
#   https://arxiv.org/pdf/1803.05787.pdf
# Some of the defense methods' code refering to Anish & Carlini's github: https://github.com/anishathalye/obfuscated-gradients

num = 8
q_table = np.ones((num,num))*30
q_table[0:4,0:4] = 25

def dct2 (block):
    return dct(dct(block.T, norm = 'ortho').T, norm = 'ortho')
def idct2(block):
    return idct(idct(block.T, norm = 'ortho').T, norm = 'ortho')

def defend_GD(img, distort_limit=0.025):
    height, width = img.shape[:2]
    num_steps = 10
 
    # Generate random distortion coefficients for x and y directions
    distort_coeffs_x = [1 + random.uniform(-distort_limit, distort_limit) for _ in range(width)]
    distort_coeffs_y = [1 + random.uniform(-distort_limit, distort_limit) for _ in range(height)]
 
    # Create meshgrid for remapping
    map_x = np.tile(np.arange(width), (height, 1))
    map_y = np.tile(np.arange(height).reshape(-1, 1), (1, width))
 
    # Apply distortion to meshgrid
    map_x_distorted = np.clip(map_x * np.array(distort_coeffs_x), 0, width - 1).astype(np.float32)
    map_y_distorted = np.clip(map_y * np.array(distort_coeffs_y), 0, height - 1).astype(np.float32)
 
    # Remap image using OpenCV's remap function
    outimg = cv2.remap(img, map1=map_x_distorted, map2=map_y_distorted, interpolation=cv2.INTER_LINEAR)
 
    return outimg

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)

# Feature distillation for single imput
def FD_fuction_sig(input_matrix):
    output = []
    input_matrix = input_matrix * 255

    h = input_matrix.shape[0]
    w = input_matrix.shape[1]
    c = input_matrix.shape[2]
    horizontal_blocks_num = w / num
    output2 = np.zeros((c, h, w))
    vertical_blocks_num = h / num

    c_block = np.split(input_matrix, c, axis=2)
    j = 0
    for ch_block in c_block:
        vertical_blocks = np.split(ch_block, vertical_blocks_num, axis=0)
        k = 0
        for block_ver in vertical_blocks:
            hor_blocks = np.split(block_ver, horizontal_blocks_num, axis=1)
            m = 0
            for block in hor_blocks:
                block = np.reshape(block, (num, num))
                block = dct2(block)
                # quantization
                table_quantized = np.matrix.round(np.divide(block, q_table))
                table_quantized = np.squeeze(np.asarray(table_quantized))
                # de-quantization
                table_unquantized = table_quantized * q_table
                IDCT_table = idct2(table_unquantized)
                if m == 0:
                    output = IDCT_table
                else:
                    output = np.concatenate((output, IDCT_table), axis=1)
                m = m + 1
            if k == 0:
                output1 = output
            else:
                output1 = np.concatenate((output1, output), axis=0)
            k = k + 1
        output2[j] = output1
        j = j + 1

    output2 = np.transpose(output2, (1, 0, 2))
    output2 = np.transpose(output2, (0, 2, 1))
    output2 = output2 / 255
    output2 = np.clip(np.float32(output2), 0.0, 1.0)
    return output2
def padresult_sig(cleandata):
    pad = augmentations.transforms.PadIfNeeded(min_height=304, min_width=304, border_mode=4)
    paddata = pad(image=cleandata)['image']
    return paddata
def cropresult_sig(paddata):
    crop = augmentations.transforms.Crop(0, 0, 299, 299)
    resultdata = crop(image=paddata)['image']
    return resultdata

def defend_FD_sig(data):
    paddata = padresult_sig(data)
    defendresult = FD_fuction_sig(paddata)
    resultdata = cropresult_sig(defendresult)
    return resultdata

# the seleted data from the imagenet validation set
cleandata = np.load(r"C:\Users\User\Desktop\Enhancement\data\clean100data.npy")
cleanlabel = np.load(r"C:\Users\User\Desktop\Enhancement\data\clean100label.npy")

orig = load_image('cat.jpg')
CORRECT_ori = 282 # tiger cat
TARGET_ori = 924 # guacamole

TARGET = 999 # toilet paper
sampleindex = 2

x = tf.placeholder(tf.float32, (299, 299, 3))
x_expanded = tf.expand_dims(x, axis=0)
logits, preds = inceptionv3.model(sess, x_expanded)

l2_x = tf.placeholder(tf.float32, (299, 299, 3))
l2_orig = tf.placeholder(tf.float32, (299, 299, 3))
normalized_l2_loss = tf.nn.l2_loss(l2_orig - l2_x) / tf.nn.l2_loss(l2_orig)

xent = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot(TARGET, 1000))
lam = tf.placeholder(tf.float32, ())
loss = xent + lam * normalized_l2_loss
grad, = tf.gradients(loss,x)

probs = tf.nn.softmax(logits)
classify = make_classify(sess, x, probs)

# demo based on a weak sample
# plot the original img and the defended version
classify(orig,correct_class=CORRECT_ori,target_class=TARGET_ori)
classify(defend_GD(defend_FD_sig(orig),0.25),correct_class=CORRECT_ori,target_class=TARGET_ori)

# 1.Using BPDA to attack the model with RDDfense as the protection
LR = 0.1
LAM =1.0

adv = np.copy(orig)
start = time.time()
for i in tqdm.tqdm(range(50)):
    fdedadv = defend_FD_sig(adv)
    adv_def = defend_GD(fdedadv)
    g, p = sess.run([grad, preds], {x: adv_def,lam: LAM, l2_x: adv, l2_orig: orig})
    print('step %d, pred=%d' % (i, p))
    adv -= LR * g
    adv = np.clip(adv, 0, 1)

end = time.time()
print('total time: ' + str(end - start))

classify(defend_GD(defend_FD_sig(adv)),correct_class=CORRECT_ori,target_class=TARGET_ori)

