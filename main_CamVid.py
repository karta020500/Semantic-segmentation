import keras_segmentation
import numpy as np
import cv2
import os
import glob
import tensorflow as tf

from tqdm import tqdm 
from keras.preprocessing.image import img_to_array, load_img
from keras_segmentation import metrics

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
# Check available GPU devices.
print("The following GPU devices are available: %s" % tf.test.gpu_device_name())

model = keras_segmentation.models.segnet.vgg_segnet(n_classes=32,  input_height=704, input_width=928  )#716 956


model.train(
    
    train_images =  "dataset2/image/train",
    train_annotations = "dataset2/label/train",
    
    val_images=None, 
    val_annotations=None,
    
    checkpoints_path = "./path_to_checkpoints/" , 
    epochs = 1000,
    batch_size = 1,
    
    validate=False , 
    val_batch_size=None , 
    auto_resume_checkpoint=False ,
    
    load_weights=None ,
    steps_per_epoch=300,
    optimizer_name='Adam'
    )

all_prs = model.predict_multiple(
    
    inp_dir = "dataset2/image/test",
    out_dir = "path_to_predictions/",
    checkpoints_path = "path_to_checkpoints/"
    )

def evaluate_camvid():
    # get seg (ground truth) and compare with pr (prediction)
    # Get seg
    segs_path = "dataset4/label/test"
    segs  =  glob.glob( os.path.join(segs_path,"*.png")  ) 

    #print(len(segs))

    imglabels = np.ndarray((len(segs),720, 960), dtype=np.uint8) 
    i=0
    for x in range(len(segs)):
        imgpath = segs[x]

        pic_name = imgpath.split('/')[-1]

        labelpath = "dataset2/label/" + pic_name.split('.')[0] + '.png'

        label = cv2.imread(labelpath)
        
        imglabels[i] = label[:,:,0]
        if i % 100 == 0:
            print('Creating testing images: {0}/{1} images'.format(i, len(segs)))
        i += 1
    #np.save('./segs_test.npy', imglabels)


    i=0
    orininal_w = 960
    orininal_h = 720
    n_classes = 32

    ious=[]
    precisions=[]
    recalls=[]

    for pr in tqdm(all_prs):
        gt = imglabels[i]
        pr = cv2.resize(pr, (orininal_w , orininal_h), interpolation=cv2.INTER_NEAREST)
        #IoU
        iou = metrics.get_iou( gt , pr , n_classes )
        ious.append( iou )
        
        # precision
        precision = metrics.get_precision( gt , pr , n_classes )
        precisions.append( precision )
        
        # recall
        recall = metrics.get_recall( gt , pr , n_classes )
        recalls.append( recall )
        
        
        i+=1

    ious = np.array( ious )
    precisions = np.array( precisions )
    recalls = np.array( recalls )

    print("Class wise IoU Class:{:d} / IoU:{:.2f}\n".format(11, np.mean(ious , axis=0 )[11]))

    print("Class wise Precision Class:{:d} / Precision:{:.2f}\n".format(11, np.mean(precisions , axis=0 )[11]))

    print("Class wise Recall Class:{:d} / Recall:{:.2f}\n".format(11, np.mean(recalls , axis=0 )[11]))
    
    # print("Class wise IoU Class:{:d} / IoU:{:.2f}\n".format(12, np.mean(ious , axis=0 )[12]))

    # print("Class wise Precision Class:{:d} / Precision:{:.2f}\n".format(12, np.mean(precisions , axis=0 )[12]))

    # print("Class wise Recall Class:{:d} / Recall:{:.2f}\n".format(12, np.mean(recalls , axis=0 )[12]))

evaluate_camvid()

'''
# dataset1 (prepared dataset, 12 calsses)
python -m keras_segmentation train --checkpoints_path="path_to_checkpoints" --train_images="dataset1/images_prepped_train/" --train_annotations="dataset1/annotations_prepped_train/"  --val_images="dataset1/images_prepped_test/" --val_annotations="dataset1/annotations_prepped_test/" --n_classes=12 --input_height=320 --input_width=640 --model_name="vgg_unet"
python -m keras_segmentation predict --checkpoints_path="./path_to_checkpoints/" --input_path="dataset2/image/test/" --output_path="path_to_predictions"
'''

'''
# dataset2 (CamVid, 32 classes)
python -m keras_segmentation train --checkpoints_path="path_to_checkpoints/" --train_images="dataset2/image/train/" --train_annotations="dataset2/trainId_label/train/"  --val_images="dataset2/image/test/" --val_annotations="dataset2/trainId_label/test/" --n_classes=32 --input_height=736 --input_width=960 --model_name="vgg_unet" --epochs=20
python -m keras_segmentation predict --checkpoints_path="./path_to_checkpoints/" --input_path="dataset2/image/test/" --output_path="path_to_predictions"
'''
