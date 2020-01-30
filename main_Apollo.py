import keras_segmentation
import numpy as np
import cv2
import os
import glob
from tqdm import tqdm
from keras.preprocessing.image import img_to_array, load_img
from keras_segmentation import metrics

# from keras_segmentation.data_utils.data_loader import get_segmentation_arr


model = keras_segmentation.models.unet.vgg_unet(n_classes=12,  input_height=800, input_width=2400)

#Show the detail of model, and total parameters 
model.summary()

model.train(
    
    train_images =  "./dataset3/image/train",
    train_annotations = "./dataset3/label/train",
    
    val_images=None, 
    val_annotations=None,
    
    verify_dataset=False,
    checkpoints_path = "./path_to_checkpoints/" , 
    epochs = 1,
    batch_size = 1,
    
    validate=False , 
    val_batch_size=None , 
    auto_resume_checkpoint=False ,
    
    load_weights=None ,
    steps_per_epoch=2,
    optimizer_name='adadelta'
    )

all_prs = model.predict_multiple(
    
    inp_dir = "./dataset3/image/test",
    out_dir = "path_to_predictions/",
    checkpoints_path = "path_to_checkpoints/"
    )

def evaluate_camvid():
    # get seg (ground truth) and compare with pr (prediction)
    # Get seg
    segs_path = "./dataset3/label/test/"
    segs  =  glob.glob( os.path.join(segs_path,"*.png")  ) 

    #print(len(segs))

    imglabels = np.ndarray((len(segs),2710, 3384), dtype=np.uint8) 
    i=0
    for x in range(len(segs)):
        imgpath = segs[x]

        pic_name = imgpath.split('/')[-1]

        labelpath = "dataset3/label/" + pic_name.split('.')[0] + '.png'

        label = load_img(labelpath, grayscale=True, target_size=[2710, 3384]) # grayscale = False
        
        label = img_to_array(label)
        
        imglabels[i] = label[:,:,0]
        if i % 100 == 0:
            print('Creating testing images: {0}/{1} images'.format(i, len(segs)))
        i += 1
    #np.save('./segs_test.npy', imglabels)


    i=0
    orininal_w = 3384
    orininal_h = 2710
    n_classes = 12

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

    ious = np.array(ious)
    precisions = np.array( precisions )
    recalls = np.array(recalls)
    
    #print("Class wise IoU Class:{:d} / IoU:{:.2f}\n".format(11, np.mean(ious , axis=0 )[11]))

    #print("Class wise Precision Class:{:d} / Precision:{:.2f}\n".format(11, np.mean(precisions , axis=0 )[11]))

    #print("Class wise Recall Class:{:d} / Recall:{:.2f}\n".format(11, np.mean(recalls , axis=0 )[11]))
    
    print("Class wise IoU "  ,  np.mean(ious , axis=0 ))
    print("Total  IoU "  ,  np.mean(ious ))
    
    # print("Class wise IoU:{:.2f}\n".format(np.mean(ious , axis=0 )))

    # print("Class wise Precision:{:.2f}\n".format(np.mean(precisions , axis=0 )))
    
    # print("Class wise Recall:{:.2f}\n".format(np.mean(recalls , axis=0 )))
#
    

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
