import os
import sys
from threading import Thread
from queue import Queue

import cv2
import numpy as np

import torch
import torch.multiprocessing as mp

from alphapose.utils.presets import SimpleTransform
from alphapose.models import builder

class DetectionLoader():
    def __init__(self, input_source, detector, cfg, opt, mode='image', batchSize=1, queueSize=128):
        self.cfg = cfg
        self.opt = opt
        self.mode = mode
        self.device = opt.device

        if mode == 'image':
            self.img_dir = opt.inputpath
            self.imglist = [os.path.join(self.img_dir, im_name.rstrip('\n').rstrip('\r')) for im_name in input_source]
            self.datalen = len(input_source)

        self.detector = detector
        self.batchSize = batchSize
        leftover = 0
        if (self.datalen) % batchSize:
            leftover = 1
        self.num_batches = self.datalen // batchSize + leftover

        self._input_size = cfg.DATA_PRESET.IMAGE_SIZE
        self._output_size = cfg.DATA_PRESET.HEATMAP_SIZE

        self._sigma = cfg.DATA_PRESET.SIGMA

        pose_dataset = builder.retrieve_dataset(self.cfg.DATASET.TRAIN)
        if cfg.DATA_PRESET.TYPE == 'simple':
            self.transformation = SimpleTransform(
                pose_dataset, scale_factor=0,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=0, sigma=self._sigma,
                train=False, add_dpg=False, gpu_device=self.device)

        # initialize the queue used to store data
        """
        image_queue: the buffer storing pre-processed images for object detection
        det_queue: the buffer storing human detection results
        pose_queue: the buffer storing post-processed cropped human image for pose estimation
        """

        self._stopped = mp.Value('b', False)
        self.image_queue = mp.Queue(maxsize=queueSize)
        self.det_queue = mp.Queue(maxsize=10 * queueSize)
        self.pose_queue = mp.Queue(maxsize=10 * queueSize)

    def start_worker(self, target):
        p = mp.Process(target=target, args=())
        p.start()
        return p

    def start(self):
        # start a thread to pre process images for object detection
        image_preprocess_worker = self.start_worker(self.image_preprocess)
        # start a thread to detect human in images
        image_detection_worker = self.start_worker(self.image_detection)
        # start a thread to post process cropped human image for pose estimation
        image_postprocess_worker = self.start_worker(self.image_postprocess)

        return [image_preprocess_worker, image_detection_worker, image_postprocess_worker]

    def stop(self):
        self.clear_queues()

    def terminate(self):
        if self.opt.sp:
            self._stopped = True
        else:
            self._stopped.value = True
        self.stop()

    def clear_queues(self):
        self.clear(self.image_queue)
        self.clear(self.det_queue)
        self.clear(self.pose_queue)

    def clear(self, queue):
        while not queue.empty():
            queue.get()

    def wait_and_put(self, queue, item):
        queue.put(item)

    def wait_and_get(self, queue):
        return queue.get()

    def image_preprocess(self):
        for i in range(self.num_batches):
            imgs, orig_imgs, im_names, im_dim_list = [], [], [], []
            for k in range(i * self.batchSize, min((i + 1) * self.batchSize, self.datalen)):
                
                if self.stopped:
                    self.wait_and_put(self.image_queue, (None, None, None, None))
                    return

                #current image
                im_name_k = self.imglist[k]

                # expected image shape like (1,3,h,w) or (3,h,w)
                img_k = self.detector.image_preprocess(im_name_k) #returns preprocessed image in the form of tensor or np array

                #if not tensor tranform to tensor
                if isinstance(img_k, np.ndarray):
                    img_k = torch.from_numpy(img_k)

                # add one dimension at the front for batch if image shape (3,h,w)
                if img_k.dim() == 3:
                    img_k = img_k.unsqueeze(0)

                #from BGR to RGB, to use both PIL and opencv
                orig_img_k = cv2.cvtColor(cv2.imread(im_name_k), cv2.COLOR_BGR2RGB) 
                
                #width and heith of images
                im_dim_list_k = orig_img_k.shape[1], orig_img_k.shape[0]

                imgs.append(img_k) #list of preprocessed images for object detection, tensor
                orig_imgs.append(orig_img_k) #list of images in RGB format
                im_names.append(os.path.basename(im_name_k)) #list of names of images
                im_dim_list.append(im_dim_list_k) #list of dimensions of images

            with torch.no_grad():
                # Human Detection
                imgs = torch.cat(imgs) #concatenates images
                im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

            self.wait_and_put(self.image_queue, (imgs, orig_imgs, im_names, im_dim_list))

    def image_detection(self):

        for i in range(self.num_batches):

            imgs, orig_imgs, im_names, im_dim_list = self.wait_and_get(self.image_queue)
            
            #if no images
            if imgs is None or self.stopped:
                self.wait_and_put(self.det_queue, (None, None, None, None, None, None, None))
                return

            with torch.no_grad():

                # repeat the first image multiple times to fill a batch
                for batch_i in range(self.batchSize - len(imgs)):
                    imgs = torch.cat((imgs, torch.unsqueeze(imgs[0], dim=0)), 0)
                    im_dim_list = torch.cat((im_dim_list, torch.unsqueeze(im_dim_list[0], dim=0)), 0)

                #detecting ogjects
                dets = self.detector.images_detection(imgs, im_dim_list)

                #if no objects
                if isinstance(dets, int) or dets.shape[0] == 0:
                    for k in range(len(orig_imgs)):
                        self.wait_and_put(self.det_queue, (orig_imgs[k], im_names[k], None, None, None, None, None))
                    continue

                #if not tensor transform to tensor
                if isinstance(dets, np.ndarray):
                    dets = torch.from_numpy(dets)
                
                boxes = dets[:, 1:5] #bounding box: center point coodinates, width and height
                scores = dets[:, 5:6] #confidence score
                ids = torch.zeros(scores.shape)

            for k in range(len(orig_imgs)):
                
                #bounding boxes from the kth image
                boxes_k = boxes[dets[:, 0] == k]

                #if there are no bounfing boxes
                if isinstance(boxes_k, int) or boxes_k.shape[0] == 0:
                    self.wait_and_put(self.det_queue, (orig_imgs[k], im_names[k], None, None, None, None, None))
                    continue
                

                inps = torch.zeros(boxes_k.size(0), 3, *self._input_size)
                cropped_boxes = torch.zeros(boxes_k.size(0), 4) #for storing the bounding boxes from current image

                self.wait_and_put(self.det_queue, (orig_imgs[k], im_names[k], boxes_k, scores[dets[:, 0] == k], ids[dets[:, 0] == k], inps, cropped_boxes))

    def image_postprocess(self):

        for i in range(self.datalen):

            with torch.no_grad():

                (orig_img, im_name, boxes, scores, ids, inps, cropped_boxes) = self.wait_and_get(self.det_queue)
                
                #if there is no image
                if orig_img is None or self.stopped:
                    self.wait_and_put(self.pose_queue, (None, None, None, None, None, None, None))
                    return
                #if the image has no detected objects
                if boxes is None or boxes.nelement() == 0:
                    self.wait_and_put(self.pose_queue, (None, orig_img, im_name, boxes, scores, ids, None))
                    continue

                # normalize data for human pose estimation
                for i, box in enumerate(boxes):
                    inps[i], cropped_box = self.transformation.test_transform(orig_img, box)
                    cropped_boxes[i] = torch.FloatTensor(cropped_box)

                self.wait_and_put(self.pose_queue, (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes))

    def read(self):
        return self.wait_and_get(self.pose_queue)

    @property
    def stopped(self):
        if self.opt.sp:
            return self._stopped
        else:
            return self._stopped.value

    @property
    def length(self):
        return self.datalen
