#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        
        #yolox_s
        self.depth = 0.33
        self.width = 0.50
        
        #yolox_m
        #self.depth = 0.67
        #self.width = 0.75
        
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        
        main_repo = '/home/ec2-user/SageMaker/chooch_vision_driftdet/'
        
        self.data_dir = main_repo + "dataset/driftdataset_5k_2k_no_night_dusk/images"
        # name of annotation file for training
        self.train_ann = main_repo + "dataset/driftdataset_5k_2k_no_night_dusk/coco_annots/bdd_train_coco.json"
        # name of annotation file for evaluation
        self.val_ann = main_repo + "dataset/driftdataset_5k_2k_no_night_dusk/coco_annots/bdd_test_coco.json"
        # name of annotation file for testing
        self.test_ann = main_repo + "dataset/driftdataset_5k_2k_no_night_dusk/coco_annots/bdd_drift_coco.json"
        self.num_classes = 10
        
        # ---------------- Original dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        # If your training process cost many memory, reduce this value.
        self.data_num_workers = 4
        # Actual multiscale ranges: [640 - 5 * 32, 640 + 5 * 32].
        # To disable multiscale training, set the value to 0.
        self.multiscale_range = 0
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)
        # dir of dataset images, if data_dir is None, this project will use `datasets` dir
        
         # --------------  Original training config --------------------- #
        # epoch number used for warmup
        self.warmup_epochs = 3
        # max training epoch
        # minimum learning rate during warmup
        self.warmup_lr = 0
        self.min_lr_ratio = 0.05
        # learning rate for one image. During training, lr will multiply batchsize.
        self.basic_lr_per_img = 0.01 / 64.0
        # name of LRScheduler
        self.scheduler = "yoloxwarmcos"
        # last #epoch to close augmention like mosaic
        self.no_aug_epochs = 20
        # apply EMA during training
        self.ema = True

        # weight decay of optimizer
        self.weight_decay = 5e-4
        # momentum of optimizer
        self.momentum = 0.9
        # log period in iter, for example,
        # if set to 1, user could see log every iteration.
        self.print_interval = 10
        # eval period in epoch, for example,
        # if set to 1, model will be evaluate after every epoch.
        self.eval_interval = 1
        # save history checkpoint or not.
        # If set to False, yolox will only save latest and best ckpt.
        self.save_history_ckpt = True
        
        
        
        self.test_conf = 0.001
        # nms threshold
        self.nmsthre = 0.65
        
        
        self.max_epoch = 100
        self.basic_lr_per_img = 0.002 / 32
        
        # uncomment to freeze backbone
        def get_model(self):
            from yolox.utils import freeze_module
            model = super().get_model()
            freeze_module(model.backbone.backbone)
            return model
