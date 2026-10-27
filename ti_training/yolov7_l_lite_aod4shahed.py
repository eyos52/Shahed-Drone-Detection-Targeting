# Finetune YOLOv7-l-lite (COCO-pretrained) on the 5-class AOD4 + Shahed set.
# Sized for one A100 on SMU M3 (partition a100, 7-day limit).
#
# Three traps this config handles, all found the hard way on Colab:
#
# 1. MultiImageMixDataset wraps the train CocoDataset for mosaic/mixup, so the
#    train override must go one level DEEPER than val/test. Put it at the wrong
#    depth and it silently does nothing.
# 2. The base schedule is a 5-epoch QuadraticWarmupLR plus a cosine with
#    T_max = max_epochs - num_last_epochs. On a short run that pins the LR near
#    1e-9 (observed: lr 3.9e-09, loss flat at 12.85) and can make T_max
#    negative. param_scheduler is replaced outright.
# 3. The base optim_wrapper literal captured lr=base_lr from its OWN namespace,
#    so redefining base_lr in a child does nothing. The LR must be set inside
#    optim_wrapper.optimizer.
_base_ = ['./yolov7_l_coco_lite.py']

classes = ('airplane', 'bird', 'drone', 'helicopter', 'shahed')
metainfo = dict(classes=classes)

# $SCRATCH/shahed on M3 -- scratch may be purged; checkpoints go to $WORK
data_root = '/lustre/scratch/client/users/muxing/shahed/tidl_dataset/'
load_from = '/lustre/scratch/client/users/muxing/shahed/yolov7_l_coco_lite_640x640.pth'

# A100 vs the T4 we measured on: batch 16 used 7.8 GB, so 32 is ~15 GB.
# num_workers 8 fixes the real bottleneck -- on Colab's 2 vCPUs data_time was
# 0.42 s of a 1.35 s step, i.e. 31% of training spent waiting on the loader
# (mosaic needs 4 image loads per sample).
max_epochs = 20
batch_size = 32
num_workers = 8

TRAIN_IMGS = 19119
ITERS = TRAIN_IMGS // batch_size            # 597 iters/epoch
TOTAL = ITERS * max_epochs                  # 11940
WARMUP = 300

model = dict(bbox_head=dict(num_classes=len(classes)))

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    dataset=dict(                           # MultiImageMixDataset
        dataset=dict(                       # the actual CocoDataset
            data_root=data_root,
            ann_file='annotations/instances_train.json',
            data_prefix=dict(img='train/'),
            metainfo=metainfo)))

val_dataloader = dict(
    batch_size=16, num_workers=num_workers,
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/instances_valid.json',
        data_prefix=dict(img='valid/'),
        metainfo=metainfo))

test_dataloader = dict(
    batch_size=16, num_workers=num_workers,
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='test/'),
        metainfo=metainfo))

val_evaluator = dict(ann_file=data_root + 'annotations/instances_valid.json',
                     classwise=True)
test_evaluator = dict(ann_file=data_root + 'annotations/instances_test.json',
                      classwise=True)

train_cfg = dict(max_epochs=max_epochs, val_interval=1)

# batch 16 @ lr 1e-3 took loss 20.4 -> 2.35 in one epoch, so 2e-3 at batch 32
# keeps the same ratio. YOLOv7's own recipe is SGD 0.01 at batch 64, which
# linear-scales to 5e-3 here, so this stays deliberately below that.
lr = 2.0e-3
optim_wrapper = dict(type='AmpOptimWrapper', optimizer=dict(lr=lr))

# a list REPLACES the base list rather than merging with it
param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False,
         begin=0, end=WARMUP),
    dict(type='CosineAnnealingLR', by_epoch=False, begin=WARMUP, end=TOTAL,
         T_max=TOTAL - WARMUP, eta_min=lr * 0.05),
]

default_hooks = dict(
    # keep the best-by-mAP checkpoint; the sbatch script evaluates that one
    checkpoint=dict(interval=1, max_keep_ckpts=3, save_best='coco/bbox_mAP',
                    rule='greater'),
    logger=dict(interval=50))
