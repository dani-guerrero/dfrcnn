import pandas as pd
import numpy as np
import cv2
import os
import hostlist
import torch
import subprocess
import torchvision
from datetime import timedelta
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
#from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader, Dataset

from torch.utils.tensorboard import SummaryWriter

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist



class TrainDataset(Dataset):
    def __init__(self, annotation_file):

        self.train_df = pd.read_csv(annotation_file)
        self.image_ids = self.train_df['file'].unique()

    def __getitem__(self, index: int):

        image_id = self.image_ids[index]
        image = cv2.imread(image_id, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0 #normalization

        bboxes = self.train_df[self.train_df['file'] == image_id]

        #ensure boxes are not out of bounds
        h,w,_ = image.shape
        # bboxes['xtl'] = bboxes['xtl'].apply(lambda x: min(x,w))
        # bboxes['ytl'] = bboxes['xtl'].apply(lambda y: min(y,h))
        # bboxes['xbr'] = bboxes['xbr'].apply(lambda x: min(x,w))
        # bboxes['ybr'] = bboxes['ybr'].apply(lambda y: min(y,h))
        boxes = bboxes[['xtl','ytl','xbr','ybr']].values

        area = (boxes[:,3]-boxes[:,1]) * (boxes[:,2]-boxes[:,0])


        #convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = torch.as_tensor(area, dtype=torch.float32)
        labels = torch.ones((bboxes.shape[0],), dtype=torch.int64)
        iscrowd = torch.zeros((bboxes.shape[0],) ,dtype=torch.int64)

        target = {}
        target['boxes']=boxes
        target['labels']=labels
        target['image_id']=torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd
        #transform image as tensor
        image = torchvision.transforms.ToTensor()(image)
        return image, target

    def __len__(self) -> int:
        return self.image_ids.shape[0]


def collate_fn(batch):
    return tuple(zip(*batch))


import sys
if __name__ == '__main__':

    writer = SummaryWriter('../log')

    train_dataset = TrainDataset(sys.argv[1])

    rank = int(os.environ['SLURM_PROCID'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=int(os.environ['SLURM_NTASKS']),rank=rank)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory = True,
        collate_fn=collate_fn,
        sampler = train_sampler
    )

    local_rank = int(os.environ['SLURM_LOCALID'])
    #gpu_ids = os.environ['SLURM_STEP_GPUS'].split(",")
    device = torch.device('cuda'+':'+str(local_rank)) if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:',device)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 2  # 1 class (mosquito) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    #params = [p for p in model.parameters() if p.requires_grad]



    num_epochs = 100
 # get distributed configuration from Slurm environment
    NODE_ID = int(os.environ['SLURM_NODEID'])
    hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']]).split()
    MASTER_ADDR = hostnames[0].decode('utf-8')
     # only available if ntasks_per_node is defined in the slurm batch script
    NTASKS_PER_NODE = int(os.environ['SLURM_NTASKS_PER_NODE'])
    # get SLURM variables
    size = int(os.environ['SLURM_NTASKS'])
    cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])

 
    # define MASTER_ADD & MASTER_PORT
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = '12345'

    # display info
    if rank == 0: # print only on master
        print(f"Training on {len(hostnames)} nodes and {size} processes, master node is {MASTER_ADDR}")
        print("Demoing ddp + model pipeline !")
        print(f"Variables for model parallel on one node: {torch.cuda.device_count()} accessible gpus  "
              f"and {NTASKS_PER_NODE} tasks per node declared")


    local_rank = int(os.environ['SLURM_LOCALID'])
    print(local_rank)

    #dataiter = iter(train_data_loader)
    #images, labels = dataiter.next()
    #writer.add_graph(model, [images])
    model.to(device)
    hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])
    os.environ['MASTER_ADDR'] = hostnames[0]
    print(local_rank,os.environ['MASTER_ADDR'], hostnames)
    # get IDs of reserved GPU
    print('device', device)
    os.environ['MASTER_PORT'] = '12350' 
    print(local_rank,os.environ['MASTER_PORT'])
    # create default process group
    world_size = int(os.environ['SLURM_NTASKS'])
    print(local_rank,'world size', world_size)
    rank = int(os.environ['SLURM_PROCID'])
    print(local_rank,rank)
    dist.init_process_group("nccl",timeout=timedelta(seconds=30), init_method='env://',rank=rank, world_size=world_size)

    ddp_model = DDP(model, device_ids=[local_rank]) 
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.00001)
    #optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    itr=0
    min_loss = float('inf')

    print("Training started")
    for epoch in range(num_epochs):
        for images, targets in train_data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
            loss_dict = ddp_model(images,targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            itr +=1
            print(loss_value, 'epoch:', epoch,', itr:', itr)
            writer.add_scalar(" Adam 0.00001 " + str(world_size)+"GPUs Loss", loss_value, itr)
        writer.add_scalar(" Adam 0.00001 " + str(world_size)+"GPUs Epoch Loss", loss_value, itr)
        print(f"Epoch #{epoch} loss: {loss_value}")

        if (loss_value < min_loss and rank == 0):
            torch.save(ddp_model.state_dict(), str(loss_value) + '-adam-model.pth')
            print(f"loss decreased from {min_loss} to {loss_value}, saving adam-model.pth")
            min_loss = loss_value

        lr_scheduler.step()

    writer.close()

