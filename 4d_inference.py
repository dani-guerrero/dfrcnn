import torch
import torchvision
import hostlist
import subprocess
import numpy as np
import cv2
from torch.utils.tensorboard import SummaryWriter
import os
import random
import csv
import sys
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from datetime import timedelta
import torch.multiprocessing as mp
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
import pickle

def valid_path(extension):
    e = extension.lower()
    if (e == '.jpg' or e=='.png' or e=='.jpeg'):
        return True
    else:
        return False

def collate_fn(batch):
    return tuple(zip(*batch))

class InferenceDataset(Dataset):
    def __init__(self, base_dir):

        self.image_ids = [os.path.join(dp, f) for dp, dn, filenames in os.walk(base_dir) for f in filenames if valid_path(os.path.splitext(f)[1])]


    def __getitem__(self, index: int):

        image_id = self.image_ids[index]

        return image_id

    def __len__(self) -> int:
        return len(self.image_ids)

#writer = SummaryWriter('../log')

#TOTAL_FILES  = 1500
#START_INDEX = 500

#PATH = 'model.pth'
#model = torch.load(PATH)
#model.eval()
#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#print('Using device:',device)
#model.to(device)
#basedir = '/gpfs/projects/nct01/nct01127/dataset/'
#species = os.listdir(basedir)
#id = 0

#if os.path.isfile('results.csv'):
#    csvfile = open('results.csv','a+')
#    csvwriter = csv.writer(csvfile)
#else:
#    csvfile = open('results.csv','w')
#    csvwriter = csv.writer(csvfile)
#    csvwriter.writerow(['file','label','class-label','proba','xtl','ytl','xbr','ybr'])
#
#itr = 0
#for specie in species:
#    if (specie!='Other-species'):
#        files=os.listdir(basedir+specie+'/best/')[START_INDEX:]
#        for imgid in files:
#            filepath = basedir + specie + '/best/' + imgid
#            itr +=1
#            img = cv2.imread(filepath, cv2.IMREAD_COLOR)
#            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
#            image_log = image
#            image /= 255.0 #normalization
#            image = torchvision.transforms.ToTensor()(image)
#            image = image.unsqueeze(1).to(device)
#            dictlist = model(image)
#            dict = dictlist[1]
#            keep = torchvision.ops.batched_nms(dict['boxes'], dict['scores'], dict['labels'], 0.1)
#            box = [round(a) for a in dict['boxes'][keep.tolist()[0]].tolist()]
#            score = dict['scores'].tolist()[0]
#            print(itr, specie, score)
#            csvwriter.writerow([filepath, 'mosquito', specie,score,box[0],box[1],box[2],box[3]])
#            if itr > TOTAL_FILES:
#                writer.close()
#                exit()
#writer.close()

def process(rank, world_size):

    base_dir = '/gpfs/projects/nct01/nct01127/dataset/'
    inference_dataset = InferenceDataset(base_dir)
    # os.environ['SLURM_PROCID'] = '0'
    # os.environ['SLURM_NTASKS'] = '1'
    # os.environ['SLURM_LOCALID'] = '0'
    # os.environ['SLURM_JOB_NODELIST'] = '0'
    # os.environ['SLURM_NTASKS'] = '1'
    # os.environ['SLURM_CPUS_PER_TASK'] = '40'
    # os.environ['SLURM_NTASKS_PER_NODE'] = '1'
    # os.environ['SLURM_NTASKS'] = '1'
    hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])
    os.environ['MASTER_ADDR'] = hostnames[0]
    os.environ['MASTER_PORT'] = '12350' 


    #rank = int(os.environ['SLURM_PROCID'])
    inference_sampler = torch.utils.data.distributed.DistributedSampler(inference_dataset,num_replicas=world_size,rank=rank, shuffle=False)

    inference_data_loader = DataLoader(
        inference_dataset,
        batch_size=1,
        num_workers=0,
        pin_memory = True,
        shuffle = False,
        collate_fn=collate_fn,
        sampler = inference_sampler
    )

    local_rank = int(os.environ['SLURM_LOCALID'])
    #world_size = int(os.environ['SLURM_NTASKS'])


    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    device = torch.device('cuda'+':'+str(local_rank)) if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 2  # 1 class (mosquito) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)




    device = torch.device('cuda'+':'+str(rank)) if torch.cuda.is_available() else torch.device('cpu')
    NODE_ID = int(os.environ['SLURM_NODEID'])
    hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']]).split()
     # only available if ntasks_per_node is defined in the slurm batch script
    NTASKS_PER_NODE = world_size #int(os.environ['SLURM_NTASKS_PER_NODE'])
    # get SLURM variables
    size = world_size #int(os.environ['SLURM_NTASKS'])
    cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    model.to(device)
    hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])
    os.environ['MASTER_ADDR'] = hostnames[0]
    os.environ['MASTER_PORT'] = '12350' 
    #world_size = int(os.environ['SLURM_NTASKS'])

    model.to(device)
    dist.init_process_group("nccl",timeout=timedelta(seconds=30), init_method='env://',rank=rank, world_size=world_size)
    ddp_model = DDP(model, device_ids=[rank]) 
    CHECKPOINT_PATH = '0.059609442949295044-adam-model.pth'
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location))

    ddp_model.eval()
    results = []
    itr = 0
    sampling_rate = 3000
    with torch.no_grad():
        for image in inference_data_loader:
            image_id = ''.join(map(lambda x: ''.join(x),image))
            print(image_id)
            image = cv2.imread(image_id, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image /= 255.0 #normalization
            #h,w,_ = image.shape
            image = torchvision.transforms.ToTensor()(image)
            image = image.unsqueeze(1).to(device)
            dictlist = ddp_model(image)
            for dict in dictlist:
                for key in dict:
                    dict[key]=dict[key].tolist()
                dict['file'] = image_id
            results.append(dictlist)
            itr += 1
            if (itr % sampling_rate == 0):
                print('['+str(rank)+']', 'itr', str(itr), ' saving file..')
                filename = 'pickle_files/'+str(rank) + '_' + str(itr) + '_' + 'results.pickle'
                with open(filename, 'wb') as handle:
                    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                results = []
            #dictlist = model(image)

def main():
    world_size = 4
    mp.spawn(process,
        args=(world_size,),
        nprocs=world_size,
        join=True)

if __name__== '__main__':
    main();