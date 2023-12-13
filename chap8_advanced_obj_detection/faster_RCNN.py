#%%
import os
from torch_snippets import *
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import glob
from sklearn.model_selection import train_test_split


#%%
IMAGE_ROOT = 'images/images'
DF_RAW = df = pd.read_csv('df.csv')

#%% define indices corresponding to labels and targets
label2target = {l:t+1 for t,l in enumerate(DF_RAW['LabelName'].unique())}
label2target['background'] = 0
target2label = {t:l for l,t in label2target.items()}
background_class = label2target['background']
num_classes = len(label2target)

#%% define the func to preprocess image
def preprocess_image(img):
    img = torch.tensor(img).permute(2, 0, 1)
    return img.to(device).float()

#%% define dataset class - OpenDataset
class OpenDataset(Dataset):
    w, h = 224, 224
    def __init__(self, df, image_dir=IMAGE_ROOT):
        self.image_dir = image_dir
        self.files = glob.glob(self.image_dir + '/*')
        self.df = df
        self.image_infos = df.ImageID.unique()
        
    def __getitem__(self, ix):
        # load images and masks
        image_id = self.image_infos[ix]
        img_path = find(image_id, self.files)
        img = Image.open(img_path).convert('RGB')
        data = df[df['ImageID'] == image_id]
        labels = data['LabelName'].values.tolist()
        data = data[['XMin', 'YMin', 'XMax', 'YMax']].values
        # convert to absolute coordinates
        data[:, [0,2]] *= self.w
        data[:, [1,3]] *= self.h
        boxes = data.astype(np.uint32).tolist()
        # torch FRCNN EXPECTS ground truths as a dict of tensors
        target = {}
        target['boxes'] = torch.Tensor(boxes).float()
        target['labels'] = torch.Tensor([label2target[i] for i in labels]).long()
        img = preprocess_image(img)
        return img, target
    
    def collate_fn(self, batch):
        return tuple(zip(*batch))
    
    def __len__(self):
        return len(self.image_infos)
        

#%% create training and validation dataloaders and datasets
trn_ids, val_ids = train_test_split(df.ImageID.unique(),
                                    test_size=0.1, random_state=99
                                    )
trn_df, val_df = df[[df['ImageID'].isin(trn_ids)], \
                    df[df['ImageID'].isin(val_ids)]
                    ]

train_ds = OpenDataset(trn_df)
test_ds = OpenDataset(val_df)

train_loader = DataLoader(train_ds, batch_size=4, collate_fn=train_ds.collate_fn,
                          drop_last=True
                          )
test_loader = DataLoader(test_ds, batch_size=4, collate_fn=test_ds.collate_fn,
                         drop_last=True
                         )


#%% define model
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_model():
    model = torchvision.models.detection\
                .fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


#%% define training and validation
def train_batch(inputs,model, optimizer):
    model.train()
    input, targets = inputs
    input = list(image.to(device) for image in input)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    optimizer.zero_grad()
    losses = model(input, targets)
    loss = sum(loss for loss in losses.values())
    loss.backward()
    optimizer.step()
    return loss, losses


#%%
@torch.no_grad()
def validate_batch(inputs, model):
    model.train()
    # to obtain losses, model needs to be in train mode only
    # note here we aint defining the model forward method 
    # hence need to work per the way model class is defined
    input, targets = inputs
    input = list(image.to(device) for image in input)
    targets = [{k: v.to(device) for k, v in t} for t in targets]
    optimizer.zero_grad()
    losses = model(input, targets)
    loss = sum(loss for loss in losses.values())
    return loss, losses


#%% initialize the model
model = get_model().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005,
                            momentum=0.9, weight_decay=0.0005
                            )
n_epochs = 5
log = Report(n_epochs)

#%% train model and cal loss values on training and test datasets
for epoch in range(n_epochs):
    _n = len(train_loader)
    for ix, inputs in enumerate(train_loader):
        loss, losses = train_batch(inputs, model, optimizer)
        loc_loss, regr_loss, loss_objectness, loss_rpn_box_reg = \
            [losses[k] for k in ['loss_classifier', 'loss_box_reg',
                                 'loss_objectness', 'loss_rpn_box_reg'
                                 ]
             ]
        pos = (epoch + (ix+1)/_n)
        log.record(pos, trn_loss=loss.item(),
                   trn_loc_loss=loc_loss.item(),
                   trn_regr_loss=regr_loss.item(),
                   trn_objectness_loss=loss_objectness.item(),
                   trn_rpn_box_reg_loss=loss_rpn_box_reg.item(),
                   end='\r'
                   )
    
    _n = len(test_loader)
    for ix, inputs in enumerate(test_loader):
        loss, losses = validate_batch(inputs, model)
        loc_loss, regr_loss, loss_objectness, loss_rpn_box_reg = \
            [losses[k] for k in ['loss_classifier', 'loss_box_reg',
                                 'loss_objectness', 'loss_rpn_box_reg'
                                 ]
             ]
        pos = (epoch + (ix+1)/_n)
        log.record(pos, val_loss=loss.item(),
                   val_loc_loss=loc_loss.item(),
                   val_regr_loss=regr_loss.item(),
                   val_objectness_loss=loss_objectness.item(),
                   val_rpn_box_reg_loss=loss_rpn_box_reg.item(),
                   end='\r'
                   )
    if (epoch+1)%(n_epochs//5)==0: log.report_avgs(epoch+1)
    
    
#%% plot variation of various loss values over increasing epochs    
log.plot_epochs(['trn_loss', 'val_loss'])


#%% predict on anew image
from torchvision.ops import nms

def decode_output(output):
    bbs = output['boxes'].cpu().detach().numpy().astype(np.uint16)
    labels = np.array([target2label[i] for i in output['labels'].cpu().detach().numpy()])
    confs = output['scores'].cpu().detach().numpy()
    ixs = nms(torch.tensor(bbs.astype(np.float32)),
              torch.tensor(confs), 0.05
              )
    if len(ixs) == 1:
        bbs, confs, labels = [np.array([tensor]) for tensor in [bbs, confs,labels]]
    return bbs.tolist(), confs.tolist(), labels.tolist()

# fetch the predictions of the boxes and classes on test imahes
model.eval()
for ix, (images, targets) in enumerate(test_loader):
    if ix == 3: break
    images = [im for im in images]
    outputs = model(images)
    for ix, output in enumerate(outputs):
        bbs, confs, labels, = decode_output(output)
        info = [f'{1}@{c:.2f}' for l, c in zip(labels, confs)]
        show(images[ix].cpu().permute(1,2,0), 
             bbs=bbs, texts=labels, sz=5
            )







