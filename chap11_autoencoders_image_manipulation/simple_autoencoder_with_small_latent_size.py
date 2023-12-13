
#%%
from torch_snippets import *
from torchvision.datasets import MNIST
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

img_transform = transforms.Compose(
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
    transforms.Lambda(lambda x: x.to(device))
)


# create train and val datasets
trn_ds = MNIST(root="/content/", transform=img_transform,
                train=True, download=False
                )

val_ds = MNIST("/content/", transform=img_transform,train=False, download=True)

#%%
batch_size = 256
trn_dl = DataLoader(trn_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False)


class AutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latend_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(in_features=64, out_features=latent_dim)
            
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=64),
            nn.ReLU(True),
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=28*28),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = x.view(len(x), -1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(len(x), 1, 28, 28)
        return x
    
    
    
#%% visualize model
from torchsummary import summary
model = AutoEncoder(3).to(device)
summary(model, torch.zeros(2,1,28,28))


def train_batch(input, model, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, input)
    loss.backward()
    optimizer.step()
    return loss

@torch.no_grad()
def validate_batch(input, model, criterion):
    model.eval()
    output = model(input)
    loss = criterion(output, input)
    return loss

model = AutoEncoder(latent_dim=3).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=0.001, weight_decay=1e-5
                              )
 
 # train e model
num_epochs = 5
log = Report(num_epochs)

for epoch in range(num_epochs):
    N = len(trn_dl)
    for ix, (data, _) in enumerate(trn_dl):
        loss = train_batch(data, model, criterion, optimizer)
        log.record(pos=(epoch + (ix+1)/N),
                   trn_Loss=loss, end='\r'
                   )  
    N = len(val_dl)
    for ix, (data, _) in enumerate(val_dl):
        loss = validate_batch(data, model, criterion)
        log.record(pos=(epoch + (ix +1)/N),
                   val_loss=loss, end='\r'
                   )
    log.report_avgs(epoch+1)
    log.plot_epochs(log=True)
    
# validate the model on val_ds dataset not provided during traing
for _ in range(3):
    ix = np.random.randint(len(val_ds))
    im, _ = val_ds[ix]
    _im = model(im[None])[0]
    fig, ax = plt.subplots(1,2, figsize=(3,3))
    show(im[0], ax=ax[0], title='input')
    show(_im[0], ax=ax[1], title="prediction")
    plt.tight_layout()
    plt.show()
        
# train networks with diff bottleneck layer sizes 2,3,5,10,50


############# convolutional encoders ##############
#. determine device type
# 2. define imgae transform
# 3. create train and val datasets with imgae transform applied
# 4 define batch size and use to create dataloads for batches
# 5. define convultional encoder class

class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
                                        nn.Conv2d(in_channels=1, out_channels=32,
                                                kernel_size=3, stride=3,
                                                padding=1
                                                ),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2),
                                        nn.Conv2d(32, 64, 3, stride=2,
                                                padding=1),
                                        nn.ReLU(True),
                                        nn.MaxPool2d(kernel_size=2, stride=1)
                                        
                                    )
        
        self.decoder = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, stride=2),
                                     nn.ReLU(True),
                                     nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5,
                                                        stride=3, padding=1),
                                     nn.ReLU(True),
                                     nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=2,
                                                        stride=2, padding=1),
                                     nn.Tanh()
                                     )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    
# 6. get sammyr of model
summary(model, torch.zeros(2, 1, 28, 28))


# 7. define train_batch func
def train_batch(data, model, criterion, optimizer):
    model.train()
    model.zero_grad()
    output = model(data)
    loss = criterion(output, data)
    loss.backward()
    loss.step()
    return loss

#8. define validate_batch
def validate_batch(data, model, criterion):
    model.eval()
    output = model(data)
    loss = criterion(output, data)
    return loss

# 9. define model train parameters
#trn_dl = DataLoader(dataset='')
model = ConvAutoEncoder()
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(params=model.parameters(),
                              lr=0.001, weight_decay=1e-5)

# 10. train batches
n_epochs = 5
log = Report(n_epochs=n_epochs)

for epoch in range(n_epochs):
    for ix, (data, _) in enumerate(trn_dl):
        N = len(data)
        loss = train_batch(data=data, model=model, criterion=criterion,
                        optimizer=optimizer
                        )
        log.record(pos=(epoch + (ix+1)/N),
                   trn_loss=loss, end='\r'
                   )
        
    for ix, (data, _) in enumerate(val_dl):
        N = len(data)
        loss = validate_batch(data=data, model=model, criterion=criterion)
        log.record(pos=(epoch + (ix+1)/N),
                   val_loss=loss, end='\r'
                   )
    log.report_avgs(epoch+1)
    log.plot_epochs(log=True)
    
    
for _ in range(3):
    ix = np.random.randint(len(val_ds))
    im, _ = val_ds[ix]
    _im = model(im[None])[0]
    fig, ax = plt.subplots(1,2, figsize=(3,3))
    show(im[0], ax=ax[0], title='input')
    show(_im[0], ax=ax[1], title="prediction")
    plt.tight_layout()
    plt.show()    
    
    
    
######### grouping similar images using t-SNE ######
# 1. INITIALIZE LIST TO STORE LATENT VECTORS AND CLASSES OF IMAGES
latent_vectors = []
classes = []

# 2. loop through images in val dataloader and store output of encoder layer

for im, clss in val_dl:
    latent_vectors.append(model.encoder(im).view(len(im), -1))
    classes.extend(clss)
    
# 3. concatenate the numpy array of latent_vectors
latent_vectors = torch.cat(latent_vectors).cpu().detach().numpy()

# 4. import tsne
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)

# fit t-SNE by running the fit_transform
clustered = tsne.fit_transform(latent_vectors)

# 6. plot data after fitting
fig = plt.figure(figsize=(12,10))
cmap = plt.get_cmap('Spectral', 10)
plt.scatter(*zip(*clustered), c=classes, cmap=cmap)

plt.colorbar(drawedges=True)

 
########## variational AutoEncoders #########
# embeddings that do not fall into same clusters have 
# difficulty in generating realistic images
latent_vectors = []    
classes = []

for im, clss in val_dl:
    latent_vectors.append(model.encoder(im))
    classes.extend(clss)
latent_vectors = torch.cat(latent_vectors).cpu().detach().numpy().reshape(10000, -1)

# generate random vectors with col mean and standard deviation
# and add noise to standard deviation and create a vector
# from them
rand_vectors = []
for col in latent_vectors.transpose(1,0):
    mu, sigma = col.mean(), col.std()
    rand_vectors.append(sigma*torch.randn(1, 100) + mu)
    
# plot images reconstructed
rand_vectors = torch.cat(rand_vectors).transpose(1,0).to(device)
fig, ax = plt.subplots(10,10,  figsize=(7,7)); ax = iter(ax.flat)
for p in rand_vectors:
    img = model.decoder(p.reshape(1,64,2,2)).view(28,28)
    show(img, ax=next(ax))