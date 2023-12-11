import torch
from torch import nn
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset, DataLoader
import copy
import numpy as np

def get_generator_block(input_dim, output_dim):      #Generator Block
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True),
    )
class GANs_Generator_FNN(nn.Module):     #Generator Model

    def __init__(self, in_dim, out_dim, hidden_dim):
        super(GANs_Generator_FNN, self).__init__()
        self.generator = nn.Sequential(
            get_generator_block(in_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim * 2),
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            #get_generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 4, out_dim),
            nn.Sigmoid()
        )
    def forward(self, noise):
        return self.generator(noise)    
   
    def get_generator(self):
        return self.generator

def get_discriminator_block(input_dim, output_dim):       #Discriminator Block
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(0.2, inplace=True)        
    )
class GANs_Discriminator_FNN(nn.Module):         #Discriminator Model
    def __init__(self, in_dim, hidden_dim):
        super(GANs_Discriminator_FNN, self).__init__()
        self.discriminator = nn.Sequential(
            get_discriminator_block(in_dim, hidden_dim * 4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, image):
        return self.discriminator(image)
    
    def get_disc(self):
        return self.discriminator

def get_conv_generator_block(input_dim, output_dim, kernel_size):      #Generator Block
    return nn.Sequential(
        nn.ConvTranspose1d(input_dim, output_dim, kernel_size, 1, 0, bias=True),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True),
    )
class GANs_Generator_CNN(nn.Module):     #Generator Model

    def __init__(self, in_dim, out_dim, hid_dim):
        super(GANs_Generator_CNN, self).__init__()
        self.generator = nn.Sequential(
            # 2 1d cnn layers followed by 2 fully connected
            #get_conv_generator_block(in_dim, hid_dim * 8, 4),
            #get_conv_generator_block(hid_dim * 8, hid_dim * 4, 4),
            #get_conv_generator_block(hid_dim * 4, hid_dim * 2, 4),
            get_conv_generator_block(1, hid_dim * 8, 3),
            get_conv_generator_block(hid_dim * 8, hid_dim * 4, 3),
            #get_conv_generator_block(hid_dim * 4, hid_dim * 2, 3),
            nn.Flatten(),
            nn.LazyLinear(out_dim),
            nn.Sigmoid()
        )
    def forward(self, noise):
        return self.generator(noise)    
   
    def get_generator(self):
        return self.generator

def get_conv_discriminator_block(input_dim, output_dim, kernel_size):       #Discriminator Block
    return nn.Sequential(
        nn.Conv1d(input_dim, output_dim, kernel_size, 1, 1, bias=True),
        nn.LeakyReLU(0.2, inplace=True)        
    )
class GANs_Discriminator_CNN(nn.Module):         #Discriminator Model
    def __init__(self, in_dim, hid_dim):
        super(GANs_Discriminator_CNN, self).__init__()
        self.discriminator = nn.Sequential(
            # 2 1d cnn layers followed by 2 fully connected
            get_conv_discriminator_block(1, hid_dim * 4, 3),
            get_conv_discriminator_block(hid_dim * 4, hid_dim * 8, 3),
            #get_conv_discriminator_block(hid_dim * 4, hid_dim * 2, 3),
            #get_conv_discriminator_block(hid_dim * 8, hid_dim * 4, 4),
            #get_conv_discriminator_block(hid_dim * 4, hid_dim * 2, 4),
            nn.Flatten(),
            nn.LazyLinear(1)

        )

    def forward(self, image):
        return self.discriminator(image)
    
    def get_disc(self):
        return self.discriminator

def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples,z_dim,device=device) 


def cnn_get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device, sample_dim):

    fake_noise = get_noise(num_images, z_dim, device=device)
    fake_noise = torch.reshape(fake_noise, (num_images,1,z_dim))
    fake = gen(fake_noise)
    fake = torch.reshape(fake, (num_images,1,sample_dim))
    disc_fake_pred = disc(fake.detach())
    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
    real = torch.reshape(real, (num_images,1,sample_dim))
    disc_real_pred = disc(real)
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
    disc_loss = (disc_fake_loss + disc_real_loss) / 2

    return disc_loss

def cnn_get_gen_loss(gen, disc, criterion, num_images, z_dim, device, sample_dim):

    fake_noise = get_noise(num_images, z_dim, device=device)
    fake_noise = torch.reshape(fake_noise, (num_images,1,z_dim))
    fake = gen(fake_noise)
    fake = torch.reshape(fake, (num_images,1,sample_dim))
    disc_fake_pred = disc(fake)
    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
    return gen_loss

def fnn_get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device, sample_dim):

    fake_noise = get_noise(num_images, z_dim, device=device)
    #fake_noise = torch.reshape(fake_noise, (num_images,1,z_dim))
    fake = gen(fake_noise)
    fake = torch.reshape(fake, (num_images,1,sample_dim))
    disc_fake_pred = disc(fake.detach())
    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
    real = torch.reshape(real, (num_images,1,sample_dim))
    disc_real_pred = disc(real)
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
    disc_loss = (disc_fake_loss + disc_real_loss) / 2

    return disc_loss

def fnn_get_gen_loss(gen, disc, criterion, num_images, z_dim, device, sample_dim):

    fake_noise = get_noise(num_images, z_dim, device=device)
    #fake_noise = torch.reshape(fake_noise, (num_images,1,z_dim))
    fake = gen(fake_noise)
    fake = torch.reshape(fake, (num_images,1,sample_dim))
    disc_fake_pred = disc(fake)
    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
    return gen_loss

def buildGanCnnModel(z_dim, lr, sample_dim, device='cpu'):
    gen = GANs_Generator_CNN(z_dim,sample_dim,128).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc = GANs_Discriminator_CNN(sample_dim, 128).to(device) 
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
    return gen, gen_opt, disc, disc_opt

def buildGanFnnModel(z_dim, lr, sample_dim, device='cpu'):
    gen = GANs_Generator_FNN(z_dim,sample_dim,128).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc = GANs_Discriminator_FNN(sample_dim, 128).to(device) 
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
    return gen, gen_opt, disc, disc_opt

def getGANDataLoader(data, batch_size):
    my_dataset = TensorDataset(torch.Tensor(data))
    return DataLoader(my_dataset, batch_size=batch_size, shuffle=True)

def trainGAN(gen, gen_opt, disc, disc_opt, dataloader, gtype, n_epochs, z_dim, device, sample_dim, lr):
    criterion = nn.BCEWithLogitsLoss()  
    test_generator = False 
    gen_loss = False
    error = False
    display_step = 1
    gens = {}
    discs = {}
    cur_step = 0
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    for epoch in range(n_epochs):
        for real in tqdm(dataloader):
            real = real[0]
            cur_batch_size = len(real)
            #real is now a tensor (batch_size, N_STEPS_IN*NFVARS+N_STEPS_OUT)
            #real = real.view(cur_batch_size, -1).to(device)
            real = real.to(device)
                
            disc_opt.zero_grad()
            if gtype == "CNN":
                disc_loss = cnn_get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device, sample_dim)
            else:
                disc_loss = fnn_get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device, sample_dim)
            disc_loss.backward(retain_graph=True)
            disc_opt.step()

            if test_generator:
                old_generator_weights = gen.gen[0][0].weight.detach().clone()

            gen_opt.zero_grad()
            if gtype == "CNN":
                gen_loss = cnn_get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device, sample_dim)
            else:
                gen_loss = fnn_get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device, sample_dim)
            
            gen_loss.backward()
            gen_opt.step()

            if test_generator:
                try:
                    assert lr > 0.0000002 or (gen.gen[0][0].weight.grad.abs().max() < 0.0005 and epoch == 0)
                    assert torch.any(gen.gen[0][0].weight.detach().clone() != old_generator_weights)
                except:
                    error = True
                    print("Generator runtime tests have failed")

            mean_discriminator_loss += disc_loss.item() / display_step
            mean_generator_loss += gen_loss.item() / display_step

            #print(f"curr step is {cur_step}, display step is {display_step}")
            if cur_step % display_step == 0 and cur_step > 0:
                print(f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
                mean_generator_loss = 0
                mean_discriminator_loss = 0
            cur_step += 1
            if epoch % 10 == 0: #step used when training fnn_gen, idk about cnn
                gens[epoch] = copy.deepcopy(gen) #I think this is what we would be returning
                discs[epoch] = copy.deepcopy(disc)
    return gens, discs

def getGAN(extreme_data, gan_dir, gtype, sample_dim,  n_epochs, batch_size, log_print):
    if gan_dir:
        #load gen and desc from gan path
        #TODO: replace loaded with official name (with epoch label) extracted from gan_dir
        gen_path = gan_dir.joinpath('generator.pt')
        gens = {'loaded':torch.jit.load(gen_path)}
        log_print(f"Loaded generator from {gen_path}")
        disc_path = gan_dir.joinpath('discriminator.pt')
        discs = {'loaded':torch.jit.load(disc_path)}
        log_print(f"Loaded generator from {disc_path}")
    else:
        device = 'cpu'
        lr = 0.00001
        z_dim = 128
        if gtype == "CNN":
            gen, gen_opt, disc, disc_opt = buildGanCnnModel(z_dim, lr, sample_dim, device=device)
        else:
            gen, gen_opt, disc, disc_opt = buildGanFnnModel(z_dim, lr, sample_dim, device=device)
        
        dataloader = getGANDataLoader(extreme_data, batch_size)
        
        log_print(f"Training {gtype} GAN with {n_epochs} epochs")
        
        if gtype == "CNN":
            gens, discs = trainGAN(gen, gen_opt, disc, disc_opt, dataloader, gtype, n_epochs, z_dim, device, sample_dim, lr)
        else:
            gens, discs = trainGAN(gen, gen_opt, disc, disc_opt, dataloader, gtype, n_epochs, z_dim, device, sample_dim, lr)
        
        
    return gens, discs

def ganResample(k_x, num_rel, gen, gen_type, device='cpu'):
    first = False
    gan_samples = []
    i = num_rel
    while i > 0:
        num_samples = i
        if i >= 1000:
            num_samples = 1000
        fake_noise = get_noise(num_samples, 128, device=device)
        if gen_type == "CNN":
            fake_noise = torch.reshape(fake_noise, (num_samples,1,128))
        res2=gen(fake_noise)
        fres2=res2.cpu().detach().numpy()
        if not first:
            first = True
            gan_samples = fres2
        else:
            gan_samples = np.append(gan_samples, fres2, axis=0)
        i -= num_samples
    #np.savetxt('myarray.txt', gan_samples)
    #np.savetxt('orig.txt', k_x)
    return np.concatenate((k_x,gan_samples),axis=0)