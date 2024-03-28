# =============================================================================
# UEFL model with extensible codebook
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from model.basemodel import Mlp, CNNEncoder, vggEncoder

class VectorQuantizer(nn.Module):
    """
    Basic codebook (discrete VQ layer)
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self.embeddings = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self.embeddings.weight.data.normal_()
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return quantized.permute(0, 3, 1, 2).contiguous(), loss, perplexity
    
class VectorQuantizerEMA(nn.Module):
    """
    VQ layer with EMA (Exponential Moving Average) for updating codebook
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self.embeddings = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self.embeddings.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self.embeddings.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return quantized.permute(0, 3, 1, 2).contiguous(), loss, perplexity

class extVQ(nn.Module):
    """
    Extensible codebook:
        use codebooks based on the silo_kind:
            if silo_kind (book_index) = 0: only shared codebook
            else: shared codebook + additional codebook
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, silo_kinds):
        super(extVQ, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        # initialize a single codebook
        self.embed = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self.embed.weight.data.normal_()
        
        # initialize extensible codebook
        self.embeddings = [] # all codewords
        self.embeddings.append(self.embed) # shared codebook
        self.embeddings.append(nn.Embedding(self._num_embeddings, self._embedding_dim))  # append an additional backup codebook for Kmeans initialization
        self.codebooks = nn.ModuleList(self.embeddings) # extensible codebook

        self._commitment_cost = commitment_cost

    def forward(self, inputs, idx, ext=False):
        """
        idx: the index of data distribution
        """
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # extend codebook (append a new codebook for next Kmeans initialization)
        if ext:
            self.codebooks.append(nn.Embedding(self._num_embeddings, self._embedding_dim))

        # accessible codewords for different silos
        if idx == 0:
            # use only shared codebook
            codes = self.codebooks[0].weight
        else:
            # Calculate distances with shared codebook and additional codebook
            codes = torch.cat((self.codebooks[0].weight, self.codebooks[idx].weight), dim=0)
            
        # Calculate distances to accessible codewords
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True) +
            torch.sum(codes ** 2, dim=1) -
            2. * torch.matmul(flat_input, codes.t())
        )
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], codes.shape[0]).to(inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, codes).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return quantized.permute(0, 3, 1, 2).contiguous(), loss, perplexity
    
class UEFL(nn.Module):
    """
    Map latent features into codewords in an extensible codebook according to data distribution
    """
    def __init__(self, input_ch, dim, depth, num_codes, data, enc, silo_kinds, seg, ema=False):
        super().__init__()
        self.num_embeddings = num_codes
        self.dim = dim*2**(depth-1) # number of channels for encoded features of different datasets (i.e. codeword length)

        # number of channels after flatten
        ch = (dim*2**(depth-1))*(32//2**depth)**2 if data == "cifar10" else 9*512
        ch = ch if enc == "cnn" else 64*dim
        if enc == "cnn":
            self.encoder = CNNEncoder(input_ch, dim, depth)
            
        elif enc == "vgg":
            self.encoder = vggEncoder(input_ch, dim, depth)
            self.dim = ch = 512
        
        # segmented codes
        self.dim = self.dim//seg
        
        # entensible codebook
        self.discretizer = extVQ(num_embeddings=self.num_embeddings, embedding_dim=self.dim, commitment_cost=0.25, silo_kinds=silo_kinds)
        
        # number of classes for different datasets
        if data == "cifar100":
            output_dim = 100
        elif data == "gtsrb":
            output_dim = 43
        else:
            output_dim = 10
        self.classifier = Mlp(in_features=ch, hidden_features=512, out_features=output_dim)

    # initialize the additional codebooks with kmeans on local data
    def init_codebooks(self, dsloader, idx, device):
        feas = []
        # obatin features for all input data
        with torch.no_grad():
            for xtr, ytr in dsloader:
                xtr, ytr = xtr.to(device), ytr.to(device)
                fea = self.encoder(xtr)
                feas.append(fea.detach())

            feas = torch.concat(feas, dim=0)
            feas = feas.permute(0, 2, 3, 1).contiguous()
            # [B, H, W, C] -> [BHW, C]
            feas = feas.reshape(-1, self.dim)

            # initialize codebooks
            kmeans = KMeans(n_clusters=self.num_embeddings, random_state=0, n_init="auto").fit(feas.cpu().numpy())
            self.discretizer.codebooks[idx].weight.data = torch.from_numpy(kmeans.cluster_centers_).to(device)
    
    # return codebooks
    def get_codebooks(self):
        codebooks = []
        for i in range(len(self.discretizer.codebooks)):
            codebooks.append(self.discretizer.codebooks[i].weight)
        return codebooks
    
    # load codebooks
    def load_codebooks(self, codebooks):
        for i in range(len(self.discretizer.codebooks)):
            self.discretizer.codebooks[i].weight.data = codebooks[i]

    # extend codebook capacity
    def extend_codebooks(self, iteration):
        if iteration > 1:
            self.discretizer.codebooks.append(nn.Embedding(self.num_embeddings, self.dim))
                        
    def forward(self, x, idx, ext=False):
        '''
        if idx (book_index) = 0: only shared codebook
        else: shared codebook + additional codebook
        '''
        fea = self.encoder(x)

        q_fea, loss, ppl = self.discretizer(fea, idx, ext)
        q_fea = q_fea.flatten(1)
        # decoder with quantized vectors
        output = self.classifier(q_fea)
        
        return output, loss, ppl
