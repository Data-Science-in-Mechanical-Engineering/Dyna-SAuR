import torch
import colorednoise as cn
import numpy as np
class PinkNoiseGenerator:
    """
    A generator for pink noise (1/f noise) with clamped values.

    This class generates batches of pink noise using powerlaw PSD (power spectral density)
    and provides them sequentially. When the batch is exhausted, a new batch is automatically
    generated.

    Attributes:
        pink_noise_length (int): The number of noise samples in each batch.
        pink_noise_shape (tuple): The shape of each noise sample.
        device (torch.device): The device (CPU/GPU) where tensors are stored.
        pink_noise_batch (torch.Tensor): The current batch of pink noise samples.
        pink_noise_counter (int): Index of the next noise sample to be returned.

    Methods:
        reset_pink(): Generates a new batch of pink noise samples.
        get_pink_action(): Returns the next pink noise sample from the current batch.

    Example:
        >>> generator = PinkNoiseGenerator(pink_noise_shape=(3,), pink_noise_length=1000, device='cuda')
        >>> noise = generator.get_pink_action()
        >>> print(noise.shape)
        torch.Size([3])
    """
    def __init__(self, pink_noise_shape, pink_noise_length, device, seed):
        self.pink_noise_length = pink_noise_length
        self.pink_noise_shape = pink_noise_shape
        self.device = device
        self.seed = seed
        self.random_state = np.random.RandomState(self.seed)
        self.reset_pink()
    def reset_pink(self):
        pink_noise_batch = cn.powerlaw_psd_gaussian(1, (*self.pink_noise_shape, self.pink_noise_length), random_state=self.random_state)
        axes = tuple([len(self.pink_noise_shape)] + list(range(len(self.pink_noise_shape))))
        pink_noise_batch = pink_noise_batch.transpose(axes)
        self.pink_noise_batch =torch.from_numpy(pink_noise_batch).float().to(self.device).clamp(-3,3)
        self.pink_noise_counter = 0
    def get_pink_action(self): # Gives you a pink noise distributed noise batch clamped between -3 and 3
        noise = self.pink_noise_batch[self.pink_noise_counter]
        self.pink_noise_counter+=1
        if self.pink_noise_counter == self.pink_noise_length:
            self.reset_pink()
        return noise
    
class PinkNoiseGeneratorIndividualTrajectories:
    """
    A generator for pink noise (1/f noise) with clamped values.

    This class generates batches of pink noise using powerlaw PSD (power spectral density)
    and provides them sequentially. When the batch is exhausted, a new batch is automatically
    generated.

    Attributes:
        pink_noise_length (int): The number of noise samples in each batch.
        pink_noise_shape (tuple): The shape of each noise sample.
        device (torch.device): The device (CPU/GPU) where tensors are stored.
        pink_noise_batch (torch.Tensor): The current batch of pink noise samples.
        pink_noise_counter (int): Index of the next noise sample to be returned.

    Methods:
        reset_pink(): Generates a new batch of pink noise samples.
        get_pink_action(): Returns the next pink noise sample from the current batch.

    Example:
        >>> generator = PinkNoiseGenerator(pink_noise_shape=(3,), pink_noise_length=1000, device='cuda')
        >>> noise = generator.get_pink_action()
        >>> print(noise.shape)
        torch.Size([3])
    """
    def __init__(self, batchsize, pink_noise_shape, pink_noise_length, device, seed):
        self.pink_noise_length = pink_noise_length
        self.pink_noise_shape = pink_noise_shape
        self.device = device
        self.seed = seed
        self.random_state = np.random.RandomState(self.seed)
        self.batch_size = batchsize
        self.indicies_counter = np.zeros(self.batch_size, dtype=int)
        self.pink_noise_batch = torch.zeros((self.pink_noise_length, self.batch_size, *self.pink_noise_shape), dtype=torch.float32, device=self.device)
        self.reset_pink(np.zeros(self.batch_size, dtype=int))
    def reset_pink(self, indices): # Assume indices a numpy list with shape (batch_size,)
        new_pink_trajecotries = indices.shape[0]
        pink_noise_batch = cn.powerlaw_psd_gaussian(1, (new_pink_trajecotries, *self.pink_noise_shape, self.pink_noise_length), random_state=self.random_state)
        axes = tuple([len(self.pink_noise_shape)+1] + list(range(len(self.pink_noise_shape)+1)))
        pink_noise_batch = pink_noise_batch.transpose(axes)
        self.pink_noise_batch[:,indices,:] =torch.from_numpy(pink_noise_batch).float().to(self.device).clamp(-3,3)
        self.indicies_counter[indices] = 0
    def get_pink_action(self): # Gives you a pink noise distributed noise batch clamped between -3 and 3
        noise = self.pink_noise_batch[self.indicies_counter,np.arange(self.batch_size),:]
        self.indicies_counter+=1
        if np.any(self.indicies_counter == self.pink_noise_length):
            assert False, "Need to reset pink noise for some trajectories before calling get_pink_action again."
        return noise