import torch

device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


if device == 'cuda':
    if 'forces_first_cuda_action_done' not in locals():

        try:
            a = torch.rand((2, 3, 2)).to('cuda')
            b = torch.rand((2, 3, 4)).to('cuda')
            a = a.permute(0, 2, 1).bmm(b)
        except:
            print('Forced first action in cuda to avoid later error.')

        torch.Tensor.ndim = property(lambda x: len(x.shape))  # so tensors can be plot
        forces_first_cuda_action_done = True
