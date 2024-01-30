from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('tensorboard/exp_1')

for i in range(10):
    writer.add_scalar('quadratic', i**2, i)
    writer.add_scalar('exponential', 2**i, i)