require 'torch'
require 'image'

c100 = torch.load('cifar10-test.t7')
print(#c100.data)
image.display{image=c100.data[{{1,1000},{},{},{}}], nrow=40}
