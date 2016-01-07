require 'nn'
require 'nngraph'
require 'optim'
require 'datasets'
require 'mytrainer'

opt = lapp[[
   -r,--learning_rate  (default 0.1)        learning rate
   -d,--learning_rate_decay   (default 5e-4)   learning rate decay
   -b,--batch_size     (default 256)          batch size
   -m,--momentum      (default 0)           momentum
   -t,--train_iter    (default 50)           number of training iterations
   -p,--pardowndata   (default 0)         reduce size of data
]]

math.randomseed(os.time())
torch.setdefaulttensortype('torch.FloatTensor')

-- Use the same cost function for everything
criterion = nn.ClassNLLCriterion()

-- Make a confusion matrix
classes = {}
for i=1,10 do classes[i]=tostring(i)end
confusion = optim.ConfusionMatrix(classes)

-- Load the data
trainset, testset =  cifar10()
if opt.pardowndata == 1 then
  local test = 1000
  trainset.data = trainset.data:sub(1,train)
  trainset.label = trainset.label:sub(1,train)

  testset.data = testset.data:sub(1,test)
  testset.label = testset.label:sub(1,test)
end
collectgarbage()

-- A simple linear network
build_linear = function()
  lastsize = 3*32*32
  view1 = nn.View(lastsize)()
  line1 = nn.Linear(lastsize,10)(view1)
  logs1 = nn.LogSoftMax()(line1)

  --
  lin = nn.gModule({view1}, {logs1})
  return lin
end


net = build_linear()

--- Train!
for i = 1,opt.train_iter do
  print('Training iteration #', i, '/', opt.train_iter)
  sgdState = train(trainset, opt, net, criterion, sgdState, confusion)
  test(testset, opt, net, confusion)
end
