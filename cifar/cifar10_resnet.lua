require 'nn'
require 'nngraph'
require 'optim'
require 'datasets'
require 'mytrainer'

local opt = lapp[[
   -r,--learning_rate  (default 0.1)        learning rate
   -d,--learning_rate_decay   (default 0.0001)   learning rate decay
   -b,--batch_size     (default 128)          batch size
   -m,--momentum      (default 0.9)           momentum
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
  local train = 1000
  local test = 1000
  trainset.data = trainset.data:sub(1,train)
  trainset.label = trainset.label:sub(1,train)

  testset.data = testset.data:sub(1,test)
  testset.label = testset.label:sub(1,test)
end
print('Datasets loaded, normalizing...')

mean = trainset:normalize()
testset:normalize(mean)

print('Normalized!')


-- A simple ResNet layer
build_reslayer = function(input, n_features)
  --- Conv1
  local conv1 = nn.SpatialConvolution(n_features,n_features,3,3,1,1,1,1)(input)
  local spatial_batch1 = nn.SpatialBatchNormalization(n_features)(conv1)
  local relu1 = nn.ReLU()(spatial_batch1)

  -- Conv2
  local conv2 = nn.SpatialConvolution(n_features,n_features,3,3,1,1,1,1)(relu1)
  local spatial_batch2 = nn.SpatialBatchNormalization(n_features)(conv2)

  -- Shortcut
  local add = nn.CAddTable()({input, spatial_batch2})
  local output = nn.ReLU()(add)

  return output
end

-- A convolutional nngraph
build_conv1 = function()
  conv1 = nn.SpatialConvolution(3,16,3,3,1,1,1,1)()
  batchnorm1 = nn.SpatialBatchNormalization(16)(conv1)
  relu1 = nn.ReLU()(batchnorm1)

  res1 = build_reslayer(relu1, 16)
  res2 = build_reslayer(res1, 16)
  res3 = build_reslayer(res2, 16)

  conv2 = nn.SpatialConvolution(16,32,3,3,2,2,1,1)(res3)
  batchnorm2 = nn.SpatialBatchNormalization(32)(conv2)
  relu2 = nn.ReLU()(batchnorm2)

  res4 = build_reslayer(relu2, 32)
  res5 = build_reslayer(res4, 32)
  res6 = build_reslayer(res5, 32)

  conv3 = nn.SpatialConvolution(32,64,3,3,2,2,1,1)(res6)
  batchnorm3 = nn.SpatialBatchNormalization(64)(conv3)
  relu3 = nn.ReLU()(batchnorm3)

  res7 = build_reslayer(relu3, 64)
  res8 = build_reslayer(res7, 64)
  res9 = build_reslayer(res8, 64)

  lastsize = 64*8*8
  view1 = nn.View(lastsize)(res9)
  line1 = nn.Linear(lastsize,10)(view1)
  logs1 = nn.LogSoftMax()(line1)
  conv = nn.gModule({conv1}, {logs1})
  return conv
end


net = build_conv1()

--- Train!
for i = 1,opt.train_iter do
  print('Training iteration #', i, '/', opt.train_iter)
  sgdState = train(trainset, opt, net, criterion, sgdState, confusion)
  test(testset, opt, net, confusion)
end
