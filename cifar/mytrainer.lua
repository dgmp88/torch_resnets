require 'datasets'
require 'optim'

function train(dataset, opt, net, criterion, sgdState, confusion)
  confusion:zero()
  local parameters,gradParameters = net:getParameters()
  n_correct = 0
  for i = 1, dataset:size(), opt.batch_size do
      -- Collect the training data for this batch
      local batch = get_batch_between(dataset,i, i+opt.batch_size)

      local inputs = batch.data
      local targets = batch.label

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- just in case:
         collectgarbage()

         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         -- reset gradients
         gradParameters:zero()

         -- evaluate function for complete mini batch
         local outputs = net:forward(inputs)
         local f = criterion:forward(outputs, targets)

         -- update confusion
         for j = 1,math.min(opt.batch_size, (#batch.label)[1]) do
            confusion:add(outputs[j], targets[j])
         end

         -- Keep up n_correct
         for j = 1,math.min(opt.batch_size, (#batch.label)[1]) do
            _, max = outputs[j]:max(1)
            if max[1] == batch.label[j] then
              n_correct = n_correct + 1
            end
         end

         -- estimate df/dW
         local df_do = criterion:backward(outputs, targets)
         net:backward(inputs, df_do)

         -- return f and df/dX
         return f,gradParameters
      end
      -- Perform SGD step:
      sgdState = sgdState or {
         learningRate = opt.learning_rate,
         momentum = opt.momentum,
         learningRateDecay = learning_rate_decay
      }
      optim.sgd(feval, parameters, sgdState)
      xlua.progress(i, dataset:size())
  end
  print('-----------Trainset----------')
  print(confusion)
  print('n_correct', n_correct, 'percent', (n_correct/dataset:size())*100)
  return sgdState
end

function test(dataset, opt, net, confusion)
  confusion:zero()
  for i = 1, dataset:size(), opt.batch_size do
      collectgarbage()
      -- Collect the training data for this batch
      local batch = get_batch_between(dataset,i, i+opt.batch_size+1)

      local inputs = batch.data
      local targets = batch.label

      local outputs = net:forward(inputs)

      -- update confusion
      for j = 1,math.min(opt.batch_size, (#batch.label)[1]) do
         confusion:add(outputs[j], targets[j])
      end
      xlua.progress(i, dataset:size())
  end

  print('-----------Testset----------')
  print(confusion)
end
