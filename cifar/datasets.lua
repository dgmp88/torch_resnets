
--- index an item frmo a table with data/labels of same length
index_metatable = function(t, i)
                    return {t.data[i], t.label[i]}
                    end

--- Format a table in the basic way needed for MNIST
format_table = function(table)
    setmetatable(table, {__index = index_metatable})
    table.data = table.data:double()/255.0 --- Normalize to 0-1
    table.data:resize(table.data:size()[1], 3, 32, 32) --- Add an empty dim
    table.size = nil --- remove this var, replace with a function
    function table:size()
        return self.data:size(1)
    end

    function table:normalize(mean_)
       local mean = mean_ or table.data:mean(1)
       for i =1,table:size() do
         xlua.progress(i, table:size())
         if i % 100 then
           collectgarbage()
         end
         table.data[i]:add(-mean)
       end
       return mean
    end

    table.label = table.label+1 --- indexes must start at 1, not 0
    return table
end

--- Load cifar100
cifar100 = function()
  local trainset = format_table(torch.load('cifar100-train.t7'))
  local testset = format_table(torch.load('cifar100-test.t7'))

  return trainset, testset
end

cifar10 = function()
  local trainset = format_table(torch.load('cifar10-train.t7'))
  local testset = format_table(torch.load('cifar10-test.t7'))

  return trainset, testset
end

--- Extract batches
get_batch_random = function(data, batch_size)
    -- Generate random subset indexes
    local batch_idxs = torch.LongTensor(batch_size)
    local n = (#data.data)[1]
    for i=1, batch_size, 1 do
        batch_idxs[i] = math.random(n)
    end

    -- Select the data
    local batch_data = torch.Tensor()
    batch_data:index(data.data, 1, batch_idxs)

    -- Select the labels
    local batch_labels = data.label:index(1, batch_idxs)

    -- Turn the batch into the right format
    local batch = {}
    batch.data = batch_data
    batch.label = batch_labels

    return batch
end

--- Extract batches
get_batch_between = function(data, from, to)
    -- Generate random subset indexes
    local data_size = #data.data

    local to = math.min(to,data_size[1])
    local batch_size = to-from
    local batch_data = torch.Tensor(batch_size, data_size[2],data_size[3],data_size[4])
    local batch_labels = torch.Tensor(batch_size)

    for i=1, batch_size, 1 do
        batch_data[i] = data.data[from+(i-1)]:clone()
        batch_labels[i] = data.label[from+(i-1)]
    end
    local batch = {}
    batch.data = batch_data
    batch.label = batch_labels
    return batch
end


--- Get minumum value and index of tensor
min = function(tensor)
    local val = tensor[1]
    local index = 1
    for i = 2, (#tensor)[1], 1 do
        if tensor[i] < val then
            val = tensor[i]
            index = i
        end
    end
    return val, index
end

--- Get maximum value and index of tensor
max = function(tensor)
    local val = tensor[1]
    local index = 1
    for i = 2, (#tensor)[1], 1 do
        if tensor[i] > val then
            val = tensor[i]
            index = i
        end
    end
    return val, index
end

percent_correct = function(labels, predictions)
  local corr = 0
  local n = (#labels)[1]
  for i=1, n, 1 do
    if labels[i] == predictions[i] then
      corr = corr + 1
    end
  end
  return (corr/n)*100
end

predictions_to_labels = function(predictions)
  local n = (#predictions)[1]
  local labels = torch.LongTensor(n)
  for i=1, n, 1 do
    val, idx = max(predictions[i])
    labels[i] = idx
  end
  return labels
end

printf = function(s,...)
   return io.write(s:format(...))
end -- function
