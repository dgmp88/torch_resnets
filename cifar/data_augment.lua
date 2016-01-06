require 'torch'
require 'image'

c100 = torch.load('cifar10-test.t7')

item = c100.data[{{1,2},{},{},{}}]


function augment_item(im)
  -- Flip or dont
  local flip = true -- math.random()>0.5
  local new_im = im:clone()

  if flip then
    local width = (#im)[4]
    local height = (#im)[3]
    for color=1,3 do
      for x=1,width do
        for y=1,height do
          new_im[{{},{color},{y},{x}}] = im[{{},{color},{y},{width-(x-1)}}]
        end
      end
    end
  end


  -- randomly pad sides and recrop
  local im = new_im:clone()

  xpad = math.random(9)-5
  ypad = math.random(9)-5

  local width = (#im)[4]
  local height = (#im)[3]
  for color=1,3 do
    for x=1,width do
      for y=1,height do
        if x <= xpad or  


        new_im[{{},{color},{y},{x}}] = im[{{},{color},{y},{x}}]
      end
    end
  end



  return new_im
end

i = item[{{1},{},{},{}}]
item[{{2},{},{},{}}] = augment_item(i)

-- image.display{image=item, nrow=2}
