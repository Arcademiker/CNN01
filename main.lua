--implementation of the Deep Learning with Torch: the 60-minute blitz (without CUDA)
--test with Cifar-10 dataset:
--The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
--There are 50000 training images and 10000 test images.
--The dataset is divided into five training batches and one test batch,
-- each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class.
--The training batches contain the remaining images in random order,
-- but some training batches may contain more images from one class than another.
--Between them, the training batches contain exactly 5000 images from each class. 
--https://github.com/soumith/cvpr2015/blob/master/Deep%20Learning%20with%20Torch.ipynb
--changes:
--to reduce confusion: in the original classifier was named classes!
--every table and variable and package was set to local
local nn = require 'nn'
local paths = require 'paths'
local torch = require 'torch'

local function main()
    --1.:
    --load test images(if not loaded already) and normalise data:
    if (not paths.filep("cifar10torchsmall.zip")) then
        os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip')
        os.execute('unzip cifar10torchsmall.zip')
    end
    local trainset = torch.load('cifar10-train.t7')
    local testset = torch.load('cifar10-test.t7')
    local classifier = {'airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
    --The dataset has to have a :size() function.
    --The dataset has to have a [i] index operator, so that dataset[i] returns the ith sample in the datset.
    setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
    );
    trainset.data = trainset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.
    function trainset:size() 
        return self.data:size(1) 
    end
    --normalising: (make your data to have a mean of 0.0 and standard-deviation of 1.0)
    local mean = {} -- store the mean, to normalize the test set in the future
    local stdv  = {} -- store the standard-deviation for the future
    for i=1,3 do -- over each image channel
        mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
        print('Channel ' .. i .. ', Mean: ' .. mean[i])
        trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
        stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
        print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
        trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
    end
    --example for index operator:
    --redChannel = trainset.data[{ {}, {1}, {}, {}  }] -- this picks {all images, 1st channel, all vertical pixels, all horizontal pixels}
    --2.:
    --set up convnet:
    local net = nn.Sequential()
    net:add(nn.SpatialConvolution(3, 6, 5, 5)) -- 3 input image channels, 6 output channels, 5x5 convolution kernel
    net:add(nn.ReLU())                       -- non-linearity 
    net:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
    net:add(nn.SpatialConvolution(6, 16, 5, 5))
    net:add(nn.ReLU())                       -- non-linearity 
    net:add(nn.SpatialMaxPooling(2,2,2,2))
    net:add(nn.View(16*5*5))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
    net:add(nn.Linear(16*5*5, 120))             -- fully connected layer (matrix multiplication between input and weights)
    net:add(nn.ReLU())                       -- non-linearity 
    net:add(nn.Linear(120, 84))
    net:add(nn.ReLU())                       -- non-linearity 
    net:add(nn.Linear(84, 10))                   -- 10 is the number of outputs of the network (in this case, 10 digits)
    net:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classification problems
    --3.:
    --define the Loss function:
    --Let us use a Log-likelihood classification loss. It is well suited for most classification problems.
    local criterion = nn.ClassNLLCriterion()
    --4.:
    --Train the neural network:
    local trainer = nn.StochasticGradient(net, criterion)
    trainer.learningRate = 0.001
    trainer.maxIteration = 5 -- just do 5 epochs of training.
    --5.:
    --Test the network, print accuracy:
    --normalize the test data with the mean and standard-deviation from the training data.
    testset.data = testset.data:double()   -- convert from Byte tensor to Double tensor
    for i=1,3 do -- over each image channel
        testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
        testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
    end
    ---[=[ example: use qlua for image.display! 
    local image = require 'image'
    local horse = testset.data[100]
    print(horse:mean(), horse:std())
    local image = require 'image'
    print(classifier[testset.label[100]])
    image.display(testset.data[100])
    local predicted = net:forward(testset.data[100])
    -- the output of the network is Log-Probabilities. To convert them to probabilities, you have to take e^x 
    print(predicted:exp())
    --tag each probability with its classifier-name for image 100:
    for i=1,predicted:size(1) do
        print(classifier[i], predicted[i])
    end
    --]=]
    --correct over the whole test set (10% would be random guessing):
    local correct = 0
    for i=1,10000 do
        local groundtruth = testset.label[i]
        local prediction = net:forward(testset.data[i])
        local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
        if groundtruth == indices[1] then
          correct = correct + 1
        end
    end
    print(correct, 100*correct/10000 .. ' % ')
    --what are the classifier that performed well, and the classifier that did not perform well 
    --(100%each would be correct 1000images*10classifier=10000testimages):
    local classifier_performance = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    for i=1,10000 do
        local groundtruth = testset.label[i]
        local prediction = net:forward(testset.data[i])
        local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
        if groundtruth == indices[1] then
            classifier_performance[groundtruth] = classifier_performance[groundtruth] + 1
        end
    end
    for i=1,#classifier do
        print(classifier[i], 100*classifier_performance[i]/1000 .. ' %')
    end
end
main()
