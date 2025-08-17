This is a handwritten digit classifier using the MNIST dataset written completely from scratch without Pytorch, Tensorflow or any other libraries that allow you to not write the backprop algorithm. Zero AI written code was used. AI advice was kept to a mininum (only one change was actually suggested by AI, and it was more of a housekeeping thing). The backpropagation was written almost completely based off videos by 3Blue1Brown. It also includes a visualizer

The first five days went decently smoothly, with the frontpropagation, network, and backpropagation algos all being finished without many bugs. The model worked when trained on one case, showing it's ability to correctly adjust weights and biases. However, training accuracy during this time was pretty terrible, plateuing at 0.126. This, being obviously unideal, was a pretty big issue. First, I implemented batch training and epochs, which did little. Then came several days of bug squashing in the backprop and forward prop algorithms. This raised accuracy to around the 0.3 to 0.4 ranges. 

Next, I edited the lost funtction. Originally, the total loss and loss derivatives used the MSE loss function. I kept the MSE as a metric of how well the model was performing, but switched to the derivatives to cross-entrophy. This did not do much for accuracy values. 

At this point it had been around 20 hours of developing, and I decided to consult AI with the code for advice. ChatGPT gave me an incorrect answer since it was unable to understand part of my code (which made me happy since I managed to stump it). Claude pointed out that while a step in my backprop was theoretically okay, it was technically unsound. 

Rather than store the error and then the gradient, I had only stored the gradient. This wasn't an issue however as I could just divide by the weight of the previous neuron to get the error again. Numbers in python are unfortunetly cut off at some point, so this could cause a small discrepancy between the calculated and expected error of the neuron in the layer above. 

After making this change, not much changed. So, I decided to switch all the middle layers to a tanh activation instead of sigmoid. This yielded around a 50 percent accuracy depending on the amount of nodes, growth factor, and training length. After this I switched the middle layers again, this time to ReLU. This increased my accuracy to 55 percent.

Online, I had also read about initialization. Unfortunetly, I underestimated the impact it would have on training, and chose to address more pressing matters. When I finally got around to implementing He initialization, accuracy rates jumped 15 percent immedietly.  


Here is a chart detailing some of the trials I did and the accuracy. All were run with a batch size of 50, and this was after I implemented a backprop change:



|Structure|Hidden Activation|Final Layer Activation|growth factor|gf shrinking|initialization|Training Examples|Accuracy|
|---------|-----------------|----------------------|--------|-|-|-|-|
|784, 10, 10|ReLU|sigmoid|0.5|no|random|7000|0.3246|
|784, 10, 10|tanh|sigmoid|10|no|random|7000|0.4421|
|784, 24, 24, 10|tanh|sigmoid|2|no|random|7000|0.5571|
|784, 10, 10|ReLU|sigmoid|0.5|no|He|15000|0.5957|
|784, 16, 16, 10|ReLU|sigmoid|0.3|no|He|7000|0.6224|
|784, 10, 10|ReLU|sigmoid|0.5|Yes|He|15000|0.6311|
|784, 24, 24, 10|ReLU|sigmoid|0.3|no|He|7000|0.6397|
|784, 24, 10|ReLU|sigmoid|1|no|He|7000|0.6863|
|784, 24, 10|ReLU|sigmoid|0.6|no|He|7000|0.688|
|784, 36, 10|ReLU|sigmoid|0.3|no|He|7000|0.6941|
|784, 24, 10|ReLU|sigmoid|0.3|no|He|7000|0.696|
|784, 10, 10|ReLU|sigmoid|0.4|no|He|15000|0.7628|
|784, 24, 10|ReLU|sigmoid|0.3|no|He|15000|0.7709|
|784, 36, 10|ReLU|SoftMax|0.3|no|He|15000|0.7753|

In the end, I was able to achieve the higehst accuracy or around 77 percent. The best few models peaked at around 0.84 and training, but this would usually fall to somewhere around the high 0.7s for testing. I would be interested to see if anyone can tweak the model even higher, as tensorflow MNIST neural networks usually get at least 90 percent testing accuracy.

If anyone is curious about modifying this program, line 203 controls the structure of the network, 223 and 224 control the amount of nodes visible on the first layer of the visualizer (since I didn't want to have 800 ish nodes on my screen in a vertical line), 273 controls the growth factor, and 274 controls whether or not to turn on the visualizer. Line 451 is the growth factor shrinking line.
