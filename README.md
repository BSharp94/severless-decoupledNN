# Severless Decoupled NN Using Delayed Gradients

8/28/19
Current this project is used to test the accuracy of neural networks with delayed gradient updates. We aim to prove two key points:

1. Delayed gradient updates can produce models with the competitive accuracy
2. Delayed gradients reach a global minimum relativly the same number of iterations

## Files 
* main.py - normal alexnet implementation of mnist dataset run over 10 epochs
* main2.py - pytorch implementation of alexnet. We need to puse pytorch due to its use of dynamic computational graphs 
* main3.py - break up the network into 3 sections. Runs synchronous but passes the gradient through communication step. 
* main4.py - network works in 3 section. Uses queues to store delayed gradient updates.


## Current Results

main.py

    - 99.0814% test accuracy over 10 epochs  


    

