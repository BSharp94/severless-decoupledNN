# Severless Decoupled NN Using Delayed Gradients

This project implements a delayed architecture for Neural networks. It targets cloud functions as an ideal infrastructure for improving run times.

## Compare Accuracy

Currently this folder is used to test that the delayed model implements the same level of accuracy with a delay in the update gradients. 

## GCloud

This folder stores the scripts for the google cloud functions.

## TODOs

* Implement a unique id system for the data batches to ensure that the backprop signals match 

* Implement a reduction method for learning rate.