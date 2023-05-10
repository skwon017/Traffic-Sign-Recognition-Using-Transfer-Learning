# Traffic-Sign-Recognition-Using-Transfer-Learning

Final Project for DS-UA 301 Advanced Topics in Data Science

### Team members
Lauren Kwon and Michelle Espinoza

## Project Description

Autonomous vehicles have recently emerged as a rising trend in artificial intelligence (AI) and deep learning (DL). Major car manufacturers such as Tesla, Toyota, Mercedes-Benz, and Ford are investing heavily in the development of self-driving car technology. Autonomous vehicles must understand and operate according to traffic rules. Consequently, cars must comprehend road markings and make appropriate decisions. Recognizing the importance of this technology, we attempt to classify traffic signs in this project.

One challenge faced in this project is the limited number of images per class in the dataset, which could lead to poor model performance. To overcome this issue, transfer learning will be employed. The approach involves using four pre-trained image classification models - ResNet-50, LeNet-5, DenseNet-161, and MobileNet v2 - to initialize the training and classification of the German Traffic Sign Recognition Benchmark dataset from Kaggle. Optimal learning rates are found using Weights & Biases.

Goal: Identifying the best model architecture and optimal learning rate when applying transfer learning for traffic sign recognition. 


![ProjectFramework](./ProjectFramework.png)


## Model Architecture

The pre-trained model is first loaded, ensuring that the weights of all layers are frozen to preserve the original learned features. A new fully connected layer is then appended to the model, replacing the original one, with an output dimension of 43 corresponding to the number of traffic sign classes. The weights for newly added fully connected layer are initialized and the bias terms of the new layer are initialized to zero. During training, only the weights of the new fully connected layer will be updated, allowing the model to fine-tune its output for traffic sign classification while leveraging the knowledge encoded in the pre-trained architecture.


## Results

### Learning Rate, Optimizer, Validation loss, Validation Accuracy

The learning rate value with the minimal validation loss is selected for each model.

- ResNet-50
  + learning rate: 0.0008861 
  + optimizer: adam
  + loss: 1.734
  + accuracy: 57.256

![plot_resnet50](./plot_resnet50.png)

- LeNet-5
  + learning rate: 0.0008741 
  + optimizer: adam
  + loss: 2.153
  + accuracy: 41.992

![plot_lenet5](./plot_lenet5.png)

- DenseNet-161
  + learning rate: 0.0008344 
  + optimizer: adam
  + loss: 1.464
  + accuracy: 68.822

![plot_densenet161](./plot_densenet161.png)

- MobileNet v2
  + learning rate: 0.0007877 
  + optimizer: adam
  + loss: 1.698
  + accuracy: 51.466

![plot_mobilenetv2](./plot_mobilenetv2.png)

### Model Performance

The performance of the model is measured in terms of accuracy on the unseen test set.

- ResNet-50
  + Accuracy: 94.98%

- LeNet-5
  + Accuracy: 93.99%

- DenseNet-161
  + Accuracy: 95.97%

- MobileNet v2
  + Accuracy: 95.45%


### Insights

1. Among the four models, DenseNet-161 achieved the highest accuracy (68.822%) and the lowest loss (1.464). This implies that DenseNet-161 is more effective at correctly identifying traffic signs in the given dataset, making fewer misclassifications compared to the other models.

2. The results demonstrate the potential of transfer learning in scenarios with limited data. All models managed to learn traffic sign classifications to some extent, leveraging their pre-trained weights. However, the performance difference between the models indicates that not all architectures are equally effective for this task, and the choice of architecture matters.

3. While DenseNet-161 performed the best among the four, an accuracy of 68.822% leaves room for improvement. Future work may involve additional data augmentation techniques, hyperparameter tuning, ensemble methods, or the use of more recent and possibly more powerful architectures. 


## Conclusion


## References

- [German Traffic Sign Recognition Benchmark (GTSRB) Dataset from Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- [Traffic sign recognition with multi-scale Convolutional Networks (Sermanet and LeCun, 2011)](https://ieeexplore.ieee.org/document/6033589)
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
- Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., ... & Rabinovich, A. (2015). Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1-9).
- Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4700-4708).
- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4510-4520).
