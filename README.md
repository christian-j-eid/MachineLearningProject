# MachineLearningProject
In this project, I created a neural net from scratch using gradient descent with 4 different loss functions, the perceptron algorithm, and linear programmming to classify linearly classifiable data. Tested the the time and success rate and created a report. 

In addition, I coded various algorithms that classified non-linearly separable data tested their efficiency and accuracy. 

This includes using mapping input to feature space to use linear programming and linear algorithms. In my project, I mapped each instance of data x to a feature space such that

<img width="137" alt="FeatureSpace" src="https://github.com/user-attachments/assets/d8ae9583-a0da-4855-8fbe-0d337a71e999">

This allows me to create larger weight vectors to classify the data.
<img width="50" alt="LargerWeight" src="https://github.com/user-attachments/assets/8ee75159-4932-4866-92e9-7da98881ea30">

The report shows speed & accuracy test results using Perceptron, Linprog, Hard-SVM, and Soft-SVM on both linear and non-linearly classifiable data using this feature space. 

I also use Kernalized algorithms for the Hard-SVM, which is applied when the loss function is calculation during gradient descent. Kernel algorithms do not use feature mapping and rely on the original training data, using them to classify new data. 

I compare results of feature mapping with Hard-SVM to using kernelized SVM with a Gaussian and Polynomial kernel function. 

Lastly, I created a larger neural net using backpropagation & the sigmoid acivation function, and compare that to the results of the Soft-SVM using Gaussian Kernelized hinge loss function. 

Report of results on test data shows that neural nets are capable of clearning to classify non-linearly classifiable data faster than the Gaussian Kernelized Soft-SVM. 
