# Introduction
Time by time, people are suffocated with the number of clothes choices and find difficulties on determining the right clothes for themselves. Hence, this project is created. This is a system that recommends clothes based on the input of images provided by the user themself. Those images may be something that the person sees often or just glances at, but the image has attracted a certain amount of interest and that person wants to search for similar products. The project uses neural networks to process the images from DeepFashion Dataset and k-NN to generate the final recommendations.

# Methodology
In this project, we propose a model that uses ResNet50 Model, k-nearest neighbors algorithm, Yolov5, BeautifulSoup and Selenium. Initial, the ResNet50 model are trained with our dataset and then another dataset containing the information and images which is crawled from some online selling website is created. When the user input searching image, that image goes through the yolov5 model to extract the clothing pieces and determine which category it belongs to. Finally, k-nearest neighbor's algorithm is used to find the most relevant products based on the input extracting image and recommendations are generated.

# Dataset
DeepLearning2 is used for two problems. First, create a dataset to train the model to recognize which category the clothes belongs to.  Second, DeepLearning2 is suitable for training yolov5 in category recognition and bounding box in an inputed image. This dataset is categorized into 13 popular categories.

[https://github.com/switchablenorms/DeepFashion2]