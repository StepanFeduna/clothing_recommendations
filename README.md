# Overview

In this project, I'm going to create AI system that provides daily recommendations to users with deep learning, based on personal preferences and fashion trends, using regular photos of the user wearing the clothes or photos of user clothes in general. Thus, the system will require minimum effort on a user's end. Recommendations are drawn from the user-owned garments. The method broadly involves two algorithms - preparation of user inventory using clothing detection on user photos and recommendation of outfit from the wardrobe. At a minimum, the user is required to upload his photos, and the rest is taken care of by the algorithm which can yield convincing outfit recommendations based on fashion trends every day.

# What problem does it solve?

The System addresses the following problems:

- According to The Telegraph, a recent study found that an average person spends almost a year in a lifetime in deciding what to wear.
- Even if the time taken each day is justified, frustrations associated with picking an outfit cannot be ignored.
- Given ten items in each category of pants, shirt, jacket, shoes, watch and accessories, the total number of possible combinations turn out to be 1 million which is something humans cannot wrap their minds around.

# Stage of development

1. Clothing Categorization
In the first part of the work, user's clothes are categorize into different categories of clothing items using. The network is trained on Deep Fashion dataset.

2. Compatibility Training
The previous network acts as a data generating stream for this network. For the training, the program is fed with a number of fashion styles that were compatible with current fashion. Based on the training data, the network generates a universal embedding that has learned the fashion style. All possible matches of users clothing are tested against the universal style embedding. The model is trained on Polyvore dataset.