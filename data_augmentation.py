"""
Data Augmentation Exercise

Objective:
After trying to improve the dataset, we will attempt another approach.
Starting from the original dataset, we will expand it by degrading certain images.
We keep the original image and add degraded copies.

The degradation will be applied in:
- Luminance (over-exposure / under-exposure)
- Blur
- Noise

Example Algorithm:
1. Analyze the brightness of all images in the database to count how many are
   under-exposed (M) and over-exposed (N) relative to a certain threshold.

2. Randomly select from the entire database (not only those already under/over-exposed):
   • M images to over-expose
   • N images to under-expose

3. Create new degraded images and add them to the dataset

Apply similar degradation for blur and noise

Train the model on this augmented dataset and compare the results

WRITE ALL REMARKS AND CONCLUSIONS IN THIS FILE AS COMMENTS
"""
