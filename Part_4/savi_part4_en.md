Part 4 - SAVI
==============
Miguel Riem Oliveira <mriem@ua.pt>
2026-2027

# Summary

- Features
- Feature matching
- Image Mosaicking

# Exercises

## Exercise 1 - OpenCV feature tutorials

Do the exercises in the OpenCV tutorial on [Feature Detection and Description](https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html).

## Exercise 2 - Compute features

Load the images in the santorini folder and compute **SIFT** _features_, limiting the number of keypoints to 500.

Visualize the features found in both images.

![Image](docs/q_features.jpg)
![Image](docs/t_features.jpg)


## Exercise 3 - Match features

Using the images in the castle folder, match the _features_ found in the previous exercise, and display the matches.

![Image](docs/castle_all_matches.png)

## Exercise 4 - Filter matches

Use David Lowe's ratio test to filter out unreliable matches computed in the previous exercise.

![Image](docs/good_matches.jpg)

## Exercise 5 - Stitch the images

Using the images in the machu pichu folder, estimate the geometric transformation between the images with the **findHomography** function, and then apply that transformation to merge the two images into one.

![Image](docs/stitched.jpg)

## Exercise 6 - Stitch images of castle

Stitch the images in the _castle_ folder. You may need to adapt the stitching algorithm developed in the previous exercise.
