# EvaluateComplexityBackground
This repository is use for evaluate the complexity (compute 'mse' and 'entropy') of the background of images which have face (The background is obtained from the image with the face removed)

To using this code, do these steps:

auto_crop02.py

+ Change your images directory:

  live_dir = "F:\Img_thang_3\Standard_Image_bonus"  # INPUT IMAGES DIRECTORY HERE (The image after remove face will store in folder 'result02')

+ python auto_crop02.py

evaluate_complexity_.py

+ This code will compute 'mse' and 'entropy' of image, the results save to file 'MSE_ENTROPY.csv'

Statistic.py

+ This code will show you the statistic of above result.
