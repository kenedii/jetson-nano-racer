record_data.py:
After xbox controller config is complete and configuration is complete in this script, This script allows the user to drive the car around the track using the XBOX controller. 
It will save the flattened RGB, IR (2 channels), and depth data (entire depth pixel map) as independent variables, and the steering and acceleration values are the target variables of the dataset. This data can be used to predict steering and acceleration values when given an image of what the car sees.

record_data2.py: 
The first script was causing the XBOX Controller's inputs to be sent to the car with a 4-5 second delay due to the large amount of data being processed, even with only 2 FPS. This script only collects the RGB image along with the depth in front of camera float value in centimetres to reduce the amount of data being processed by the Jetson every second. Using this script will considerably reduce the input delay between the steering/acceleration controls from the XBOX Controller and the RC Car while collecting training data driving the car.
