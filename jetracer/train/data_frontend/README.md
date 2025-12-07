frontend.py:

Easily manage your training data with this Streamlit app to browse multiple RC car run folders, view frames, visualize controls, and delete bad frames (removes image file + CSV row).

Run: ```streamlit run frontend.py```

dataset_csv_creator.py: Takes a folder filled with runs recorded while driving the RC Car and combines the image with the csv data into one csv. The csv format is timestamp,steer_us,throttle_us,steer_norm,throttle_norm,depth_front,R1,G1,B1...
