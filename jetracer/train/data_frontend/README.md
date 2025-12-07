frontend.py:

Easily manage your training data with this Streamlit app to browse multiple RC car run folders, view frames, visualize controls, and delete bad frames (removes image file + CSV row).

Run: ```streamlit run frontend.py```


![demo](https://cdn.discordapp.com/attachments/1146243338832457751/1447347533746802829/Screenshot_29.png?ex=69374aee&is=6935f96e&hm=5206c0e4390c59ac9f297cfbe0399db17b9db31921cd322a88f4cf007c102b91)


dataset_csv_creator.py: Takes a folder filled with runs recorded while driving the RC Car and combines the image with the csv data into one csv. The csv format is timestamp,steer_us,throttle_us,steer_norm,throttle_norm,depth_front,R1,G1,B1...
