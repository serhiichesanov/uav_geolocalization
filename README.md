# UAV geolocalization using matching method for aerial and UAV camera images
The repository implements a vision-based geo-localization system for unmanned aerial vehicles (UAVs) that estimates the geographic coordinates of a UAV from its camera images by matching them with aerial/satellite reference imagery. In contrast to traditional GNSS/GPS-based positioning, this approach uses computer vision and image matching to determine pose when GNSS signals might be unreliable or unavailable.
# Training pipeline
<img width="1136" height="851" alt="image" src="https://github.com/user-attachments/assets/3d620f32-ac00-497d-ae5e-1f578a419239" />

Both image types are first augmented, standardized, and resized, then passed through a **Siamese MobileNet**  with shared weights. The network extracts comparable features from different viewpoints and projects them into **L2-normalized embeddings**.

Training uses **online triplet mining**, where anchor, positive, and negative samples are selected dynamically. **Euclidean distances** between embeddings are optimized using a **soft triplet loss**, pulling matching UAV–satellite pairs closer while pushing non-matching pairs apart.

The result is a viewpoint-invariant representation suitable for vision-based UAV geo-localization.
# Dataset
<p align=center><img width="827" height="369" alt="image" src="https://github.com/user-attachments/assets/3903999b-0e2f-4d47-84cd-b7e209af4e63" /></p>

The experiments are based on the [DenseUAV dataset](https://github.com/Dmmm1997/DenseUAV), a large-scale benchmark specifically designed for vision-based UAV self-positioning in low-altitude urban environments. DenseUAV contains paired UAV-view and satellite-view images collected with dense spatial sampling

The dataset includes 27,297 images covering 14 university campuses, capturing significant viewpoint, scale, and appearance variations between UAV and satellite imagery.
## <p align=center>Dataset info</p>
<p align=center><img width="777" height="388" alt="image" src="https://github.com/user-attachments/assets/26c544e1-0ae6-4b21-b353-512430933221" /></p>



The ratio of training and test data: 75% and 25%, respectively.

# Real-time matching module
<p align="center"><img width="968" height="789" alt="image" src="https://github.com/user-attachments/assets/e5a4f32d-add5-4a33-b166-559c0f493550" /></p>

The trained model is applied in real-time geolocation. A database of satellite images covering the predicted flight area is provided. They are passed to the network to compute their vector representations. After that, the desired video is selected, which is divided into frames, which are also computed by the model. Then the Euclidean distance between the frame and each satellite image from the database is calculated, sorted in ascending order, and the smallest distance is taken. The index of such a distance is the model's prediction, indicating the current photo from the database. Thus, longitude and latitude information is obtained.

# Inference
## Train
![TRAIN-ezgif com-optimize](https://github.com/user-attachments/assets/5c53af57-e3e0-4188-b872-4c01166742ff)

## Test Case 1 (Better Case)
![TEST1-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/f79abf8e-5cef-4e9f-ab03-90eacbb9fbc5)

## Test Case 2 (Worse Case)
![TEST0-ezgif com-optimize](https://github.com/user-attachments/assets/e6372a3a-43ba-4429-a44f-5bbbf571f6dc)

## Test on ALTO dataset
![ALTO-ezgif com-optimize](https://github.com/user-attachments/assets/75497616-914f-4cfb-8e8e-247d7e3c84f0)





