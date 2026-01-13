# UAV geolocalization using matching method for aerial and UAV camera images
The repository implements a vision-based geo-localization system for unmanned aerial vehicles (UAVs) that estimates the geographic coordinates of a UAV from its camera images by matching them with aerial/satellite reference imagery. In contrast to traditional GNSS/GPS-based positioning, this approach uses computer vision and image matching to determine pose when GNSS signals might be unreliable or unavailable.
# Training pipeline
<img width="1136" height="851" alt="image" src="https://github.com/user-attachments/assets/3d620f32-ac00-497d-ae5e-1f578a419239" />

Both image types are first augmented, standardized, and resized, then passed through a **Siamese MobileNet**  with shared weights. The network extracts comparable features from different viewpoints and projects them into **L2-normalized embeddings**.

Training uses **online triplet mining**, where anchor, positive, and negative samples are selected dynamically. **Euclidean distances** between embeddings are optimized using a **soft triplet loss**, pulling matching UAV–satellite pairs closer while pushing non-matching pairs apart.

The result is a viewpoint-invariant representation suitable for vision-based UAV geo-localization.
# Dataset
<img width="861" height="378" alt="image" src="https://github.com/user-attachments/assets/502ec5a6-4bdc-486b-9368-54367a60362a" />

The experiments are based on the [DenseUAV dataset](https://github.com/Dmmm1997/DenseUAV), a large-scale benchmark specifically designed for vision-based UAV self-positioning in low-altitude urban environments. DenseUAV contains paired UAV-view and satellite-view images collected with dense spatial sampling

The dataset includes 27,297 images covering 14 university campuses, capturing significant viewpoint, scale, and appearance variations between UAV and satellite imagery.
<img width="811" height="391" alt="image" src="https://github.com/user-attachments/assets/23611368-cf66-45ab-bc6e-c301e3a99eb3" />

The ratio of training and test data: 75% and 25%, respectively.

