# UAV geolocalization using matching method for aerial and UAV camera images
The repository implements a vision-based geo-localization system for unmanned aerial vehicles (UAVs) that estimates the geographic coordinates of a UAV from its camera images by matching them with aerial/satellite reference imagery. In contrast to traditional GNSS/GPS-based positioning, this approach uses computer vision and image matching to determine pose when GNSS signals might be unreliable or unavailable.
# System architecture
<img width="1136" height="851" alt="image" src="https://github.com/user-attachments/assets/3d620f32-ac00-497d-ae5e-1f578a419239" />
The pipeline starts with two different visual modalities: a satellite image and a UAV image. Although their viewpoints and appearance differ significantly, both inputs go through the same sequence of data preprocessing steps. These steps include augmentation to increase robustness to viewpoint and illumination changes, standardization to normalize pixel distributions, and resizing to ensure a fixed input resolution for the network.

After preprocessing, both image streams are passed into a Siamese neural network built on top of MobileNet. The key idea here is weight sharing: the same MobileNet encoder processes satellite and UAV images, forcing the network to learn modality-invariant visual representations. This shared backbone extracts high-level semantic features that are comparable across views.

The output of each MobileNet branch is projected into a low-dimensional embedding space and then L2-normalized. Normalization constrains embeddings to lie on a unit hypersphere, which stabilizes training and makes Euclidean distance a meaningful similarity measure.

To train the network, the architecture relies on online triplet mining. During training, embeddings are dynamically grouped into triplets consisting of an anchor, a positive sample that corresponds to the same geographic location, and a negative sample from a different location. This mining strategy selects informative triplets on the fly, focusing learning on hard or semi-hard examples.

Distances between the anchor–positive and anchor–negative embeddings are computed using Euclidean distance and fed into a soft triplet loss. The loss encourages the network to minimize the distance between matching UAV–satellite pairs while maximizing the distance to non-matching pairs, with a smooth margin that improves convergence compared to the standard triplet loss.

Overall, this architecture learns a shared embedding space where geographically corresponding UAV and satellite images are close to each other despite drastic viewpoint differences. Such a representation is well suited for vision-based UAV geo-localization, image retrieval, and GNSS-denied navigation scenarios.
