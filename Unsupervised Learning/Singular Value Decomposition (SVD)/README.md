# Singular Value Decomposition (SVD) for Image Compression

This repository demonstrates how **Singular Value Decomposition (SVD)** can be used for **image compression**. SVD is a mathematical technique that decomposes a matrix into three components: **U**, **S**, and **V**, which can be utilized to approximate the original matrix while reducing the amount of data required to represent it. In this model, SVD is applied to the individual color channels (Red, Green, Blue) of an image to compress it and achieve significant reduction in size while maintaining visual quality.

## Project Overview

This project compresses an image by performing the following steps:

1. **Image Loading**: An image is loaded and split into its individual color channels (Red, Green, and Blue).
2. **SVD Decomposition**: Each color channel is decomposed using SVD, which results in three matrices—**U**, **S**, and **V**—for each channel.
3. **Compression**: The rank (number of singular values) is reduced for each channel, resulting in a compressed representation of the image.
4. **Reconstruction**: The image is reconstructed from the compressed components, and the compression ratio is calculated.
5. **Visualization**: The effect of compression is visualized by testing different ranks and displaying the compressed image at various compression levels.
6. **Interactive Compression**: The interactive widget allows for real-time adjustments of the compression rank to explore the trade-off between compression and image quality.

## Key Features

- **Image Compression**: The image is compressed by reducing the number of singular values in each color channel.
- **Rank Control**: You can control the compression level by specifying the rank (number of singular values) retained for each channel.
- **Compression Ratio**: The compression ratio is calculated and displayed as a percentage, showing how much the image size is reduced compared to the original.
- **Visualization**: The original image, individual color channels, and the compressed image are displayed to compare the effects of compression.
- **Interactive Exploration**: Interactive widgets are provided to adjust the compression rank dynamically and visualize the results.

## Project Workflow

1. **Load and Display Image**: The image (`dog.jpg`) is loaded and displayed to show the original content.
2. **SVD Decomposition**: Each of the color channels (Red, Green, Blue) is processed using SVD, which decomposes the image into three matrices.
3. **Compression Function**: By adjusting the rank, fewer singular values are used to approximate each channel, resulting in image compression.
4. **Visualize Compression at Different Ranks**: The compression effect is demonstrated by testing different ranks (e.g., 5, 20, and 100).
5. **Compression Ratio**: The compression ratio is calculated based on the rank and displayed to show the percentage of data reduction.

## Interactive Features

- **Compression Exploration**: Use the interactive widget to adjust the compression rank dynamically. The rank range is from 1 to 500, allowing you to explore the compression effect in real-time.

## Usage

1. **Install Necessary Libraries**: Ensure you have the following libraries installed:
   - `numpy`
   - `matplotlib`
   - `ipywidgets`

   Install them using the following:

   ```bash
   pip install numpy matplotlib ipywidgets
   ```

2. **Run the Compression Model**: After loading the image, adjust the rank to compress the image and explore the effect of different compression levels.
   
3. **Interactive Compression**: Use the interactive widget to adjust the rank dynamically and observe how the compression quality changes as the rank varies.

## Conclusion

This project showcases the power of **Singular Value Decomposition (SVD)** for **image compression**, offering a way to significantly reduce the size of an image while retaining key features. The compression rank can be tuned to control the trade-off between image quality and compression size, making it a valuable tool for applications requiring efficient image storage and transmission.
