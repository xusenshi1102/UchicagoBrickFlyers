<!-- PROJECT SHIELDS -->
<!--

-->
[![MIT License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/xusenshi1102/UchicagoBrickFlyers">
    <img src="images/logo.png" alt="Logo" width="1449" height="151">
  </a>

  <h3 align="center">3D MODELING to BLUEPRINT TECHNOLOGY</h3>

  <p align="center">
    By BrickFlyer @ University of Chicago
    <br />
    <a></a>
    <br />
    <br />
    <a href="https://youtu.be/vi7t1NEUvjg">View Demo</a>
    ·
    <a href="https://docs.google.com/presentation/d/16eW59hRi-5u8jI3KI2rahYbW-pX7z3u7/edit?usp=sharing&ouid=113572346800658150948&rtpof=true&sd=true">View Presentation</a>
  </p>
</div>


<!-- ABOUT THE PROJECT -->
## About The Project

![Product Pipeline Screen Shot][product-pipeline-screenshot]

3D Model Capture and LEGO Blueprint Generation Using Drones and Computer Vision is an innovative system that integrates autonomous drone navigation, computer vision, and 3D modeling to convert real-world objects into customizable LEGO blueprints. This project automates the complex process of object capture, segmentation, and 3D reconstruction to create a foundational LEGO model that users can modify and expand.
Key features of the project include:
* Autonomous Object Detection and Segmentation: Utilizes YOLO8-world and SAM2 for real-time detection and segmentation of target objects captured by a Tello drone.
* 3D Model Reconstruction: Employs DUSt3R to convert multi-angle images into detailed 3D models.
* LEGO Blueprint Generation: Transforms the 3D model into a LEGO design plan with step-by-step instructions using BrickLink Studio.

The system is designed for accessibility, focusing on small, everyday objects found in classrooms, such as desks and chairs, making it ideal for educational, creative, and prototyping applications.

By bridging advanced technology with LEGO's simplicity, this project lowers barriers to complex LEGO design, fostering innovation and creativity across diverse domains.


<!-- GETTING STARTED -->
## Getting Started

Follow these steps to use the system and generate a LEGO blueprint for a suitcase or any object of your choice.

### Prerequisites

To ensure a smooth workflow, make sure the following prerequisites are met:

1. [**Conda Virtual Environment**](https://www.anaconda.com/download?utm_source=anacondadocs&utm_medium=documentation&utm_campaign=download&utm_content=topnavalldocs)  
2. [**Tello Drone (DJI Tello EDU)**](https://www.amazon.com/DJI-CP-TL-00000026-02-Tello-EDU/dp/B07TZG2NNT)  
3. [**BrickLink LEGO Studio Software**](https://www.bricklink.com/v3/studio/download.page)

### Installation

1. Clone the repo
```bash
git clone https://github.com/xusenshi1102/UchicagoBrickFlyers.git
```

2. Create virtual environment
```bash
conda create -n brickflyer python=3.11
conda activate brickflyer
```

3. Install packages
```bash
pip install -r requirements.txt
```

4. [Install SAM2](https://github.com/facebookresearch/sam2)
```bash
cd external
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .
```

5. [Install DUST3R](https://github.com/naver/dust3r)
```bash
cd external
git clone --recursive https://github.com/naver/dust3r && cd dust3r
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install -r requirements.txt
```

6. Check Drone
Ensure that your drone and computer are connected to the same network.

<!-- USAGE EXAMPLES -->
## Usage
Ensure your drone faces toward the object
```bash
python3 main.py --target_object --drone_ip
```

For a detailed demonstration, you can watch the usage video on YouTube:  
[Watch the Video](https://youtu.be/vi7t1NEUvjg)


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<!-- MARKDOWN LINKS & IMAGES -->
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/xusenshi1102/UchicagoBrickFlyers/blob/main/LICENSE
[product-pipeline-screenshot]: images/project-pipeline.png