---
layout: home
permalink: index.html

# Please update this with your repository name and title
repository-name: eYY-4yp-project-template
title:
---

[comment]: # "This is the standard layout for the project, but you can clean this and use your own template"

# DETECTION OF BIOLOGICAL DEGRADATION OF CONCRETE STRUCTURES USING COMPUTER VISION

#### Team

- E/19/371, Shantosh M., [email](mailto:e19371@eng.pdn.ac.lk) [Portfolio](https://www.thecn.com/SM3178)
- E/19/133, Harishanth A., [email](mailto:e19133@eng.pdn.ac.lk), [Portfolio](https://www.thecn.com/HA930)
- E/19/137, Hayanan T., [email](mailto:e19137@eng.pdn.ac.lk), [Portfolio](https://www.thecn.com/TH1357)

#### Supervisors

- Ms.Yasodha Vimukthi, [email](mailto:yasodhav@eng.pdn.ac.lk)
- Dr. J. A. S. C. Jayasinghe, [email](mailto:supunj@eng.pdn.ac.lk )

#### Table of content

1. [Abstract](#abstract)
2. [Related works](#related-works)
3. [Methodology](#methodology)
4. [Experiment Setup and Implementation](#experiment-setup-and-implementation)
5. [Results and Analysis](#results-and-analysis)
6. [Conclusion](#conclusion)
7. [Publications](#publications)
8. [Links](#links)

---

<!-- 
DELETE THIS SAMPLE before publishing to GitHub Pages !!!
This is a sample image, to show how to add images to your page. To learn more options, please refer [this](https://projects.ce.pdn.ac.lk/docs/faq/how-to-add-an-image/)
![Sample Image](./images/sample.png) 
-->


## Abstract
Concrete structures are vulnerable to both biological and chemical degradation, which can compromise their durability and structural integrity over time. This research explores the use of computer vision techniques to detect and classify biological degradation, such as algae, mold, and fungi growth, as well as chemical deterioration, including carbonation, sulfate attack, and chloride-induced corrosion. By leveraging deep learning models, image processing algorithms, and pattern recognition techniques, this study aims to develop an automated system for early detection and assessment of concrete degradation. High-resolution image analysis is employed to identify degradation patterns, enhancing the accuracy and efficiency of monitoring methods. The findings of this research could contribute to the development of advanced, data-driven solutions for the proactive maintenance of concrete infrastructure.

## Introduction
Concrete is a fundamental material in modern infrastructure, used in bridges, buildings, roads, and dams due to its strength and durability. However, prolonged environmental exposure leads to biological and chemical degradation, impacting structural safety and longevity. Biological degradation occurs due to the growth of microorganisms such as algae, fungi, and bacteria, which cause surface damage and cracks. Chemical degradation, including sulfate attack, chloride-induced corrosion, and carbonation, alters the material’s composition, leading to deterioration. Traditional inspection methods, such as manual assessments and non-destructive testing (NDT), often lack efficiency and scalability, necessitating automated, real-time monitoring solutions.

Computer vision and deep learning technologies offer high-precision damage detection for concrete structures. These advancements enable automated image analysis to identify cracks, discoloration, and degradation patterns with greater accuracy. Deep learning models, including CNN-based architectures (e.g., ResNet, EfficientNet) and object detection models (e.g., YOLO, Faster R-CNN), improve classification and detection capabilities. While existing research primarily focuses on mechanical damage such as cracks and corrosion, limited studies address biological degradation. Additionally, a comprehensive approach to chemical degradation detection is needed. This research integrates both aspects into a unified system for enhanced structural monitoring and predictive maintenance.

## Related works

## Methodology
The research methodology for developing a deep learning-based system for detecting and quantifying biological and chemical degradation in concrete structures involves several structured stages. The Figure 5.1 explained the stages encompass data collection, data pre-processing and annotation, model development, training and evaluation, and system validation. The methodology is designed to ensure the development of a robust, scalable, and efficient system for structural health monitoring.

![image](https://github.com/user-attachments/assets/79154cb2-d8fb-4270-a660-8d4bcc403369)

### Data Collection
Data collection is a crucial step in the methodology, as the quality and diversity of data directly influence the performance of deep learning models. The research begins by gathering real-world images from infrastructure sites, such as bridges, highways, and industrial buildings, where concrete degradation is common. Since existing datasets primarily focus on mechanical damage like cracks and spalling, there is a significant gap in datasets that include biological (e.g., algae, fungi) and chemical (e.g., sulfate attack, carbonation) degradation. To fill this gap, synthetic data generation techniques are employed to augment the dataset, allowing for a broader representation of degradation types. Additionally, multimodal imaging techniques are employed, including RGB imaging and thermal imaging, which are crucial for capturing detailed information on surface textures and temperature variations of the concrete surface, respectively.

### Data Pre-processing and Annotation
After data collection, the images undergo pre-processing to improve their quality and make them suitable for deep learning. This step includes noise reduction, contrast enhancement, and normalization of images to address inconsistencies that might affect model performance. The pre-processing ensures that the images are clear and of consistent quality across the dataset. Following pre-processing, expert annotations are carried out to label images according to the type of degradation (e.g., microbial growth, chemical attack). Additionally, image segmentation techniques are applied to segment out areas of degradation, allowing the model to focus on these regions during training. This segmentation is crucial for tasks that involve detecting localized degradation patterns, as it helps the model differentiate between degraded and non-degraded areas on the concrete surface.

### Model Development
In this stage, various deep learning models are developed to detect and classify degradation patterns. The study primarily explores Convolutional Neural Networks (CNNs) for feature extraction, as CNNs are highly effective for processing and identifying spatial patterns in image data. To tackle the challenge of real-time degradation detection, the You Only Look Once (YOLO) architecture is also explored. YOLO is a fast and efficient object detection algorithm that enables the system to detect multiple degradation types in a single pass, making it suitable for real-time applications. Additionally, U-Net, a popular architecture for image segmentation, is employed for pixel-level localization of degradation areas, which is essential for accurately quantifying the extent of damage. To further enhance the models' capabilities, transfer learning is applied. Pre-trained models (such as those trained on large-scale datasets like ImageNet) are fine-tuned on the degradation dataset, which helps in leveraging learned features for more efficient training and better performance.

### Model Training and Evaluation
Training the deep learning models involves feeding the pre-processed and annotated images into the selected architectures. During the training process, various hyper-parameters, such as learning rate, batch size, and epochs, are tuned to optimize the model's performance. The dataset is augmented using techniques like rotation, scaling, and flipping, which helps the model generalize across diverse concrete surfaces and environmental conditions. Once the model is trained, it is evaluated using metrics like accuracy, precision, recall, and Intersection over Union (IoU) to assess how well it detects and classifies degradation types. Cross-validation is also performed to ensure that the model generalizes well to unseen data and avoids overfitting. The evaluation process helps identify potential areas of improvement and ensures that the model achieves high performance in both classification and segmentation tasks.

### System Validation and Testing
The final stage of the methodology involves validating the developed system in real-world conditions. The trained model is integrated into a prototype application designed for degradation detection. The system is deployed on concrete surfaces in various environmental conditions to assess its performance in practical settings. The predictions made by the model are compared with the ground truth data from manual inspections to verify its accuracy and reliability. Additionally, the system’s performance is benchmarked against existing methods used in structural health monitoring to demonstrate its superiority. The system is also evaluated for scalability, with considerations made for deploying it on large-scale infrastructure projects, ensuring that it can handle high volumes of data and operate efficiently.

Through these stages, the research aims to create an accurate, scalable, and efficient  solution for the early detection of both biological and chemical degradation in concrete structures, thereby enhancing the long-term durability and maintenance of infrastructure.

## Experiment Setup and Implementation

## Results and Analysis

## Conclusion
This research investigates the integration of computer vision and deep learning technologies to address the challenges of detecting and quantifying both biological and chemical degradation in concrete structures. The study aims to bridge the gap in current inspection methodologies by developing a scalable solution for structural health monitoring. By incorporating advanced deep learning models, including object detection and segmentation techniques, alongside multimodal imaging data such as thermal imaging, the proposed approach enhances the accuracy of degradation detection. The use of these innovative techniques allows for the identification of surface and subsurface damage, enabling a more comprehensive understanding of the structural health of concrete. This framework not only addresses the limitations of traditional inspection methods but also offers a promising pathway to improve the efficiency and effectiveness of infrastructure maintenance, potentially leading to significant cost savings and increased longevity of concrete structures.

## Publications
[//]: # "Note: Uncomment each once you uploaded the files to the repository"

<!-- 1. [Semester 7 report](./) -->
<!-- 2. [Semester 7 slides](./) -->
<!-- 3. [Semester 8 report](./) -->
<!-- 4. [Semester 8 slides](./) -->
<!-- 5. Author 1, Author 2 and Author 3 "Research paper title" (2021). [PDF](./). -->


## Links

[//]: # ( NOTE: EDIT THIS LINKS WITH YOUR REPO DETAILS )

- [Project Repository]([https://github.com/cepdnaclk/repository-name](https://github.com/cepdnaclk/e19-4yp-Detection-of-Biological-Degradation-on-Concrete-Structures-Using-Computer-Vision))
- [Project Page]([https://cepdnaclk.github.io/repository-name](https://cepdnaclk.github.io/e19-4yp-Detection-of-Biological-Degradation-on-Concrete-Structures-Using-Computer-Vision/))
- [Department of Computer Engineering](http://www.ce.pdn.ac.lk/)
- [University of Peradeniya](https://eng.pdn.ac.lk/)

[//]: # "Please refer this to learn more about Markdown syntax"
[//]: # "https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet"
