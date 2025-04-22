# Learning Multi-Mode Musculoskeletal Motion for Natural Walking and Running Gaits

## Abstract 

Traditional musculoskeletal control policies primarily focus on individual gait patterns, with limited success in developing a single model capable of handling multiple locomotion modes, such as walking and running. While prior work has explored learning different gaits within a shared framework, existing approaches typically train independent models rather than a unified, adaptable policy. In this paper, we propose a multi-mode controller learning framework that leverages Feature-wise Linear Transformation (FiLM) to integrate mode-specific dynamics into a unified controller. By incorporating FiLM layers into the control policy, we enable efficient adaptation between locomotion modes while maintaining network simplicity. Furthermore, we employ skeleton scaling and muscle condition randomization to expand the model’s state-space exploration, significantly improving transition stability. We validate our framework by extending Generative GaitNet to support running gaits using novel reward design, integrating it with its original walking model to achieve a unified controller with minimal increase in complexity. The resulting model produces natural-looking gaits across both walking and running, demonstrating the effectiveness of FiLM in multi-mode musculoskeletal control.

## Publications

The 9th International Digital Human Modelling Symposium, DHM 2025

## Installation & Compile
Refer to Bidirectional Gaitnet https://github.com/namjohn10/BidirectionalGaitNet

