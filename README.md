<h1 align="center">TʀᴀᴄᴋGʀᴀᴘʜ: Online Open-Vocabulary 3D Scene Graphs via Image-Space Tracking</h1>
<div align="center">
  <a href="https://ntnu-arl.github.io/trackgraph-site/"><img src="https://img.shields.io/badge/Homepage-1E88E5?style=flat-square" alt="Homepage"></a>
  <a href="https://arxiv.org/abs/2609.31005"><img src="https://img.shields.io/badge/arXiv-78909C?style=flat-square" alt="arXiv"></a>
  <a href="https://www.youtube.com/watch?v=cfQyDyOGiNU"><img src="https://img.shields.io/badge/YouTube-E57373?style=flat-square" alt="YouTube"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-BSD--3--Clause-green" alt="License: BSD-3-Clause"></a>
  <img src="https://img.shields.io/badge/ROS_2-Jazzy-blue" alt="ROS 2: Jazzy">
</div>

TʀᴀᴄᴋGʀᴀᴘʜ constructs online open-vocabulary 3D scene graphs by propagating and tracking mask
identities in image space using dense DINOv3-features before fusing them into persistent, class-agnostic 3D
segments. 3D association reconciles tracking interruptions and revisits, while
compact multi-view CLIP galleries enable language- and image-guided object search.

This repository contains the TʀᴀᴄᴋGʀᴀᴘʜ mapping core, which extends [Hydra](https://github.com/MIT-SPARK/Hydra) with online open-vocabulary mapping.

<img width="1000" alt="trackgraph-image-for-readme" src="https://github.com/user-attachments/assets/386f9446-5d68-4eb4-aeca-2ed37a30b5d2" />

## Setup and usage

Follow the [TʀᴀᴄᴋGʀᴀᴘʜ setup guide](https://github.com/ntnu-arl/trackgraph_ros/tree/main#setup)
for Ubuntu 24.04 and ROS 2 Jazzy, including dataset and robot launch instructions.
The guide imports all required repositories and selects the matching branches.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{hellesylt2026trackgraph,
  title={{TRACKGRAPH}: Online Open-Vocabulary 3D Scene Graphs via Image-Space Tracking},
  author={Hellesylt, Peder Borge and Gassol Puigjaner, Albert and Alexis, Kostas and Stahl, Annette},
  journal={arXiv preprint arXiv:2609.31005},
  year={2026},
  url={https://arxiv.org/abs/2609.31005}
}
```

## License

Released under BSD-3-Clause.

## Acknowledgements

This work was supported in part by:

- **Research Council of Norway** through **NCEI** (Grant No. **357451**).
- **European Commission** through **SYNERGISE** under the **Horizon Europe Programme** (Grant No. **101121321**).
