# ImageCLEFmedical-gan-2023

Scripts, figures, and working notes for our team's participation in [ImageCLEFmed GAN task 2023](https://www.imageclef.org/2023/medical/gans), part of the [ImageCLEF labs](https://www.imageclef.org/2023) at the [14th CLEF Conference, 2023](https://clef2023.clef-initiative.eu/index.php).

**Implementation Stack**: Python, PyTorch, Scikit-learn.

## Quick Links

- [Manuscript [PDF]](https://ceur-ws.org/Vol-3497/paper-116.pdf) describing the methods, rationale, and results.
- [Contest Description and Resources](https://www.imageclef.org/2023/medical/gans).
- [Model Pipelines](./src/).


## Cite Us

[Link to the Research Paper](https://ceur-ws.org/Vol-3497/paper-116.pdf)

If you find our work useful in your research, don't forget to cite us!

```
@article{hb2023correlating,
  url = {https://ceur-ws.org/Vol-3497/paper-116.pdf},
  title={Correlating Biomedical Image Fingerprints between GAN-generated and Real Images using a ResNet Backbone with ML-based Downstream Comparators and Clustering: ImageCLEFmed GANs, 2023},
  author={Bharathi, Haricharan and Bhaskar, Anirudh and Venkataramani, Vishal and Desingu, Karthik and Kalinathan, Lekshmi},
  year={2023},
  keywords={Ensemble Learning, Convolutional Neural Networks, Gradient Boosting Ensemble, Metadata-aided Classification, Image Classification, Transfer Learning},
  journal={Conference and Labs of the Evaluation Forum},
  publisher={Conference and Labs of the Evaluation Forum},
  ISSN={1613-0073},  
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## Key Highlights

- A deep learning -based feature extraction and subsequent boosting ensemble approach for fungi species classification is proposed.
- Leverages state-of-the-art deep learning architectures such as ResNeXt and Efficient-Net among others and trains them by transfer learning onto a fungi image dataset for feature extraction.
- Finally, integrates the output representation vectors with geographic metadata to train a gradient boosting ensemble classifier that predicts the fungi species. 
- The authors trained multiple deep learning architectures, assessed their individual performance, and selected effective feature extraction models.

