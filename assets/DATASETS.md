# Biomedical Domain Datasets

Our study includes 15 biomedical image classification datasets. Place all the datasets in one directory under `data` to ease management. The file structure looks like

```
data/
|–– BRISC/
|–– BTMRI/
|–– BTMRI-P/
|–– BTMRI-S/
|–– BUID/
|–– BUSBRA/
|–– BUSI/
|–– CHMNIST/
|–– COVID_19/
|–– CTKidney/
|–– Kvasir/
|–– LungColon/
|–– OCTMNIST/
|–– RETINA/
|–– UDIAT/
```

## Datasets Description
| **Modality**               | **Organ(s)**      | **Name**                                                                                           | **Classes**                                                                                                       | **# train/val/test** |
|:---------------------------:|:-----------------:|:-------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------:|:--------------------:|
| Computerized Tomography     | Kidney            | [CTKidney](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone)| Kidney Cyst, Kidney Stone, Kidney Tumor, Normal Kidney                                                            | 6221/2487/3738       |
| Dermatoscopy                | Skin              | [DermaMNIST](https://medmnist.com/)                                                                | Actinic Keratosis, Basal Cell Carcinoma, Benign Keratosis, Dermatofibroma, Melanocytic nevus, Melanoma, Vascular Lesion | 7007/1003/2005       |
| Endoscopy                   | Colon             | [Kvasir](https://www.kaggle.com/datasets/abdallahwagih/kvasir-dataset-for-classification-and-segmentation)| Dyed Lifted Polyps, Normal Cecum, Esophagitis, Dyed Resection Margins, Normal Pylorus, Normal Z Line, Polyps, Ulcerative Colitis | 2000/800/1200        |
| Fundus Photography          | Retina            | [RETINA](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)               | Cataract, Diabetic Retinopathy, Glaucoma, Normal Retina                                                           | 2108/841/1268        |
| Histopathology              | Lung, Colon       | [LC25000](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)| Colon Adenocarcinoma, Colon Benign Tissue, Lung Adenocarcinoma, Lung Benign Tissue, Lung Squamous Cell Carcinoma   | 12500/5000/7500      |
| Histopathology              | Colorectal        | [CHMNIST](https://www.kaggle.com/datasets/kmader/colorectal-histology-mnist)                        | Adipose Tissue, Complex Stroma, Debris, Empty Background, Immune Cells, Normal Mucosal Glands, Simple Stroma, Tumor Epithelium | 2496/1000/1504       |
| Magnetic Resonance Imaging  | Brain             | [BTMRI](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)                  | Glioma Tumor, Meningioma Tumor, Normal Brain, Pituitary Tumor                                                     | 2854/1141/1717       |
| Magnetic Resonance Imaging  | Brain             | [BTMRI-P](https://www.kaggle.com/datasets/pradeep2665/brain-mri)                  | Glioma Tumor, Meningioma Tumor, Normal Brain, Pituitary Tumor                                                     | 8000/0/1000       |
| Magnetic Resonance Imaging  | Brain             | [BTMRI-S](https://www.kaggle.com/datasets/mohamedmetwalysherif/braintumordataset)                  | Glioma Tumor, Meningioma Tumor, Normal Brain, Pituitary Tumor                                                     | 2451/0/529       |
| Magnetic Resonance Imaging  | Brain             | [BRISC](https://www.kaggle.com/datasets/briscdataset/brisc2025)                  | Glioma Tumor, Meningioma Tumor, Normal Brain, Pituitary Tumor                                                     | 2498/0/1000       |
| Optical Coherence Tomography| Retina            | [OCTMNIST](https://medmnist.com/)                                                                 | Choroidal Neovascularization, Drusen, Diabetic Macular Edema, Normal                                             | 97477/10832/1000     |
| Ultrasound                  | Breast            | [BUSI](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)                | Benign Tumors, Malignant Tumors, Normal Scans                                                                    | 389/155/236          |
| Ultrasound                  | Breast            | [BUID](https://qamebi.com/breast-ultrasound-images-database)                | Benign Tumors, Malignant Tumors                                                                    | 162/0/36          |
| Ultrasound                  | Breast            | [BUSBRA](https://www.kaggle.com/datasets/orvile/bus-bra-a-breast-ultrasound-dataset)                | Benign Tumors, Malignant Tumors                                                                   | 1311/0/283          |
| Ultrasound                  | Breast            | [UDIAT](https://www.kaggle.com/datasets/ayush02102001/udiat-segmentation-dataset)                | Benign Tumors, Malignant Tumors                                                                | 113/0/26           |
| X-Ray                       | Chest             | [COVID-QU-Ex](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)          | COVID-19, Lung Opacity, Normal Lungs, Viral Pneumonia                                                             | 10582/4232/6351      |
| X-Ray                       | Knee              | [KneeXray](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity) | No, Doubtful, Minimal, Moderate, and Severe Osteoarthritis                                                       | 5778/826/1656        |


### Download the datasets
All the datasets can be found [here](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/tree/main) on HuggingFace. Download each dataset seperately:

#### Few-shot Evaluation Datasets

- <b>BTMRI</b> [[Drive](https://drive.google.com/file/d/1_lJLZRUmczqZqoN-dNqkAzGzmi4ONoU5/view?usp=sharing) | [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/BTMRI.zip)]
- <b>BUSI</b> [[Drive](https://drive.google.com/file/d/1hB5M7wcAUTV9EtiYrijACoQ36R6VmQaa/view?usp=sharing) | [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/BUSI.zip)]
- <b>CHMNIST</b> [[Drive](https://drive.google.com/file/d/1tyQiYQmqAGNaY4SCK_8U5vEbbaa1AD-g/view?usp=sharing) | [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/CHMNIST.zip)]
- <b>COVID_19</b> [[Drive](https://drive.google.com/file/d/1zMLN5q5e_tmH-deSZQiY4Xq0M1EqCrML/view?usp=sharing) | [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/COVID_19.zip)]
- <b>CTKidney</b> [[Drive](https://drive.google.com/file/d/1PBZ299k--mZL8JU7nhC1Wy8yEmlqmVDh/view?usp=sharing) | [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/CTKidney.zip)]
- <b>Kvasir</b> [[Drive](https://drive.google.com/file/d/1T_cqnNIjmGazNeg6gziarvCNWGsFEkRi/view?usp=sharing) | [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/Kvasir.zip)]
- <b>LungColon</b> [[Drive](https://drive.google.com/file/d/1YIu5fqMXgyemisiL1L1HCvES2nVpCtun/view?usp=sharing) | [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/LungColon.zip)]
- <b>OCTMNIST</b> [[Drive](https://drive.google.com/file/d/1mYZNWxbPxnnVvcwHQYybA8gdMzQAoOem/view?usp=sharing) | [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/OCTMNIST.zip)]
- <b>RETINA</b> [[Drive](https://drive.google.com/file/d/18U-Gc22h5QryomNNzY4r4Qfrq52yf5EO/view?usp=sharing) | [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/RETINA.zip)]

#### Domain Generalization Datasets

- <b>BTMRI-P</b> [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/BTMRI-P.zip)
- <b>BTMRI-S</b> [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/BTMRI-S.zip)
- <b>BRISC</b> [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/BRISC.zip)
- <b>BUID</b> [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/BUID.zip)
- <b>BUSBRA</b> [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/BUSBRA.zip)
- <b>UDIAT</b> [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/UDIAT.zip)

After downloading each dataset, unzip and place each under its respective directory like the following

```
BTMRI/
|–– BTMRI/
|   |–– glioma_tumor/
|   |–– meningioma_tumor/
|   |–– normal_brain/
|   |–– pituitary_tumor/
|–– split_BTMRI.json
```
