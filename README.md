## Prediction of Pneumonia Based in Chest X-Ray Images with Streamlit-Based Website

### Project Description 
Health issues are a global concern due to their direct impact on humans, such as lung disorders that can lead to death if ignored. Lung disorders like shortness of breath, chest pain, and productive cough often indicate Pneumonia, a lung infection caused by viruses, bacteria, or fungi.

Globally, Pneumonia cases range from 1.5 to 14 per 1,000 people per year. In the United States, there are 24.8 cases per 10,000 people, while in the Philippines and Malaysia, the rates are 1425 and 99 cases per 10,000 people, respectively, among discharged hospital patients.

Pneumonia detection starts with symptom examination, followed by a chest X-ray, which is the "gold standard" in radiological examination. The X-ray results are then analyzed by a doctor to confirm the diagnosis.

To assist doctors in diagnosing Pneumonia, a prediction tool based on chest X-ray images is needed. Thus, a Streamlit-based website tool has been developed, allowing users to upload chest X-ray images and automatically receive predictions and confidence percentages of the results without replacing the doctor's role.

### Model Deep Learning
The deep learning model used is a convolutional neural network (CNN), specifically a modified ResNet50. These modifications were made to improve the accuracy and balance other evaluation metrics such as precision, recall, and F1-score.

The modifications include adding batch normalization layers, ReLU activation, dropout, and fully connected layers at the end of the architecture. The dropout rate used is 0.5. Additionally, the model will be trained using parameters with a batch size of 128, 23 epochs, a learning rate of 0.0001, and a weight decay of 0.0001.

![Modifikasi ResNet50 drawio](https://github.com/user-attachments/assets/0810abcb-7a15-42ff-9e8f-e597c8cd11a8)

![Detail ResNet50 drawio (1)](https://github.com/user-attachments/assets/519172a1-28e7-4304-80fc-3380657f0ee9)

### Dataset
The dataset used is from Kermany [Kermany D., et al.](https://data.mendeley.com/datasets/rscbjbr9sj/3), consisting of Chest X-Ray images. This dataset is divided into three parts (train, test, validation) and contains 5,885 X-Ray images (JPEG) with two categories in each part (Pneumonia and Normal). The chest X-ray images (anterior and posterior) were selected from pediatric patients aged one to five years old from the Women and Children’s Medical Center, Guangzhou.

The training data consists of 1,346 normal images and 3,875 pneumonia images, which are further split into 80% for model training and 20% for model validation. The test data includes 234 normal images and 410 pneumonia images used for testing the trained model. The [validation data](https://github.com/fariqfirdaus/Streamlit_Pneumonia_pytoch/tree/main/val) comprises 10 normal images and 10 pneumonia images for final validation of the model integrated into Streamlit.

### Screenshots (Results & [Web Interface](https://prediksi-pneumonia.streamlit.app/))
- Web Interface - Home Page

![Home Page](https://github.com/user-attachments/assets/a68b2838-de31-4220-8eb9-530232ec4fe9)

- Web Interface - Classification Results via Web Interface

![Prediction-Result](https://github.com/user-attachments/assets/3be5b8db-a9a5-4c57-8a27-2f7204e19533)

- Classifier Evaluation - Loss and Accuration

![Loss and Accuration](https://github.com/user-attachments/assets/061da0c4-899e-42bd-a4e4-a8a1157cf748)

- Classifier Evaluation - Evaluation Model

![Evaluation model](https://github.com/user-attachments/assets/ecb1ba1c-a105-4075-af8c-84ea6a62293b)

- Classifier Evaluation - Confusion Matrix

![Confusion Matrix](https://github.com/user-attachments/assets/caf6f4f7-44f1-4991-b105-59c96f0b2049)


### About
The Product was developed by Moh. Fariq Firdaus Karim
