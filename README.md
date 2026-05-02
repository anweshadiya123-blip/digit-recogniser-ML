### Installation

### Prerequisites
- Python 3.9+
- CUDA 11.7 (optional, for GPU support)

### Install Dependencies

1. Clone the repository:
```bash
git clone https://github.com/anweshadiya123-blip/digit-recogniser-ML.git
cd digit-recogniser-ML
```
2.Create and activate virtual environment
``` bash
python -m venv my_venv
venv\Scripts\activate
source venv/bin/activate
```
3.Install requirements
```bash
pip install -r requirement.txt
```
4.Training the model
```
python new_train.py
```
5.Testing the model
```
python test.py
```
### 6.Model Features

1.Accuracy: High accuracy on MNIST test set
2.Confusion Matrix: Visual evaluation of prediction performance
3.Training Progress: Real-time loss and accuracy tracking
4.Model Persistence: Save and load trained models
