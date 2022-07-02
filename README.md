# DEMOC_code
## Requirement:

Python --- 3.7.4
Keras --- 2.3.1
Tensorflow --- 1.15.0

## Usage:

### Input CITE-seq dataset
   View1: RNA data, a M*N matrix with M genes and N cells
   View2: Protein data, a D*N matrix with D proteins and N cells
   View3: Imputed RNA data, a M*N matrix with M proteins and N cells

### Input scRNA-seq dataset
   View1: RNA data, a M*N matrix with M genes and N cells
   View2: Imputed RNA data, a M*N matrix with M proteins and N cells

### settings in main.py
  (Default) testing = True, when testing = True, the code just test the trained DEMOC model  
  
  (Default) train_ae = False, when train_ae = Ture, the code will pre-train the autoencoders first, and the fine-turn the model with DEMOC

### run the code：
  Input CITE-seq dataset: python main.py --Multi_view=3
  Input scRNA-seq dataset: python main.py --Multi_view=2
