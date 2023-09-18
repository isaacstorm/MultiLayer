# MultiLayer
<p>Transfer integral is a crucial parameter determines the charge mobility of organic semiconductors. The quantum chemical calculation of transfer integrals for all the molecular pairs in organic materials is usually an unaffordable task. Luckily this can be accelerated by the data-driven machine learning method. In this project, we develop machine learning models based on artificial neutral networks to predict transfer integrals accurately and efficiently. </p>
<p>This project contains script files required for feature generation, feature screening, model training and transfer integral prediction for four organic semiconductor molecules</p>
<p><strong>########## Software/Libraries Requirement ##########</strong></p>
<ul>
<li>Python 3.8</li>
<li>Scikit Learn version 0.24.1</li>
<li>Numpy version 1.16.2</li>
<li>PyTorch version 1.8.0</li>
</ul>

<p>Folder model contain scripts required for dataset generation, model training and transfer integral prediction for quadruple thiophene packing with both dynamic and static disorders</p>
<li>data0.txt #XYZ coordinates of each atom in a molecular pair extracted from MD simulations, unit in nm</li>
<li>ce0 #cell vectors of periodic box used in MD simulations, arranged by XX YY ZZ XY XZ YX YZ ZX ZY, unit in nm</li>
<li>id0 #nearest neighbor list file  , serial number starting from 1</li>
<li>elelist #PDB format file of a dimer, new structure needs to be generated in this order to ensure that scripts read the atomic order correctly.</li>
<li>import.txt #Importance ranking obtained using the feature filtering method  </li>
<li>make321mutil.py #Scripts to generate overlap matrix, the parameters for element specificity already included in the script. Overlap matrix elements of hydrogen atoms or intra-molecular terms will not be created.</li>
<li>jefflod0.txt #Effective transfer integrals obtained by quantum chemical calculations and Lowdin’s orthogonalization. Units in Hartree and non-physical phase has not been corrected.</li>
<li>makefilter.py #Scripts for feature filtering</li>
<li>pred.py #Scripts for transfer integral prediction</li>
<li>finetune.py #Neural network training program for fine-tuning The fine tuned model will be saved as mlplossyGPB0&90.pth, which can be called by the torch.load </li>

<p><strong>########## Download Model and Dataset  ##########</strong></p>
The dataset, pre-trained model mlp.pth and fine-tuned model mlplossyGPB0&90.pth can be downloaded at , and use

  ```
  torch.load(PATH)
  ```
to load the model

<p><strong>########## Prediction with Models ##########</strong></p>
<ol>
<li>Prepare your own 3D coordinate files, lattice vector files and nearest neighbor list files according to the sample files provided, making sure that makefilter.py, make321mutil.py, id0, data0.txt, ce0 and import.txt are all placed in the same folder and the atomic order and file format are the same as those provided in the project which can be obtained by opening the elelist with a visualization program. </li>
</li>
<li>Modify make321mutil.py and makefilter.py  and run </li>
  <li>Run </li>

  ```
  ./make321mutil.py
  ```
<li>A feature file A321exx0.txt will be generated </li>
<li>Modify  makefilter.py  and run </li>

  ```
  ./makefilter.py
  ```
 
<li>Filtered feature file A321exx0_edit1.txt will be generated </li>
<li>Modify pred.py and run</li>
   
  ```
  ./pred.py
  ```
<li>Predict file will be generated</li>
</ol>


<p><strong>########## Training yourself models ##########</strong></p>
<ol>
<li>1.	make sure you have download the pre-trained model and dataset file, You can also generate your own dataset file form Filtered feature file and transfer integrals file by run</li>
</ol>
  ```
  X =open('A321exx0_edit1.txt','r')
  X = X.read()
  X = X.split()
  X = np.array(X).reshape((-1,3200))
  X = X.astype(float)
  Y =open('jefflod0.txt','r') 
  Y = Y.read()
  Y = Y.split()
  Y = np.array(Y)
  Y = Y.astype(float)
  np.savez('dataset.npz',X,Y)
  ```
<ol>
<li>2.	Modify the python interpreter path to your own dataset path and pre-trained model path </li>
<li>Run </li>
  
  ```
  ./finetune.py
  ```
  The performance of the model will be printed to the screen during training.
<li>Model file mlplossyGPB0&90.pth will be generated</li>
</ol>
