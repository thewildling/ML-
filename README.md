# Managing data for CM4Smart :floppy_disk:

This project looks at comparing ResNet50 and KNN for classifying X-rays of pneumonia. 

## Projectstructure :clipboard:

![alt text](https://github.com/anderf2706/ML-/blob/main/Images/structure.png)

**Structure**

## Getting Started :checkered_flag:

To get started, download this repo either by cloning it from your IDE, or download it as a zip. 
The majority of methods can be found in ML.py and KNN_classifier.py.

The images used for training, should be put in the respective folders described in the readme-files found in the data folders.

## Running the system :rocket:

Both of the main py-files contain a main method at the bottom of the files, and can be run as is. 

## Methods :star:

methods from ML.py and KNN_classifier.py

ML.py:

### make_pred(Learner learn, string file):
Make predictions on a single image.

### make_and_train():
Creates the data used for the model. Then creates the model for the ResNet50, and trains it for 2-4-20-10 epochs. To save the model, uncomment the learn.save() line.

### interp():
creates a confusion matrix for the learner.

### load_model():
takes in the saved model from the folder specified in the train method, and loads it back, to be used without re-training. 

KNN_classifier.py:

### dataset():
Takes the images from the x-ray folder, and creates test, train and validation sets to be used for the model. 

### save_model(Trainer classifier):
saves the model to be used later.

### load_model():
loads the model saved by the save_model method to be used for classifyinfg 

### plot_best_K():
plots a graph showing what the best K-value is for the model. 


### create_active_table()
making active table for M3 without rotation and temperature, removed nan values, and duplicates

### creating_all_frames()
run when first cloning project, creates all fullframes

### Adding new machines :heavy_plus_sign:
Since many of the machines are different in the info they give us, the procedure for adding a new machine, could be fairly easy, if it follows the same structure, 
and gives the same sensoroutputs as one of the 4 machines made. 

### step 1:
The first thing you need to do, is put the csv files you have been given, into the Data folder udner CM4Smart->src->Data

### step 2:
Now you have to import them into the system. The way you do this, is going to the readingCSV-file, and down to the loadcsv_to_pickle method. Here there is 3 different
types. It is commented in the code, which to use where. Copy one of the lines, and type in a string being the name of the file in data-folder.

![alt text](https://github.com/anderf2706/CM4Smart/blob/master/docs/img/loadcsv.png)

**The file to change is on the bottom**

When you have set everything up here, go to Analyze_data. On line 23, change pickles(False) -> True. Do not run anything else. You will then see outputs to your console,
followed by a crash. This is normal, and isn't a problem. 

### step 3:
Go to indexing_files in Analyzer-pack, and copy one block of code as is shown here:

![alt text](https://github.com/anderf2706/CM4Smart/blob/master/docs/img/index_files.png)

**Code-snippet**

Take this codesnippet and change the name of the machine-nr first, then change the path to the files you generated using loadcsv. These are located in src->pickle_data.

### part 4:
Go to the Lists of all readings section near the bottom, looking like this:

![alt text](https://github.com/anderf2706/CM4Smart/blob/master/docs/img/list_files.png)

**Lists-section. Add one**

Add in a new list, by copying one of the existing ones, and change both names of new files in first input, second is abreviation of name, look to previous to see, 
Then third is the new machine nr.

### part 5:
This path is a little more work to get done. Becuase of the nature of the data we are given, the machines dosen't follow the same rules when it comes to structure, and what they return. This means that making a default method for indexing, is very difficult. So to add a new machine, you have to see the structure, and see if they have the same as the 4 machines already in. If they do, you have to go through index_on_month, and index_on_order and add the new machine to the right place. You have to walk through the method to add to all the places where machine-nr is used. 
If the data looks different than the machines already here, you have to add new methods. You can copy the index method already there, and use it as a template. 

### part 6:
When these methods are fixes, then you can run sort and sort_on_order. An example of this is in picture under run system at the top of this doc. When these are generated, a order folder should be found under pickle_data. If you have these files, you can start using all the other methods for manipulating the data. 


### Issue for now:
So many of the machines give different outputs, both regarding data and format, so making a standard method is not easy. 

### Prerequisites :white_check_mark:
-------------------


-pandas-profiling(need to download with conda)

numpy==1.18.5

pandas == 1.1.1

seaborn == 0.10.1

matplotlib == 3.3.1


-for installing fastai:

conda install fastai pytorch=1.0.0 -c fastai -c pytorch -c conda-forge
 
-for installing pandas-profiling:

conda install -c conda-forge pandas-profiling

## Authors :pencil2:

Anders Fredriksen and Fridtjof Høyer 
