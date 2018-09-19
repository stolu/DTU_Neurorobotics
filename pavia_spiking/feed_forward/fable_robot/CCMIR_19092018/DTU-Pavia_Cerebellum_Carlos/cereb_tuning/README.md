
#############################################################
#HOW TO RUN AN EXPERIMENT IN THE NRP WITH THE SPIKING CEREB?#
#############################################################
#1. Modifiy the initial values for the test 
1.1. cd to: ~/Documents/NRP/Models/brain_model/
     Copy the directory /DTU_spiking_cereb in the "brain_model" directory

1.2. Modify the different values depending on your needs in the file "init_parmeters_dtu_pavia_cererb.py"

1.2.1. It is IMPORTANT that everytime you are runing a new test you modify:
	- TEST: 
	set the TEST variable to a test number that does not exist.
1.2.2. Select the results directory and create a test directory
	- RESULTS_PATH: (this one does not have to be change everytime you tun a test!!) 
	<your-desired-location>/cereb_tuning/results
	- MANUALLY create a directory with the name TEST inside the RESULTS_PATH directory.
	this is the path were all the spike_detector of the test TEST will be stored when running an 		experiment.
1.2.3. Modify the rest of the parameters depending on which step of the tuning you are

1.3. Now you can launch your experiment and press "play"... but do not press "leave" yet or you may loose some information. Now you can go to section 2.
 
#2. Save CSV files if needed
#This step will be most commonly done when the static tuning has been accomplished.
2.1. After runing your experiment, before clicking "leave" in the frontend of the NRP:
	- Press "pause" to pause the simulation
	- Open the "toggle experiment editors" and in the "transfer function" tab, click on "Save CSV". 	This will save two .csv files with information related to the joints and the control inputs.
	- Go to /home/<your-username>/.opt/nrpStorage/<name-of-the-current-experiment>
	In my case it looked like this <name-of-the-current-experiment>: 				fable_manipulation_dtu_spiking_cereb_v2.0_1
	- In this directory, everytime you click on "Save CSV" a new folder will be created containing two files.
	- Copy this files to the RESULTS_PATH/TEST (this directory was manually created in step 1.2.2):
	For example, if TEST=3:
	<your-desired-location>/cereb_tuning/results/3
2.2 Now you can leave the experiment with no problem

######################################################################################
#HOW TO ANALYZE THE RESULTS TO PROCEED WITH THE TUNING OF A SPIKING CEREB IN THE NRP?#
######################################################################################
#3. Ensure you have run the prevous section before continuing with the "how to?", otherwise you will probably run into problems.

#4. cd to: <your-desired-location>/cereb_tuning/src
4.1 Open and modify the "main_tuning.py"
	- TEST: This is the test number that you want to analyze. The available ones are stored in 		RESULTS_PATH (from the previous section 1.3)
	- JOINTS: list of the joints that you want to analyze
	- MODEL: name of the model and number of neurons per population of the model
	- After instanciating the CerebTestTune class, you can call whatever method you need. Notice that 
4.2 Run the "main_tuning.py"
