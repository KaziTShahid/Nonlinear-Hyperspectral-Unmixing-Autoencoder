# Nonlinear-Hyperspectral-Unmixing-Autoencoder
This is the code for an autoencoder that performs nonlinear pixel unmixing on hyperspectral images (based on the Fan, Bilinear, PPNM models, and also higher order nonlinear terms)

The three mixing models used here are the Fan, Bilinear and PPNM models.

Also, you can have higher order nonlinear terms, like 3rd or 4th degree cross-products instead of only upto the 2nd degree cross products. For this, change the "upto_how_many_degrees" parameter.

# Main Files

"autoencoder_main.py" is the main file to run. It uses one dataset currently, the "PaviaU" dataset. If you want to add more datasets, download more from the link below
http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes
Once you download other datasets, make sure to include the filenames in the "dataset_choices" variable.

"rbf_kazi.py" is the first layer, finding abundances.

"nonlin_layer_kazi.py" unmixes according to the Fan or the Bilinear model.
"ppnm_layer_kazi.py" unmixes according to the PPNM model. The model to choose will come from the "mixing_models" variable.

# Citation

If you wish to use this code, please cite the link where the datasets come from.
http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes
Also, cite the link where this code was from
https://github.com/KaziTShahid/Nonlinear-Hyperspectral-Unmixing-Autoencoder
The paper published which uses this code will be added later on.
