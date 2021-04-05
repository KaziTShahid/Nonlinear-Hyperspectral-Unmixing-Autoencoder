# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 14:58:51 2020

@author: Kazi T Shahid and Ioannis D. Schizas

This code is used for the paper ""

This will generate an autoencoder for unsupervised nonlinear hyperspectral unmixing

If you wish to run the code as-is, download the "Pavia University" dataset from the URL below:
http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes

If you wish to add more datasets, add their names to the list called 'dataset_choices' in the subsection 'hyperparameters'
Also, specify the classes to choose to create the datasets in the subsection 'building data'

If you wish to use this code, please cite the URL given above for the dataset, and also the URL where this code was downloaded from:
https://github.com/KaziTShahid/Nonlinear-Hyperspectral-Unmixing-Autoencoder


"""



def lin_mixing(ref_pixels,all_percentages,num_classes):
    
    all_pixels = []    
        
    for i in range(len(all_percentages)):
        
        abundances = all_percentages[i]
        current_pixel = np.zeros([1,len(ref_pixels[0])])
        
        for j in range(num_classes):        
            current_pixel += abundances[j]*ref_pixels[j]
        all_pixels.append(np.transpose(current_pixel))
        
    return all_pixels


def bilin_mixing(ref_pixels,all_percentages,num_classes,TAKE_REPEATING_PRODUCTS,gamma,upto_how_many_degrees):
    
    
    classes = list(range(num_classes))
    pairs=[]
    for i in range(2,upto_how_many_degrees+1):    
    
        if (TAKE_REPEATING_PRODUCTS==1):
             pairs += list(itertools.product(classes,repeat = i))
        else: pairs += list(itertools.combinations(classes, i))
    
    all_pixels = []    
        
    for i in range(len(all_percentages)):
        
        abundances = all_percentages[i]
        current_pixel = np.zeros([1,len(ref_pixels[0])])
                
        for j in range(num_classes):        
            current_pixel += abundances[j]*ref_pixels[j]
        for j in range(len(pairs)):
            
            pair = pairs[j]
            current_pixel += gamma * (abundances[pair[0]]*ref_pixels[pair[0]]) * (abundances[pair[1]]*ref_pixels[pair[1]])
        
        all_pixels.append(np.transpose(current_pixel))
    
    return all_pixels

def ppnm_mixing(ref_pixels,all_percentages,num_classes,b_s):
    
    all_pixels = []    
        
    for i in range(len(all_percentages)):
        
        abundances = all_percentages[i]
        current_pixel = np.zeros([1,len(ref_pixels[0])])
        
        for j in range(num_classes):        
            current_pixel += abundances[j]*ref_pixels[j]
        current_pixel += np.multiply(current_pixel,current_pixel) * b_s[i]
        all_pixels.append(np.transpose(current_pixel))
        
    return all_pixels



import math
import scipy.io
import random
import tensorflow as tf
import logging
logging.getLogger('tensorflow').disabled = True
import time
import numpy as np
# import matplotlib.pyplot as plt
from scipy import linalg
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds, eigs
from collections import Counter
import itertools
# import pickle

from sklearn.cluster import KMeans
from rbf_kazi import RBFLayer
from nonlin_layer_kazi import NONLIN_Layer
from ppnm_layer_kazi import PPNM_Layer


#%% hyperparameters

iterations = 5 #how many times the synthetic data will be generated again
timer = 0 # calculating how much total time the network takes for all iterations, across all datasets

upto_how_many_degrees = 2 #upto how many degree cross-product terms to consider
if upto_how_many_degrees < 2:
    raise Exception('upto_how_many_degrees has to be equal to at least 2')


b_s_lb = -0.3 # lower bound for scaling factor with ppnm method
b_s_ub = 0.3 # upper bound for scaling factor with ppnm method

optimizer = tf.optimizers.Adam(learning_rate=0.0001)
num_epochs = 50

SNR_values = [0,5,10,15,20]

pixels_per_class = 1000 # how many mixed pixels will have majority abundance of each class

gamma = 1 # scaling factor for bilinear model (setting gamma=0 is the same as making Linear Mixing Model)

main_material_percentage_max = 90 #upper bound for majority abundance of one class
main_material_percentage_min = 80 #lower bound for majority abundance of one class

if main_material_percentage_max <= main_material_percentage_min:
    raise Exception('choose max that is higher than the min')


    

mixing_models = ['fan', 'bilin', 'ppnm']
dataset_choices = ['PaviaU']
        
rmse_values = np.zeros([len(mixing_models),len(dataset_choices),len(SNR_values),iterations])    
all_sad_rec = np.zeros([len(mixing_models),len(dataset_choices),len(SNR_values),iterations])    
all_sad_cent = np.zeros([len(mixing_models),len(dataset_choices),len(SNR_values),iterations])     
# all_est_abundances = np.zeros([len(mixing_models),len(dataset_choices),len(SNR_values),iterations,total_pixels,num_classes])    


#%% choosing between FAN and BILINEAR model
    
for mix_model_index in range(len(mixing_models)):
        
    mixing_model = mixing_models[mix_model_index]
    unmixing_layer = mixing_model
    
    if mixing_model == 'fan':
        TAKE_REPEATING_PRODUCTS = 0
    elif mixing_model == 'bilin': 
        TAKE_REPEATING_PRODUCTS = 1
    
    #%% choosing dataset

    for dataset_index in range(len(dataset_choices)):
            
        dataset_choice = dataset_choices[dataset_index]


        #%% building data
        
        ground_truth = scipy.io.loadmat(dataset_choice+'_gt.mat')
        data = scipy.io.loadmat(dataset_choice+'.mat')
        
        if dataset_choice == 'PaviaU':
            dataset_gt = ground_truth['paviaU_gt']
            dataset = data['paviaU']            
            data_to_choose = [1,4,5,9] #chooses which classes to take reference pixels from
        
            
        #%% automatically determined parameters    
        dims = dataset.shape        
        num_classes = len(data_to_choose)
        total_pixels = pixels_per_class*num_classes
        
        if upto_how_many_degrees > num_classes:
            raise Exception('upto_how_many_degrees cannot exceed value of num_classes')
        
        dataset = dataset/np.max(dataset) #normalizing data to unit peak
        
        for all_index in range(iterations):

            print('['+str(mix_model_index)+' '+str(dataset_index)+' '+str(all_index)+' '+'] out of ['+str(len(mixing_models)-1)+' '+str(len(dataset_choices)-1)+' '+str(iterations-1)+' '+']')
        
        #%% finding reference pixels
            
            if dataset_choice != 'Cuprite':  
            
                all_pixel_locations = []
                ref_pixels = []
                for i in range(num_classes):
                    current_class = data_to_choose[i]
                    current_class_locations = np.where(dataset_gt==current_class) #finding where pixels of current class exist
                    x = current_class_locations[0]
                    y = current_class_locations[1]
                    random.seed(all_index*i)
                    loc = random.randint(0,len(x)-1) #randomly choosing reference pixel(endmember) of current class
                    ref_pixels.append(dataset[x[loc],y[loc],:])
    
            else: 
                ref_pixels = []
                for i in range(num_classes):
                    ref_pixels.append(dataset[data_to_choose[i]-1,:]) #in Cuprite, we only have a dictionary of reference pixels
                    
                    
            # %% generating mixed pixels

            all_percentages = []
            
            tally = 0
            for i in range(num_classes):
                for j in range(pixels_per_class):
                    
                    tally += 1
                    random.seed(all_index*tally)
                    main_percentage = random.randint(main_material_percentage_min,main_material_percentage_max) #abundance of the one majority endmember, randomly chosen within predefined bounds
                    remaining_percentages = np.zeros([num_classes-1,1])
            
                    for k in range(num_classes-1):
                        random.seed(all_index*tally*k)
                        remaining_percentages[k,0] = random.random()
                    remaining_percentages = remaining_percentages/sum(remaining_percentages)*(100-main_percentage) #remaining abundances are randomly chosen
                    current_percentages = np.zeros([num_classes,1])
                    current_percentages[i] = main_percentage
                    current_percentages[list(set(list(range(num_classes)))-set([i]))] = remaining_percentages
                    
                    current_percentages /= 100
                    all_percentages.append(np.squeeze(current_percentages))
                
            np.random.seed(all_index*1000) # can be replaced with iteration number afterwards
            b_s = np.random.uniform(b_s_lb, b_s_ub, len(all_percentages))
            
            #%% choice of mixture model        

            if mixing_model=='lin':
                all_pixels = np.squeeze(lin_mixing(ref_pixels,all_percentages,num_classes))
            elif mixing_model=='fan':
                all_pixels = np.squeeze(bilin_mixing(ref_pixels,all_percentages,num_classes,TAKE_REPEATING_PRODUCTS,gamma,upto_how_many_degrees))
            elif mixing_model=='bilin':
                all_pixels = np.squeeze(bilin_mixing(ref_pixels,all_percentages,num_classes,TAKE_REPEATING_PRODUCTS,gamma,upto_how_many_degrees))
            elif mixing_model=='ppnm':
                all_pixels = np.squeeze(ppnm_mixing(ref_pixels,all_percentages,num_classes,b_s))

            orig_data = all_pixels #keeping original mixed pixels before adding noise     
         
            for SNR_index in range(len(SNR_values)):
                
                #%% adding noise
                
                noise_dB = SNR_values[SNR_index]
                
                all_pixels = orig_data
                
                k = 1 / (10** (noise_dB/10) )
                all_pixels = all_pixels + np.random.normal(scale=k*np.max(all_pixels), size=[total_pixels,ref_pixels[0].shape[0]]) # adding noise
                
               
                
                #%% kmeans
                
                kmeans = KMeans(n_clusters=num_classes, random_state=0).fit(np.squeeze(all_pixels))
                kmeans_centers = kmeans.cluster_centers_

        
        
                #%% initializing betas (betas = the inverse of the sigma^2 in the paper, times 2)
                
                input_array = all_pixels
                dists = np.zeros(total_pixels)
                for i in range(total_pixels):
                    temp_dists = np.zeros(num_classes)
                    for j in range(num_classes):
                        temp_dists[j] = np.linalg.norm(input_array[i,:]-kmeans_centers[j,:])
                    dists[i] = np.min(temp_dists)
                    
                betas_kmeans = (1/np.mean(dists)) ** 2
                
                
                if all_index == 0:
                    print('betas from kmeans: ' +str(betas_kmeans))
                    
                    
                #%% defining the autoencoder
                
                
                class Nonlinear_Unmixing_AutoEncoder(tf.keras.Model):
                        
                    def __init__(self):
                        super(Nonlinear_Unmixing_AutoEncoder, self).__init__()
                        
                        self.flatten_layer = tf.keras.layers.Flatten()
                        
                        self.rbflayer = RBFLayer(num_classes, betas_kmeans, centers = kmeans_centers)                
                        
                        self.nonlin_layer = NONLIN_Layer(TAKE_REPEATING_PRODUCTS,upto_how_many_degrees,initial_endmembers = kmeans_centers)
                        
                        self.ppnm_layer = PPNM_Layer(initial_endmembers = kmeans_centers)
                        
                    def call(self, inp):
                
                        x_reshaped = self.flatten_layer(inp)
                        x = x_reshaped
                        
                        rbf_vector = self.rbflayer(x)
                        
                        x1 = tf.divide(rbf_vector, tf.math.reduce_sum(rbf_vector,axis=1)[:, np.newaxis]) # normalizing to unit sum
                
                        classes = list(range(num_classes))
                        
                        pairs=[]
                        for i in range(2,upto_how_many_degrees+1):    
                        
                            if (TAKE_REPEATING_PRODUCTS==1):
                                 pairs += list(itertools.product(classes,repeat = i))
                            else: pairs += list(itertools.combinations(classes, i))  
                        
                        x = tf.pad(x1, ((0,0),(0,len(pairs))), 'constant', constant_values=(0)) #padding
                        
                        mask = []
                        for i in range(len(pairs)):
    
                            pair = pairs[i]
                            abundance_cross_product = tf.ones(total_pixels)
                            for j in range(len(pair)): 
                                abundance_cross_product = tf.math.multiply(abundance_cross_product, x[:,pair[j]])
                                
                            mask.append(abundance_cross_product)
                        
                            
                        mask_tf = tf.convert_to_tensor(mask)
                        mask_tf = tf.pad(mask_tf, ((num_classes,0),(0,0)), 'constant', constant_values=(0)) 
                        new_x = x + tf.transpose(mask_tf) #creating nonlinear abundance vector
                        
                        
                        if (unmixing_layer == 'bilin') or (unmixing_layer == 'fan'):
                            x = self.nonlin_layer(new_x)
                        elif (unmixing_layer == 'ppnm'):
                            x = self.ppnm_layer(x1)
                        
                        
                        return x, x_reshaped, x1
                    
                
                def loss(x, x_bar):
                    return tf.losses.mean_squared_error(x, x_bar)
                def grad(model, inputs):
                    with tf.GradientTape() as tape:
                        reconstruction, inputs_reshaped, x1 = model(inputs)
                        loss_value = loss(inputs_reshaped, reconstruction)
                    return loss_value, tape.gradient(loss_value, model.trainable_variables), inputs_reshaped, reconstruction, x1
                
                
                #%% defining parameters
                
                model = Nonlinear_Unmixing_AutoEncoder()
                global_step = tf.Variable(0)
                reconstructed_pixels = []
                all_estimated_endmembers = []
                norm_errors = []
                norm_errors_endmembers = []
                norm_errors_endmembers_smoothed = []
                est_abundances = []
                estimates = [[] for i in range(num_classes)]
                loss_values = []
                all_estimated_endmembers_rec = []
                all_estimated_endmembers_centers = []
                all_weights = []
                mse = 0
                arranged_mse = 0
                sads = np.zeros(total_pixels)
                
                
                
                                
                #%% solving done with batch_size = total_pixels
                
                
                batch_size = total_pixels
                
                if np.mod(total_pixels,batch_size) != 0:
                    raise Exception('choose batch size that can split "total_pixels"')
                    
                start_time = time.time()
                 
                for epoch in range(num_epochs):
                    
                    if (np.mod(epoch,10)==0):
                        print("Epoch: ", epoch)
                        
                    for x in range(0, total_pixels, batch_size):
                        x_inp = all_pixels[x : x + batch_size]
                        
                        loss_value, grads, inputs_reshaped, reconstruction, out_bottleneck = grad(model, x_inp)
                
                        optimizer.apply_gradients(zip(grads, model.trainable_variables))
                        
                        out_rec = np.squeeze(reconstruction)
                        reconstructed_pixels.append(out_rec)
                        weights = model.get_weights()
                        all_weights.append(weights)
                        estimated_endmembers_rec = weights[2] #estimated endmembers from final layer(reconstructing input)
                        estimated_endmembers_centers = weights[0] #estimated endmembers from kmeans centers
                        all_estimated_endmembers_rec.append(estimated_endmembers_rec)
                        all_estimated_endmembers_centers.append(estimated_endmembers_centers)
                        loss_values.append(np.squeeze(loss_value))
                        
                        
                        if epoch == num_epochs-1:
                            est_abundances.append(np.squeeze(out_bottleneck)) #this way adapts for varying batch sizes
                            
                    if np.mean(np.squeeze(loss_value)) <= 0.00004:
                        est_abundances.append(np.squeeze(out_bottleneck)) #determines that we reached convergence, and stops iterating
                        break
                
                print('elapsed time: ' +str(time.time()-start_time) )  
                timer += time.time()-start_time
                est_abundances = est_abundances[0]

                #%% rearranging for kmeans, since kmeans does not show centers in correct order
                ## first the abundances are compared with the original abundances and their columns are sorted accordingly,
                ## then the kmeans centers are rearranged in the same order
                ## measuring kmeans centers first is more problematic since high noise can mess it up
                
                
                maxes = np.zeros(num_classes)
                maxes_loc = np.zeros(num_classes)
                
                for i in range(num_classes):
                    
                    section = est_abundances[i*pixels_per_class:i*pixels_per_class+pixels_per_class,:]
                    section_avg = np.mean(section,axis=0)
                    maxes_loc[i] = np.where(section_avg==max(section_avg))[0][0]
                    maxes[i] = max(section_avg)
                
                maxes_loc = maxes_loc.astype(int)
                rearranged_est_abundances = np.zeros(est_abundances.shape)
                rearranged_estimated_endmembers_rec = np.zeros(estimated_endmembers_rec.shape)
                rearranged_estimated_endmembers_centers = np.zeros(estimated_endmembers_centers.shape)
                
                if len( set(list(range(num_classes))) - set(maxes_loc) ) == 0:
                
                    for i in range(num_classes):
                        
                        est_location = int(maxes_loc[i])
                        rearranged_est_abundances[:,i] = est_abundances[:,est_location]
                        rearranged_estimated_endmembers_rec[i,:] = estimated_endmembers_rec[est_location,:]
                        rearranged_estimated_endmembers_centers[i,:] = estimated_endmembers_centers[est_location,:]
                        
                else: 
                    
                    
                    unused_indices_unarranged = set(list(range(num_classes))) - set(maxes_loc)
                 
                    aa = np.asarray(list(Counter(list(maxes_loc)).items()))
                    sorting = np.argsort(aa[:,1]) #1st entry is column index, 2nd entry is # of instances
                    aa = aa[np.argsort(aa[:,1]),:]
                    
                    used_indices_arranged = set([])
                    for i in range(len(aa)):                
                        pair = aa[i]
                        
                        if pair[1] == 1:
                            assign_loc = np.where(maxes_loc==pair[0])[0][0]
                            rearranged_est_abundances[:,assign_loc] = est_abundances[:,pair[0]] 
                            rearranged_estimated_endmembers_rec[assign_loc,:] = estimated_endmembers_rec[pair[0],:] 
                            rearranged_estimated_endmembers_centers[assign_loc,:] = estimated_endmembers_centers[pair[0],:] 
                            used_indices_arranged.add(assign_loc)
                            
                        else:
                            assign_loc = np.where(maxes_loc==pair[0])[0][0]
                            rearranged_est_abundances[:,assign_loc] = est_abundances[:,pair[0]] 
                            rearranged_estimated_endmembers_rec[assign_loc,:] = estimated_endmembers_rec[pair[0],:] 
                            rearranged_estimated_endmembers_centers[assign_loc,:] = estimated_endmembers_centers[pair[0],:]
                            used_indices_arranged.add(assign_loc)
                            
                    unused_indices_arranged = set(list(range(num_classes))) - used_indices_arranged
                    unused_indices_arranged = list(unused_indices_arranged)
                    unused_indices_unarranged = list(unused_indices_unarranged)
                    
                    
                    for j in range(len(unused_indices_arranged)):
                        assign_loc = unused_indices_arranged[j]
                        assign_loc_unarranged = unused_indices_unarranged[j]
                        rearranged_est_abundances[:,assign_loc] = est_abundances[:,assign_loc_unarranged]
                        rearranged_estimated_endmembers_rec[assign_loc,:] = estimated_endmembers_rec[assign_loc_unarranged,:] 
                        rearranged_estimated_endmembers_centers[assign_loc,:] = estimated_endmembers_centers[assign_loc_unarranged,:] 
        
                est_abundances = rearranged_est_abundances
                estimated_endmembers_rec = rearranged_estimated_endmembers_rec
                estimated_endmembers_centers = rearranged_estimated_endmembers_centers
                
                #%% calculating rmse (abundance)
                
                
                all_percentages_array = np.array(all_percentages)   
                
                rmse = (np.sum((est_abundances - all_percentages_array)**2) / (total_pixels*num_classes) )**0.5
                # print('rmse :'+str(rmse))
                
                rmse_values[mix_model_index,dataset_index,SNR_index,all_index] = rmse
                
            
                
                #%% calculating sads(endmembers)
                
                sad = 0
                for i in range(num_classes):
                    sad += math.acos( sum(np.multiply(ref_pixels[i],estimated_endmembers_rec[i,:])) / ( np.linalg.norm(ref_pixels[i]) * np.linalg.norm(estimated_endmembers_rec[i,:]) ) )
                
                sads = sad/num_classes
                
                # print('sads (reconstructed est endmembers ) :'+str(sads))                
                all_sad_rec[mix_model_index,dataset_index,SNR_index,all_index] = sads
                
                sad = 0
                for i in range(num_classes):
                    sad += math.acos( sum(np.multiply(ref_pixels[i],estimated_endmembers_centers[i,:])) / ( np.linalg.norm(ref_pixels[i]) * np.linalg.norm(estimated_endmembers_centers[i,:]) ) )
                
                sads = sad/num_classes
                
                # print('sads (kernel center est endmembers ) :'+str(sads))                
                all_sad_cent[mix_model_index,dataset_index,SNR_index,all_index] = sads
                
                

                    
                    
print('total time(for autoencoder only): ' + str(timer))


rmse_values_mean = np.zeros(rmse_values.shape[0:3])
all_sad_rec_mean = np.zeros(rmse_values.shape[0:3])
all_sad_cent_mean = np.zeros(rmse_values.shape[0:3])

#%% if divergence occurs, this will find average while excluding the "nan" values
## i don't have divergence currently, but kept it nonetheless        

nan_counter = 0
if np.isnan(np.sum(rmse_values)):
    
    print('there are some nan values here')
    
    
    for i in range(rmse_values.shape[0]):
        
        for j in range(rmse_values.shape[1]):

            for k in range(rmse_values.shape[2]):
                ind = 0
                rmse_values_avg = 0
                all_sad_rec_avg = 0
                all_sad_cent_avg = 0
        
                for l in range(rmse_values.shape[3]):
                    if np.isnan(rmse_values[i,j,k,l])==False:
                        rmse_values_avg += rmse_values[i,j,k,l]
                        all_sad_rec_avg += all_sad_rec[i,j,k,l]
                        all_sad_cent_avg += all_sad_cent[i,j,k,l]                          
                        ind += 1
                    else: nan_counter += 1
                        
                rmse_values_mean[i,j,k] = rmse_values_avg/ind
                all_sad_rec_mean[i,j,k] = all_sad_rec_avg/ind
                all_sad_cent_mean[i,j,k] = all_sad_cent_avg/ind
            print('there are '+str(nan_counter)+' nan values found')

else:
    
    rmse_values_mean = np.mean(rmse_values,axis=3)
    all_sad_rec_mean = np.mean(all_sad_rec,axis=3)
    all_sad_cent_mean = np.mean(all_sad_cent,axis=3)

    


#%% saving results to a file

## I made the title to contain the hyperparameters' values, this makes it easier to tune the values

# title = 'all_results_kazi_' + str(len(mixing_models)) + '_mixmodels_' + str(iterations)  + '_iters_'+str(upto_how_many_degrees)+'_degrees.mat'
# with open(title, 'wb') as f:
#     pickle.dump([mixing_models, dataset_choices, rmse_values, all_sad_rec, all_sad_cent], f)

 
