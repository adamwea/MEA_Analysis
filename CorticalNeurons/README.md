# Cortical Neurons - axonal electrical images
---

Extracellular electrical activity recorded from primary cortical neurons using CMOS-based HD-MEA system (MaxWell Biosystems AG). Datasets comprise electrophysiological recordings obtained from 50 cortical neurons. Recorded signals were up-sampled to 200 kHz following the Whitaker-Shannon interpolation formula. Recorded signals were spike-sorted and spike-trigger averaged across an entire array (26400 electrodes). Such signal representation is referred to as ‘axonal electrical image’. 


## File structure

Electrophysiological data obtained from individual neurons are stored in separate .mat files. Each file follows the same structure (myNeuron#.mat):

myNeuron.type ---- Neuron type (cortical / motor neuron)
myNeuron.temp ---- Averaged signals recorded from 26400 electrodes 
myNeuron.elcs ---- Electrode IDs (from 0 to 26400)
myNeuron.xpos ---- Electrode X-positions (µm)
myNeuron.ypos ---- Electrode Y-positions (µm)



## Code/Software

Python 3.7 and Matlab R2020a were used for data analysis.
Spyking-Circus algorithm (1) was used to spike-sort recorded signals.

1. Yger, P. et al. A spike sorting toolbox for up to thousands of electrodes validated with ground truth recordings in vitro and in vivo. Elife 7, e34518 (2018).