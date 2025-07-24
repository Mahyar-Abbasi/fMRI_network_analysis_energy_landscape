# fMRI_network_analysis_phase_transition

In this research collaboration, under the supervision of Prof. Reza Jafari, we aimed to investigate network disorder
 and entropy in the dynamical network configurations of subjects with Parkinson’s disease and Autism (data sourced
 from open datasets including ABIDE). Utilizing phase transition theory and the Boltzmann Machine algorithm, we
 plotted phase diagrams based on brain states: normal and autistic. This investigation stems
 from the Critical Brain Hypothesis, which posits that the normal brain operates near a phase transition point. However, we could not show that there is a critical point between these two states as we have not captured a significant difference between network energy between the two groups.

# fMRI data processing:
I've made use of "nilearn" package, an efficient library for fMRI atlas importing, filtering and preprocessing. I did my analysis on ABIDE dataset which can be easily downloaded thorugh nilearn simple codes that I wrote in codes.
AAL atlas(automated anatomical labeling brain atlas) has been taken for network analysis. 
