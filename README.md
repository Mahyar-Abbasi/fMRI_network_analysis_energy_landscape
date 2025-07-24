# fMRI_network_analysis_phase_transition

In this research collaboration, under the supervision of Prof. Reza Jafari, we aimed to investigate network disorder
 and entropy in the dynamical network configurations of subjects with Parkinson’s disease and Autism (data sourced
 from open datasets including ABIDE). Utilizing phase transition theory and the Boltzmann Machine algorithm, we
 plotted phase diagrams based on brain states: normal and autistic. This investigation stems
 from the Critical Brain Hypothesis, which posits that the normal brain operates near a phase transition point. However, we could not demonstrate the existence of a critical point between the two states, as no significant difference in network energy was observed between the two groups.

fMRI Data Processing:
I utilized the nilearn package—an efficient library for fMRI atlas importing, filtering, and preprocessing. My analysis was conducted on the ABIDE dataset, which can be easily downloaded using simple nilearn-based scripts that I developed.
For network analysis, I employed the AAL atlas (Automated Anatomical Labeling brain atlas).

for more detailed discussion of energy landscape related to fMRI neural signals please view this article:
https://pubmed.ncbi.nlm.nih.gov/29410486/
