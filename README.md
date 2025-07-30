# AI electrode and stimulus design for Functional Eelectrical Stimulation research: a pilot study

[Siobhan Mackenzie Hall](https://github.com/smhall97), [Javier Sáez](https://github.com/javiersaez1) , [Panagiotis Tsakonas](https://github.com/PanosTsakonas), [Rodrigo Osorio L.](https://github.com/RodOsorio), [Francisco Saavedra](https://github.com/Fasaavedra), Pablo Aqueveque, James FitzGerald, Brian Andrews

This work is predicted to impact the field of functional electrical stimulation and electrode design by supporting iterative and fast exploratory phases that complement the SOTA numerical methods. We also foresee these being used in low resource settings (e.g. a computer in a clinical department) to be used by clinicians and physiotherapists in real-time to determine parameters suitable for a specific patient. We also hope to facilitate wider participation from multiple disciplines in the electrode design process.

This work explores the hypothesis that a neural network can be trained to predict whether or not an axon within a femoral nerve bundle will activate under a range of applied geometries and stimuli. Parameters include the axon depth below the skin surface, the electrode shape and configuration, and the specific stimulation protocol. We present the methods employed to investigate this:

1) Simulated Data Preparation: the data based on the state-of-the-art numerical methods used to train the surrogate FFNN
2) Training and running inference on the surrogate model


## Simulated Data Preparation - COMSOL LiveLink for MATLAB
This project integrates COMSOL Multiphysics® and MATLAB® using LiveLink™ to extract simulation data and prepare inputs for a Feedforward Neural Network (FFNN) to approximate voltage solutions in Finite Element Models (FEM) for Functional Electrical Stimulation (FES) research

## Files
- AI_surrogate.mlx: Main Live Script for running the data extraction pipeline
- *.txt: Tissue dielectric property tables
- thigh_test.mph: COMSOL model file
- Experiment details.xlsx: Spreadsheets containing the details of each iteration (geometry, current, pulse width, etc). This values must be declared as variables in the COMSOL model to iterate their values.

## Requirements
- MATLAB R2023b or later
- COMSOL Multiphysics 6.1 with LiveLink™ for MATLAB or later

## Workflow
1. **Model Initialization**
   - Loads a `.mph` model.

2. **Material Property**
   - Loads dielectric properties of biological tissues.

3. **Reads Spreadsheets values**
   - For each iteration extracts the parameters of the simulation.

4. **Simulation Loop**
   - For each geometry parameter:
     - Updates the geometry parameters in the COMSOL model.
   - For each `I_stim` value:
     - Updates the current parameter in the COMSOL model.
     - Solves the FEM problem via LiveLink.
     - Extracts voltage distribution data.

5. **MRG Model**
   - For the voltage extracted in each iteration the MRG is runned with different pulse width values.
   - The activation is labeled as 0 or 1.

## Training and running inference on the surrogate model

- Link to simulated data

- Link to train and run inference on the ANN - an ANN is trained on a single electrode configuration


