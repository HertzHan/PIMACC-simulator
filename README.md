# PIMACC

PIMACC is an end-to-end computing-in-memory (CIM) accuracy simulator, which evaluates the output accuracy of a neural network running on a CIM accelerator under the impact of non-ideal factors. PIMACC works with task/data mappings and instructions given by a CIM compiler. PIMACC is devoted to simulator non-ideal effects in CIM inference, such as quantization error, error of device resistance, IR-drop effects, etc. Error injections are flexible and can be configured in a hardware configuration file.

PIMACC is based on PIMCOMP-NN, a CIM compiler, whose link is given hear: https://github.com/sunxt99/PIMCOMP-NN. The code of PIMCOMP-NN is included in this package. To run PIMACC, you need to first compile your target NN network with PIMCOMP-NN, including front end and back end. Please refer to the user guide of PIMCOMP-NN to see how to compile and use PIMCOMP-NN. To make sure you can run PIMACC after compilation, use `–v=YES` in backend compilation of PIMCOMP-NN.

# Usage

## Requirements

In order to run the PIMACC verification program, you need to install the following python package.

- onnx
- onnxruntime
- numpy
- torch
- torchvision
- cv2

## Simulation
First compile PIMCOMP-NN. Please refer to the user guide of PIMCOMP-NN.

After PIMCOMP-NN is compiled, use it to compile your network model. This will generate a file storing the instructions.

Before running PIMACC, you need to make sure you have the same onnx model before compilation, and the compilation result in `PIMACC/output/VerificationInfo.json` after compilation.

```shell
cd PIMACC/verification/
python verification.py --model_path ../models/ONNX/resnet18.onnx --pipeline_type element --image_num 1000 --batchsize 10
```
Simuilator will run and give the accuracy result.

## Configaration
Non-ideal effects are configured in PIMACC/config.json, you can set values of different effects.
- "cell_precision": cell precision
- "reference_conductance_state": the conductance of low-resistance-state
- "R_ratio": on-off ratio, the ratio of high and low resistances
- "bitline_resistance": wire resistance between two devices on bitline
- "wordline_resistance":wire resistance between two devices on wordline
- "variation": device resistance variation, in form of standard deviation
- "SAF_flag": whether to consider SAF fault
- "p_SA0": probability of SA0
- "p_SA1": probability of SA1

